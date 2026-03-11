import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import math
import numpy as np
from multiprocessing import Pool, cpu_count
import json

class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_withoutDiff(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LLM_Diff_interaction_as_condition_mask_both_tag_EMA_withoutDiff, self).__init__(config, dataloader)

        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.diff_weight = config['diff_weight']
        self.bpr_loss = BPRLoss()

        self.emb_user = nn.Embedding(
            self.num_users_src + self.num_users_tgt - self.num_users_overlap + 1,
            self.embedding_dim,
            padding_idx=0
        )
        self.emb_item_src = nn.Embedding(self.num_items_src + 1, self.embedding_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1, self.embedding_dim, padding_idx=0)

        # 用户/物品text emb
        semantic_emb = dataloader.get_modality_embs()['CrossDomain_semantics_both_tag']
        semantic_emb = torch.from_numpy(semantic_emb).float()  # [U + Is + It, D]

        # 检查text emb数量是否正确
        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        num_src_items = self.num_items_src
        num_tgt_items = self.num_items_tgt
        assert semantic_emb.shape[0] == num_users + num_src_items + num_tgt_items, \
            "Semantic embedding size mismatch."

        user_semantic_emb = semantic_emb[:num_users]
        src_item_semantic_emb = semantic_emb[num_users: num_users + num_src_items]
        tgt_item_semantic_emb = semantic_emb[num_users + num_src_items:]
        pad_user = torch.zeros(1, user_semantic_emb.shape[1])
        pad_src_item = torch.zeros(1, src_item_semantic_emb.shape[1])
        pad_tgt_item = torch.zeros(1, tgt_item_semantic_emb.shape[1])
        user_semantic_emb = torch.cat([pad_user, user_semantic_emb], dim=0)
        src_item_semantic_emb = torch.cat([pad_src_item, src_item_semantic_emb], dim=0)
        tgt_item_semantic_emb = torch.cat([pad_tgt_item, tgt_item_semantic_emb], dim=0)

        self.register_buffer("user_text_emb", user_semantic_emb)
        self.register_buffer("src_item_text_emb", src_item_semantic_emb)
        self.register_buffer("tgt_item_text_emb", tgt_item_semantic_emb)

        # 构建伪交互字典，一个用户 -> 一个tgt物品set
        self.src_only_pseudo_pos = self._build_src_only_pseudo_positives()
        # 构建伪交互采样池
        self.pseudo_ui_pairs = []
        for u, items in self.src_only_pseudo_pos.items():
            for i in items:
                self.pseudo_ui_pairs.append((u, i))

        # 两个扩散模型，分别用于构建src/tgt适配的用户兴趣
        # self.diff_src = Diffusion(config)
        # self.diff_tgt = Diffusion(config)
        self.diff_src = MLPBackbone(config)
        self.diff_tgt = MLPBackbone(config)

        # EMA
        self.diff_tgt_ema = None
        self.ema_initialized = False
        self.global_step = 0

        # 两个聚合器，分别聚合用户的src、tgt交互的物品emb
        self.src_interaction_agg = InteractionAggregator(config)
        self.tgt_interaction_agg = InteractionAggregator(config)

        # 构造用户交互历史
        (
            self.history_src_user_src,
            self.history_src_user_tgt,
            self.history_tgt_user_src,
            self.history_tgt_user_tgt
        ) = self._build_padded_history(dataloader)

        self.apply(xavier_uniform_initialization)
        self.emb_user.weight.data[0, :] = 0
        self.emb_item_src.weight.data[0, :] = 0
        self.emb_item_tgt.weight.data[0, :] = 0

    @torch.no_grad()
    def _build_src_only_pseudo_positives(self):
        top_k = self.config["top_k_pos"]

        pseudo_pos = {}

        # 1. 取 src-only 用户 text emb
        u_start = self.num_users_overlap + 1
        u_end = self.num_users_src + 1
        user_emb = self.user_text_emb[u_start:u_end]  # [U_s, D]

        # 2. tgt item emb（去掉 padding 0）
        item_emb = self.tgt_item_text_emb[1:]  # [I_t, D]

        # 3. 相似度矩阵
        # user_emb = F.normalize(user_emb, dim=1)
        # item_emb = F.normalize(item_emb, dim=1)
        sim = torch.matmul(user_emb, item_emb.T)  # [U_s, I_t]

        # 4. Top-K
        topk_vals, topk_idx = torch.topk(sim, k=top_k, dim=1)

        # 5. 构造 dict（注意 item id +1）
        for i, u_global in enumerate(range(u_start, u_end)):
            pseudo_items = (topk_idx[i] + 1).tolist()
            pseudo_pos[u_global] = set(pseudo_items)

        return pseudo_pos

    def _build_padded_history(self, dataloader):
        L = int(self.config["history_len"])

        # ---------- src user space ----------
        history_src_user_src = torch.zeros(
            (self.num_users_src + 1, L), dtype=torch.long, device=self.device
        )
        history_src_user_tgt = torch.zeros(
            (self.num_users_src + 1, L), dtype=torch.long, device=self.device
        )

        # ---------- tgt user space ----------
        history_tgt_user_src = torch.zeros(
            (self.num_users_tgt + 1, L), dtype=torch.long, device=self.device
        )
        history_tgt_user_tgt = torch.zeros(
            (self.num_users_tgt + 1, L), dtype=torch.long, device=self.device
        )

        # ===== src domain interactions =====
        for u, items in dataloader.dataset.positive_items_src.items():
            if not items:
                continue
            items = list(items)[-L:]
            history_src_user_src[u, :len(items)] = torch.tensor(items, device=self.device)

            # overlap users: also visible in tgt-user space
            if u <= self.num_users_overlap:
                history_tgt_user_src[u, :len(items)] = torch.tensor(items, device=self.device)

        # ===== tgt domain interactions =====
        for u, items in dataloader.dataset.positive_items_tgt.items():
            if not items:
                continue
            items = list(items)[-L:]
            history_tgt_user_tgt[u, :len(items)] = torch.tensor(items, device=self.device)

            # overlap users: also visible in src-user space
            if u <= self.num_users_overlap:
                history_src_user_tgt[u, :len(items)] = torch.tensor(items, device=self.device)

        return (
            history_src_user_src,
            history_src_user_tgt,
            history_tgt_user_src,
            history_tgt_user_tgt
        )

    def batch_random_mask(self, seq: torch.Tensor, mask_rate: float, min_keep: int = 1):
        """
        seq: [B, L] item ids, 0 is padding
        return: masked_seq [B, L] with some non-zero positions set to 0
        """
        # 按比例mask_rate，mask掉用户部分交互的物品
        if mask_rate <= 0:
            return seq

        # valid positions
        valid = seq != 0  # [B, L]
        valid_cnt = valid.sum(dim=1)  # [B]

        # keep at least min_keep (and at least 1 if you want)
        keep_cnt = (valid_cnt.float() * (1 - mask_rate)).long()
        keep_cnt = torch.clamp(keep_cnt, min=min_keep)
        # also cannot exceed valid_cnt
        keep_cnt = torch.minimum(keep_cnt, valid_cnt)

        # random score per position, invalid positions get large so they go to the end
        rand = torch.rand_like(seq.float())  # [B, L]
        rand = rand.masked_fill(~valid, 2.0)

        # smaller rand = kept (top-k smallest)
        order = rand.argsort(dim=1)  # [B, L]
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        keep_mask_in_order = pos < keep_cnt.unsqueeze(1)  # [B, L]

        keep_mask = torch.zeros_like(valid)
        keep_mask.scatter_(1, order, keep_mask_in_order)  # [B, L] True means keep

        out = seq.clone()
        out[~keep_mask] = 0
        return out

    def sample_tgt_neg_excluding_pseudo(self, users, num_items_tgt, max_retry=5):
        neg_items = []
        users_list = users.tolist()

        for u in users_list:
            forbid = self.src_only_pseudo_pos.get(u, None) #这个src-only用户的伪正tgt物品set

            neg = None
            for _ in range(max_retry):
                cand = torch.randint(
                    low=1,
                    high=num_items_tgt + 1,
                    size=(1,),
                    device=self.device
                ).item()
                if cand not in forbid:
                    neg = cand
                    break
            # 极端兜底（几乎不会发生）
            if neg is None:
                neg = cand
            neg_items.append(neg)

        return torch.tensor(neg_items, device=self.device)

    def pre_epoch_processing(self, epoch_idx):
        # 一开始不初始化EMA，只有在伪交互构建完成后，才初始化EMA_diff
        if epoch_idx >= self.config["pseudo_start_epoch"]:
            if not self.ema_initialized:
                import copy
                self.diff_tgt_ema = copy.deepcopy(self.diff_tgt)
                for p in self.diff_tgt_ema.parameters():
                    p.requires_grad = False
                self.ema_initialized = True

    def post_batch_processing(self, epoch_idx, batch_idx):
        # 每个batch的diff更新完成后，再更新EMA_diff
        if epoch_idx >= self.config["pseudo_start_epoch"]:
            self._momentum_update()
            self.global_step += 1

    @torch.no_grad()
    def _momentum_update(self):
        # EMA 更新方法
        if self.diff_tgt_ema is None:
            return
        m = self.config["ema_momentum"]
        for p, p_ema in zip(self.diff_tgt.parameters(),
                            self.diff_tgt_ema.parameters()):
            p_ema.data.mul_(m).add_(p.data, alpha=1 - m)

    @torch.no_grad()
    def rebuild_src_only_pseudo_with_ema(self):
        assert self.diff_tgt_ema is not None

        top_k = self.config["top_k_pos"]
        batch_size = 2048   # EMA的batch size

        u_start = self.num_users_overlap + 1
        u_end = self.num_users_src + 1

        item_emb = self.emb_item_tgt.weight[1:]  # [I_t, D]

        new_pseudo_pos = {}
        new_pairs = []
        for s in range(u_start, u_end, batch_size):
            e = min(s + batch_size, u_end)
            users = torch.arange(s, e, device=self.device)  # 小 batch

            u = self.emb_user(users)

            hist_src_items = self.history_src_user_src[users]
            hist_tgt_items = self.history_src_user_tgt[users]
            hist_src = self.emb_item_src(hist_src_items)
            hist_tgt = self.emb_item_tgt(hist_tgt_items)
            cond_src = self.src_interaction_agg(hist_src, u)
            cond_tgt = self.tgt_interaction_agg(hist_tgt, u)

            u_denoised = self.diff_tgt_ema(u, cond_src, cond_tgt)

            u_final = u + self.config["lambda_user_emb"] * u_denoised

            sim = torch.matmul(u_final, item_emb.T)
            topk_idx = torch.topk(sim, k=top_k, dim=1).indices + 1

            for i, uid in enumerate(range(s, e)):
                items = set(topk_idx[i].tolist())
                new_pseudo_pos[uid] = items
                for it in items:
                    new_pairs.append((uid, it))

        self.src_only_pseudo_pos = new_pseudo_pos
        self.pseudo_ui_pairs = new_pairs

    def calculate_loss(self, interaction, epoch_idx):
        # 根据当前Epoch，判断是否需要用伪交互
        use_pseudo = epoch_idx >= self.config["pseudo_start_epoch"]

        # 如果EMA已经更新了pseudo_update_interval轮，就用EMA_diff更新伪交互集合
        if (
                epoch_idx >= self.config["pseudo_start_epoch"]
                and self.diff_tgt_ema is not None
                and self.global_step > 0
                and self.global_step % self.config["pseudo_update_interval"] == 0
        ):
            self.rebuild_src_only_pseudo_with_ema()

        users_src = interaction['users_src']
        pos_items_src = interaction['pos_items_src']
        neg_items_src = interaction['neg_items_src']
        users_tgt = interaction['users_tgt']
        pos_items_tgt = interaction['pos_items_tgt']
        neg_items_tgt = interaction['neg_items_tgt']

        # src loss，包括rec和diff
        u_src = self.emb_user(users_src)  # [B, D]
        i_pos_src = self.emb_item_src(pos_items_src)  # [B, D]
        i_neg_src = self.emb_item_src(neg_items_src)  # [B, D]

        # 聚合src用户的src和tgt交互
        hist_src_items = self.history_src_user_src[users_src]  # [B, L]
        hist_tgt_items = self.history_src_user_tgt[users_src]  # [B, L]
        hist_src_items = self.batch_random_mask(hist_src_items, self.config['mask_rate'], min_keep=1)
        hist_tgt_items = self.batch_random_mask(hist_tgt_items, self.config['mask_rate'], min_keep=1)
        hist_src = self.emb_item_src(hist_src_items)  # [B, L, D]
        hist_tgt = self.emb_item_tgt(hist_tgt_items)  # [B, L, D]
        cond_src = self.src_interaction_agg(hist_src, u_src)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_src)


        # 这里t是随机采的，不是对称采样
        u_src_denoised = self.diff_src(u_src, cond_src, cond_tgt)
        diff_loss_src = torch.tensor(0.0, device=u_src.device)
        u_src_final = u_src + self.config["lambda_user_emb"] * u_src_denoised # 残差连接

        pos_score_src = (u_src_final * i_pos_src).sum(dim=-1)
        neg_score_src = (u_src_final * i_neg_src).sum(dim=-1)
        bpr_loss_src = self.bpr_loss(pos_score_src, neg_score_src)

        # tgt loss，包括rec和diff
        users_tgt_local = users_tgt
        offset = self.num_users_src - self.num_users_overlap
        users_tgt_global = users_tgt_local + (users_tgt_local > self.num_users_overlap).long() * offset # tgt单域用户需要加一个偏移值，从而取到正确的id emb
        u_tgt = self.emb_user(users_tgt_global)  # [B, D]
        i_pos_tgt = self.emb_item_tgt(pos_items_tgt)  # [B, D]
        i_neg_tgt = self.emb_item_tgt(neg_items_tgt)  # [B, D]

        # 聚合tgt用户的src和tgt交互
        hist_src_items = self.history_tgt_user_src[users_tgt]  # [B, L]
        hist_tgt_items = self.history_tgt_user_tgt[users_tgt]  # [B, L]
        hist_src_items = self.batch_random_mask(hist_src_items, self.config['mask_rate'], min_keep=1)
        hist_tgt_items = self.batch_random_mask(hist_tgt_items, self.config['mask_rate'], min_keep=1)
        hist_src = self.emb_item_src(hist_src_items)
        hist_tgt = self.emb_item_tgt(hist_tgt_items)
        cond_src = self.src_interaction_agg(hist_src, u_tgt)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_tgt)

        B = u_tgt.size(0)
        u_tgt_denoised = self.diff_tgt(u_tgt, cond_src, cond_tgt)
        diff_loss_tgt = torch.tensor(0.0, device=u_src.device)
        u_tgt_final = u_tgt + self.config["lambda_user_emb"] * u_tgt_denoised # 残差连接

        pos_score_tgt = (u_tgt_final * i_pos_tgt).sum(dim=-1)
        neg_score_tgt = (u_tgt_final * i_neg_tgt).sum(dim=-1)
        bpr_loss_tgt = self.bpr_loss(pos_score_tgt, neg_score_tgt)

        # 伪交互 Loss
        pseudo_loss = 0.0
        if use_pseudo:
            num_pseudo = self.config["pseudo_num_per_batch"]

            # 随机采伪交互
            idx = torch.randint(
                low=0,
                high=len(self.pseudo_ui_pairs),
                size=(num_pseudo,),
                device=self.device
            )
            u_list = []
            i_list = []
            for j in idx.tolist():
                u, i = self.pseudo_ui_pairs[j]
                u_list.append(u)
                i_list.append(i)
            pseudo_users = torch.tensor(u_list, device=self.device)
            pseudo_items = torch.tensor(i_list, device=self.device)
            # 给src-only用户负采样tgt域物品
            neg_items = self.sample_tgt_neg_excluding_pseudo(
                pseudo_users,
                self.num_items_tgt
            )

            u_pseudo = self.emb_user(pseudo_users)
            # 取 pseudo_users 的 src/tgt 历史（src-user space）
            hist_src_items = self.history_src_user_src[pseudo_users]  # [Bp, L]
            hist_tgt_items = self.history_src_user_tgt[pseudo_users]  # [Bp, L]
            hist_src_items = self.batch_random_mask(hist_src_items, self.config['mask_rate'], min_keep=1)
            hist_tgt_items = self.batch_random_mask(hist_tgt_items, self.config['mask_rate'], min_keep=1)
            hist_src = self.emb_item_src(hist_src_items)  # [Bp, L, D]
            hist_tgt = self.emb_item_tgt(hist_tgt_items)  # [Bp, L, D]
            cond_src = self.src_interaction_agg(hist_src, u_pseudo)  # [Bp, D]
            cond_tgt = self.tgt_interaction_agg(hist_tgt, u_pseudo)  # [Bp, D]
            Bp = u_pseudo.size(0)
            u_pseudo_denoised = self.diff_tgt(u_pseudo, cond_src, cond_tgt)
            u_pseudo_final = u_pseudo + self.config["lambda_user_emb"] * u_pseudo_denoised  # 残差连接

            i_emb = self.emb_item_tgt(pseudo_items)  # [Bp, D]
            neg_emb = self.emb_item_tgt(neg_items)

            pos_score = (u_pseudo_final * i_emb).sum(dim=-1)
            neg_score = (u_pseudo_final * neg_emb).sum(dim=-1)
            pseudo_loss = self.bpr_loss(pos_score, neg_score)

        # loss = loss_rec + loss_dif + loss_pseudo
        loss = bpr_loss_src + bpr_loss_tgt
        if use_pseudo:
            loss = loss + self.config["pseudo_rec_weight"] * pseudo_loss
        return loss

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()  # [B]
        device = users.device

        if is_warm:
            # 目标域用户offset
            offset = self.num_users_src - self.num_users_overlap
            users_global = users + (users > self.num_users_overlap).long() * offset
            u = self.emb_user(users_global)

            hist_src = self.emb_item_src(self.history_tgt_user_src[users])  # [B, L, D]
            hist_tgt = self.emb_item_tgt(self.history_tgt_user_tgt[users])  # [B, L, D]
        else:
            u = self.emb_user(users)

            hist_src = self.emb_item_src(self.history_src_user_src[users])  # [B, L, D]
            hist_tgt = self.emb_item_tgt(self.history_src_user_tgt[users])  # [B, L, D]

        cond_src = self.src_interaction_agg(hist_src, u)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u)
        u_denoised = self.diff_tgt(u, cond_src, cond_tgt)

        u_final = u + self.config["lambda_user_emb"] * u_denoised

        item_emb = self.emb_item_tgt.weight
        scores = torch.matmul(u_final, item_emb.t())
        scores[:, 0] = 0.0
        return scores


# 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂 扩散模型相关 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(
        - beta_min / timesteps
        - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps)
    )
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """
    a: [T]
    t: [B]  (same device as a)
    return: [B, 1, 1, ...] broadcastable to x_shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = int(config['timesteps'])
        self.beta_start = float(config['beta_start'])
        self.beta_end = float(config['beta_end'])
        self.embedding_dim = int(config['embedding_dim'])
        self.beta_sche = config['beta_sche']

        if self.beta_sche == 'linear':
            betas = linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.beta_sche == 'exp':
            betas = exp_beta_schedule(self.timesteps)
        elif self.beta_sche == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        elif self.beta_sche == 'sqrt':
            betas = betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 1e-4))
        else:
            raise ValueError(f"Unknown beta_sche: {self.beta_sche}")

        self.register_buffer("betas", betas.float())
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("posterior_variance", posterior_variance)

        self.w_q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.w_k = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.w_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_q); init(self.w_k); init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)

    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def selfAttention(self, features: torch.Tensor):
        mask = (features.abs().sum(dim=-1) > 0) # [B, N]

        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)
        attn_logits = (q * (self.embedding_dim ** -0.5)) @ k.transpose(-1, -2)  # [B, N, N]
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = attn_logits.softmax(dim=-1)  # [B, N, N]
        out = attn @ v  # [B, N, D]
        out = out * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return out.sum(dim=1) / denom

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t = t.to(x_start.device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond_src=None, cond_tgt=None, loss_type="l2"):
        device = x_start.device
        t = t.to(device)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)  # [B, D] on device
        tokens = torch.stack([x_noisy, t_emb, cond_src, cond_tgt], dim=1) # [B, 4, D]
        predicted_x0 = self.selfAttention(tokens)

        if loss_type == "l2":
            loss = F.mse_loss(predicted_x0, x_start)
        elif loss_type == "l1":
            loss = F.l1_loss(predicted_x0, x_start)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_x0, x_start)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        return loss, predicted_x0

    @torch.no_grad()
    def p_sample(self, x_t, t, t_index, cond_src, cond_tgt):
        device = x_t.device
        t = t.to(device)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)
        tokens = torch.stack([x_t, t_emb, cond_src, cond_tgt], dim=1)
        x_start = self.selfAttention(tokens)

        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean

        var = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, x_start, cond_src, cond_tgt):
        device = x_start.device

        noise_x = torch.randn_like(x_start)
        t_init = torch.full(
            (x_start.shape[0],),
            self.timesteps - 1,
            dtype=torch.long,
            device=device
        )
        x_t = self.q_sample(x_start=x_start, t=t_init, noise=noise_x)

        x_quarter = x_t
        x_half = x_t
        x_three_quarter = x_t

        for n in reversed(range(self.timesteps)):
            t = torch.full((x_t.shape[0],), n, dtype=torch.long, device=device)
            x_t = self.p_sample(x_t=x_t, t=t, t_index=n, cond_src=cond_src, cond_tgt=cond_tgt)

            if n == int((self.timesteps - 1) * 0.75):
                x_quarter = x_t
            if n == int((self.timesteps - 1) * 0.5):
                x_half = x_t
            if n == int((self.timesteps - 1) * 0.25):
                x_three_quarter = x_t

        return x_start, x_t, x_quarter, x_half, x_three_quarter

class MLPBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['embedding_dim']

        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, u, cond_src, cond_tgt):
        x = torch.cat([u, cond_src, cond_tgt], dim=-1)
        return self.net(x)

# 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂 交互物品emb聚合器 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂

class InteractionAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.aggregator = config['aggregator']
        self.embedding_dim = config['embedding_dim']
        dropout_rate = config['dropout']

        self.W_agg = nn.Linear(config['embedding_dim'], config['embedding_dim'], bias=False)

        if self.aggregator in ["user_attention"]:
            self.W_att = nn.Sequential(
                nn.Linear(config['embedding_dim'], config['embedding_dim']),
                nn.Tanh()
            )
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None


    def forward(self, item_emb, user_emb=None):
        """
        item_emb: [B, L, D]
        user_emb: [B, D] (required for user_attention)
        return:   [B, D]
        """
        # padding mask
        mask = (item_emb.abs().sum(dim=-1) > 0)  # [B, L]

        if self.aggregator == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (item_emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            return self.W_agg(pooled)

        elif self.aggregator == "user_attention":
            assert user_emb is not None

            key = self.W_att(item_emb)                     # [B, L, D]
            att = torch.bmm(key, user_emb.unsqueeze(-1))   # [B, L, 1]
            att = att.squeeze(-1)                          # [B, L]
            att = att.masked_fill(~mask, -1e9)
            att = torch.softmax(att, dim=1)

            if self.dropout is not None:
                att = self.dropout(att)

            pooled = torch.bmm(att.unsqueeze(1), item_emb).squeeze(1)  # [B, D]
            return self.W_agg(pooled)

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

# 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂 用户/物品 text 处理 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂



def _extract_item_profile(meta: dict) -> dict:
    """
    负责从 metadata 里抽取一个‘适合给 LLM 看’的 item 描述
    缺失字段置为"None"
    """
    if meta is None:
        return {
            "title": "None",
            "description": "None",
            "categories": "None",
            "price": "None",
            "brand": "None",
        }

    # title
    title = meta.get("title")
    title = title if isinstance(title, str) and title.strip() else "None"

    # description
    description = meta.get("description")
    description = description if isinstance(description, str) and description.strip() else "None"

    # categories (flatten + deduplicate)
    raw_categories = meta.get("categories")
    categories = []
    if isinstance(raw_categories, list):
        for path in raw_categories:
            if isinstance(path, list):
                categories.extend([c for c in path if isinstance(c, str) and c.strip()])
            elif isinstance(path, str) and path.strip():
                categories.append(path)
    categories = list(dict.fromkeys(categories))  # deduplicate, keep order
    categories = ", ".join(categories) if categories else "None"

    # price
    price = meta.get("price")
    price = str(price) if isinstance(price, (int, float)) else "None"

    # brand
    brand = meta.get("brand")
    brand = brand if isinstance(brand, str) and brand.strip() else "None"

    return {
        "title": title, # String / "None" (都是字符串)
        "description": description, # String / "None" (都是字符串)
        "categories": categories, # String / "None" (都是字符串)
        "price": price, # String / "None" (都是字符串)
        "brand": brand, # String / "None" (都是字符串)
    }


def _normalize_review(review):
    """
    LLM 输入里永远有 review 字段
    """
    if isinstance(review, str) and review.strip():
        return review
    return "None"


def _collect_domain_user_items(
    interaction_df,
    domain: str,
    id_mapping: dict,
    metadata: dict,
    reviews: dict,
    user_item_dict: dict,
):
    """
    Collect item profiles + reviews for one domain (src or tgt).
    """
    for row in interaction_df.itertuples(index=False):
        user_id = row.user
        item_id = row.item

        raw_user_id = id_mapping[domain]["id2user"][user_id]
        raw_item_id = id_mapping[domain]["id2item"][item_id]

        if raw_user_id not in user_item_dict:
            user_item_dict[raw_user_id] = {"src": [], "tgt": []}

        meta = metadata[domain].get(raw_item_id)
        review = reviews[domain].get((raw_user_id, raw_item_id))

        item_profile = _extract_item_profile(meta)
        item_profile["review"] = _normalize_review(review)

        user_item_dict[raw_user_id][domain].append(item_profile)


def _collect_domain_items(
    domain: str,
    id_mapping: dict,
    metadata: dict,
    item_profile_dict: dict,
):
    id2item = id_mapping[domain]["id2item"]  # list, index 0 is padding

    # iterate all items in mapping (skip padding idx 0)
    for item_id in range(1, len(id2item)):
        raw_item_id = id2item[item_id]

        if raw_item_id in item_profile_dict:
            continue

        meta = metadata[domain].get(raw_item_id)
        item_profile_dict[raw_item_id] = _extract_item_profile(meta)


def _user_domain_items_to_string(user_profile):
    """
    Convert a user's item-level profile (list[dict[str, str]])
    into a JSON-like string.

    Contract:
    - item is dict[str, str]
    - all values are already strings (including "None")
    - this function only does formatting + minimal escaping
    """
    if not user_profile:
        return "[]"

    ordered_keys = ["title", "description", "categories", "price", "brand", "review"]

    lines = ["["]
    for item in user_profile:
        if not isinstance(item, dict):
            continue

        fields = []
        for k in ordered_keys:
            v = item[k]  # 这里假设一定存在、一定是 str
            v_escaped = v.replace("\\", "\\\\").replace("\"", "\\\"")
            fields.append(f"\"{k}\": \"{v_escaped}\"")

        item_str = "{ " + ", ".join(fields) + " }"
        lines.append(item_str)

    lines.append("]")
    return "\n".join(lines)


def _user_item_profile_to_string(user_profile: dict) -> dict:
    """
    Convert one user's cross-domain item profiles to strings.

    Returns:
        {
            "src": "<string>",
            "tgt": "<string>"
        }
    """
    return {
        "src": _user_domain_items_to_string(user_profile["src"]),
        "tgt": _user_domain_items_to_string(user_profile["tgt"]),
    }


def _build_user_prompt_string(user_profile_string: dict) -> str:
    """
    Build a single prompt string for one user.

    Input:
        user_profile_string = {
            "src": "<JSON-like string>",
            "tgt": "<JSON-like string>"
        }

    Output:
        prompt_str (str)
    """
    return (
        "INTERACTIONS FROM ELECTRONICS:\n"
        f"{user_profile_string['src']}\n\n"
        "INTERACTIONS FROM PHONES:\n"
        f"{user_profile_string['tgt']}"
    )


def _build_all_user_prompt_strings(user_profile_strings: dict) -> dict:
    """
    Build prompt strings for all users.

    Returns:
        { raw_user_id: prompt_string }
    """
    user_prompts = {}

    for user_id, profile_string in user_profile_strings.items():
        user_prompts[user_id] = _build_user_prompt_string(profile_string)

    return user_prompts


def _build_item_prompt_string(item_profile: dict) -> str:
    """
    Build user prompt for ONE item.
    """
    return json.dumps(item_profile, indent=2)


def _extract_string_list(text: str):
    """
    Try to extract a List[str] from LLM output.
    Accepts JSON-style or Python-style lists.
    Returns list[str] or None.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()

    # fast path: JSON list
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, str) for x in obj):
            return obj
    except Exception:
        pass

    # fallback: Python literal list
    try:
        import ast
        obj = ast.literal_eval(text)
        if isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, str) for x in obj):
            return obj
    except Exception:
        pass

    return None


def _process_single_user(args):
    """
    Worker function for one user.
    Each user has at most max_retry attempts.
    """
    user_id, prompt, system_prompt, max_retry = args

    from openai import OpenAI
    client = OpenAI()

    last_response_text = None

    for _ in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.2,
                # max_tokens=512,
                # extra_body={
                #     "chat_template_kwargs": {"enable_thinking": False},
                # },
            )

            text = response.choices[0].message.content.strip()
            last_response_text = text

            tags = _extract_string_list(text)
            if tags is not None:
                return {
                    "user_id": user_id,
                    "summary": tags,
                    "success": True,
                }

        except Exception as e:
            last_response_text = str(e)

    return {
        "user_id": user_id,
        "summary": last_response_text,
        "success": False,
    }



def _run_user_summarization_multiprocess(
    user_prompts: dict,
    system_prompt: str,
    max_retry: int = 2,
    num_workers: int = None,
):
    """
    Run LLM summarization for all users using multiprocessing.
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    tasks = [
        (user_id, prompt, system_prompt, max_retry)
        for user_id, prompt in user_prompts.items()
    ]

    results = {}
    error_users = {}

    with Pool(processes=num_workers) as pool:
        for out in pool.imap_unordered(_process_single_user, tasks):
            user_id = out["user_id"]
            if out["success"]:
                results[user_id] = out["summary"]
            else:
                error_users[user_id] = out["summary"]

    return results, error_users


def extract_CrossDomain_semantics_both_tag_modality_data(config, modality, interaction, id_mapping, raw_data_list):
    if config['dataset'] == "Amazon2014":
        reviews, metadata = raw_data_list
    elif config['dataset'] == "Douban":
        reviews = raw_data_list[0]
        metadata = raw_data_list[2]

    """
    Step 1:
    Collect per-user cross-domain item profiles + reviews.
    """
    # 分别收集用户在每个域 {交互过的物品的meta信息 + 用户评论信息}
    user_item_profiles = {}

    _collect_domain_user_items(
        interaction_df=interaction["src"],
        domain="src",
        id_mapping=id_mapping,
        metadata=metadata,
        reviews=reviews,
        user_item_dict=user_item_profiles,
    )
    _collect_domain_user_items(
        interaction_df=interaction["tgt"],
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        reviews=reviews,
        user_item_dict=user_item_profiles,
    )

    # 分别收集两个域的物品的metadata
    item_profiles_src = {}
    item_profiles_tgt = {}

    _collect_domain_items(
        domain="src",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_src,
    )

    _collect_domain_items(
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_tgt,
    )

    # 检查user_item_profiles中的src用户数量、tgt用户数量、重叠用户数量是否正确
    profile_users = set(user_item_profiles.keys()) # user_item_profiles 中的用户
    src_users = set(id_mapping["src"]["id2user"][1:])  # src domain 的所有 raw 用户
    tgt_users = set(id_mapping["tgt"]["id2user"][1:])  # tgt domain 的所有 raw 用户
    profile_src_users = {u for u, v in user_item_profiles.items() if len(v["src"]) > 0} # 在 profile 中有src交互的用户
    profile_tgt_users = {u for u, v in user_item_profiles.items() if len(v["tgt"]) > 0} # 在 profile 中有tgt交互的用户
    mapping_overlap_users = src_users & tgt_users # id_mapping定义下的重叠用户
    profile_overlap_users = profile_src_users & profile_tgt_users # user_item_profiles定义下的重叠用户
    assert profile_users.issubset(src_users | tgt_users), \
        "user_item_profiles contains users not in src or tgt id_mapping"
    assert profile_src_users == src_users, \
        f"Mismatch in src users: profile={len(profile_src_users)}, mapping={len(src_users)}"
    assert profile_tgt_users == tgt_users, \
        f"Mismatch in tgt users: profile={len(profile_tgt_users)}, mapping={len(tgt_users)}"
    assert profile_overlap_users == mapping_overlap_users, \
        f"Mismatch in overlap users: profile={len(profile_overlap_users)}, mapping={len(mapping_overlap_users)}"

    # 检查item_profiles_src、item_profiles_tgt中的物品数量是否正确
    src_items = set(id_mapping["src"]["id2item"][1:])
    tgt_items = set(id_mapping["tgt"]["id2item"][1:])
    profile_src_items = set(item_profiles_src.keys())
    profile_tgt_items = set(item_profiles_tgt.keys())
    assert profile_src_items == src_items, \
        f"Mismatch in src items: profile={len(profile_src_items)}, mapping={len(src_items)}"
    assert profile_tgt_items == tgt_items, \
        f"Mismatch in tgt items: profile={len(profile_tgt_items)}, mapping={len(tgt_items)}"

    # 处理user_item_profiles中每个用户每个域的交互List，把它变成一个字符串
    user_profile_strings = {
        user_id: _user_item_profile_to_string(profile)
        for user_id, profile in user_item_profiles.items()
    }

    # 处理item_prompts_src中每个物品的metadata，变成一个字符串
    item_prompts_src = {
        item_id: _build_item_prompt_string(profile)
        for item_id, profile in item_profiles_src.items()
    }

    # 处理item_prompts_tgt中每个物品的metadata，变成一个字符串
    item_prompts_tgt = {
        item_id: _build_item_prompt_string(profile)
        for item_id, profile in item_profiles_tgt.items()
    }

    user_prompts = _build_all_user_prompt_strings(user_profile_strings)

    """
    Step 2:
    call LLM to summarize users (multiprocess).
    """
    import os
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    os.environ["OPENAI_BASE_URL"] = config['openai_base_url']

    # 用户总结
    results, error_users = _run_user_summarization_multiprocess(
        user_prompts=user_prompts,
        system_prompt=CROSSDOMAIN_USER_SYSTEM_PROMPT,
        max_retry=2,
        num_workers=20,
    )

    # src物品总结
    data_items_src, error_items_src = _run_user_summarization_multiprocess(
        user_prompts=item_prompts_src,
        system_prompt=SRC_ITEM_SYSTEM_PROMPT,
        max_retry=2,
        num_workers=20,
    )

    # tgt物品总结
    data_items_tgt, error_items_tgt = _run_user_summarization_multiprocess(
        user_prompts=item_prompts_tgt,
        system_prompt=TGT_ITEM_SYSTEM_PROMPT,
        max_retry=2,
        num_workers=20,
    )

    return {
        "data_users": results,
        "error_users": error_users,
        "data_items_src": data_items_src,
        "error_items_src": error_items_src,
        "data_items_tgt": data_items_tgt,
        "error_items_tgt": error_items_tgt,
    }


def generate_CrossDomain_semantics_both_tag_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    """
    Generate cross-domain user semantic embeddings.

    Output shape:
        [num_src_user + num_tgt_user - num_overlap_user, embedding_dim]

    User order:
        1 ~ num_overlap_user                 : overlap users
        num_overlap_user+1 ~ num_src_user    : src-only users
        num_overlap_user+1 ~ num_tgt_user    : tgt-only users
    """
    if len(modality_data["error_users"]):
        raise ValueError(f"There are error users in modality data json file.")
    if len(modality_data["error_items_src"]) > 0:
        raise ValueError("There are error src items in modality data json file.")
    if len(modality_data["error_items_tgt"]) > 0:
        raise ValueError("There are error tgt items in modality data json file.")

    # =========================================================
    # Step 0: infer number of overlap users (prefix-based)
    # =========================================================
    src_id2user = id_mapping["src"]["id2user"]
    tgt_id2user = id_mapping["tgt"]["id2user"]

    num_overlap_user = 0
    max_check = min(len(src_id2user), len(tgt_id2user))
    for i in range(1, max_check):
        if src_id2user[i] == tgt_id2user[i]:
            num_overlap_user += 1
        else:
            break

    num_src_user = len(src_id2user) - 1  # exclude padding idx 0
    num_tgt_user = len(tgt_id2user) - 1

    # =========================================================
    # Step 1: prepare user summaries
    # =========================================================
    # modality_data = {"data": {raw_user_id: summary, ...}, "error_users": {...}}
    user_summaries = modality_data["data_users"]
    item_summaries_src = modality_data["data_items_src"]
    item_summaries_tgt = modality_data["data_items_tgt"]

    embedding_model = modality["emb_model"]
    batch_size = modality["emb_batch_size"]
    normalize = modality.get("normalize_semantic_emb", False)

    # =========================================================
    # Step 2: build ordered user list (CRITICAL)
    # =========================================================
    ordered_raw_users = []
    ordered_tag_lists = []  # List[List[str]]


    # (1) overlap users
    for uid in range(1, num_overlap_user + 1):
        raw_user = src_id2user[uid]
        if raw_user not in user_summaries:
            raise ValueError(f"Missing summary for overlap user {raw_user}")
        ordered_raw_users.append(raw_user)
        ordered_tag_lists.append(user_summaries[raw_user])

    # (2) src-only users
    for uid in range(num_overlap_user + 1, num_src_user + 1):
        raw_user = src_id2user[uid]
        if raw_user not in user_summaries:
            raise ValueError(f"Missing summary for src-only user {raw_user}")
        ordered_raw_users.append(raw_user)
        ordered_tag_lists.append(user_summaries[raw_user])

    # (3) tgt-only users
    for uid in range(num_overlap_user + 1, num_tgt_user + 1):
        raw_user = tgt_id2user[uid]
        if raw_user not in user_summaries:
            raise ValueError(f"Missing summary for tgt-only user {raw_user}")
        ordered_raw_users.append(raw_user)
        ordered_tag_lists.append(user_summaries[raw_user])

    # =========================================================
    # Step 2.5: build ordered item lists (CRITICAL)
    # =========================================================
    # src items
    src_id2item = id_mapping["src"]["id2item"]
    ordered_src_items = []
    ordered_src_item_tags = []

    for iid in range(1, len(src_id2item)):
        raw_item = src_id2item[iid]
        if raw_item not in item_summaries_src:
            raise ValueError(f"Missing summary for src item {raw_item}")
        ordered_src_items.append(raw_item)
        ordered_src_item_tags.append(item_summaries_src[raw_item])

    # tgt items
    tgt_id2item = id_mapping["tgt"]["id2item"]
    ordered_tgt_items = []
    ordered_tgt_item_tags = []

    for iid in range(1, len(tgt_id2item)):
        raw_item = tgt_id2item[iid]
        if raw_item not in item_summaries_tgt:
            raise ValueError(f"Missing summary for tgt item {raw_item}")
        ordered_tgt_items.append(raw_item)
        ordered_tgt_item_tags.append(item_summaries_tgt[raw_item])

    # =========================================================
    # Step 3: batch encode summaries
    # =========================================================
    # user
    flat_texts = []  # 所有 tag（List[str]）
    flat_user_idx = []  # 每个 tag 属于哪个 user（index）
    for user_idx, tag_list in enumerate(ordered_tag_lists):
        for tag in tag_list:
            flat_texts.append(tag)
            flat_user_idx.append(user_idx)

    # item
    flat_item_texts = []
    flat_item_idx = []  # index in [src_items + tgt_items]
    # src items first
    for item_idx, tag_list in enumerate(ordered_src_item_tags):
        for tag in tag_list:
            flat_item_texts.append(tag)
            flat_item_idx.append(item_idx)
    src_item_offset = len(ordered_src_items)
    # tgt items
    for item_idx, tag_list in enumerate(ordered_tgt_item_tags):
        for tag in tag_list:
            flat_item_texts.append(tag)
            flat_item_idx.append(src_item_offset + item_idx)

    # user 和 item 的所有tag全部放入text-emb处理
    all_flat_texts = flat_texts + flat_item_texts
    num_flat = len(all_flat_texts)
    if num_flat == 0:
        raise ValueError("No tag texts to embed.")

    num_users = len(ordered_raw_users)
    num_items_total = len(ordered_src_items) + len(ordered_tgt_items)
    # 下面 emb_dim 还不知道，等拿到第一批 embedding 再初始化
    sum_user_embs = None
    cnt_user = np.zeros((num_users,), dtype=np.int32)
    sum_item_embs = None
    cnt_item = np.zeros((num_items_total,), dtype=np.int32)
    num_user_tags = len(flat_texts)  # user tags 在 all_flat_texts 里的前缀长度

    # if 'text-embedding-3' in modality['emb_model']:
    #     from openai import OpenAI
    #
    #     client = OpenAI(
    #         api_key=config["openai_api_key"],
    #         base_url=config.get("openai_base_url", None),
    #     )
    #
    #     flat_embs = []
    #     for start in range(0, num_flat, batch_size):
    #         end = min(start + batch_size, num_flat)
    #         batch_texts = all_flat_texts[start:end]
    #
    #         response = client.embeddings.create(
    #             model=embedding_model,
    #             input=batch_texts,
    #         )
    #
    #         for emb_obj in response.data:
    #             emb = np.asarray(emb_obj.embedding, dtype=np.float32)
    #             if normalize:
    #                 emb = emb / (np.linalg.norm(emb) + 1e-12)
    #             flat_embs.append(emb)
    #
    #     flat_embs = np.stack(flat_embs, axis=0)  # [num_tags, D]
    if 'text-embedding-3' in modality['emb_model']:
        from openai import OpenAI

        client = OpenAI(
            api_key=config["openai_api_key"],
            base_url=config.get("openai_base_url", None),
        )

        for start in range(0, num_flat, batch_size):
            end = min(start + batch_size, num_flat)
            batch_texts = all_flat_texts[start:end]

            response = client.embeddings.create(
                model=embedding_model,
                input=batch_texts,
            )

            # response.data 的顺序与 input 对齐
            for j, emb_obj in enumerate(response.data):
                emb = np.asarray(emb_obj.embedding, dtype=np.float32)

                if normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-12)

                # 第一次拿到 embedding 时初始化 sum arrays
                if sum_user_embs is None:
                    emb_dim = emb.shape[0]
                    sum_user_embs = np.zeros((num_users, emb_dim), dtype=np.float32)
                    sum_item_embs = np.zeros((num_items_total, emb_dim), dtype=np.float32)

                global_idx = start + j  # 这条 embedding 在 all_flat_texts 中的全局位置

                # 0 .. num_user_tags-1 是 user tags
                if global_idx < num_user_tags:
                    uidx = flat_user_idx[global_idx]
                    sum_user_embs[uidx] += emb
                    cnt_user[uidx] += 1
                else:
                    # item tags 的 idx 从 0 开始数，所以要减去 num_user_tags
                    k = global_idx - num_user_tags
                    iidx = flat_item_idx[k]  # 这是在 [src_items + tgt_items] 的 index
                    sum_item_embs[iidx] += emb
                    cnt_item[iidx] += 1
    else:
        raise NotImplementedError("Only OpenAI embedding models are handled here.")

    # # 重新从tag emb，通过mean pooling聚合成user text emb
    # emb_dim = flat_embs.shape[1]
    # num_users = len(ordered_raw_users)
    # num_user_tags = len(flat_texts)
    #
    # sum_embs = np.zeros((num_users, emb_dim), dtype=np.float32)
    # cnt = np.zeros((num_users,), dtype=np.int32)
    #
    # for i in range(num_user_tags):
    #     uidx = flat_user_idx[i]
    #     sum_embs[uidx] += flat_embs[i]
    #     cnt[uidx] += 1
    #
    # raw_user_to_emb = {}
    #
    # for uidx, raw_user in enumerate(ordered_raw_users):
    #     user_emb = sum_embs[uidx] / cnt[uidx]
    #     raw_user_to_emb[raw_user] = user_emb
    #
    # # 聚合item
    # num_items_total = len(ordered_src_items) + len(ordered_tgt_items)
    #
    # sum_item_embs = np.zeros((num_items_total, emb_dim), dtype=np.float32)
    # cnt_item = np.zeros((num_items_total,), dtype=np.int32)
    #
    # item_offset = num_user_tags  # item embeddings start after user tags
    #
    # for i in range(len(flat_item_texts)):
    #     idx = flat_item_idx[i]
    #     sum_item_embs[idx] += flat_embs[item_offset + i]
    #     cnt_item[idx] += 1
    #
    # item_embs = sum_item_embs / cnt_item[:, None]
    #
    # # Sanity check
    # assert len(raw_user_to_emb) == len(ordered_raw_users), \
    #     "Some user embeddings are missing after batch encoding"

    if sum_user_embs is None:
        raise ValueError("Embedding failed: sum_user_embs was not initialized.")

    # 防止除 0（理论上不会，因为每个 user/item 至少 1 个 tag）
    cnt_user_safe = np.clip(cnt_user, 1, None).astype(np.float32)
    cnt_item_safe = np.clip(cnt_item, 1, None).astype(np.float32)

    user_embeddings = sum_user_embs / cnt_user_safe[:, None]
    item_embs = sum_item_embs / cnt_item_safe[:, None]

    # =========================================================
    # Step 4: stack embeddings in final order
    # =========================================================
    # user_embeddings = np.stack(
    #     [raw_user_to_emb[raw_user] for raw_user in ordered_raw_users],
    #     axis=0
    # )

    final_embeddings = np.concatenate(
        [
            user_embeddings,
            item_embs[:len(ordered_src_items)],  # src items
            item_embs[len(ordered_src_items):],  # tgt items
        ],
        axis=0,
    )

    return final_embeddings



def generate_CrossDomain_semantics_both_tag_final_embs(config, modality, interaction, id_mapping, modality_embs):
    input_dim = modality_embs.shape[1]
    target_dim = modality["emb_pca"]

    if input_dim == target_dim:
        return modality_embs.astype(np.float32)

    from sklearn.decomposition import PCA
    pca = PCA(
        n_components=target_dim,
        random_state=config.get("seed", 999),
    )
    final_embs = pca.fit_transform(modality_embs)

    if modality["normalize_semantic_emb"]:
        norm = np.linalg.norm(final_embs, axis=1, keepdims=True) + 1e-12
        final_embs = final_embs / norm

    return final_embs.astype(np.float32)


# 需要针对不同src+tgt，修改源域，目标域的说明:The source domain focuses on
CROSSDOMAIN_USER_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Your task is to summarize a user's interests based on their interactions with items from two different types of product categories.
One category is about electronics.
The other category is about cell phones and accessories.

The information I will give you:
INTERACTIONS FROM ELECTRONICS: A LIST of user interactions with items related to electronics.
INTERACTIONS FROM PHONES: A LIST of user interactions with items related to cell phones and accessories.

Each interaction is described in JSON format with the following attributes, where missing values are set to "None".
The attributes include the item's information and the user's review on that item:
{
  "title": "the name of the item"
  "description": "a description of the item"
  "categories": "several tags describing the item"
  "price": "the price of the item"
  "brand": "the brand of the item"
  "review": "the user's review on the item"
}

Requirements:
1. Extract a set of high-level, abstract user preference tags from the user's interactions.
   The tags should reflect:
   - emotional attitudes,
   - value orientations,
   - comfort or reliability expectations,
   - lifestyle or usage preferences,
   as inferred from item attributes and user reviews.
   Do NOT describe:
   - specific products,
   - item categories,
   - functions,
   - materials,
   - usage scenarios,
   or any concrete physical objects.
2. Output only a list of tags, following this structure: ["tag1", "tag2", "tag3", "..."]
    - tags must be: typically 1–2 words (at most 3)
    - the number of tags should be 2-4
3. Do not provide any other text outside the list.
"""



SRC_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Your task is to analyze ONE product item related to electronics,
and summarize what types of users this item is likely to attract.

The information I will give you is the item's metadata in JSON format,
where any missing field is set to "None":
{
  "title": "item name",
  "description": "item description",
  "categories": "category tags",
  "price": "price",
  "brand": "brand"
}

Requirements:
1. Infer high-level user preference tags that describe what kinds of users may like this item.
   Tags should reflect abstract user tendencies such as:
   - style orientation,
   - comfort or quality expectations,
   - value sensitivity,
   - lifestyle or aesthetic preferences.
   Do NOT mention:
   - specific product names,
   - brands,
   - materials,
   - item categories,
   - concrete usage scenarios.
2. Output ONLY a list of tags, following this format: ["tag1", "tag2", "tag3", "..."]
   - tags must be: typically 1–2 words (at most 3)
   - the number of tags should be 2–4
3. Do not provide any other text outside the list.
"""



TGT_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Your task is to analyze ONE product item related to cell phones and accessories,
and summarize what types of users this item is likely to attract.

The information I will give you is the item's metadata in JSON format,
where any missing field is set to "None":
{
  "title": "item name",
  "description": "item description",
  "categories": "category tags",
  "price": "price",
  "brand": "brand"
}

Requirements:
1. Infer high-level user preference tags that describe what kinds of users may like this item.
   Tags should reflect abstract user tendencies such as:
   - performance orientation,
   - durability or reliability expectations,
   - activity intensity preferences,
   - outdoor or fitness lifestyle traits.
   Do NOT mention:
   - specific product names,
   - brands,
   - materials,
   - item categories,
   - concrete usage scenarios.
2. Output ONLY a list of tags, following this format: ["tag1", "tag2", "tag3", "..."]
   - tags must be: typically 1–2 words (at most 3)
   - the number of tags should be 2–4
3. Do not provide any other text outside the list.
"""




