import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import math
import numpy as np
from multiprocessing import Pool, cpu_count

class MuSiC(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(MuSiC, self).__init__(config, dataloader)

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

        self.diff_src = Diffusion(config)
        self.diff_tgt = Diffusion(config)

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

        # 读semantic emb
        semantic = dataloader.get_modality_embs()["CrossDomain_semantics_music"]

        s = 0
        src_user = semantic[s:s + self.num_users_src];
        s += self.num_users_src
        tgt_user = semantic[s:s + self.num_users_tgt];
        s += self.num_users_tgt
        pad = np.zeros((1, src_user.shape[1]), dtype=np.float32)

        self.register_buffer(
            "src_user_semantic_emb",
            torch.from_numpy(np.vstack([pad, src_user]))  # [num_users_src+1, D_sem]
        )
        self.register_buffer(
            "tgt_user_semantic_emb",
            torch.from_numpy(np.vstack([pad, tgt_user]))  # [num_users_tgt+1, D_sem]
        )

        semantic_dim = semantic.shape[1]
        self.semantic_proj = nn.Linear(semantic_dim, self.embedding_dim)  # -> [*, embedding_dim]

        self.apply(xavier_uniform_initialization)
        self.emb_user.weight.data[0, :] = 0
        self.emb_item_src.weight.data[0, :] = 0
        self.emb_item_tgt.weight.data[0, :] = 0

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

    def _get_text_cond_for_src_users(self, users_src: torch.Tensor):
        """
        users_src: src user-id space [B], 1..num_users_src
        returns:
          src_text: [B, D]  (always available from src buffer)
          tgt_text: [B, D]  (ONLY overlap users get tgt buffer; others are padding zeros)
        """
        device = users_src.device
        src_text = self.semantic_proj(self.src_user_semantic_emb[users_src])

        tgt_text = torch.zeros(users_src.size(0), self.embedding_dim, device=device, dtype=src_text.dtype)

        mask = users_src <= self.num_users_overlap
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
            tgt_text[idx] = self.semantic_proj(self.tgt_user_semantic_emb[users_src[idx]])

        return src_text, tgt_text

    def _get_text_cond_for_tgt_users(self, users_tgt: torch.Tensor):
        """
        users_tgt: tgt user-id space [B], 1..num_users_tgt
        returns:
          src_text: [B, D]  (ONLY overlap users get src buffer; others are padding zeros)
          tgt_text: [B, D]  (always available from tgt buffer)
        """
        device = users_tgt.device
        tgt_text = self.semantic_proj(self.tgt_user_semantic_emb[users_tgt])

        src_text = torch.zeros(users_tgt.size(0), self.embedding_dim, device=device, dtype=tgt_text.dtype)

        mask = users_tgt <= self.num_users_overlap
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
            src_text[idx] = self.semantic_proj(self.src_user_semantic_emb[users_tgt[idx]])

        return src_text, tgt_text

    def calculate_loss(self, interaction, epoch_idx):
        users_src = interaction['users_src']
        pos_items_src = interaction['pos_items_src']
        neg_items_src = interaction['neg_items_src']
        users_tgt = interaction['users_tgt']
        pos_items_tgt = interaction['pos_items_tgt']
        neg_items_tgt = interaction['neg_items_tgt']

        # src
        u_src = self.emb_user(users_src)  # [B, D]
        i_pos_src = self.emb_item_src(pos_items_src)  # [B, D]
        i_neg_src = self.emb_item_src(neg_items_src)  # [B, D]

        # 聚合src用户的src和tgt交互
        # hist_src = self.emb_item_src(self.history_src_user_src[users_src])
        # cond_src = self.src_interaction_agg(hist_src, u_src)
        # hist_tgt = self.emb_item_tgt(self.history_src_user_tgt[users_src])
        # cond_tgt = self.tgt_interaction_agg(hist_tgt, u_src)

        hist_src_items = self.history_src_user_src[users_src]  # [B, L]
        hist_tgt_items = self.history_src_user_tgt[users_src]  # [B, L]
        hist_src_items = self.batch_random_mask(hist_src_items, self.config['mask_rate'], min_keep=1)
        hist_tgt_items = self.batch_random_mask(hist_tgt_items, self.config['mask_rate'], min_keep=1)
        hist_src = self.emb_item_src(hist_src_items)  # [B, L, D]
        hist_tgt = self.emb_item_tgt(hist_tgt_items)  # [B, L, D]
        cond_src = self.src_interaction_agg(hist_src, u_src)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_src)


        # 这里t是随机采的，不是对称采样
        B = u_src.size(0)
        t_src = torch.randint(low=0, high=self.diff_src.timesteps, size=(B,), device=u_src.device)
        src_text_src, tgt_text_src = self._get_text_cond_for_src_users(users_src)
        diff_loss_src, u_src_denoised = self.diff_src.p_losses(
            x_start=u_src,
            t=t_src,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            src_text=src_text_src,
            tgt_text=tgt_text_src,
            loss_type="l2"
        )
        u_src_final = u_src + self.config["lambda_user_emb"] * u_src_denoised # 残差连接

        pos_score_src = (u_src_final * i_pos_src).sum(dim=-1)
        neg_score_src = (u_src_final * i_neg_src).sum(dim=-1)
        bpr_loss_src = self.bpr_loss(pos_score_src, neg_score_src)

        # tgt
        users_tgt_local = users_tgt
        offset = self.num_users_src - self.num_users_overlap
        users_tgt_global = users_tgt_local + (users_tgt_local > self.num_users_overlap).long() * offset # tgt单域用户需要加一个偏移值，从而取到正确的id emb
        u_tgt = self.emb_user(users_tgt_global)  # [B, D]
        i_pos_tgt = self.emb_item_tgt(pos_items_tgt)  # [B, D]
        i_neg_tgt = self.emb_item_tgt(neg_items_tgt)  # [B, D]

        # 聚合tgt用户的src和tgt交互
        # hist_src = self.emb_item_src(self.history_tgt_user_src[users_tgt])
        # cond_src = self.src_interaction_agg(hist_src, u_tgt)
        # hist_tgt = self.emb_item_tgt(self.history_tgt_user_tgt[users_tgt])
        # cond_tgt = self.tgt_interaction_agg(hist_tgt, u_tgt)

        hist_src_items = self.history_tgt_user_src[users_tgt]  # [B, L]
        hist_tgt_items = self.history_tgt_user_tgt[users_tgt]  # [B, L]
        hist_src_items = self.batch_random_mask(hist_src_items, self.config['mask_rate'], min_keep=1)
        hist_tgt_items = self.batch_random_mask(hist_tgt_items, self.config['mask_rate'], min_keep=1)
        hist_src = self.emb_item_src(hist_src_items)
        hist_tgt = self.emb_item_tgt(hist_tgt_items)
        cond_src = self.src_interaction_agg(hist_src, u_tgt)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_tgt)

        B = u_tgt.size(0)
        t_tgt = torch.randint(low=0, high=self.diff_tgt.timesteps, size=(B,), device=u_tgt.device)
        src_text_tgt, tgt_text_tgt = self._get_text_cond_for_tgt_users(users_tgt)
        diff_loss_tgt, u_tgt_denoised = self.diff_tgt.p_losses(
            x_start=u_tgt,
            t=t_tgt,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            src_text=src_text_tgt,
            tgt_text=tgt_text_tgt,
            loss_type="l2"
        )
        u_tgt_final = u_tgt + self.config["lambda_user_emb"] * u_tgt_denoised # 残差连接

        pos_score_tgt = (u_tgt_final * i_pos_tgt).sum(dim=-1)
        neg_score_tgt = (u_tgt_final * i_neg_tgt).sum(dim=-1)
        bpr_loss_tgt = self.bpr_loss(pos_score_tgt, neg_score_tgt)

        # loss = loss_rec + loss_dif
        loss = bpr_loss_src+ bpr_loss_tgt+ self.diff_weight * (diff_loss_src + diff_loss_tgt)
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
            src_text, tgt_text = self._get_text_cond_for_tgt_users(users)  # users 是 tgt-space
        else:
            u = self.emb_user(users)

            hist_src = self.emb_item_src(self.history_src_user_src[users])  # [B, L, D]
            hist_tgt = self.emb_item_tgt(self.history_src_user_tgt[users])  # [B, L, D]
            src_text, tgt_text = self._get_text_cond_for_src_users(users)  # users 是 src-space

        cond_src = self.src_interaction_agg(hist_src, u)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u)
        _, u_denoised, _, _, _ = self.diff_tgt.sample(
            x_start=u,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            src_text=src_text,
            tgt_text=tgt_text
        )

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

    def p_losses(self, x_start, t, cond_src=None, cond_tgt=None, src_text=None, tgt_text=None, loss_type="l2"):
        device = x_start.device
        t = t.to(device)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)  # [B, D] on device
        tokens = torch.stack([x_noisy, t_emb, cond_src, cond_tgt, src_text, tgt_text], dim=1)  # [B, 6, D]
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
    def p_sample(self, x_t, t, t_index, cond_src, cond_tgt, src_text, tgt_text):
        device = x_t.device
        t = t.to(device)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)
        tokens = torch.stack([x_t, t_emb, cond_src, cond_tgt, src_text, tgt_text], dim=1)
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
    def sample(self, x_start, cond_src, cond_tgt, src_text, tgt_text):
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
            x_t = self.p_sample(x_t=x_t, t=t, t_index=n, cond_src=cond_src, cond_tgt=cond_tgt, src_text=src_text, tgt_text=tgt_text)

            if n == int((self.timesteps - 1) * 0.75):
                x_quarter = x_t
            if n == int((self.timesteps - 1) * 0.5):
                x_half = x_t
            if n == int((self.timesteps - 1) * 0.25):
                x_three_quarter = x_t

        return x_start, x_t, x_quarter, x_half, x_three_quarter


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


# 🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭 LLM总结 🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭

def _collect_music_user_text(uid, interaction_df, id_mapping, domain, reviews):
    rows = interaction_df[interaction_df["user"] == uid]

    if len(rows) == 0:
        return None

    texts = []
    raw_user = id_mapping[domain]["id2user"][uid]

    for iid in rows["item"].values:
        raw_item = id_mapping[domain]["id2item"][iid]
        review = reviews[domain].get((raw_user, raw_item))

        if isinstance(review, str) and review.strip():
            texts.append(review.strip())

    if not texts:
        return None

    return "\n".join(texts[:50])


def _music_call_llm(text, config):
    from openai import OpenAI

    system_prompt = _get_music_user_prompt(config)
    client = OpenAI(
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url"),
    )

    resp = client.chat.completions.create(
        model=config.get("music_llm_model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )

    return resp.choices[0].message.content.strip()


def _music_llm_worker(args):
    raw_uid, text, config = args

    try:
        summary = _music_call_llm(text, config)
        return raw_uid, summary
    except Exception as e:
        return raw_uid, f"ERROR: {e}"


def _run_music_llm_parallel(user_text_dict, config, num_workers=None):

    if num_workers is None:
        num_workers = min(cpu_count(), 20)

    tasks = [(u, t, config) for u, t in user_text_dict.items()]

    results = {}

    with Pool(num_workers) as pool:
        for uid, summary in pool.imap_unordered(_music_llm_worker, tasks):
            results[uid] = summary

    return results


def _get_music_user_prompt(config):
    dataset = config["dataset"]

    if dataset == "Douban":
        return MUSIC_USER_SYSTEM_PROMPT_ZH
    else:
        return MUSIC_USER_SYSTEM_PROMPT_EN


def _music_encode_text_list(text_list, config, batch_size):
    from openai import OpenAI
    import numpy as np

    client = OpenAI(
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url"),
    )

    model = config.get("music_embedding_model", "text-embedding-3-large")

    embs = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]

        resp = client.embeddings.create(
            model=model,
            input=batch,
        )

        # ⭐ API 返回顺序与 input 对齐
        for item in resp.data:
            embs.append(item.embedding)

    return np.array(embs, dtype=np.float32)


def extract_CrossDomain_semantics_music_modality_data(config, modality, interaction, id_mapping, raw_data_list):
    reviews = raw_data_list[0]

    user_text_src = {}
    user_text_tgt = {}

    # ===== collect src user text =====
    for uid in id_mapping["src"]["user2id"].values():
        text = _collect_music_user_text(uid, interaction["src"], id_mapping, "src", reviews)

        if text:
            raw_uid = id_mapping["src"]["id2user"][uid]
            user_text_src[raw_uid] = text

    # ===== collect tgt user text =====
    for uid in id_mapping["tgt"]["user2id"].values():
        text = _collect_music_user_text(uid, interaction["tgt"], id_mapping, "tgt", reviews)

        if text:
            raw_uid = id_mapping["tgt"]["id2user"][uid]
            user_text_tgt[raw_uid] = text

    # ⭐ parallel LLM
    src_summary = _run_music_llm_parallel(user_text_src, config)
    tgt_summary = _run_music_llm_parallel(user_text_tgt, config)

    return {
        "src_user": src_summary,
        "tgt_user": tgt_summary,
    }

def generate_CrossDomain_semantics_music_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    # =========================
    # 1️⃣ user emb (MuSiC LLM summary)
    # =========================

    src_user_list = id_mapping["src"]["id2user"][1:]
    tgt_user_list = id_mapping["tgt"]["id2user"][1:]

    src_summary = modality_data["src_user"]
    tgt_summary = modality_data["tgt_user"]

    # order text
    src_texts = [src_summary[u] for u in src_user_list]
    tgt_texts = [tgt_summary[u] for u in tgt_user_list]

    # encode
    src_user_emb = _music_encode_text_list(src_texts, config, modality['emb_batch_size'])
    tgt_user_emb = _music_encode_text_list(tgt_texts, config, modality['emb_batch_size'])

    # =========================
    # 2️⃣ item emb (from LLM-Diff)
    # =========================

    item_emb_path = modality["item_emb_path"]
    import os
    assert os.path.exists(item_emb_path), f"{item_emb_path} not found"

    llm_diff_emb = np.load(item_emb_path)

    # ---------- count users ----------
    src_users = set(id_mapping["src"]["id2user"][1:])
    tgt_users = set(id_mapping["tgt"]["id2user"][1:])

    num_src_user = len(src_users)
    num_tgt_user = len(tgt_users)
    num_overlap = len(src_users & tgt_users)

    num_all_user = num_src_user + num_tgt_user - num_overlap

    # ---------- count items ----------
    num_src_item = len(id_mapping["src"]["id2item"]) - 1
    num_tgt_item = len(id_mapping["tgt"]["id2item"]) - 1

    expected_total = num_all_user + num_src_item + num_tgt_item

    assert llm_diff_emb.shape[0] == expected_total, \
        f"LLM-Diff emb size mismatch: {llm_diff_emb.shape[0]} vs {expected_total}"

    # ---------- slice ----------
    start = 0
    end = num_all_user

    src_item_emb = llm_diff_emb[end:end+num_src_item]
    tgt_item_emb = llm_diff_emb[end+num_src_item:end+num_src_item+num_tgt_item]

    # =========================
    # 3️⃣ concat
    # =========================

    final_emb = np.concatenate(
        [
            src_user_emb,
            tgt_user_emb,
            src_item_emb,
            tgt_item_emb,
        ],
        axis=0,
    )

    return final_emb



def generate_CrossDomain_semantics_music_final_embs(config, modality, interaction, id_mapping, modality_embs):
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

    return final_embs.astype(np.float32)




MUSIC_USER_SYSTEM_PROMPT_EN = """
You are an expert in recommendation systems.

Summarize the user's overall preferences based on the given texts.

Requirements:
- Capture high-level preference tendencies.
- Avoid specific item names.
- Write in English.
- Output a concise description.
""".strip()

MUSIC_USER_SYSTEM_PROMPT_ZH = """
你是一名推荐系统专家。

请基于给定文本，总结该用户的整体偏好。

要求：
- 概括高层次偏好倾向；
- 不要提及具体物品名称；
- 使用中文输出；
- 输出一段简洁描述。
""".strip()


