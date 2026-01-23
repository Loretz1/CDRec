import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import math
import numpy as np

class LLM_Diff_interaction_as_condition_mask(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LLM_Diff_interaction_as_condition_mask, self).__init__(config, dataloader)

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
        diff_loss_src, u_src_denoised = self.diff_src.p_losses(x_start=u_src, t=t_src, cond_src=cond_src,
                                                               cond_tgt=cond_tgt, loss_type="l2")
        u_src_final = u_src + u_src_denoised # 残差连接

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
        diff_loss_tgt, u_tgt_denoised = self.diff_tgt.p_losses(x_start=u_tgt, t=t_tgt, cond_src=cond_src,
                                                               cond_tgt=cond_tgt, loss_type="l2")
        u_tgt_final = u_tgt + u_tgt_denoised # 残差连接

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
        else:
            u = self.emb_user(users)

            hist_src = self.emb_item_src(self.history_src_user_src[users])  # [B, L, D]
            hist_tgt = self.emb_item_tgt(self.history_src_user_tgt[users])  # [B, L, D]

        cond_src = self.src_interaction_agg(hist_src, u)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u)
        _, u_denoised, _, _, _ = self.diff_tgt.sample(x_start=u, cond_src=cond_src, cond_tgt=cond_tgt)

        u_final = u + u_denoised

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



