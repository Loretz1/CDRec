import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import math
import numpy as np

class LLM_Diff_ddim(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LLM_Diff_ddim, self).__init__(config, dataloader)

        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.diff_weight = config['diff_weight']

        self.emb_user = nn.Embedding(
            self.num_users_src + self.num_users_tgt - self.num_users_overlap + 1,
            self.embedding_dim,
            padding_idx=0
        )
        self.emb_item_src = nn.Embedding(self.num_items_src + 1, self.embedding_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1, self.embedding_dim, padding_idx=0)

        self.diff_src = Diffusion(config)
        self.diff_tgt = Diffusion(config)

        self.bpr_loss = BPRLoss()

        self.apply(xavier_uniform_initialization)
        self.emb_user.weight.data[0, :] = 0
        self.emb_item_src.weight.data[0, :] = 0
        self.emb_item_tgt.weight.data[0, :] = 0

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

        # 这里t是随机采的，不是对称采样
        B = u_src.size(0)
        t_src = torch.randint(low=0, high=self.diff_src.timesteps, size=(B,), device=u_src.device)
        diff_loss_src, u_src_denoised = self.diff_src.p_losses(x_start=u_src, t=t_src, loss_type="l2")
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

        B = u_tgt.size(0)
        t_tgt = torch.randint(low=0, high=self.diff_tgt.timesteps, size=(B,), device=u_tgt.device)
        diff_loss_tgt, u_tgt_denoised = self.diff_tgt.p_losses(x_start=u_tgt, t=t_tgt, loss_type="l2")
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
            _, u_denoised, _, _, _ = self.diff_tgt.sample(u)
        else:
            u = self.emb_user(users)
            _, u_denoised, _, _, _ = self.diff_tgt.sample(u)

        u_final = u + u_denoised

        item_emb = self.emb_item_tgt.weight
        scores = torch.matmul(u_final, item_emb.t())
        scores[:, 0] = 0.0
        return scores


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

        # DDIM
        self.linespace = config["linespace"]
        indices = list(range(0, self.timesteps + 1, self.linespace))
        self.sub_timesteps = len(indices)
        indices_now = [indices[i] - 1 for i in range(len(indices))]
        indices_now[0] = 0
        self.register_buffer("alphas_cumprod_ddim", self.alphas_cumprod[indices_now])
        self.register_buffer("alphas_cumprod_ddim_prev", F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0))
        self.register_buffer("sqrt_recipm1_alphas_cumprod_ddim", torch.sqrt(1. / self.alphas_cumprod_ddim - 1))
        self.register_buffer(
            "posterior_ddim_coef1",
            torch.sqrt(self.alphas_cumprod_ddim_prev)
            - torch.sqrt(1. - self.alphas_cumprod_ddim_prev)
            / self.sqrt_recipm1_alphas_cumprod_ddim
        )
        self.register_buffer(
            "posterior_ddim_coef2",
            torch.sqrt(1. - self.alphas_cumprod_ddim_prev)
            / torch.sqrt(1. - self.alphas_cumprod_ddim)
        )

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
        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)
        attn = (q * (self.embedding_dim ** -0.5)) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out.mean(dim=1)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t = t.to(x_start.device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, loss_type: str = "l2"):
        device = x_start.device
        t = t.to(device)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)  # [B, D] on device
        tokens = torch.stack([x_noisy, t_emb], dim=1)  # [B, 2, D]
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

    # DDIM
    @torch.no_grad()
    def ddim_sample_step(self, x_t, t, step_idx):
        t_emb = self.get_timestep_embedding(t, self.embedding_dim)
        tokens = torch.stack([x_t, t_emb], dim=1)
        x_start = self.selfAttention(tokens)

        x_prev = (
                self.posterior_ddim_coef1[step_idx] * x_start +
                self.posterior_ddim_coef2[step_idx] * x_t
        )
        return x_prev

    # DDPM
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, t_index: int):
        device = x_t.device
        t = t.to(device)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)
        tokens = torch.stack([x_t, t_emb], dim=1)
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

    # DDIM
    @torch.no_grad()
    def sample(self, x_start):
        device = x_start.device
        B = x_start.shape[0]

        t_init = torch.full(
            (B,),
            self.timesteps - 1,
            device=device,
            dtype=torch.long
        )
        noise = torch.randn_like(x_start)
        x = self.q_sample(x_start=x_start, t=t_init, noise=noise)

        for i in reversed(range(self.sub_timesteps)):
            t = torch.full(
                (B,),
                i * self.linespace,
                device=device,
                dtype=torch.long
            )
            x = self.ddim_sample_step(x, t, i)

        return x_start, x, None, None, None

    # DDPM
    # @torch.no_grad()
    # def sample(self, x_start: torch.Tensor):
    #     device = x_start.device
    #
    #     noise_x = torch.randn_like(x_start)
    #     t_init = torch.full(
    #         (x_start.shape[0],),
    #         self.timesteps - 1,
    #         dtype=torch.long,
    #         device=device
    #     )
    #     x_t = self.q_sample(x_start=x_start, t=t_init, noise=noise_x)
    #
    #     x_quarter = x_t
    #     x_half = x_t
    #     x_three_quarter = x_t
    #
    #     for n in reversed(range(self.timesteps)):
    #         t = torch.full((x_t.shape[0],), n, dtype=torch.long, device=device)
    #         x_t = self.p_sample(x_t, t, n)
    #
    #         if n == int((self.timesteps - 1) * 0.75):
    #             x_quarter = x_t
    #         if n == int((self.timesteps - 1) * 0.5):
    #             x_half = x_t
    #         if n == int((self.timesteps - 1) * 0.25):
    #             x_three_quarter = x_t
    #
    #     return x_start, x_t, x_quarter, x_half, x_three_quarter

