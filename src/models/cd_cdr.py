import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_, normal_
import numpy as np
import math

def diffusion_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Linear layers. For bias in nn.Linear layers, using constant 0 to initialize.
    Use normal_ for embeddings.
    """
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data, 0, 1)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

## Edit from DreamRec
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

## Edit from DreamRec
def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

## Edit from DreamRec
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

## Edit from DreamRec
def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

## Edit from DreamRec
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

## Edit from DreamRec
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CD_CDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        self.feature_dim = config["feature_dim"]
        self.dropout = config["dropout"]
        self.loss_type = config["loss_type"].lower()
        self.history_len = config.get("history_len", 50)
        self.loss_type = config["loss_type"]
        self.timesteps = config['timestep']
        # self.linespace = 100
        self.linespace = max(self.timesteps // 20, 1)
        self.w = config['uncon_w']
        self.p = config['uncon_p']
        self.beta_sche = config['beta_sche']
        self.beta_start = 0.0001
        self.beta_end = 0.02
        if self.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif self.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # DDIM Reverse Process
        indices = list(range(0, self.timesteps+1, self.linespace)) # [0,100,...,2000]
        self.sub_timesteps = len(indices)
        indices_now = [indices[i]-1 for i in range(len(indices))]
        indices_now[0] = 0
        self.alphas_cumprod_ddim = self.alphas_cumprod[indices_now]
        self.alphas_cumprod_ddim_prev = F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0)
        self.sqrt_recipm1_alphas_cumprod_ddim = torch.sqrt(1. / self.alphas_cumprod_ddim - 1)
        self.posterior_ddim_coef1 = torch.sqrt(self.alphas_cumprod_ddim_prev) - torch.sqrt(1.-self.alphas_cumprod_ddim_prev)/ self.sqrt_recipm1_alphas_cumprod_ddim
        self.posterior_ddim_coef2 = torch.sqrt(1.-self.alphas_cumprod_ddim_prev) / torch.sqrt(1. - self.alphas_cumprod_ddim)
        self.diffuser_type = config["diffuser_type"]


        self.item_embedding_src = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)
        self.item_embedding_tgt = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)
        self.W_k = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim), nn.Tanh()
        )

        self.W_q = nn.Linear(self.feature_dim, 1, bias=False)

        layer_norm = config['layer_norm']
        if layer_norm:
            self.ln_1 = nn.LayerNorm(self.feature_dim)
            self.ln_2 = nn.LayerNorm(self.feature_dim)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()

        self.UI_map = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        self.gate_proj = nn.Linear(2 * self.feature_dim, self.feature_dim)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.n_heads = config["n_heads"]
        self.domain_emb = nn.Embedding(2, embedding_dim=self.feature_dim)
        self.domain_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.n_heads,
            batch_first=True
        )

        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.feature_dim,
        )

        #TimestepEmbedder slightly difference
        self.step_mlp = nn.Sequential( # time vector mlp
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        if self.diffuser_type =='mlp1':
            self.diffu_mlp = nn.Sequential( # diffusion mlp, in: x,c,t; out: x
                nn.Linear(self.feature_dim*3, self.feature_dim)
        )
        elif self.diffuser_type =='mlp2':
            self.diffu_mlp = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim*2),
            nn.GELU(),
            nn.Linear(self.feature_dim*2, self.feature_dim)
        )



        self._build_history(dataloader)
        self.apply(diffusion_initialization)
        self.item_embedding_tgt.weight.data[0, :] = 0
        self.item_embedding_src.weight.data[0, :] = 0

    def _build_history(self, dataloader):
        """
        Build fixed-length user interaction history matrices.
        - history_len is a constant hyper-parameter
        - If the number of interactions > history_len: randomly drop extra items
        - If the number of interactions < history_len: pad with 0
        """
        device = self.device
        max_len = self.history_len

        H_src = torch.zeros((self.num_users_src + 1, max_len), dtype=torch.long, device=device)
        L_src = torch.zeros(self.num_users_src + 1, dtype=torch.long, device=device)

        H_tgt = torch.zeros((self.num_users_tgt + 1, max_len), dtype=torch.long, device=device)
        L_tgt = torch.zeros(self.num_users_tgt + 1, dtype=torch.long, device=device)

        for u, items in dataloader.dataset.positive_items_src.items():
            # Convert set to list for torch processing
            items = list(items)
            n = len(items)
            if n == 0:
                continue
            if n > max_len:
                # random truncation
                rand_idx = torch.randperm(n)[:max_len]
                sampled = torch.tensor([items[i] for i in rand_idx], dtype=torch.long, device=device)
                H_src[u] = sampled
                L_src[u] = max_len
            else:
                sampled = torch.tensor(items, dtype=torch.long, device=device)
                H_src[u, :n] = sampled
                L_src[u] = n

        for u, items in dataloader.dataset.positive_items_tgt.items():
            items = list(items)
            n = len(items)
            if n == 0:
                continue
            if n > max_len:
                rand_idx = torch.randperm(n)[:max_len]
                sampled = torch.tensor([items[i] for i in rand_idx], dtype=torch.long, device=device)
                H_tgt[u] = sampled
                L_tgt[u] = max_len
            else:
                sampled = torch.tensor(items, dtype=torch.long, device=device)
                H_tgt[u, :n] = sampled
                L_tgt[u] = n

        self.history_items_src = H_src
        self.history_len_src = L_src
        self.history_items_tgt = H_tgt
        self.history_len_tgt = L_tgt

    def _remove_pos_from_history(self, pos_item, history_item, history_len):
        pos_item_expanded = pos_item.unsqueeze(1)
        mask = (history_item != pos_item_expanded)
        remove_counts = (~mask).sum(dim=1)
        updated_history_len = history_len - remove_counts
        filtered_history_item = history_item * mask.long()
        return filtered_history_item, updated_history_len

    def get_user_representation(self, history_item, history_len, domain):
        if domain == 0:
            history_item_e = self.item_embedding_src(history_item)
        else:
            history_item_e = self.item_embedding_tgt(history_item)
        UI_aggregation_e = self.get_UI_aggregation(history_item_e, history_len)
        return UI_aggregation_e

    def get_UI_aggregation(self, history_item_e, history_len):
        mask = (torch.abs(history_item_e).sum(dim=-1) > 1e-8).int()
        history_item_e = self.ln_1(history_item_e)  # [user_num, max_history_len, embedding_size]
        key = self.W_k(history_item_e)
        attention = self.W_q(key).squeeze(2)
        attention_stable = attention - attention.max(dim=1, keepdim=True).values
        e_attention = torch.exp(attention_stable)
        e_attention = e_attention * mask

        attention_weight = e_attention / (e_attention.sum(dim=1, keepdim=True) + 1.0e-10)
        out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
        out = self.UI_map(out)
        UI_aggregation_e = out
        # UI_aggregation_e = self.ln_2(UI_aggregation_e)
        return UI_aggregation_e

    def domain_condition_generator(self, UI_aggregation_e, domain):
        if 1:
            Q = self.domain_emb(domain).unsqueeze(1)  # [B, 1, D]
            K = V = UI_aggregation_e.unsqueeze(1)  # [B, 1, D]
            output, _ = self.domain_attn(Q, K, V)  # [B, 1, D]
        return output.squeeze(1)  # [B, D]

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            #noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def i_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        # cf guidance

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)

        x_t = x
        model_mean = (
                self.posterior_ddim_coef1[t_index] * x_start +
                self.posterior_ddim_coef2[t_index] * x_t
        )

        return model_mean

    def add_uncon(self, h):
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - self.p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0], device = self.device)) * (1-mask)
        return h

    ## Edit from DreamRec
    # DreamRec_backbone forward
    def denoise_step(self, x, h, step):
        t = self.step_mlp(step)
        res = self.diffu_mlp(torch.cat((x, h, t), dim=1))
        return res

    # DreamRec_backbone forward_uncon
    def denoise_uncon(self, x, step): # with out condition
        h = self.none_embedding(torch.tensor([0], device = self.device))
        h = torch.cat([h.view(1, self.feature_dim)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        res = self.diffu_mlp(torch.cat((x, h, t), dim=1))
        return res

    # DreamRec_backbone sample_from_noise
    @torch.no_grad()
    def sample_from_noise(self, model_forward, model_forward_uncon, h):

        x = torch.randn_like(h)

        #for n in reversed(range(0, self.timesteps, self.linespace)):
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((h.shape[0], ), n*self.linespace, device=h.device, dtype=torch.long)
            x = self.i_sample(model_forward, model_forward_uncon, x, h, step, n)

        return x

    def run_diffusion_process(self, target_item_emb, condition_emb):
        B = target_item_emb.shape[0]
        n = torch.randint(0, self.timesteps, (B,), device=self.device).long()  # [B]
        noise = torch.randn_like(target_item_emb)  # [B, D]

        # 加噪
        x_noisy = self.q_sample(x_start=target_item_emb, t=n, noise=noise)  # [B, D]

        # 条件注入（如 DreamRec 中的 unconditional mixing）
        h = self.add_uncon(condition_emb)  # [B, D]

        # 去噪预测
        predicted_x = self.denoise_step(x=x_noisy, h=h, step=n)  # [B, D]

        return target_item_emb, predicted_x

    def calculate_loss(self, interaction, epoch_idx):
        users = [interaction["users_src"], interaction["users_tgt"]]
        pos = [interaction["pos_items_src"], interaction["pos_items_tgt"]]

        losses = []
        for domain in [0, 1]:
            user = users[domain]
            pos_item = pos[domain]

            if domain == 0:
                s_items = self.history_items_src[user]  # [B, L]
                s_len = self.history_len_src[user]

                u_tgt = user.clone()
                u_tgt[u_tgt > self.num_users_overlap] = 0
                t_items = self.history_items_tgt[u_tgt]
                t_len = self.history_len_tgt[u_tgt]

                s_items, s_len = self._remove_pos_from_history(pos_item, s_items, s_len)
            else:
                t_items = self.history_items_tgt[user]  # [B, L]
                t_len = self.history_len_tgt[user]

                u_src = user.clone()
                u_src[u_src > self.num_users_overlap] = 0
                s_items = self.history_items_src[u_src]
                s_len = self.history_len_src[u_src]

                t_items, t_len = self._remove_pos_from_history(pos_item, t_items, t_len)

            s_UI_aggregation_e = self.get_user_representation(s_items, s_len, domain=0)  # [B, D]
            t_UI_aggregation_e = self.get_user_representation(t_items, t_len, domain=1)  # [B, D]
            pos_item_e = self.item_embedding_src(pos_item) if domain == 0 else self.item_embedding_tgt(pos_item)

            # Domain condition generator
            gate = torch.sigmoid(self.gate_proj(torch.cat([s_UI_aggregation_e, t_UI_aggregation_e], dim=-1)))
            UI_aggregation_e = gate * s_UI_aggregation_e + (1 - gate) * t_UI_aggregation_e  # [B, D]
            domain_idx = torch.full_like(user, domain)  # [B], e.g., all 0 for source
            UI_aggregation_e = self.domain_condition_generator(UI_aggregation_e, domain_idx)  # [B, D]

            x_start, predicted_x = self.run_diffusion_process(pos_item_e, UI_aggregation_e)
            if self.loss_type == 'l1':
                loss = F.l1_loss(x_start, predicted_x)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, predicted_x)
            elif self.loss_type == "huber":
                loss = F.smooth_l1_loss(x_start, predicted_x)
            else:
                raise NotImplementedError()
            losses.append(loss)
        return_loss = 0.2 * losses[0] + 0.8 * losses[1]
        return return_loss

    def full_sort_predict(self, interaction, is_warm):
        user = interaction[0].long()

        if not is_warm:
            s_items = self.history_items_src[user]  # [B, L]
            s_len = self.history_len_src[user]

            u_tgt = user.clone()
            u_tgt[u_tgt > self.num_users_overlap] = 0
            t_items = self.history_items_tgt[u_tgt]
            t_len = self.history_len_tgt[u_tgt]
        else:
            t_items = self.history_items_tgt[user]  # [B, L]
            t_len = self.history_len_tgt[user]

            u_src = user.clone()
            u_src[u_src > self.num_users_overlap] = 0
            s_items = self.history_items_src[u_src]
            s_len = self.history_len_src[u_src]
        s_UI_aggregation_e = self.get_user_representation(s_items, s_len, domain=0)
        t_UI_aggregation_e = self.get_user_representation(t_items, t_len, domain=1)
        gate = torch.sigmoid(self.gate_proj(torch.cat([s_UI_aggregation_e, t_UI_aggregation_e], dim=-1)))
        UI_aggregation_e = gate * s_UI_aggregation_e + (1 - gate) * t_UI_aggregation_e
        domain_idx = torch.full_like(user, 1)
        UI_aggregation_e = self.domain_condition_generator(UI_aggregation_e, domain_idx)

        h = UI_aggregation_e
        x = self.sample_from_noise(self.denoise_step, self.denoise_uncon, h)

        target_all_item_emb = self.item_embedding_tgt.weight
        II_cos = torch.matmul(x, target_all_item_emb.T)
        return II_cos

    def set_train_stage(self, stage_id):
        super().set_train_stage(stage_id)
