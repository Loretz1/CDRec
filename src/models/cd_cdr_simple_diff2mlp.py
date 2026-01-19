import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from common.abstract_recommender import GeneralRecommender
from torch.nn.init import xavier_normal_, constant_, normal_
from common.loss import BPRLoss

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
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

def gather_indexes(output, gather_index):
    """Gathers the vectors at the specific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)


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


class CD_CDR_simple_diff2mlp(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(CD_CDR_simple_diff2mlp, self).__init__(config, dataloader)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        # self.margin = config["margin"]
        # self.negative_weight = config["negative_weight"]
        self.gamma = config["gamma"]
        self.neg_seq_len = config['neg_seq_len']
        self.aggregator = config["aggregator"]
        self.loss_n = config["loss_n"]
        if self.aggregator not in ["mean", "user_attention", "self_attention", "transformer"]:
            raise ValueError(
                "aggregator must be mean, user_attention, self_attention or transformer"
            )
        self.n_heads = config["n_heads"]
        if self.aggregator == "transformer":
            self.dropout_prob = config["dropout"]
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=self.n_heads,
                # dim_feedforward=self.embedding_size * 4,
                dim_feedforward=self.embedding_size,
                dropout=self.dropout_prob,
                activation='gelu',
                batch_first=True
            )

        self.total_num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        self.user_embedding = nn.Embedding(self.total_num_users + 1, self.embedding_size, padding_idx=0)
        self.item_emb_src = nn.Embedding(self.num_items_src + 1, self.embedding_size, padding_idx=0)
        self.item_emb_tgt = nn.Embedding(self.num_items_tgt + 1, self.embedding_size, padding_idx=0)
        ## Edit from DreamRec
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.embedding_size,
        )

        # Domain condition generator
        self.domain_emb = nn.Embedding(2, embedding_dim=self.embedding_size)
        self.domain_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_size,
            num_heads=self.n_heads,
            batch_first=True
        )
        self.gate_proj = nn.Linear(2 * self.embedding_size, 1)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        # feature space mapping matrix of user and item
        self.UI_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False, )
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
            if self.aggregator == "self_attention":
                self.W_q = nn.Linear(self.embedding_size, 1, bias=False)
        elif self.aggregator == "transformer":
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=self.transformer_layer,
                num_layers=config["n_layers"]
            )
            self.UI_map = nn.Identity()
            self.mean_pooling = config["mean_pooling"]

        self.sigmoid = nn.Sigmoid()
        self.bprloss = BPRLoss()
        self.bceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()

        ## Edit from DreamRec
        self.diffuser_type = config["diffuser_type"]
        self.timesteps = config['timestep']  # 200, diffusion steps
        self.linespace = 100
        self.w = config['uncon_w']  # 2, the weight of conditioned diffusion in inference phase
        self.p = config['uncon_p']  # 0.1, how much prob does train phase use unconditioned diffusion
        self.beta_sche = config['beta_sche']  # exp, the schedule of beta sequence
        self.dropout = config['dropout']
        self.emb_dropout = nn.Dropout(self.dropout)
        layer_norm = config['layer_norm']
        if layer_norm:
            self.ln_1 = nn.LayerNorm(self.embedding_size)
            self.ln_2 = nn.LayerNorm(self.embedding_size)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
        # TimestepEmbedder slightly difference
        self.step_mlp = nn.Sequential(  # time vector mlp
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, self.embedding_size),
            nn.GELU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        if self.diffuser_type == 'mlp1':
            self.diffu_mlp = nn.Sequential(  # diffusion mlp, in: x,c,t; out: x
                nn.Linear(self.embedding_size * 3, self.embedding_size)
            )
        elif self.diffuser_type == 'mlp2':
            self.diffu_mlp = nn.Sequential(
                nn.Linear(self.embedding_size * 3, self.embedding_size * 2),
                nn.GELU(),
                nn.Linear(self.embedding_size * 2, self.embedding_size)
            )
        elif self.diffuser_type == 'self_attention':
            self.ln_diffusion = nn.LayerNorm(self.embedding_size)
            self.w_q = nn.Linear(self.embedding_size, self.embedding_size)
            self.w_k = nn.Linear(self.embedding_size, self.embedding_size)
            self.w_v = nn.Linear(self.embedding_size, self.embedding_size)
        # without_diffusion & diff2mlp
        for p in self.diffu_mlp.parameters():
            p.requires_grad = False
        for p in self.step_mlp.parameters():
            p.requires_grad = False
        # without_diffusion & diff2mlp
        # diff2mlp
        self.mlp_generator = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.GELU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size)
        )
        # diff2mlp
        self.beta_start = 0.0001
        self.beta_end = 0.02
        if self.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif self.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()
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
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # DDIM Reverse Process, edit from alphafuse
        indices = list(range(0, self.timesteps + 1, self.linespace))  # [0,100,...,2000]
        self.sub_timesteps = len(indices)
        indices_now = [indices[i] - 1 for i in range(len(indices))]
        indices_now[0] = 0
        self.alphas_cumprod_ddim = self.alphas_cumprod[indices_now]
        self.alphas_cumprod_ddim_prev = F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0)
        self.sqrt_recipm1_alphas_cumprod_ddim = torch.sqrt(1. / self.alphas_cumprod_ddim - 1)
        self.posterior_ddim_coef1 = torch.sqrt(self.alphas_cumprod_ddim_prev) - torch.sqrt(
            1. - self.alphas_cumprod_ddim_prev) / self.sqrt_recipm1_alphas_cumprod_ddim
        self.posterior_ddim_coef2 = torch.sqrt(1. - self.alphas_cumprod_ddim_prev) / torch.sqrt(
            1. - self.alphas_cumprod_ddim)

        self.history_items_src, self.history_len_src, self.history_items_tgt, self.history_len_tgt = self._build_history(
            config, dataloader)
        self.apply(xavier_normal_initialization)
        self.user_embedding.weight.data[0, :] = 0
        self.item_emb_src.weight.data[0, :] = 0
        self.item_emb_tgt.weight.data[0, :] = 0

    def _build_history(self, config, dataloader):
        """
        Build fixed-length user interaction history matrices.
        - history_len is a constant hyper-parameter
        - If the number of interactions > history_len: randomly drop extra items
        - If the number of interactions < history_len: pad with 0
        """
        device = self.device
        max_len = config['history_len']

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
        return H_src, L_src, H_tgt, L_tgt

    def _remove_pos_from_history(self, pos_item, history_item, history_len):
        pos_item_expanded = pos_item.unsqueeze(1)
        mask = (history_item != pos_item_expanded)
        remove_counts = (~mask).sum(dim=1)
        updated_history_len = history_len - remove_counts
        filtered_history_item = history_item * mask.long()
        return filtered_history_item, updated_history_len

    def _to_total_user_id(self, user_id, domain):
        if domain == 0:
            return user_id
        u = user_id.clone()
        mask = u > self.num_users_overlap
        u[mask] = (u[mask] - self.num_users_overlap) + self.num_users_src
        return u

    ## Edit from DreamRec
    def q_sample(self, x_start, t, noise=None):  # add noise to x_start according to a series of timestamp t
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    ## Edit from DreamRec
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t,
                 t_index):  # inference one step: denoising from x generating x_start
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

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

    @torch.no_grad()
    def sample_from_noise(self, model_forward, model_forward_uncon, h):

        x = torch.randn_like(h)

        # for n in reversed(range(0, self.timesteps, self.linespace)):
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((h.shape[0],), n * self.linespace, device=h.device, dtype=torch.long)
            x = self.i_sample(model_forward, model_forward_uncon, x, h, step, n)

        return x

    def selfAttention(self, features):
        # features: [bs, #modality(id, text, img, time), d]
        #         ipdb.set_trace()

        features = self.ln_diffusion(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)

        # 目的就是做这 4 个 token 之间的自注意力融合，让模型学出“当前噪声状态该更多依赖哪种模态/时间信息
        # [bs, #modality, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        # average pooling
        y = features.mean(dim=-2)  # [bs, d]

        return y

    ## Edit from DreamRec
    def denoise_step(self, x, h, step):
        t = self.step_mlp(step)
        if self.diffuser_type in ("mlp1", "mlp2"):
            res = self.diffu_mlp(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == "self_attention":
            tokens = torch.stack([x, h, t], dim=1)  # [B, 3, D]
            res = self.selfAttention(tokens)
        return res

    def denoise_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0], device=self.device))
        h = torch.cat([h.view(1, self.embedding_size)] * x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type in ("mlp1", "mlp2"):
            res = self.diffu_mlp(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == "self_attention":
            tokens = torch.stack([x, h, t], dim=1)  # [B, 3, D]
            res = self.selfAttention(tokens)
        return res

    def get_user_representation(self, user, history_item, history_len, user_domain, item_domain):
        if item_domain == 0:
            history_item_e = self.item_emb_src(history_item)  # [B, L, D]
        else:
            history_item_e = self.item_emb_tgt(history_item)  # [B, L, D]
        user_total = self._to_total_user_id(user, user_domain)  # [B]
        user_e = self.user_embedding(user_total)  # [B, D]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)
        return UI_aggregation_e

    def get_UI_aggregation(self, user_e, history_item_e, history_len):
        r"""Get the combined vector of user and historically interacted items

        Args:
            user_e (torch.Tensor): User's feature vector, shape: [user_num, embedding_size]
            history_item_e (torch.Tensor): History item's feature vector,
                shape: [user_num, max_history_len, embedding_size]
            history_len (torch.Tensor): User's history length, shape: [user_num]

        Returns:
            torch.Tensor: Combined vector of user and item sequences, shape: [user_num, embedding_size]
        """
        if self.aggregator == "mean":
            pos_item_sum = history_item_e.sum(dim=1)
            # [user_num, embedding_size]
            out = pos_item_sum / (history_len + 1.e-10).unsqueeze(1)
        elif self.aggregator in ["user_attention", "self_attention"]:
            mask = (torch.abs(history_item_e).sum(dim=-1) > 1e-8).int()
            history_item_e = self.ln_1(history_item_e)  # [user_num, max_history_len, embedding_size]
            key = self.W_k(history_item_e)
            if self.aggregator == "user_attention":
                attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)  # [user_num, max_history_len]
            elif self.aggregator == "self_attention":
                attention = self.W_q(key).squeeze(2)
            attention_stable = attention - attention.max(dim=1, keepdim=True).values
            e_attention = torch.exp(attention_stable)
            e_attention = e_attention * mask

            attention_weight = e_attention / (e_attention.sum(dim=1, keepdim=True) + 1.0e-10)
            out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
            # out = self.ln_2(out)
        elif self.aggregator == "transformer":
            mask = torch.abs(history_item_e).sum(dim=-1) < 1e-8
            fully_padded = mask.all(dim=1)
            mask[fully_padded, 0] = False
            transformer_input = history_item_e
            transformer_input = self.ln_1(transformer_input)
            transformer_input = self.emb_dropout(transformer_input)
            transformer_output = self.transformer_encoder(
                transformer_input,
                src_key_padding_mask=mask
            )
            if self.mean_pooling:
                mask = ~mask.unsqueeze(-1)  # [B, L, 1]
                masked_output = transformer_output * mask.float()
                sum_output = masked_output.sum(dim=1)
                valid_length = history_len.unsqueeze(1).float()
                out = sum_output / (valid_length + 1e-10)
            else:
                last_valid_pos = (mask.size(1) - 1) - mask.flip(dims=[1]).int().argmin(dim=1)  # [batch_size]
                last_valid_pos[mask.all(dim=1)] = 0
                out = gather_indexes(transformer_output, last_valid_pos)
        # Combined vector of user and item sequences
        out = self.UI_map(out)
        UI_aggregation_e = self.gamma * user_e + (1 - self.gamma) * out
        # UI_aggregation_e = self.ln_2(UI_aggregation_e)
        return UI_aggregation_e

    def run_diffusion_process(self, target_item_emb, condition_emb):
        """
        Args:
            target_item_emb: x_start, [B, D]
            condition_emb: h (condition), [B, D]

        Returns:
            x_start: [B, D]
            predicted_x: [B, D]
        """
        B = target_item_emb.shape[0]
        n = torch.randint(0, self.timesteps, (B,), device=self.device).long()  # [B]
        noise = torch.randn_like(target_item_emb)  # [B, D]
        x_noisy = self.q_sample(x_start=target_item_emb, t=n, noise=noise)  # [B, D]
        h = self.add_uncon(condition_emb)  # [B, D]
        predicted_x = self.denoise_step(x=x_noisy, h=h, step=n)  # [B, D]
        return target_item_emb, predicted_x

    def add_uncon(self, h):
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - self.p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        h = h * mask + self.none_embedding(torch.tensor([0], device=self.device)) * (1 - mask)
        return h

    def domain_condition_generator(self, s_UI_aggregation_e, t_UI_aggregation_e, domain):
        """
        domain embedding as Q, UI_aggregation_e as K=V

        Args:
            s_UI_aggregation_e: [B, D]
            t_UI_aggregation_e: [B, D]
            domain: scalar (e.g. 0 or 1)

        Returns:
            [B, D]
        """

        if 1:
            UI_aggregation_e = t_UI_aggregation_e if domain == 1 else s_UI_aggregation_e
            return UI_aggregation_e
        else:
            gate_input = torch.cat([s_UI_aggregation_e, t_UI_aggregation_e], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, D] 或 [B, 1]
            UI_aggregation_e = gate * s_UI_aggregation_e + (1 - gate) * t_UI_aggregation_e  # [B, D]

            domain_idx = (torch.ones([s_UI_aggregation_e.shape[0]],
                                     device=s_UI_aggregation_e.device) * domain).int()  # [B], e.g., all 0 for source
            """Q = self.domain_emb(domain_idx).unsqueeze(1)      # [B, 1, D]
            K = V = UI_aggregation_e.unsqueeze(1)         # [B, 1, D]
            output, _ = self.domain_attn(Q, K, V)         # [B, 1, D]"""
            domain_bias = self.domain_emb(domain_idx)  # [B, D]
            output = UI_aggregation_e + domain_bias
            return output.squeeze(1)  # [B, D]

    def calculate_loss(self, interaction, epoch_idx):
        user_id = [interaction["users_src"], interaction["users_tgt"]] # source, target
        item_id = [interaction["pos_items_src"], interaction["pos_items_tgt"]] # source, target
        neg_item_id = [interaction["neg_items_src"], interaction["neg_items_tgt"]]

        losses = []
        for domain in [0, 1]:  # source, target
            user = user_id[domain]
            pos_item = item_id[domain]
            neg_item = neg_item_id[domain]

            neg_item_seq = neg_item.reshape((self.neg_seq_len, -1)).T  # [B, neg_seq_len]
            user_number = int(len(user) / self.neg_seq_len)
            user = user[0:user_number]  # [B]
            pos_item = pos_item[0:user_number]  # [B]

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

            # Aggregated Domain-Specific Interactions
            s_UI_aggregation_e = self.get_user_representation(user, s_items, s_len, user_domain=domain, item_domain=0)  # [B, D]
            t_UI_aggregation_e = self.get_user_representation(user, t_items, t_len, user_domain=domain, item_domain=1)  # [B, D]
            if domain == 0:
                pos_item_e = self.item_emb_src(pos_item)  # [B, D]
            else:
                pos_item_e = self.item_emb_tgt(pos_item)  # [B, D]

            # Domain condition generator
            UI_aggregation_e = self.domain_condition_generator(s_UI_aggregation_e, t_UI_aggregation_e, domain)  # [B, D]

            if domain == 0:
                neg_item_seq_e = self.item_emb_src(neg_item_seq)  # [B, neg_seq_len, D]
            else:
                neg_item_seq_e = self.item_emb_tgt(neg_item_seq)  # [B, neg_seq_len, D]
            pos_item_score = torch.mul(UI_aggregation_e.unsqueeze(1), pos_item_e.unsqueeze(1)).sum(dim=2)
            neg_item_score = torch.mul(UI_aggregation_e.unsqueeze(1), neg_item_seq_e).sum(dim=2)

            if self.loss_n == 'bce':
                pos_label = torch.ones_like(pos_item_score)
                neg_label = torch.zeros_like(neg_item_score)
                loss = self.bceloss(self.sigmoid(pos_item_score), pos_label) + \
                       self.bceloss(self.sigmoid(neg_item_score), neg_label)
            elif self.loss_n == 'bpr':
                loss = self.bprloss(pos_item_score, neg_item_score)
            elif self.loss_n == 'mse':
                loss = self.mseloss(UI_aggregation_e, pos_item_e)
            # without_diffusion & diff2mlp
            # x_start, predicted_x = self.run_diffusion_process(pos_item_e, UI_aggregation_e)
            # loss += F.mse_loss(x_start, predicted_x)
            # without_diffusion & diff2mlp
            # diff2mlp
            pred_item_e = self.mlp_generator(UI_aggregation_e)
            loss += F.mse_loss(pred_item_e, pos_item_e)
            # diff2mlp
            losses.append(loss)
        # return_loss = losses[0]
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

        user_domain = 1 if is_warm else 0
        # no need remove pos_item from history because history only contains training interactions
        s_UI_aggregation_e = self.get_user_representation(user, s_items, s_len, user_domain=user_domain, item_domain=0)  # [B, D]
        t_UI_aggregation_e = self.get_user_representation(user, t_items, t_len, user_domain=user_domain, item_domain=1)  # [B, D]
        # Domain condition generator
        UI_aggregation_e = self.domain_condition_generator(s_UI_aggregation_e, t_UI_aggregation_e, 1)  # [B, D]

        # without_diffusion & diff2mlp
        # x = self.sample_from_noise(self.denoise_step, self.denoise_uncon, UI_aggregation_e)
        # x = UI_aggregation_e
        x = self.mlp_generator(UI_aggregation_e)
        # without_diffusion & diff2mlp
        all_item_emb = self.item_emb_tgt.weight
        II_cos = torch.matmul(x, all_item_emb.T)
        II_cos[:, 0] = -1e9
        return II_cos