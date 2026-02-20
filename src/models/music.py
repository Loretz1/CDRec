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

class MuSiC(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(MuSiC, self).__init__(config, dataloader)

        self.config = config
        self.embedding_dim = config["embedding_dim"]

        # =========================
        # 1️⃣ load semantic emb
        # =========================
        semantic = dataloader.get_modality_embs()["CrossDomain_semantics_music"]

        # =========================
        # 2️⃣ split semantic layout
        # =========================
        s = 0
        src_user = semantic[s:s+self.num_users_src]
        s += self.num_users_src
        tgt_user = semantic[s:s+self.num_users_tgt]
        s += self.num_users_tgt
        src_item = semantic[s:s+self.num_items_src]
        s += self.num_items_src
        tgt_item = semantic[s:s+self.num_items_tgt]

        # =========================
        # 3️⃣ register domain buffers (+padding)
        # =========================
        pad = np.zeros((1, src_user.shape[1]), dtype=np.float32)

        self.register_buffer(
            "src_user_semantic_emb",
            torch.from_numpy(np.vstack([pad, src_user]))
        )

        self.register_buffer(
            "tgt_user_semantic_emb",
            torch.from_numpy(np.vstack([pad, tgt_user]))
        )

        self.register_buffer(
            "src_item_semantic_emb",
            torch.from_numpy(np.vstack([pad, src_item]))
        )

        self.register_buffer(
            "tgt_item_semantic_emb",
            torch.from_numpy(np.vstack([pad, tgt_item]))
        )

        semantic_dim = semantic.shape[1]
        self.semantic_proj = nn.Linear(semantic_dim, self.embedding_dim)

        # item ID embedding
        self.src_item_id_emb = nn.Embedding(self.num_items_src + 1, self.embedding_dim)
        self.tgt_item_id_emb = nn.Embedding(self.num_items_tgt + 1, self.embedding_dim)

        # =========================
        # 5️⃣ diffusion module
        # =========================
        self.T =self.config["T"]

        self.diffusion = MuSiCDiffusion(
            dim=self.embedding_dim,
            cond_dim=self.embedding_dim,
            T=self.T,
        )

        # =========================
        # 6️⃣ beta schedule
        # =========================
        beta = torch.linspace(self.config['beta_start'], self.config['beta_end'], self.T)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha", alpha)
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alpha))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - self.alpha_bar))

        # =========================
        # 7️⃣ rating loss
        # =========================
        self.bpr = BPRLoss()

        # =========================
        # 7️⃣ history
        # =========================
        (
            self.history_src_user_src,
            self.history_src_user_tgt,
            self.history_tgt_user_src,
            self.history_tgt_user_tgt
        ) = self._build_padded_history(dataloader)

        # =========================
        # 8️⃣ init weights
        # =========================
        self.apply(xavier_uniform_initialization)


    def _p_sample(self, x_t, t, cond):
        """
        One reverse step: x_t -> x_{t-1} using predicted noise eps_theta.
        x_t: [B, d]
        t:  [B] long
        cond: [B, d]
        """
        t_emb = sinusoidal_embedding(t, self.embedding_dim)
        eps = self.diffusion(x_t, t_emb, cond)  # predict noise

        beta_t = self.beta[t].unsqueeze(-1)
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].unsqueeze(-1)

        # DDPM mean
        mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_ab_t * eps)

        # noise (no noise when t==0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().unsqueeze(-1)

        # variance = beta_t (DDPM original)
        x_prev = mean + nonzero_mask * torch.sqrt(beta_t) * noise
        return x_prev

    def _p_sample_loop(self, cond, steps=None):
        """
        Run reverse diffusion starting from Gaussian noise.
        cond: [B, d]
        steps: if None -> use self.T; else run a shortened schedule of 'steps'
        """
        B, d = cond.shape
        device = cond.device

        if steps is None:
            steps = self.T

        # Use a shortened schedule by selecting timesteps linearly
        if steps == self.T:
            timesteps = torch.arange(self.T - 1, -1, -1, device=device)
        else:
            # e.g., steps=50 from T=1000
            idx = torch.linspace(self.T - 1, 0, steps, device=device).long()
            timesteps = idx

        x = torch.randn(B, d, device=device)

        for ti in timesteps:
            t = torch.full((B,), int(ti.item()), device=device, dtype=torch.long)
            x = self._p_sample(x, t, cond)

        return x

    def _build_padded_history(self, dataloader):

        L = int(self.config["history_len"])
        device = self.device
        dataset = dataloader.dataset

        # ---------- src user space ----------
        history_src_user_src = torch.zeros(
            (self.num_users_src + 1, L),
            dtype=torch.long,
            device=device
        )
        history_src_user_tgt = torch.zeros(
            (self.num_users_src + 1, L),
            dtype=torch.long,
            device=device
        )

        # ---------- tgt user space ----------
        history_tgt_user_src = torch.zeros(
            (self.num_users_tgt + 1, L),
            dtype=torch.long,
            device=device
        )
        history_tgt_user_tgt = torch.zeros(
            (self.num_users_tgt + 1, L),
            dtype=torch.long,
            device=device
        )

        # ===== src domain interactions =====
        for u, items in dataset.positive_items_src.items():
            if not items:
                continue

            items = list(items)[-L:]
            history_src_user_src[u, :len(items)] = torch.tensor(items, device=device)

            # overlap users visible in tgt-user space
            if u <= self.num_users_overlap:
                history_tgt_user_src[u, :len(items)] = torch.tensor(items, device=device)

        # ===== tgt domain interactions =====
        for u, items in dataset.positive_items_tgt.items():
            if not items:
                continue

            items = list(items)[-L:]
            history_tgt_user_tgt[u, :len(items)] = torch.tensor(items, device=device)

            # overlap users visible in src-user space
            if u <= self.num_users_overlap:
                history_src_user_tgt[u, :len(items)] = torch.tensor(items, device=device)

        return (
            history_src_user_src,
            history_src_user_tgt,
            history_tgt_user_src,
            history_tgt_user_tgt
        )

    def _aggregate_hist(self, users, domain):
        """
        Aggregate item id embeddings from padded history.
        users: [B]
        domain: "src" or "tgt"
        """

        if domain == "src":
            hist = self.history_tgt_user_src[users]  # 注意 user space
            emb_table = self.src_item_id_emb
        else:
            hist = self.history_tgt_user_tgt[users]
            emb_table = self.tgt_item_id_emb

        # hist: [B, L]
        mask = hist > 0

        # [B, L, d]
        emb = emb_table(hist)

        # mask padding
        emb = emb * mask.unsqueeze(-1)

        # mean
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        out = emb.sum(1) / denom

        return out

    def calculate_loss(self, interaction, epoch_idx):
        users = interaction["users"]
        pos_items = interaction["pos_items"]
        neg_items = interaction["neg_items"]

        device = users.device
        B = users.size(0)

        # =========================================================
        # 1️⃣ semantic → projection
        # =========================================================
        tgt_u_sem = self.tgt_user_semantic_emb[users]
        tgt_u = self.semantic_proj(tgt_u_sem)  # [B, d]
        src_hist = self._aggregate_hist(users, "src")
        tgt_hist = self._aggregate_hist(users, "tgt")
        tgt_u = (1- self.config['id_w']) * tgt_u + self.config['id_w'] * (src_hist + tgt_hist) / 2


        # =========================================================
        # 2️⃣ sample timestep
        # =========================================================
        t = torch.randint(0, self.T, (B,), device=device)

        # =========================================================
        # 3️⃣ overlap mask
        # =========================================================
        overlap = users <= self.num_users_overlap
        side = ~overlap

        # =========================================================
        # 4️⃣ prepare containers
        # =========================================================
        pred_full = torch.zeros_like(tgt_u)

        loss_overlap = torch.tensor(0.0, device=device)
        loss_side = torch.tensor(0.0, device=device)

        # =========================================================
        # 5️⃣ overlap diffusion
        # =========================================================
        if overlap.any():
            idx = overlap.nonzero(as_tuple=True)[0]

            tgt = tgt_u[idx]

            src_sem = self.src_user_semantic_emb[users[idx]]
            src = self.semantic_proj(src_sem)

            xt, noise = _q_sample(tgt, t[idx], self.alpha_bar)
            t_emb = sinusoidal_embedding(t[idx], self.embedding_dim)

            pred = self.diffusion(xt, t_emb, src)

            pred_full[idx] = pred
            loss_overlap = F.mse_loss(pred, noise)

        # =========================================================
        # 6️⃣ side diffusion
        # =========================================================
        if side.any():
            idx = side.nonzero(as_tuple=True)[0]

            tgt = tgt_u[idx]

            xt, noise = _q_sample(tgt, t[idx], self.alpha_bar)
            t_emb = sinusoidal_embedding(t[idx], self.embedding_dim)

            pred = self.diffusion(xt, t_emb, tgt)

            pred_full[idx] = pred
            loss_side = F.mse_loss(pred, noise)

        # =========================================================
        # 7️⃣ unified residual user
        # =========================================================

        # residual refinement (embedding diffusion common trick)
        u_final = tgt_u - self.config['w_u'] * pred_full

        # =========================================================
        # 8️⃣ ranking loss
        # =========================================================
        pos_sem = self.tgt_item_semantic_emb[pos_items]
        neg_sem = self.tgt_item_semantic_emb[neg_items]
        pos_sem = self.semantic_proj(pos_sem)
        neg_sem = self.semantic_proj(neg_sem)
        pos_id = self.tgt_item_id_emb(pos_items)
        neg_id = self.tgt_item_id_emb(neg_items)

        pos = (1 - self.config['id_w']) * pos_sem + self.config['id_w'] * pos_id
        neg = (1 - self.config['id_w']) * neg_sem + self.config['id_w'] * neg_id

        pos_score = (u_final * pos).sum(-1)
        neg_score = (u_final * neg).sum(-1)

        loss_bpr = self.bpr(pos_score, neg_score)

        # =========================================================
        # 9️⃣ total
        # =========================================================
        loss = self.config['lambda_diff'] * (loss_overlap + loss_side) + loss_bpr

        return loss

    def full_sort_predict(self, interaction, is_warm):
        # cold-only: users are from src domain id space
        users = interaction[0].long()

        # 1) cond from src semantic
        src_sem = self.src_user_semantic_emb[users]
        cond = self.semantic_proj(src_sem)  # [B, d]
        src_hist = self._aggregate_hist(users, "src")
        cond = (1- self.config['id_w']) * cond + self.config['id_w'] * (src_hist)

        # 2) reverse diffusion to get tgt user embedding
        sample_steps = self.T
        u_gen = self._p_sample_loop(cond, steps=sample_steps)  # [B, d]

        # (optional) residual mix with cond (often stabilizes)
        u_final = cond + self.config['w_u'] * (u_gen - cond)

        # 3) all tgt item embeddings
        tgt_items_sem = self.tgt_item_semantic_emb[1:]  # remove padding
        tgt_items_sem = self.semantic_proj(tgt_items_sem)  # [It, d]
        tgt_items_id = self.tgt_item_id_emb.weight[1:]
        tgt_items = (1 - self.config['id_w']) * tgt_items_sem + self.config['id_w'] * tgt_items_id

        # 4) full sort scores
        scores = torch.matmul(u_final, tgt_items.t())  # [B, It]
        return scores


# 🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭 扩散 🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭🤭

class MuSiCDiffusion(nn.Module):
    def __init__(self, dim, cond_dim, T=1000):
        super().__init__()

        self.T = T
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x_t, t_emb, cond):
        h = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(h)


def sinusoidal_embedding(t, dim):
    device = t.device
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


def _q_sample(x0, t, alpha_bar):
    noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alpha_bar[t]).unsqueeze(-1)
    sqrt_1mab = torch.sqrt(1 - alpha_bar[t]).unsqueeze(-1)
    xt = sqrt_ab * x0 + sqrt_1mab * noise
    return xt, noise


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




