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

class LLM4CDSR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LLM4CDSR, self).__init__(config, dataloader)

        self.config = config
        self.embedding_dim = int(config["embedding_dim"])
        self.history_len = int(config.get("history_len", 40))
        self.bpr_loss = BPRLoss()

        # ========= 1) trainable local embedding layers =========
        # 用户全局编码
        self.emb_user = nn.Embedding(
            self.num_users_src + self.num_users_tgt - self.num_users_overlap + 1,
            self.embedding_dim,
            padding_idx=0
        )
        self.emb_item_src = nn.Embedding(self.num_items_src + 1, self.embedding_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1, self.embedding_dim, padding_idx=0)

        # ========= 2) load frozen LLM modality embeddings =========
        # dataloader.get_modality_embs()[modality_name] 形状: [U + Is + It, D_text]
        modality_name = "CrossDomain_semantics_LLM4CDSR"
        semantic_emb = dataloader.get_modality_embs()[modality_name]
        semantic_emb = torch.from_numpy(semantic_emb).float()  # [U + Is + It, D_text]

        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        num_src_items = self.num_items_src
        num_tgt_items = self.num_items_tgt
        assert semantic_emb.shape[0] == num_users + num_src_items + num_tgt_items, \
            "Semantic embedding size mismatch for LLM4CDSR modality."

        # split (no padding inside semantic_emb)
        user_profile_text = semantic_emb[:num_users]  # [U, D_text]
        src_item_text = semantic_emb[num_users: num_users + num_src_items]  # [Is, D_text]
        tgt_item_text = semantic_emb[num_users + num_src_items:]  # [It, D_text]

        # pad to align embedding tables (index 0 is padding)
        pad_u = torch.zeros(1, user_profile_text.shape[1])
        pad_is = torch.zeros(1, src_item_text.shape[1])
        pad_it = torch.zeros(1, tgt_item_text.shape[1])

        user_profile_text = torch.cat([pad_u, user_profile_text], dim=0)   # [U+1, D_text]
        src_item_text = torch.cat([pad_is, src_item_text], dim=0)          # [Is+1, D_text]
        tgt_item_text = torch.cat([pad_it, tgt_item_text], dim=0)          # [It+1, D_text]

        # frozen buffers
        self.register_buffer("user_text_emb", user_profile_text)  # for alignment
        self.register_buffer("src_item_text_emb", src_item_text)          # for global item emb
        self.register_buffer("tgt_item_text_emb", tgt_item_text)

        text_dim = int(user_profile_text.shape[1])

        # ========= 3) adapter for items: E_LLM -> E_tilde (d) =========
        # paper: “trainable adapter transform LLM item emb into final item emb”
        self.adapter = nn.Sequential(
            nn.Linear(text_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # ========= 4) g(·) for profile alignment: P_LLM -> p_tilde (d) =========
        # paper 3.4.3: two-layer MLP g(·) to shrink to d
        self.profile_mlp = nn.Sequential(
            nn.Linear(text_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # ========= 5) Tri-thread encoders =========
        # three threads: src sequence, tgt sequence, mixed sequence
        self.encoder_src = SimpleSAEncoder(self.embedding_dim)
        self.encoder_tgt = SimpleSAEncoder(self.embedding_dim)
        self.encoder_global = SimpleSAEncoder(self.embedding_dim)

        # ========= 6) init =========
        self.apply(xavier_uniform_initialization)
        self.emb_user.weight.data[0, :] = 0
        self.emb_item_src.weight.data[0, :] = 0
        self.emb_item_tgt.weight.data[0, :] = 0

        # ========= 7) build user history =========
        self._build_user_history(dataloader)

    def _build_user_history(self, dataloader):
        """
        构建四个 history tensor：
          history_src_user_src
          history_src_user_tgt
          history_tgt_user_src
          history_tgt_user_tgt

        用户顺序严格遵循你的编码规则：
          1 ~ num_overlap
          overlap+1 ~ src-only
          ... ~ tgt-only
        """

        positive_items_src = dataloader.dataset.positive_items_src
        positive_items_tgt = dataloader.dataset.positive_items_tgt

        num_union_users = (
                self.num_users_src
                + self.num_users_tgt
                - self.num_users_overlap
        )

        L = self.history_len

        # allocate
        self.history_src_user_src = torch.zeros(
            (num_union_users + 1, L), dtype=torch.long, device=self.device
        )
        self.history_src_user_tgt = torch.zeros(
            (num_union_users + 1, L), dtype=torch.long, device=self.device
        )
        self.history_tgt_user_src = torch.zeros(
            (num_union_users + 1, L), dtype=torch.long, device=self.device
        )
        self.history_tgt_user_tgt = torch.zeros(
            (num_union_users + 1, L), dtype=torch.long, device=self.device
        )

        # 1️⃣ 填充 src 域用户的 src 历史
        for user_id, item_list in positive_items_src.items():
            items = list(item_list)[:L]
            if len(items) > 0:
                self.history_src_user_src[user_id, :len(items)] = torch.LongTensor(items)

        # 2️⃣ 填充 tgt 域用户的 tgt 历史
        for user_id, item_list in positive_items_tgt.items():
            items = list(item_list)[:L]
            if len(items) > 0:
                self.history_tgt_user_tgt[user_id, :len(items)] = torch.LongTensor(items)

        # 3️⃣ overlap 用户的跨域历史
        # overlap 编码为 1 ~ num_users_overlap
        for u in range(1, self.num_users_overlap + 1):

            # overlap 在 tgt 的历史 → src_user_tgt
            if u in positive_items_tgt:
                items = list(positive_items_tgt[u])[:L]
                if len(items) > 0:
                    self.history_src_user_tgt[u, :len(items)] = torch.LongTensor(items)

            # overlap 在 src 的历史 → tgt_user_src
            if u in positive_items_src:
                items = list(positive_items_src[u])[:L]
                if len(items) > 0:
                    self.history_tgt_user_src[u, :len(items)] = torch.LongTensor(items)

    def _to_global_user_id(self, users: torch.Tensor, is_warm: bool) -> torch.Tensor:
        """
        users: [B] from sampler space
        is_warm:
          True  -> users are tgt-space ids
          False -> users are src-space ids
        return:
          global union-space ids for self.emb_user
        """
        if not is_warm:
            return users  # src-space id already in union encoding for overlap+src-only

        # warm: tgt-space users
        # overlap users stay same: 1..num_overlap
        # tgt-only users (id > num_overlap) need offset by (num_users_src - num_overlap)
        offset = (self.num_users_src - self.num_users_overlap)
        return torch.where(
            users <= self.num_users_overlap,
            users,
            users + offset
        )

    def calculate_loss(self, interaction, epoch_idx):
        """
        Paper:
          - SRS loss: Eq.(4), with logit fusion Eq.(3)
          - Reg loss: Eq.(6)
          - Profile align: Eq.(7)
          - Total: Eq.(8)
        """
        users_src = interaction["users_src"]          # [B1]
        pos_src   = interaction["pos_items_src"]      # [B1]
        neg_src   = interaction["neg_items_src"]      # [B1]

        users_tgt = interaction["users_tgt"]          # [B2]
        pos_tgt   = interaction["pos_items_tgt"]      # [B2]
        neg_tgt   = interaction["neg_items_tgt"]      # [B2]

        alpha = float(self.config.get("alpha", 1.0))
        beta  = float(self.config.get("beta",  1.0))
        gamma = float(self.config.get("gamma", 0.2))   # Eq.(6) 温度
        tau   = float(self.config.get("tau",   0.2))   # Eq.(7) 温度

        # ---------- helpers: build three-thread user representations ----------
        def encode_src_users(u: torch.Tensor):
            """
            Return:
              uA:     local-src user repr  [B, d]  (Eq.(2): f_{thetaA}(E^A))
              u_til:  global user repr      [B, d]  (Eq.(2): f_{theta~}(E~))
            """
            # local seq in domain A
            seqA_ids = self.history_src_user_src[u]               # [B, L]
            maskA = (seqA_ids != 0)
            seqA_emb = self.emb_item_src(seqA_ids)                # [B, L, d]
            uA = self.encoder_src(seqA_emb, maskA)                # [B, d]

            # mixed/global seq: 用两域拼接(无时间时的简化 mixed)，先转 global item emb 再 encoder_global
            seqA_g_ids = self.history_src_user_src[u]             # [B, L]
            seqB_g_ids = self.history_src_user_tgt[u]             # [B, L]
            # global emb：adapter(LLM_emb)
            gA = self.adapter(self.src_item_text_emb[seqA_g_ids]) # [B, L, d]
            gB = self.adapter(self.tgt_item_text_emb[seqB_g_ids]) # [B, L, d]
            seqG = torch.cat([gA, gB], dim=1)                     # [B, 2L, d]
            maskG = torch.cat([(seqA_g_ids != 0), (seqB_g_ids != 0)], dim=1)  # [B, 2L]
            u_til = self.encoder_global(seqG, maskG)              # [B, d]

            u_tilde = self.config['user_llm_emb_w'] * u_til + self.emb_user(u)  # [B, d]
            return uA, u_tilde

        def encode_tgt_users(u: torch.Tensor):
            """
            Return:
              uB:     local-tgt user repr   [B, d]
              u_til:  global user repr      [B, d]
            """
            # local seq in domain B
            seqB_ids = self.history_tgt_user_tgt[u]               # [B, L]
            maskB = (seqB_ids != 0)
            seqB_emb = self.emb_item_tgt(seqB_ids)                # [B, L, d]
            uB = self.encoder_tgt(seqB_emb, maskB)                # [B, d]

            # mixed/global seq (拼接)
            seqA_g_ids = self.history_tgt_user_src[u]             # [B, L]
            seqB_g_ids = self.history_tgt_user_tgt[u]             # [B, L]
            gA = self.adapter(self.src_item_text_emb[seqA_g_ids]) # [B, L, d]
            gB = self.adapter(self.tgt_item_text_emb[seqB_g_ids]) # [B, L, d]
            seqG = torch.cat([gA, gB], dim=1)                     # [B, 2L, d]
            maskG = torch.cat([(seqA_g_ids != 0), (seqB_g_ids != 0)], dim=1)
            u_til = self.encoder_global(seqG, maskG)              # [B, d]

            u_global_users = self._to_global_user_id(u, is_warm=False)  # [B]
            u_tilde = self.config['user_llm_emb_w'] * u_til + self.emb_user(u_global_users)
            return uB, u_tilde

        # ---------- (1) SRS loss for src domain: Eq.(3)(4) ----------
        uA, uG_src = encode_src_users(users_src)                  # [B1,d], [B1,d]

        # local item emb e_i^A
        eA_pos = self.emb_item_src(pos_src)                       # [B1,d]
        eA_neg = self.emb_item_src(neg_src)                       # [B1,d]
        # global item emb e~_i = adapter(e_i^LLM)
        # LLM global embedding
        g_pos_llm = self.adapter(self.src_item_text_emb[pos_src])
        g_neg_llm = self.adapter(self.src_item_text_emb[neg_src])
        g_pos = self.config['item_llm_emb_w'] * g_pos_llm + eA_pos
        g_neg = self.config['item_llm_emb_w'] * g_neg_llm + eA_neg

        # Eq.(3) logit fusion is concat-dot = dot(u~,e~)+dot(uA,eA)
        pos_score_src = (uG_src * g_pos).sum(dim=-1) + (uA * eA_pos).sum(dim=-1)  # [B1]
        neg_score_src = (uG_src * g_neg).sum(dim=-1) + (uA * eA_neg).sum(dim=-1)  # [B1]
        loss_srs_src = -F.logsigmoid(pos_score_src - neg_score_src).mean()

        # ---------- (2) SRS loss for tgt domain: Eq.(3)(4) ----------
        uB, uG_tgt = encode_tgt_users(users_tgt)

        eB_pos = self.emb_item_tgt(pos_tgt)
        eB_neg = self.emb_item_tgt(neg_tgt)

        g_pos2_llm = self.adapter(self.tgt_item_text_emb[pos_tgt])
        g_neg2_llm = self.adapter(self.tgt_item_text_emb[neg_tgt])
        g_pos2 = self.config['item_llm_emb_w'] * g_pos2_llm + eB_pos
        g_neg2 = self.config['item_llm_emb_w'] * g_neg2_llm + eB_neg

        pos_score_tgt = (uG_tgt * g_pos2).sum(dim=-1) + (uB * eB_pos).sum(dim=-1)
        neg_score_tgt = (uG_tgt * g_neg2).sum(dim=-1) + (uB * eB_neg).sum(dim=-1)
        loss_srs_tgt = -F.logsigmoid(pos_score_tgt - neg_score_tgt).mean()

        # ---------- (3) Contrastive regularization: Eq.(6) ----------
        # 论文正样本：同一用户 mixed 序列中 co-occur 的 A/B item。
        # 这里用“同一 batch 中同时出现于 users_src & users_tgt 的 overlap 用户”的 (pos_src,pos_tgt) 做正对。
        loss_reg = torch.tensor(0.0, device=self.device)

        # build user->pos dict for current batch
        src_map = {}
        for u, i in zip(users_src.tolist(), pos_src.tolist()):
            src_map[u] = i
        tgt_map = {}
        for u, i in zip(users_tgt.tolist(), pos_tgt.tolist()):
            tgt_map[u] = i

        overlap_users = [u for u in src_map.keys() if u in tgt_map]
        if len(overlap_users) >= 2:
            # Z pairs
            src_items_z = torch.tensor([src_map[u] for u in overlap_users], device=self.device)
            tgt_items_z = torch.tensor([tgt_map[u] for u in overlap_users], device=self.device)

            eA = self.adapter(self.src_item_text_emb[src_items_z])   # [Z,d]
            eB = self.adapter(self.tgt_item_text_emb[tgt_items_z])   # [Z,d]
            # Eq.(6) is in-batch contrastive; symmetric anchor swap
            loss_reg = _info_nce_symmetric(eA, eB, temperature=gamma)

        # ---------- (4) Profile alignment loss: Eq.(7) ----------
        # 对齐 u~ 与 p~ = g(P_LLM)
        loss_profile = torch.tensor(0.0, device=self.device)

        # src batch alignment
        if users_src.numel() >= 2:
            p_src = self.profile_mlp(self.user_text_emb[users_src])   # [B1,d]
            loss_profile = loss_profile + _info_nce_symmetric(uG_src, p_src, temperature=tau)

        # tgt batch alignment
        if users_tgt.numel() >= 2:
            p_tgt = self.profile_mlp(self.user_text_emb[users_tgt])   # [B2,d]
            loss_profile = loss_profile + _info_nce_symmetric(uG_tgt, p_tgt, temperature=tau)

        # ---------- (5) Total loss: Eq.(8) ----------
        total = (loss_srs_src + loss_srs_tgt) + alpha * loss_reg + beta * loss_profile
        return total

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()  # [B]
        device = users.device

        # -------- select histories by user space --------
        if is_warm:
            # warm eval uses tgt-user histories
            seq_src_ids = self.history_tgt_user_src[users].to(device)  # [B, L]
            seq_tgt_ids = self.history_tgt_user_tgt[users].to(device)  # [B, L]
        else:
            # cold/src eval uses src-user histories
            seq_src_ids = self.history_src_user_src[users].to(device)  # [B, L]
            seq_tgt_ids = self.history_src_user_tgt[users].to(device)  # [B, L]

        # -------- global (mixed) preference u_tilde --------
        # global item emb = adapter(LLM item emb)
        g_src = self.adapter(self.src_item_text_emb[seq_src_ids])  # [B, L, d]
        g_tgt = self.adapter(self.tgt_item_text_emb[seq_tgt_ids])  # [B, L, d]
        seq_global = torch.cat([g_src, g_tgt], dim=1)  # [B, 2L, d]
        mask_global = torch.cat([seq_src_ids != 0, seq_tgt_ids != 0], dim=1)  # [B, 2L]
        u_tilde = self.encoder_global(seq_global, mask_global)  # [B, d]
        u_global_users = self._to_global_user_id(users, is_warm=is_warm)
        u_tilde = self.config['user_llm_emb_w'] * u_tilde + self.emb_user(u_global_users)

        # -------- local target preference u_B --------
        # For warm users: real tgt history
        # For cold/src users: tgt history likely all-0, u_B becomes near-zero (safe)
        tgt_seq_emb = self.emb_item_tgt(seq_tgt_ids)  # [B, L, d]
        tgt_mask = (seq_tgt_ids != 0)  # [B, L]
        u_B = self.encoder_tgt(tgt_seq_emb, tgt_mask)  # [B, d]

        # -------- build item embeddings for scoring --------
        # local tgt items: e_i^B
        E_local = self.emb_item_tgt.weight  # [It+1, d]
        # global tgt items: e_tilde_i = adapter(E_LLM_i)
        E_llm = self.adapter(self.tgt_item_text_emb)  # [It+1, d]
        E_global = self.config['item_llm_emb_w'] * E_llm + E_local  # fused global item embedding

        # -------- Eq.(3)-style fusion scoring --------
        scores = torch.matmul(u_tilde, E_global.t()) + torch.matmul(u_B, E_local.t())  # [B, It+1]
        scores[:, 0] = 0.0
        return scores


class SimpleSAEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        """
        x: [B, L, D]
        pad_mask: [B, L]  True = valid
        return: [B, D]
        """

        # 如果全 padding，直接返回 0
        valid_counts = pad_mask.sum(dim=1)  # [B]

        # mask
        mask = pad_mask.unsqueeze(-1).float()  # [B, L, 1]
        x = x * mask

        # mean pool
        denom = valid_counts.clamp(min=1).unsqueeze(-1).float()
        pooled = x.sum(dim=1) / denom

        # 对完全空序列用户，强制设为 0
        pooled[valid_counts == 0] = 0.0

        return pooled


def _info_nce_symmetric(x: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric in-batch InfoNCE:
      L = L(x->y) + L(y->x)
    x,y: [Z, D]
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    logits_xy = (x @ y.t()) / temperature          # [Z, Z]
    logits_yx = (y @ x.t()) / temperature          # [Z, Z]
    labels = torch.arange(x.size(0), device=x.device)

    loss_xy = F.cross_entropy(logits_xy, labels)
    loss_yx = F.cross_entropy(logits_yx, labels)
    return loss_xy + loss_yx

# 😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀 Profile总结↓ 😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀

def _build_subseq_prompt_from_titles(titles: list) -> str:
    """
    EXACT template from LLM4CDSR paper (Section 3.4.2)
    """
    body = ""
    for t in titles:
        if isinstance(t, str) and t.strip():
            body += f"{t.strip()};\n"

    if body == "":
        body = "None;\n"

    return (
        "Assume you are a consumer who is shopping online.\n"
        "You have shown interest in the following commodities:\n"
        f"{body}"
        "The commodities are segmented by '\\n'.\n"
        "Please conclude it not beyond 50 words. "
        "Do not only evaluate one specific commodity but illustrate the interests overall."
    )


def _build_overall_prompt_from_summaries(summaries: list) -> str:
    """
    EXACT template from LLM4CDSR paper (Section 3.4.2)
    """
    body = ""
    for s in summaries:
        if isinstance(s, str) and s.strip():
            body += f"{s.strip()};\n"

    if body == "":
        body = "None;\n"

    return (
        "Assume you are a consumer and there are preference\n"
        "demonstrations from several aspects as follows:\n"
        f"{body}"
        "Please illustrate your preference with fewer than 100 words"
    )


def _compute_counts_from_id_mapping(id_mapping):
    src_users = set(id_mapping["src"]["id2user"][1:])
    tgt_users = set(id_mapping["tgt"]["id2user"][1:])

    overlap_users = src_users & tgt_users
    union_users = src_users | tgt_users

    num_users = len(union_users)
    num_src_items = len(id_mapping["src"]["id2item"]) - 1
    num_tgt_items = len(id_mapping["tgt"]["id2item"]) - 1

    return {
        "num_users": num_users,
        "num_src_items": num_src_items,
        "num_tgt_items": num_tgt_items,
        "num_overlap_users": len(overlap_users),
    }


def _load_and_slice_item_embs(path, counts):
    arr = np.load(path)

    expected = counts["num_users"] + counts["num_src_items"] + counts["num_tgt_items"]
    if arr.shape[0] != expected:
        raise ValueError(
            f"Embedding size mismatch: got {arr.shape[0]}, "
            f"expected {expected} (= users + src_items + tgt_items)"
        )

    start = counts["num_users"]
    mid = start + counts["num_src_items"]
    end = mid + counts["num_tgt_items"]

    src_item_embs = arr[start:mid]
    tgt_item_embs = arr[mid:end]

    return np.concatenate([src_item_embs, tgt_item_embs], axis=0)


def _build_user_mixed_sequences(interaction, id_mapping):
    user2seq = {}

    for dom in ["src", "tgt"]:
        df = interaction[dom]
        id2user = id_mapping[dom]["id2user"]
        id2item = id_mapping[dom]["id2item"]

        for row in df.itertuples(index=False):
            raw_u = id2user[row.user]
            raw_i = id2item[row.item]
            user2seq.setdefault(raw_u, [])
            user2seq[raw_u].append((dom, raw_i))

    return user2seq


def _get_title(raw_item, metadata_dom, dataset_name: str):
    meta = metadata_dom.get(raw_item)
    if not isinstance(meta, dict):
        return "None"

    if dataset_name == "Amazon2014":
        title = meta.get("title")
        return title.strip() if isinstance(title, str) and title.strip() else "None"

    elif dataset_name == "Douban":
        labels = meta.get("labels")
        return labels.strip() if isinstance(labels, str) and labels.strip() else "None"

    else:
        return "None"


def _raw_item_to_global_index(dom, raw_item, id_mapping, counts):
    if dom == "src":
        local_id = id_mapping["src"]["item2id"][raw_item]
        return local_id - 1
    else:
        local_id = id_mapping["tgt"]["item2id"][raw_item]
        return counts["num_src_items"] + (local_id - 1)


def _process_single_prompt(args):
    key, prompt, model_name, max_retry, api_key, base_url = args
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)

    last_err = None
    for _ in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return {"key": key, "text": text, "success": True}
        except Exception as e:
            last_err = str(e)

    return {"key": key, "text": last_err, "success": False}


def _run_prompts(prompts, model_name, api_key, base_url, max_retry=2, num_workers=8):
    tasks = [
        (k, p, model_name, max_retry, api_key, base_url)
        for k, p in prompts.items()
    ]

    ok, err = {}, {}
    with Pool(processes=min(num_workers, cpu_count())) as pool:
        for out in pool.imap_unordered(_process_single_prompt, tasks):
            if out["success"]:
                ok[out["key"]] = out["text"]
            else:
                err[out["key"]] = out["text"]

    return ok, err


def _embed_texts_openai(config, embedding_model, texts, batch_size=1024, normalize=False):
    from openai import OpenAI

    client = OpenAI(
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url", None),
    )

    all_embs = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]

        resp = client.embeddings.create(
            model=embedding_model,
            input=batch,
        )

        for obj in resp.data:
            emb = np.asarray(obj.embedding, dtype=np.float32)
            if normalize:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            all_embs.append(emb)

    return np.stack(all_embs, axis=0)


def extract_CrossDomain_semantics_LLM4CDSR_modality_data(
    config, modality, interaction, id_mapping, raw_data_list
):
    """
    物品总结的text-emb没有啥创新，直接复用其它方法处理好的item-text-emb
    1. 读取 item_emb_save_path
    2. 切出 item embedding
    3. KMeans 分簇
    4. 用户 K 子总结 + overall 总结
    """

    # ---- raw metadata ----
    if config["dataset"] == "Amazon2014":
        _, metadata = raw_data_list
    else:
        metadata = raw_data_list[2]

    # ---- counts ----
    counts = _compute_counts_from_id_mapping(id_mapping)

    # ---- load item embeddings ----
    item_emb_path = modality["item_emb_path"]
    all_item_embs = _load_and_slice_item_embs(item_emb_path, counts)

    # ---- clustering ----
    from sklearn.cluster import KMeans
    K = int(modality.get("cluster_k", 2))
    kmeans = KMeans(n_clusters=K, random_state=999, n_init="auto")
    cluster_ids = kmeans.fit_predict(all_item_embs)

    # ---- user sequences ----
    user2seq = _build_user_mixed_sequences(interaction, id_mapping)

    # ---- build subseq prompts ----
    subseq_prompts = {}
    for raw_user, seq in user2seq.items():
        buckets = [[] for _ in range(K)]

        for dom, raw_item in seq:
            idx = _raw_item_to_global_index(dom, raw_item, id_mapping, counts)
            cid = int(cluster_ids[idx])

            title = _get_title(raw_item, metadata[dom], config["dataset"])
            buckets[cid].append(title)

        for k in range(K):
            key = f"{raw_user}__sub{k}"
            subseq_prompts[key] = _build_subseq_prompt_from_titles(buckets[k])

    # ---- run LLM for subseq ----
    data_sub, err_sub = _run_prompts(
        subseq_prompts,
        model_name=modality.get("llm_chat_model", "gpt-4o-mini"),
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url"),
        max_retry=modality.get("max_retry", 2),
        num_workers=modality.get("num_workers", 20),
    )

    # =========================
    # Early stop if any subseq error
    # =========================
    if len(err_sub) > 0:
        return {
            "counts": counts,
            "cluster_k": K,
            "data_subseq": data_sub,
            "error_subseq": err_sub,
            "data_users": {},  # MUST be empty
            "error_users": {},  # MUST be empty
        }

    # ---- build overall prompts ----
    overall_prompts = {}
    for raw_user in user2seq.keys():
        subs = []
        for k in range(K):
            key = f"{raw_user}__sub{k}"
            subs.append(data_sub.get(key, "None"))
        overall_prompts[raw_user] = _build_overall_prompt_from_summaries(subs)

    # ---- run LLM for overall ----
    data_overall, err_overall = _run_prompts(
        overall_prompts,
        model_name=modality.get("llm_chat_model", "gpt-4o-mini"),
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url"),
        max_retry=modality.get("max_retry", 2),
        num_workers=modality.get("num_workers", 20),
    )

    return {
        "counts": counts,
        "cluster_k": K,
        "data_subseq": data_sub,
        "error_subseq": err_sub,
        "data_users": data_overall,
        "error_users": err_overall,
    }


def generate_CrossDomain_semantics_LLM4CDSR_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    """
    将 user summary 文本转成 embedding，
    并替换原始 npy 中的 user 部分，
    最终返回 [users | src_items | tgt_items] embedding。
    """

    # ---------- counts ----------
    counts = modality_data["counts"]
    num_users = counts["num_users"]
    num_src_items = counts["num_src_items"]
    num_tgt_items = counts["num_tgt_items"]

    # ---------- 读取原始 embedding ----------
    item_emb_save_path = modality["item_emb_path"]
    full_arr = np.load(item_emb_save_path)

    # sanity check
    expected_rows = num_users + num_src_items + num_tgt_items
    if full_arr.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding row mismatch: got {full_arr.shape[0]}, expected {expected_rows}"
        )

    # ---------- 构造 union user 顺序 ----------
    src_id2user = id_mapping["src"]["id2user"][1:]
    src_users_set = set(src_id2user)
    tgt_id2user = id_mapping["tgt"]["id2user"][1:]
    tgt_only_users = [u for u in tgt_id2user if u not in src_users_set]
    ordered_users = list(src_id2user) + tgt_only_users

    if len(ordered_users ) != num_users:
        raise ValueError("Union user size mismatch")

    # ---------- 取出 user summary ----------
    data_users = modality_data["data_users"]

    user_texts = []
    missing_users = []

    for raw_user in ordered_users :
        if raw_user in data_users:
            user_texts.append(data_users[raw_user])
        else:
            user_texts.append("None")
            missing_users.append(raw_user)

    # 可选：如果你希望缺失就报错
    # if len(missing_users) > 0:
    #     raise ValueError(f"Missing user summaries for {len(missing_users)} users")

    # ---------- 生成 user embedding ----------
    user_text_emb = _embed_texts_openai(
        config=config,
        embedding_model=modality["emb_model"],
        texts=user_texts,
        batch_size=modality.get("emb_batch_size", 1024),
        normalize=modality.get("normalize_semantic_emb", False),
    )

    # ---------- 切出 item 部分 ----------
    start = num_users
    mid = start + num_src_items
    end = mid + num_tgt_items

    src_item_emb = full_arr[start:mid]
    tgt_item_emb = full_arr[mid:end]

    # ---------- 拼接 ----------
    final_emb = np.concatenate(
        [user_text_emb, src_item_emb, tgt_item_emb],
        axis=0
    )

    return final_emb


def generate_CrossDomain_semantics_LLM4CDSR_final_embs(config, modality, interaction, id_mapping, modality_embs):
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