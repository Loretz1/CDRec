import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import math
import numpy as np
import json
from multiprocessing import Pool, cpu_count

class LLM_Diff_ddim_userprofile_tag(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LLM_Diff_ddim_userprofile_tag, self).__init__(config, dataloader)

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

        semantic_emb = dataloader.get_modality_embs()['CrossDomain_semantics_tag']
        pad = torch.zeros(1, semantic_emb.shape[1], dtype=torch.float32)
        semantic_emb = torch.cat([pad, torch.from_numpy(semantic_emb)], dim=0)
        self.register_buffer(
            "user_text_emb",
            semantic_emb
        )

        # 映射文本emb -> id emb
        text_dim = semantic_emb.shape[1]
        self.text_mapper = nn.Sequential(
            nn.Linear(text_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

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
        text_src = self.user_text_emb[users_src]  # [B, text_dim]
        text_src = self.text_mapper(text_src)  # [B, D]
        i_pos_src = self.emb_item_src(pos_items_src)  # [B, D]
        i_neg_src = self.emb_item_src(neg_items_src)  # [B, D]

        # 这里t是随机采的，不是对称采样
        B = u_src.size(0)
        t_src = torch.randint(low=0, high=self.diff_src.timesteps, size=(B,), device=u_src.device)
        diff_loss_src, u_src_denoised = self.diff_src.p_losses(x_start=u_src, cond=text_src, t=t_src, loss_type="l2")
        u_src_final = u_src + self.config["lambda_user_emb"] * u_src_denoised # 残差连接

        pos_score_src = (u_src_final * i_pos_src).sum(dim=-1)
        neg_score_src = (u_src_final * i_neg_src).sum(dim=-1)
        bpr_loss_src = self.bpr_loss(pos_score_src, neg_score_src)

        # tgt
        users_tgt_local = users_tgt
        offset = self.num_users_src - self.num_users_overlap
        users_tgt_global = users_tgt_local + (users_tgt_local > self.num_users_overlap).long() * offset # tgt单域用户需要加一个偏移值，从而取到正确的id emb
        u_tgt = self.emb_user(users_tgt_global)  # [B, D]
        text_tgt = self.user_text_emb[users_tgt_global]  # [B, text_dim]
        text_tgt = self.text_mapper(text_tgt)  # [B, D]
        i_pos_tgt = self.emb_item_tgt(pos_items_tgt)  # [B, D]
        i_neg_tgt = self.emb_item_tgt(neg_items_tgt)  # [B, D]

        B = u_tgt.size(0)
        t_tgt = torch.randint(low=0, high=self.diff_tgt.timesteps, size=(B,), device=u_tgt.device)
        diff_loss_tgt, u_tgt_denoised = self.diff_tgt.p_losses(x_start=u_tgt, cond=text_tgt, t=t_tgt, loss_type="l2")
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
            text = self.user_text_emb[users_global]
            text = self.text_mapper(text)
            _, u_denoised, _, _, _ = self.diff_tgt.sample(u, cond=text)
        else:
            u = self.emb_user(users)
            text = self.user_text_emb[users]
            text = self.text_mapper(text)
            _, u_denoised, _, _, _ = self.diff_tgt.sample(u, cond=text)

        u_final = u + self.config["lambda_user_emb"] * u_denoised

        item_emb = self.emb_item_tgt.weight
        scores = torch.matmul(u_final, item_emb.t())
        scores[:, 0] = 0.0
        return scores


# 😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄 扩散模型 😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄


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

    def p_losses(self, x_start, cond, t, loss_type = "l2"):
        device = x_start.device
        t = t.to(device)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)  # [B, D] on device
        tokens = torch.stack([x_noisy, cond, t_emb], dim=1)  # [B, 3, D]
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
    def ddim_sample_step(self, x_t, cond, t, step_idx):
        t_emb = self.get_timestep_embedding(t, self.embedding_dim)
        tokens = torch.stack([x_t, cond, t_emb], dim=1)
        x_start = self.selfAttention(tokens)

        x_prev = (
                self.posterior_ddim_coef1[step_idx] * x_start +
                self.posterior_ddim_coef2[step_idx] * x_t
        )
        return x_prev

    # DDIM
    @torch.no_grad()
    def sample(self, x_start, cond):
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
            x = self.ddim_sample_step(x, cond, t, i)

        return x_start, x, None, None, None



# 😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄 用户Profile 😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄


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
        "INTERACTIONS FROM CLOTHING:\n"
        f"{user_profile_string['src']}\n\n"
        "INTERACTIONS FROM SPORTS:\n"
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


def extract_CrossDomain_semantics_tag_modality_data(config, modality, interaction, id_mapping, raw_data_list):
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

    # 处理user_item_profiles中每个用户每个域的交互List，把它变成一个字符串
    user_profile_strings = {
        user_id: _user_item_profile_to_string(profile)
        for user_id, profile in user_item_profiles.items()
    }

    user_prompts = _build_all_user_prompt_strings(user_profile_strings)

    """
    Step 2:
    call LLM to summarize users (multiprocess).
    """
    import os
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    os.environ["OPENAI_BASE_URL"] = config['openai_base_url']

    results, error_users = _run_user_summarization_multiprocess(
        user_prompts=user_prompts,
        system_prompt=CROSSDOMAIN_USER_SYSTEM_PROMPT,
        max_retry=2,
        num_workers=20,
    )

    return {
        "data": results,
        "error_users": error_users
    }


def generate_CrossDomain_semantics_tag_embs(
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
    user_summaries = modality_data["data"]

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
    # Step 3: batch encode summaries
    # =========================================================

    flat_texts = []  # 所有 tag（List[str]）
    flat_user_idx = []  # 每个 tag 属于哪个 user（index）
    for user_idx, tag_list in enumerate(ordered_tag_lists):
        for tag in tag_list:
            flat_texts.append(tag)
            flat_user_idx.append(user_idx)

    num_flat = len(flat_texts)
    if num_flat == 0:
        raise ValueError("No tag texts to embed.")

    if 'text-embedding-3' in modality['emb_model']:
        from openai import OpenAI

        client = OpenAI(
            api_key=config["openai_api_key"],
            base_url=config.get("openai_base_url", None),
        )

        flat_embs = []
        for start in range(0, num_flat, batch_size):
            end = min(start + batch_size, num_flat)
            batch_texts = flat_texts[start:end]

            response = client.embeddings.create(
                model=embedding_model,
                input=batch_texts,
            )

            for emb_obj in response.data:
                emb = np.asarray(emb_obj.embedding, dtype=np.float32)
                if normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-12)
                flat_embs.append(emb)

        flat_embs = np.stack(flat_embs, axis=0)  # [num_tags, D]

    emb_dim = flat_embs.shape[1]
    num_users = len(ordered_raw_users)

    sum_embs = np.zeros((num_users, emb_dim), dtype=np.float32)
    cnt = np.zeros((num_users,), dtype=np.int32)

    for i in range(num_flat):
        uidx = flat_user_idx[i]
        sum_embs[uidx] += flat_embs[i]
        cnt[uidx] += 1

    raw_user_to_emb = {}

    for uidx, raw_user in enumerate(ordered_raw_users):
        user_emb = sum_embs[uidx] / cnt[uidx]
        raw_user_to_emb[raw_user] = user_emb

    # Sanity check
    assert len(raw_user_to_emb) == len(ordered_raw_users), \
        "Some user embeddings are missing after batch encoding"

    # =========================================================
    # Step 4: stack embeddings in final order
    # =========================================================
    final_embeddings = [
        raw_user_to_emb[raw_user] for raw_user in ordered_raw_users
    ]

    final_embeddings = np.stack(final_embeddings, axis=0)

    return final_embeddings



def generate_CrossDomain_semantics_tag_final_embs(config, modality, interaction, id_mapping, modality_embs):
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
One category is about clothing, shoes and jewelry.
The other category is about sports and outdoors.

The information I will give you:
INTERACTIONS FROM CLOTHING: A LIST of user interactions with items related to clothing, shoes and jewelry.
INTERACTIONS FROM SPORTS: A LIST of user interactions with items related to sports and outdoors.

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

