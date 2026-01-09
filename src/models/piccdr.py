import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from common.init import xavier_uniform_initialization
import scipy.sparse as sp
import math
import json

class PicCDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(PicCDR, self).__init__(config, dataloader)
        self.config=config
        self.feature_dim = config['feature_dim']
        self.tau = config['tau']
        self.lambda_dda = config['lambda_dda']

        self.user_emb_src = nn.Embedding(self.num_users_overlap, self.feature_dim)
        self.user_emb_tgt = nn.Embedding(self.num_users_overlap, self.feature_dim)
        self.item_emb_src = nn.Embedding(self.num_items_src, self.feature_dim)
        self.item_emb_tgt = nn.Embedding(self.num_items_tgt, self.feature_dim)

        semantic_emb = dataloader.get_modality_embs()['PicCDR_semantics']
        semantic_emb = torch.from_numpy(semantic_emb).float()
        assert semantic_emb.shape[0] == self.num_users_overlap * 3
        self.register_buffer(
            "semantic_emb_sep_src",
            semantic_emb[0:self.num_users_overlap]
        )
        self.register_buffer(
            "semantic_emb_sep_tgt",
            semantic_emb[self.num_users_overlap:2 * self.num_users_overlap]
        )
        self.register_buffer(
            "semantic_emb_inv",
            semantic_emb[2 * self.num_users_overlap:3 * self.num_users_overlap]
        )

        # Transfer Learning Net
        self.transfer_inv = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.transfer_src = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.transfer_tgt = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        # 投影语义emb -> id emb
        self.semantic_projector = nn.Sequential(
            nn.Linear(self.semantic_emb_inv.shape[1], self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.club_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.W1 = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.W2 = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        self.apply(xavier_uniform_initialization)

    def info_nce(self, x, y, tau):
        # x, y: [B, d]
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        logits = x @ y.T / tau  # [B, B]
        labels = torch.arange(x.size(0), device=x.device)
        return F.cross_entropy(logits, labels)

    def club(self, x, y):
        """
        x, y: [B, d]
        returns MI upper bound estimate
        """
        x = F.normalize(x, dim=1)
        y_proj = F.normalize(self.club_projector(y), dim=1)

        pos = F.cosine_similarity(x, y_proj, dim=1)
        y_neg = y_proj[torch.randperm(y_proj.size(0))]
        neg = F.cosine_similarity(x, y_neg, dim=1)

        return (pos - neg).mean()

    def conditional_club(self, s, h1, h2):
        # s, h1, h2: [B, d]
        x = F.normalize(self.W1(self.club_projector(s)) * h2, dim=1)
        y = F.normalize(self.W2(self.club_projector(h1)) * h2, dim=1)

        pos = F.cosine_similarity(x, y, dim=1)
        y_neg = y[torch.randperm(y.size(0))]
        neg = F.cosine_similarity(x, y_neg, dim=1)

        return (pos - neg).mean()

    def calculate_loss(self, interaction, epoch_idx):
        users = interaction['users'] - 1
        src_pos = interaction['pos_items_src'] - 1
        src_neg = interaction['neg_items_src'] - 1
        tgt_pos = interaction['pos_items_tgt'] - 1
        tgt_neg = interaction['neg_items_tgt'] - 1

        uniq_users, inv_idx = torch.unique(users, return_inverse=True)
        h_u_src_u = self.user_emb_src(uniq_users)
        h_u_tgt_u = self.user_emb_tgt(uniq_users)
        h_cat_u = torch.cat([h_u_src_u, h_u_tgt_u], dim=1)

        s_u_inv = self.transfer_inv(h_cat_u)
        s_u_src = self.transfer_src(h_cat_u)
        s_u_tgt = self.transfer_tgt(h_cat_u)

        t_inv_proj = self.semantic_projector(self.semantic_emb_inv[uniq_users].detach())
        t_src_proj = self.semantic_projector(self.semantic_emb_sep_src[uniq_users].detach())
        t_tgt_proj = self.semantic_projector(self.semantic_emb_sep_tgt[uniq_users].detach())

        # L_sem
        L_sem = (self.info_nce(s_u_inv, t_inv_proj, self.tau)
                 + self.info_nce(s_u_src, t_src_proj, self.tau) + self.info_nce(s_u_tgt, t_tgt_proj, self.tau))

        # L_mul_maxTerm: MI maximization term
        L_mul_maxTerm = self.info_nce(s_u_inv, h_u_src_u, self.tau) + self.info_nce(s_u_inv, h_u_tgt_u, self.tau)

        # L_dis
        L_dis = self.club(s_u_inv, s_u_src) + self.club(s_u_inv, s_u_tgt)

        # L_mul_minTerm: MI minimization term
        L_mul_minTerm = (
                self.conditional_club(s_u_inv, h_u_src_u, h_u_tgt_u) +
                self.conditional_club(s_u_inv, h_u_tgt_u, h_u_src_u)
        )

        L_dda = L_sem + L_mul_maxTerm - L_dis - L_mul_minTerm

        # 下面是推荐损失
        s_inv = s_u_inv[inv_idx]
        s_src = s_u_src[inv_idx]
        s_tgt = s_u_tgt[inv_idx]
        z_src = s_inv + s_src
        z_tgt = s_inv + s_tgt

        i_pos_src = self.item_emb_src(src_pos)
        i_neg_src = self.item_emb_src(src_neg)
        i_pos_tgt = self.item_emb_tgt(tgt_pos)
        i_neg_tgt = self.item_emb_tgt(tgt_neg)

        pos_src = torch.sum(z_src * i_pos_src, dim=1)
        neg_src = torch.sum(z_src * i_neg_src, dim=1)
        L_rec_src = -torch.log(torch.sigmoid(pos_src - neg_src) + 1e-8).mean()
        pos_tgt = torch.sum(z_tgt * i_pos_tgt, dim=1)
        neg_tgt = torch.sum(z_tgt * i_neg_tgt, dim=1)
        L_rec_tgt = -torch.log(torch.sigmoid(pos_tgt - neg_tgt) + 1e-8).mean()
        L_rec = L_rec_src + L_rec_tgt

        # 最后是推荐损失 + dda损失
        loss = L_rec + self.lambda_dda * L_dda
        return loss

    def full_sort_predict(self, interaction, is_warm):
        user = interaction[0].long() - 1

        h_u_src = self.user_emb_src(user)  # [B, d]
        h_u_tgt = self.user_emb_tgt(user)  # [B, d]

        h_cat = torch.cat([h_u_src, h_u_tgt], dim=1)  # [B, 2d]

        s_u_inv = self.transfer_inv(h_cat)  # [B, d]
        s_u_tgt = self.transfer_tgt(h_cat)  # [B, d]
        z_u_tgt = s_u_inv + s_u_tgt  # [B, d]

        all_item_emb = self.item_emb_tgt.weight  # [num_items_tgt, d]

        scores = torch.matmul(z_u_tgt, all_item_emb.T)  # [B, num_items_tgt]
        padding = torch.zeros((user.size(0), 1), device=self.device) # [B, 1]
        scores = torch.concat((padding, scores), dim=1)  # [B, num_items_tgt+1]
        return scores

def _extract_item_profile(meta):
    if meta is None:
        return {
            "title": "None",
            "description": "None",
            "categories": "None",
            "price": "None",
            "brand": "None"
        }

    # -------- title --------
    title = meta.get("title")
    if not title or not isinstance(title, str):
        title = "None"

    # -------- description --------
    description = meta.get("description")
    if not description or not isinstance(description, str):
        description = "None"

    # -------- categories (flatten -> quoted string list) --------
    raw_categories = meta.get("categories")
    categories_str = "None"

    if isinstance(raw_categories, list):
        flat_categories = []

        for path in raw_categories:
            if isinstance(path, list):
                for c in path:
                    if isinstance(c, str) and c.strip():
                        flat_categories.append(c.strip())
            elif isinstance(path, str) and path.strip():
                flat_categories.append(path.strip())

        # 去重，保持顺序
        seen = set()
        flat_categories = [
            c for c in flat_categories
            if not (c in seen or seen.add(c))
        ]

        if flat_categories:
            joined = ", ".join(flat_categories)
            categories_str = f"\"{joined}\""

    # -------- price --------
    price = meta.get("price")
    if isinstance(price, (int, float)):
        price = str(price)
    else:
        price = "None"

    # -------- brand --------
    brand = meta.get("brand")
    if not brand or not isinstance(brand, str):
        brand = "None"

    return {
        "title": title,   # String / "None" (都是字符串)
        "description": description, # String / "None" (都是字符串)
        "categories": categories_str, # String / "None" (都是字符串)
        "price": price, # String / "None" (都是字符串)
        "brand": brand # String / "None" (都是字符串)
    }

def _reviews_list_to_string(reviews_list):
    """
    Convert a list of review strings into the PicCDR-style string format:
    [
    "review1"
    "review2"
    ...
    ]
    """
    if not reviews_list:
        return "[]"

    lines = ["["]
    for r in reviews_list:
        if not isinstance(r, str):
            continue
        lines.append(f"\"{r}\"")
    lines.append("]")

    return "\n".join(lines)

def _user_profile_to_string(user_profile):
    """
    Convert a user's item-level profile (list of dict[str, str])
    into a JSON-like string block.

    IMPORTANT:
    - All values are strings (including "None")
    - Values are preserved verbatim
    - No semantic processing is performed here
    """
    if not user_profile:
        return "[]"

    lines = ["["]

    for item in user_profile:
        if not isinstance(item, dict):
            continue

        fields = []
        for k, v in item.items():
            # v 必须是字符串，包括 "None"
            # 只做最小必要的转义，防止破坏结构
            v_escaped = v.replace("\\", "\\\\").replace("\"", "\\\"")
            fields.append(f"\"{k}\": \"{v_escaped}\"")

        item_str = "{ " + ", ".join(fields) + " }"
        lines.append(item_str)

    lines.append("]")
    return "\n".join(lines)

def _build_user_semantic_string(profile_str, src_reviews_str, tgt_reviews_str):
    """
    Build the final user-level semantic string with the following order:
    1) Source domain feedback
    2) Target domain feedback
    3) User profile
    """
    return (
        "SOURCE DOMAIN FEEDBACK:\n"
        f"{src_reviews_str}\n\n"
        "TARGET DOMAIN FEEDBACK:\n"
        f"{tgt_reviews_str}\n\n"
        "USER PROFILE:\n"
        f"{profile_str}"
    )

def extract_first_json_block(text: str):
    """
    Extract the first valid JSON object from a string.
    Returns dict if successful, otherwise None.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()

    # 快速路径：本身就是合法 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试从文本中提取第一个 {...} 块
    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None

def _is_valid_piccdr_json(obj):
    """
    Check whether obj satisfies the required PicCDR JSON schema.
    """
    if not isinstance(obj, dict):
        return False

    required_top_keys = {
        "common_preference",
        "source_specific_preference",
        "target_specific_preference"
    }

    if set(obj.keys()) != required_top_keys:
        return False

    for k in required_top_keys:
        block = obj.get(k)
        if not isinstance(block, dict):
            return False
        if "summarization" not in block or "reasoning" not in block:
            return False
        if not isinstance(block["summarization"], str):
            return False
        if not isinstance(block["reasoning"], str):
            return False

    return True

def extract_PicCDR_semantics_modality_data(config, modality, interaction, id_mapping, reviews, metadata):
    users_raw_id = id_mapping['src']['id2user'][1:]

    users_reviews_src = [[] for _ in range(len(users_raw_id))]
    users_reviews_tgt = [[] for _ in range(len(users_raw_id))]
    users_profile = [[] for _ in range(len(users_raw_id))]

    # 取出每个用户在src、tgt的所有评论记录，分别组成一个List
    # 同时取出用户在src+tgt交互过的所有物品的元信息，组成users_profile
    for row in interaction['src'].itertuples(index=False):
        user_id = row.user
        item_id = row.item
        raw_user_id = id_mapping['src']['id2user'][user_id]
        raw_item_id = id_mapping['src']['id2item'][item_id]

        review = reviews['src'] .get((raw_user_id, raw_item_id))
        if review is not None:
            users_reviews_src[user_id - 1].append(review)

        meta = metadata['src'].get(raw_item_id)
        users_profile[user_id - 1].append(
            _extract_item_profile(meta)
        )
    for row in interaction['tgt'].itertuples(index=False):
        user_id = row.user
        item_id = row.item
        raw_user_id = id_mapping['tgt']['id2user'][user_id]
        raw_item_id = id_mapping['tgt']['id2item'][item_id]

        review = reviews['tgt'] .get((raw_user_id, raw_item_id))
        if review is not None:
            users_reviews_tgt[user_id - 1].append(review)

        meta = metadata['tgt'].get(raw_item_id)
        users_profile[user_id - 1].append(
            _extract_item_profile(meta)
        )

    # 把每个用户评论List，转化为一个字符串
    users_reviews_src_str = [_reviews_list_to_string(user_reviews) for user_reviews in users_reviews_src]
    users_reviews_tgt_str = [_reviews_list_to_string(user_reviews) for user_reviews in users_reviews_tgt]

    # 把用户Profile转化为字符串
    users_profile_str = [_user_profile_to_string(profile) for profile in users_profile]

    # 最终输入到LLM的每个用户的String，融合了上面三个字符串
    prompts = [
        _build_user_semantic_string(
            users_profile_str[i],
            users_reviews_src_str[i],
            users_reviews_tgt_str[i]
        )
        for i in range(len(users_profile_str))
    ]

    # system prompt写在本文件内
    system_prompt = PICCDR_USER_SYSTEM_PROMPT

    # 开始生成用户解耦的Profile
    from openai import OpenAI
    client = OpenAI(
        api_key=config['openai_api_key'],
        base_url=config['openai_base_url'],
    )
    results = {}  # raw_user_id -> dict or str，如果LLM生成有效json，存为dict，否则是str
    error_users = []  # raw_user_id list (final failed users)

    for idx, raw_user_id in enumerate(users_raw_id):
        prompt = prompts[idx]

        success = False
        last_response_text = None

        for attempt in range(2):  # 每个用户最多两次机会
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                response_text = response.choices[0].message.content.strip()
                last_response_text = response_text

                # 尝试解析 JSON
                parsed = extract_first_json_block(response_text)

                # 校验 schema
                if parsed is not None and _is_valid_piccdr_json(parsed):
                    results[raw_user_id] = parsed
                    success = True
                    break

            except Exception:
                # 包含 JSONDecodeError / API error / 其他异常
                continue

        if not success:
            # 两次都失败
            results[raw_user_id] = last_response_text
            error_users.append(raw_user_id)

    print(f"[PicCDR LLM] Total users: {len(users_raw_id)}")
    print(f"[PicCDR LLM] Error users: {len(error_users)}")
    if error_users:
        print(f"[PicCDR LLM] Error user raw-ids: {error_users}")

    return {
        "data": results,
        "error_users": error_users
    }

def generate_PicCDR_semantics_embs(config, modality, interaction, id_mapping, modality_data):
    # 输出一个 shape = [num_users * 3, emb_dim] 的 numpy array：
    # [
    #     u1_src, u2_src, ..., uN_src,
    #     u1_tgt, u2_tgt, ..., uN_tgt,
    #     u1_common, u2_common, ..., uN_common
    # ]

    users_raw_id = id_mapping['src']['id2user'][1:]  # 去掉第一个占位符
    results = modality_data['data']
    src_sentences = []
    tgt_sentences = []
    common_sentences = []

    # 构建Emb Model的输入，即每个用户的解耦用户总结
    for raw_uid in users_raw_id:
        user_dict = results[raw_uid]
        src_sentences.append(user_dict['source_specific_preference']['summarization'])
        tgt_sentences.append(user_dict['target_specific_preference']['summarization'])
        common_sentences.append( user_dict['common_preference']['summarization'])
    all_sentences = src_sentences + tgt_sentences + common_sentences

    # 生成Semantic embeddings
    if 'text-embedding-3' in modality['emb_model']:
        if not config.get('openai_api_key'):
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        try:
            from openai import OpenAI
            from tqdm import tqdm
            import time

            client_kwargs = {'api_key': config['openai_api_key']}
            if config.get('openai_base_url'):
                client_kwargs['base_url'] = config['openai_base_url']
            client = OpenAI(**client_kwargs)

            embs = []
            bs = modality['emb_batch_size']
            for i in tqdm(range(0, len(all_sentences), bs), desc='Encoding PicCDR semantics'):
                batch = all_sentences[i:i + bs]

                try:
                    responses = client.embeddings.create(
                        input=batch,
                        model=modality['emb_model']
                    )
                    for r in responses.data:
                        embs.append(r.embedding)
                except Exception as e:
                    print(f'[Embedding Error] batch {i}-{i+bs}: {e}')

                    # retry with truncation
                    new_batch = [
                        sent[:8000] if isinstance(sent, str) and len(sent) > 8000 else sent
                        for sent in batch
                    ]

                    time.sleep(2)
                    responses = client.embeddings.create(
                        input=new_batch,
                        model=modality['emb_model']
                    )
                    for r in responses.data:
                        embs.append(r.embedding)
            embs = np.array(embs, dtype=np.float32)
        except ImportError:
            raise ImportError("Please install openai")
    else:
        raise ValueError(f"Unsupported embedding model: {modality['emb_model']}")
    return embs

def generate_PicCDR_semantics_final_embs(config, modality, interaction, id_mapping, embs):
    # 输入的embs：
    #     shape = [num_users * 3, emb_dim]
    #     order = [src_all, tgt_all, common_all]
    # 输出pca_embs: [num_users * 3, emb_pca]，在YAML模态中配置的emb_pca，代表着最终嵌入的维度

    num_users = len(id_mapping['src']['id2user']) - 1  # 去掉padding
    emb_dim = embs.shape[1]
    # 检查一下上一步输入的embs是否满足 num_users * 3
    assert embs.shape[0] == num_users * 3, \
        f"Expected {num_users * 3} user embeddings, got {embs.shape[0]}"

    src_embs = embs[0:num_users]
    tgt_embs = embs[num_users:2 * num_users]
    common_embs = embs[2 * num_users:3 * num_users]

    if modality['emb_pca'] == emb_dim:
        pca_src = src_embs
        pca_tgt = tgt_embs
        pca_common = common_embs
    elif modality['emb_pca'] < emb_dim:
        try:
            # 三种表征分开PCA降维
            from sklearn.decomposition import PCA
            pca_src_model = PCA(
                n_components=modality['emb_pca'],
                whiten=True
            )
            pca_src = pca_src_model.fit_transform(src_embs)

            pca_tgt_model = PCA(
                n_components=modality['emb_pca'],
                whiten=True
            )
            pca_tgt = pca_tgt_model.fit_transform(tgt_embs)

            pca_common_model = PCA(
                n_components=modality['emb_pca'],
                whiten=True
            )
            pca_common = pca_common_model.fit_transform(common_embs)
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")

    else:
        raise ValueError(
            f"The dimension of emb_pca must be <= emb_dim "
            f"(got emb_pca={modality['emb_pca']}, emb_dim={emb_dim})"
        )

    # 降维后重新拼回 [num_users * 3, emb_pca]
    pca_embs = np.concatenate(
        [pca_src, pca_tgt, pca_common],
        axis=0
    ).astype(np.float32)
    return pca_embs


PICCDR_USER_SYSTEM_PROMPT = """
You will serve as an assistant to summarize the user’s common and domain-specific content based on his reviews in both two domains and his user profile. The goal is to disentangle the user’s preferences into shared preferences and preferences that are specific to the source domain and the target domain.
Here are the instructions:
1. Each interacted product in the user profile is described in JSON format
with the following attributes, where missing values are set to "None":
{
    "title": "the name of the product",
    "description": "a description of the product",
    "categories": "several tags describing the product",
    "price": "the price of the product",
    "brand": "the brand of the product"
}
2. User feedback in each domain is provided in the following List format:
[
    "the first feedback",
    "the second feedback",
    "the third feedback",
    ...
]

The information I will give you:
SOURCE DOMAIN FEEDBACK: A list of user reviews from the source domain.
TARGET DOMAIN FEEDBACK: A list of user reviews from the target domain.
USER PROFILE: A list of JSON strings describing the products the user has interacted with.

Requirements:
1. Please provide your answer in JSON format, following this structure:
{
  "common_preference": {
    "summarization": "summarize the user’s preferences that are shared across both the source and target domains",
    "reasoning": "briefly explain your reasoning for the summarization"
  },
  "source_specific_preference": {
    "summarization": "summarize the user’s preferences that are only reflected in the source domain",
    "reasoning": "briefly explain your reasoning for the summarization"
  },
  "target_specific_preference": {
    "summarization": "summarize the user’s preferences that are only reflected in the target domain",
    "reasoning": "briefly explain your reasoning for the summarization"
  }
}
2. The three summarizations should be disentangled and mutually exclusive, representing shared preferences, source-specific preferences, and target-specific preferences.
3. Each "summarization" should be no longer than 100 words.
4. Each "reasoning" has no word limit.
5. Do not provide any other text outside the JSON string.
""".strip()

