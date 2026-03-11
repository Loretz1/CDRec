import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import math
import numpy as np

class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_elec(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LLM_Diff_interaction_as_condition_mask_both_tag_EMA_elec, self).__init__(config, dataloader)

        self.config = config
        self.feature_dim = config['feature_dim']
        self.max_len = config["history_len"]
        self.dropout = config["dropout"]
        self.bpr_loss = BPRLoss()
        self.history_items_src, self.history_scores_src = self._build_history(dataloader, 'src')
        self.history_items_tgt, self.history_scores_tgt = self._build_history(dataloader, 'tgt')

        self.emb_user = nn.Embedding(
            self.num_users_src + self.num_users_tgt - self.num_users_overlap + 1,
            self.feature_dim,
            padding_idx=0
        )
        self.emb_item_src = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)

        self.agg = BehaviorAggregator(config)

        self.diff_src = Diffusion(config)
        self.diff_tgt = Diffusion(config)
        self.src_interaction_agg = InteractionAggregator(config)
        self.tgt_interaction_agg = InteractionAggregator(config)

        # EMA
        semantic_emb = dataloader.get_modality_embs()['CrossDomain_semantics_both_tag']  # 用户/物品text emb
        semantic_emb = torch.from_numpy(semantic_emb).float()  # [U + Is + It, D]

        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap # 检查text emb数量是否正确
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

        self.diff_tgt_ema = None
        self.ema_initialized = False
        self.global_step = 0

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

            # 固定的低噪声t
            t = torch.zeros(users.size(0), dtype=torch.long, device=self.device)

            _, u_denoised = self.diff_tgt_ema.p_losses(
                x_start=u,
                t=t,
                cond_src=cond_src,
                cond_tgt=cond_tgt,
                loss_type="l2"
            )

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



    def _build_history(self, dataloader, domain: str):
        assert domain in ["src", "tgt"]
        if domain == "src":
            num_users = self.num_users_src
            pos_items_dict = dataloader.dataset.positive_items_src
        else:
            num_users = self.num_users_tgt
            pos_items_dict = dataloader.dataset.positive_items_tgt
        history_items = torch.zeros((num_users + 1, self.max_len), dtype=torch.long, device=self.device)
        history_scores = torch.full((num_users + 1, self.max_len), fill_value=-100.0, dtype=torch.float,
                                    device=self.device)

        for u, items in pos_items_dict.items():
            items = list(items)
            n = len(items)
            if n == 0:
                continue
            if n > self.max_len:
                rand_idx = torch.randperm(n, device=self.device)[:self.max_len]
                sampled_items = torch.tensor([items[i] for i in rand_idx], dtype=torch.long, device=self.device)
                history_items[u] = sampled_items
                history_scores[u] = 1.0
            else:
                history_items[u, :n] = torch.tensor(items, dtype=torch.long, device=self.device)
                history_scores[u, :n] = 1.0
        return history_items, history_scores

    def batch_random_mask(self, items, scores, mask_rate):
        """
        items:  [B, L]  padding=0
        scores: [B, L]  valid=1.0, pad/mask=-100
        """
        B, L = items.shape

        valid = (items != 0) & (scores > -50)
        rand = torch.rand_like(scores)
        rand = rand.masked_fill(~valid, 2.0)
        valid_cnt = valid.sum(dim=1)
        keep_cnt = (valid_cnt.float() * (1 - mask_rate)).long()
        keep_cnt = torch.clamp(keep_cnt, min=1)
        order = rand.argsort(dim=1)
        pos = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        keep_mask = pos < keep_cnt.unsqueeze(1)

        scatter_mask = torch.zeros_like(valid)
        scatter_mask.scatter_(1, order, keep_mask)

        out_items = items.clone()
        out_scores = scores.clone()
        out_items[~scatter_mask] = 0
        out_scores[~scatter_mask] = -100.0
        return out_items, out_scores

    def batch_build_global(self, users: torch.Tensor, cur_domain: str, ):
        B = users.size(0)
        L = self.max_len

        global_item = torch.zeros((B, 2, L), dtype=torch.long, device=self.device)
        global_score = torch.full((B, 2, L), -100.0, dtype=torch.float, device=self.device)
        overlap_mask = (users >= 1) & (users <= self.num_users_overlap)
        # mask_other = (self.num_users_src == self.num_users_tgt) and (self.num_users_src == self.num_users_overlap)

        if cur_domain == "src":
            src_items = self.history_items_src[users]
            src_scores = self.history_scores_src[users]
            # if mask_other:
            m_items, m_scores = self.batch_random_mask(src_items, src_scores, self.config["mask_rate"])
            src_items, src_scores = m_items, m_scores
            global_item[:, 0, :] = src_items
            global_score[:, 0, :] = src_scores
            if overlap_mask.any():
                u = users[overlap_mask]
                tgt_items = self.history_items_tgt[u]
                tgt_scores = self.history_scores_tgt[u]
                # if mask_other:
                tgt_items, tgt_scores = self.batch_random_mask(tgt_items, tgt_scores, self.config["mask_rate"])
                global_item[overlap_mask, 1, :] = tgt_items
                global_score[overlap_mask, 1, :] = tgt_scores
        else:
            tgt_items = self.history_items_tgt[users]
            tgt_scores = self.history_scores_tgt[users]
            # if mask_other:
            m_items, m_scores = self.batch_random_mask(tgt_items, tgt_scores, self.config["mask_rate"])
            tgt_items, tgt_scores = m_items, m_scores
            global_item[:, 1, :] = tgt_items
            global_score[:, 1, :] = tgt_scores
            if overlap_mask.any():
                u = users[overlap_mask]
                src_items = self.history_items_src[u]
                src_scores = self.history_scores_src[u]
                # if mask_other:
                src_items, src_scores = self.batch_random_mask(src_items, src_scores, self.config["mask_rate"])
                global_item[overlap_mask, 0, :] = src_items
                global_score[overlap_mask, 0, :] = src_scores
        return global_item, global_score

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

        # src user
        uid_src = users_src
        user_emb_src = self.emb_user(uid_src)

        global_item_src, global_score_src = self.batch_build_global(users_src, "src") # [B, 2, L]
        global_item_emb_src = torch.cat([
            self.emb_item_src(global_item_src[:, 0, :]),  # [B, L, D]
            self.emb_item_tgt(global_item_src[:, 1, :])  # [B, L, D]
        ], dim=1)  # [B, 2L, D]

        u_base_src = self.agg(user_emb_src, global_item_emb_src, global_score_src)
        u_base_src = F.dropout(u_base_src, self.dropout, training=self.training)   # 这个是输入扩散之前的src user表征

        hist_src_items = global_item_src[:, 0, :]  # [B, L]
        hist_tgt_items = global_item_src[:, 1, :]  # [B, L]
        hist_src = self.emb_item_src(hist_src_items)  # [B, L, D]
        hist_tgt = self.emb_item_tgt(hist_tgt_items)  # [B, L, D]
        cond_src = self.src_interaction_agg(hist_src, u_base_src)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_base_src)

        B = u_base_src.size(0)
        t_src = torch.randint(low=0, high=self.diff_src.timesteps, size=(B,), device=u_base_src.device)
        diff_loss_src, u_denoised_src = self.diff_src.p_losses(x_start=u_base_src, t=t_src, cond_src=cond_src,
                                                           cond_tgt=cond_tgt, loss_type="l2")
        u_final_src = u_base_src + self.config["lambda_user_emb"] * u_denoised_src

        # tgt user
        uid_tgt = users_tgt.clone()
        mask = users_tgt > self.num_users_overlap
        uid_tgt[mask] = users_tgt[mask] + (self.num_users_src - self.num_users_overlap)
        user_emb_tgt = self.emb_user(uid_tgt)  # [B, D]

        global_item_tgt, global_score_tgt = self.batch_build_global(users_tgt, "tgt") # [B, 2, L]
        global_item_emb_tgt = torch.cat([
            self.emb_item_src(global_item_tgt[:, 0, :]),  # [B, L, D]
            self.emb_item_tgt(global_item_tgt[:, 1, :])  # [B, L, D]
        ], dim=1)  # [B, 2L, D]

        u_base_tgt = self.agg(user_emb_tgt, global_item_emb_tgt, global_score_tgt)
        u_base_tgt = F.dropout(u_base_tgt, self.dropout, training=self.training)   # 这个是输入扩散之前的src user表征

        hist_src_items = global_item_tgt[:, 0, :]  # [B, L]
        hist_tgt_items = global_item_tgt[:, 1, :]  # [B, L]
        hist_src = self.emb_item_src(hist_src_items)  # [B, L, D]
        hist_tgt = self.emb_item_tgt(hist_tgt_items)  # [B, L, D]
        cond_src = self.src_interaction_agg(hist_src, u_base_tgt)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_base_tgt)

        B = u_base_tgt.size(0)
        t_tgt = torch.randint(low=0, high=self.diff_tgt.timesteps, size=(B,), device=u_base_tgt.device)
        diff_loss_tgt, u_denoised_tgt = self.diff_tgt.p_losses(x_start=u_base_tgt, t=t_tgt, cond_src=cond_src,
                                                           cond_tgt=cond_tgt, loss_type="l2")
        u_final_tgt = u_base_tgt + self.config["lambda_user_emb"] * u_denoised_tgt

        # src item
        item_pos_emb_src = self.emb_item_src(pos_items_src)
        item_pos_emb_src = F.dropout(item_pos_emb_src, self.dropout, training=self.training)
        item_neg_emb_src = self.emb_item_src(neg_items_src)
        item_neg_emb_src = F.dropout(item_neg_emb_src, self.dropout, training=self.training)

        # tgt item
        item_pos_emb_tgt = self.emb_item_tgt(pos_items_tgt)
        item_pos_emb_tgt = F.dropout(item_pos_emb_tgt, self.dropout, training=self.training)
        item_neg_emb_tgt = self.emb_item_tgt(neg_items_tgt)
        item_neg_emb_tgt = F.dropout(item_neg_emb_tgt, self.dropout, training=self.training)

        # scores
        scores_pos_src = (u_final_src * item_pos_emb_src).sum(dim=-1)
        scores_neg_src = (u_final_src * item_neg_emb_src).sum(dim=-1)
        scores_pos_tgt = (u_final_tgt * item_pos_emb_tgt).sum(dim=-1)
        scores_neg_tgt = (u_final_tgt * item_neg_emb_tgt).sum(dim=-1)
        loss_bpr_src = -torch.log(torch.sigmoid(scores_pos_src - scores_neg_src) + 1e-12).mean()
        loss_bpr_tgt = -torch.log(torch.sigmoid(scores_pos_tgt - scores_neg_tgt) + 1e-12).mean()

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
            hist_src = self.emb_item_src(hist_src_items)  # [Bp, L, D]
            hist_tgt = self.emb_item_tgt(hist_tgt_items)  # [Bp, L, D]
            cond_src = self.src_interaction_agg(hist_src, u_pseudo)  # [Bp, D]
            cond_tgt = self.tgt_interaction_agg(hist_tgt, u_pseudo)  # [Bp, D]
            Bp = u_pseudo.size(0)
            t_pseudo = torch.randint(low=0, high=self.diff_tgt.timesteps, size=(Bp,), device=u_pseudo.device)
            diff_loss_pseudo, u_pseudo_denoised = self.diff_tgt.p_losses(
                x_start=u_pseudo, t=t_pseudo, cond_src=cond_src, cond_tgt=cond_tgt, loss_type="l2"
            )
            u_pseudo_final = u_pseudo + self.config["lambda_user_emb"] * u_pseudo_denoised  # 残差连接

            i_emb = self.emb_item_tgt(pseudo_items)  # [Bp, D]
            neg_emb = self.emb_item_tgt(neg_items)

            pos_score = (u_pseudo_final * i_emb).sum(dim=-1)
            neg_score = (u_pseudo_final * neg_emb).sum(dim=-1)
            pseudo_loss = self.bpr_loss(pos_score, neg_score)

        loss = (
                loss_bpr_src
                + loss_bpr_tgt
                + self.config["diff_weight"] * (diff_loss_src + diff_loss_tgt)
        )
        if use_pseudo:
            loss = loss + self.config["pseudo_rec_weight"] * pseudo_loss
        return loss

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()  # domain-local IDs
        B = users.size(0)

        if is_warm:
            # tgt-domain IDs
            uid = users.clone()
            mask = users > self.num_users_overlap
            uid[mask] = users[mask] + (self.num_users_src - self.num_users_overlap)
        else:
            # src-domain IDs
            uid = users

        user_emb = self.emb_user(uid)  # [B, D]

        global_item = torch.zeros((B, 2, self.max_len), dtype=torch.long, device=self.device)
        global_score = torch.full((B, 2, self.max_len), -100.0, device=self.device)

        if is_warm:
            global_item[:, 1, :] = self.history_items_tgt[users]
            global_score[:, 1, :] = self.history_scores_tgt[users]
            global_item[:, 0, :] = self.history_items_src[users]
            global_score[:, 0, :] = self.history_scores_src[users]
        else:
            global_item[:, 0, :] = self.history_items_src[users]
            global_score[:, 0, :] = self.history_scores_src[users]

        global_item_emb = torch.cat([
            self.emb_item_src(global_item[:, 0, :]),
            self.emb_item_tgt(global_item[:, 1, :])
        ], dim=1)
        u_base = self.agg(user_emb, global_item_emb, global_score)

        # 过Diff
        hist_src_items = global_item[:, 0, :]
        hist_tgt_items = global_item[:, 1, :]
        hist_src = self.emb_item_src(hist_src_items)
        hist_tgt = self.emb_item_tgt(hist_tgt_items)
        cond_src = self.src_interaction_agg(hist_src, u_base)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_base)
        _, u_denoised, _, _, _ = self.diff_tgt.sample(x_start=u_base, cond_src=cond_src, cond_tgt=cond_tgt)

        u_final = u_base + self.config["lambda_user_emb"] * u_denoised

        scores = torch.matmul(u_final, self.emb_item_tgt.weight.t())
        scores[:, 0] = 0.0
        return scores


# 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂 交互物品emb聚合器 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂

class BehaviorAggregator(nn.Module):
    def __init__(self, opt):
        super(BehaviorAggregator, self).__init__()
        self.opt = opt
        self.aggregator = opt["aggregator"]
        self.lambda_a = opt["lambda_a"]
        feature_dim = opt["feature_dim"]
        dropout_rate = opt["dropout"]

        self.W_agg = nn.Linear(feature_dim, feature_dim, bias=False)
        if self.aggregator in ["user_attention"]:
            self.W_att = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, id_emb, sequence_emb, score):
        out = id_emb
        if self.aggregator == "mean":
            out = self.mean_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self.user_attention_pooling(id_emb, sequence_emb)
        # elif self.aggregator == "item_similarity":
        #     out = self.item_similarity_pooling(sequence_emb, score)
        else:
            print("a wrong aggregater!!")
            exit(0)
        return self.lambda_a * id_emb + (1 - self.lambda_a) * out

    def user_attention_pooling(self, id_emb, sequence_emb):
        key = self.W_att(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, id_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_agg(output)

    def mean_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
        return self.W_agg(mean)

    def item_similarity_pooling(self, sequence_emb, score):
        if len(score.size()) != 2:
            score = score.view(score.size(0), -1)
        score = F.softmax(score, dim = -1)
        score = score.unsqueeze(-1)
        ans = (score * sequence_emb).sum(dim=1)
        return self.W_agg(ans)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, -1e9)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-12)

class InteractionAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.aggregator = config['aggregator']
        self.feature_dim = config['feature_dim']
        dropout_rate = config['dropout']

        self.W_agg = nn.Linear(config['feature_dim'], config['feature_dim'], bias=False)

        if self.aggregator in ["user_attention"]:
            self.W_att = nn.Sequential(
                nn.Linear(config['feature_dim'], config['feature_dim']),
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
        self.feature_dim = int(config['feature_dim'])
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

        self.w_q = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.w_k = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.w_v = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        init(self.w_q); init(self.w_k); init(self.w_v)
        self.ln = nn.LayerNorm(self.feature_dim, elementwise_affine=False)

    def get_timestep_embedding(self, timesteps: torch.Tensor, feature_dim: int):
        assert len(timesteps.shape) == 1
        half_dim = feature_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if feature_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def selfAttention(self, features: torch.Tensor):
        mask = (features.abs().sum(dim=-1) > 0) # [B, N]

        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)
        attn_logits = (q * (self.feature_dim ** -0.5)) @ k.transpose(-1, -2)  # [B, N, N]
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

        t_emb = self.get_timestep_embedding(t, self.feature_dim)  # [B, D] on device
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

        t_emb = self.get_timestep_embedding(t, self.feature_dim)
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