import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss

class UniCDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(UniCDR, self).__init__(config, dataloader)

        self.config = config
        self.feature_dim = config['feature_dim']
        self.max_len = config["history_len"]
        self.dropout = config["dropout"]
        self.criterion = nn.BCEWithLogitsLoss()
        self.history_items_src, self.history_scores_src = self._build_history(dataloader, 'src')
        self.history_items_tgt, self.history_scores_tgt = self._build_history(dataloader, 'tgt')

        self.emb_user_src_specific = nn.Embedding(self.num_users_src + 1, self.feature_dim, padding_idx=0)
        self.emb_user_tgt_specific = nn.Embedding(self.num_users_tgt + 1, self.feature_dim, padding_idx=0)
        self.emb_user_shared = nn.Embedding(self.num_users_src + self.num_users_tgt - self.num_users_overlap + 1, self.feature_dim, padding_idx=0)
        self.emb_item_src = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)

        self.agg_list = nn.ModuleList(
            [BehaviorAggregator(config), BehaviorAggregator(config), BehaviorAggregator(config)])
        self.dis_list = nn.ModuleList([nn.Bilinear(config["feature_dim"], config["feature_dim"], 1),
                                       nn.Bilinear(config["feature_dim"], config["feature_dim"], 1)])

        self.apply(xavier_uniform_initialization)
        self.emb_user_src_specific.weight.data[0, :] = 0
        self.emb_user_tgt_specific.weight.data[0, :] = 0
        self.emb_user_shared.weight.data[0, :] = 0
        self.emb_item_src.weight.data[0, :] = 0
        self.emb_item_tgt.weight.data[0, :] = 0

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
        return items, scores

    def batch_build_global(self, users: torch.Tensor, cur_domain: str, ):
        B = users.size(0)
        L = self.max_len

        global_item = torch.zeros((B, 2, L), dtype=torch.long, device=self.device)
        global_score = torch.full((B, 2, L), -100.0, dtype=torch.float, device=self.device)
        overlap_mask = (users >= 1) & (users <= self.num_users_overlap)
        mask_other = (self.num_users_src == self.num_users_tgt) and (self.num_users_src == self.num_users_overlap)

        if cur_domain == "src":
            if overlap_mask.any():
                u = users[overlap_mask]
                items = self.history_items_tgt[u]
                scores = self.history_scores_tgt[u]
                if mask_other:
                    items, scores = self.batch_random_mask(items, scores, self.config["mask_rate"])
                global_item[overlap_mask, 1, :] = items
                global_score[overlap_mask, 1, :] = scores
        else:
            if overlap_mask.any():
                u = users[overlap_mask]
                items = self.history_items_src[u]
                scores = self.history_scores_src[u]
                if mask_other:
                    items, scores = self.batch_random_mask(items, scores, self.config["mask_rate"])
                global_item[overlap_mask, 0, :] = items
                global_score[overlap_mask, 0, :] = scores

        # without domain masking
        # if cur_domain == "src":
        #     src_items = self.history_items_src[users]
        #     src_scores = self.history_scores_src[users]
        #     if mask_other:
        #         m_items, m_scores = self.batch_random_mask(src_items, src_scores, self.config["mask_rate"])
        #         src_items, src_scores = m_items, m_scores
        #     global_item[:, 0, :] = src_items
        #     global_score[:, 0, :] = src_scores
        #     if overlap_mask.any():
        #         u = users[overlap_mask]
        #         tgt_items = self.history_items_tgt[u]
        #         tgt_scores = self.history_scores_tgt[u]
        #         if mask_other:
        #             tgt_items, tgt_scores = self.batch_random_mask(tgt_items, tgt_scores, self.config["mask_rate"])
        #         global_item[overlap_mask, 1, :] = tgt_items
        #         global_score[overlap_mask, 1, :] = tgt_scores
        # else:
        #     tgt_items = self.history_items_tgt[users]
        #     tgt_scores = self.history_scores_tgt[users]
        #     if mask_other:
        #         m_items, m_scores = self.batch_random_mask(tgt_items, tgt_scores, self.config["mask_rate"])
        #         tgt_items, tgt_scores = m_items, m_scores
        #     global_item[:, 1, :] = tgt_items
        #     global_score[:, 1, :] = tgt_scores
        #     if overlap_mask.any():
        #         u = users[overlap_mask]
        #         src_items = self.history_items_src[u]
        #         src_scores = self.history_scores_src[u]
        #         if mask_other:
        #             src_items, src_scores = self.batch_random_mask(src_items, src_scores, self.config["mask_rate"])
        #         global_item[overlap_mask, 0, :] = src_items
        #         global_score[overlap_mask, 0, :] = src_scores
        # without domain masking
        return global_item, global_score

    def getLossContrastive(self, specific_user, share_user, domain_id):
        random_label = (torch.arange(0, share_user.size(0), 1).cuda(share_user.device)
                        + torch.randint(1, share_user.size(0),(1,)).item()) % share_user.size(0)

        pos = self.dis_list[domain_id](specific_user, share_user).view(-1)
        neg = self.dis_list[domain_id](specific_user[random_label], share_user).view(-1)

        pos_label, neg_label = torch.ones(pos.size()).to(self.device), torch.zeros(neg.size()).to(self.device)
        critic_loss = self.criterion(pos, pos_label) + self.criterion(neg, neg_label)
        return critic_loss

    def calculate_loss(self, interaction, epoch_idx):
        users_src = interaction['users_src']
        pos_items_src = interaction['pos_items_src']
        neg_items_src = interaction['neg_items_src']
        users_tgt = interaction['users_tgt']
        pos_items_tgt = interaction['pos_items_tgt']
        neg_items_tgt = interaction['neg_items_tgt']

        context_item_src, context_score_src = self.batch_random_mask(self.history_items_src[users_src],
                                                                     self.history_scores_src[users_src],
                                                                     self.config["mask_rate"])
        global_item_src, global_score_src = self.batch_build_global(users_src, "src")
        context_item_tgt, context_score_tgt = self.batch_random_mask(self.history_items_tgt[users_tgt],
                                                                     self.history_scores_tgt[users_tgt],
                                                                     self.config["mask_rate"])
        global_item_tgt, global_score_tgt = self.batch_build_global(users_tgt, "tgt")

        # src user
        user_emb_src = self.emb_user_src_specific(users_src)
        context_item_emb_src = self.emb_item_src(context_item_src)
        specific_user_src = self.agg_list[0](user_emb_src, context_item_emb_src, context_score_src)
        specific_user_src = F.dropout(specific_user_src, self.dropout, training=self.training)

        global_user_emb_src = self.emb_user_shared(users_src)
        global_item_emb_src = torch.cat(
            [self.emb_item_src(global_item_src[:, 0, :]),
             self.emb_item_tgt(global_item_src[:, 1, :])], dim=1)
        share_user_src = self.agg_list[-1](global_user_emb_src, global_item_emb_src, global_score_src)
        share_user_src = F.dropout(share_user_src, self.dropout, training=self.training)

        # without_contrastive_loss
        loss_cont_src = self.getLossContrastive(specific_user_src, share_user_src, domain_id = 0)
        # without_contrastive_loss
        user_src_emb = specific_user_src + share_user_src

        # tgt user
        user_emb_tgt = self.emb_user_tgt_specific(users_tgt)
        context_item_emb_tgt = self.emb_item_tgt(context_item_tgt)
        specific_user_tgt = self.agg_list[1](user_emb_tgt, context_item_emb_tgt, context_score_tgt)
        specific_user_tgt = F.dropout(specific_user_tgt, self.dropout, training=self.training)

        shared_user_id_tgt = users_tgt.clone()
        mask_tgt_only = users_tgt > self.num_users_overlap
        shared_user_id_tgt[mask_tgt_only] = (
            users_tgt[mask_tgt_only]
            + (self.num_users_src - self.num_users_overlap)
        )
        global_user_emb_tgt = self.emb_user_shared(shared_user_id_tgt)
        global_item_emb_tgt = torch.cat([self.emb_item_src(global_item_tgt[:, 0, :]),
                                         self.emb_item_tgt(global_item_tgt[:, 1, :])], dim=1)
        share_user_tgt = self.agg_list[-1](global_user_emb_tgt, global_item_emb_tgt, global_score_tgt)
        share_user_tgt = F.dropout(share_user_tgt, self.dropout, training=self.training)

        # without_contrastive_loss
        loss_cont_tgt = self.getLossContrastive(specific_user_tgt, share_user_tgt, domain_id=1)
        # without_contrastive_loss
        user_tgt_emb = specific_user_tgt + share_user_tgt

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
        scores_pos_src = (user_src_emb * item_pos_emb_src).sum(dim=-1)
        scores_neg_src = (user_src_emb * item_neg_emb_src).sum(dim=-1)
        scores_pos_tgt = (user_tgt_emb * item_pos_emb_tgt).sum(dim=-1)
        scores_neg_tgt = (user_tgt_emb * item_neg_emb_tgt).sum(dim=-1)
        loss_bpr_src = -torch.log(torch.sigmoid(scores_pos_src - scores_neg_src) + 1e-12).mean()
        loss_bpr_tgt = -torch.log(torch.sigmoid(scores_pos_tgt - scores_neg_tgt) + 1e-12).mean()

        # without_contrastive_loss
        loss = (self.config["lambda_loss"] * (loss_bpr_src + loss_bpr_tgt)
                + (1 - self.config["lambda_loss"]) * (loss_cont_src + loss_cont_tgt))
        # loss = self.config["lambda_loss"] * (loss_bpr_src + loss_bpr_tgt)
        # without_contrastive_loss

        return loss

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        B = users.size(0)

        # generate user representation
        if is_warm:
            # specific
            user_emb_tgt = self.emb_user_tgt_specific(users)
            context_item_emb_tgt = self.emb_item_tgt(self.history_items_tgt[users])
            specific_user = self.agg_list[1](user_emb_tgt, context_item_emb_tgt, self.history_scores_tgt[users])

            # shared
            shared_user_id = users.clone()
            mask_tgt_only = users > self.num_users_overlap
            shared_user_id[mask_tgt_only] = (users[mask_tgt_only] + (self.num_users_src - self.num_users_overlap))
            global_user_emb = self.emb_user_shared(shared_user_id)

            global_item = torch.zeros((B, 2, self.max_len), dtype=torch.long, device=self.device)
            global_score = torch.full((B, 2, self.max_len), -100.0, device=self.device)
            global_item[:, 1, :] = self.history_items_tgt[users]
            global_score[:, 1, :] = self.history_scores_tgt[users]
            overlap_mask = users <= self.num_users_overlap
            if overlap_mask.any():
                u_overlap = users[overlap_mask]
                global_item[overlap_mask, 0, :] = self.history_items_src[u_overlap]
                global_score[overlap_mask, 0, :] = self.history_scores_src[u_overlap]
            global_item_emb = torch.cat([
                self.emb_item_src(global_item[:, 0, :]),
                self.emb_item_tgt(global_item[:, 1, :])
            ], dim=1)
            shared_user = self.agg_list[-1](
                global_user_emb,
                global_item_emb,
                global_score
            )
            user_emb = specific_user + shared_user
        else:
            # shared
            global_user_emb = self.emb_user_shared(users)
            global_item = torch.zeros((B, 2, self.max_len), dtype=torch.long, device=self.device)
            global_score = torch.full((B, 2, self.max_len), -100.0, device=self.device)
            global_item[:, 0, :] = self.history_items_src[users]
            global_score[:, 0, :] = self.history_scores_src[users]
            global_item_emb = torch.cat([
                self.emb_item_src(global_item[:, 0, :]),
                self.emb_item_tgt(global_item[:, 1, :])
            ], dim=1)
            user_emb = self.agg_list[-1](
                global_user_emb,
                global_item_emb,
                global_score
            )

        item_emb_tgt = self.emb_item_tgt.weight  # [num_items_tgt+1, D]
        scores = torch.matmul(user_emb, item_emb_tgt.t())  # [B, num_items_tgt+1]
        scores[:, 0] = 0.0
        return scores

class BehaviorAggregator(nn.Module):
    def __init__(self, opt):
        super(BehaviorAggregator, self).__init__()
        self.opt = opt
        self.aggregator = opt["aggregator"]
        self.lambda_a = opt["lambda_a"]
        embedding_dim = opt["feature_dim"]
        dropout_rate = opt["dropout"]

        self.W_agg = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["user_attention"]:
            self.W_att = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
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
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-12)
