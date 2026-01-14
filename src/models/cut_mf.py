import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization


class CUT_MF(GeneralRecommender):
    """
    CUT with MF backbone
    """
    def __init__(self, config, dataloader):
        super(CUT_MF, self).__init__(config, dataloader)
        self.config = config
        self.feature_dim = config['feature_dim']

        self.user_emb_stage0 = nn.Embedding(self.num_users_tgt, self.feature_dim)
        self.item_emb_stage0 = nn.Embedding(self.num_items_tgt, self.feature_dim)

        self.user_emb_all = nn.Embedding(self.num_users_src + self.num_users_tgt - self.num_users_overlap,self.feature_dim)
        self.item_emb_all = nn.Embedding(self.num_items_src + self.num_items_tgt, self.feature_dim)

        self.transform_weight = config['transform_weight']
        self.user_transform_matrix_r = nn.Parameter(torch.zeros(self.feature_dim, self.feature_dim))
        self.register_buffer(
            'user_transform_matrix',
            torch.eye(self.feature_dim)
        )

        self.apply(xavier_uniform_initialization)

    def transform_user(self, users_all, user_emb):
        Uo = self.num_users_overlap
        Us = self.num_users_src
        is_target = (users_all < Uo) | (users_all >= Us)
        assert all(is_target)
        is_target = is_target.unsqueeze(1).float()
        W = self.user_transform_matrix + self.transform_weight * self.user_transform_matrix_r
        transformed = torch.matmul(user_emb, W)
        return is_target * transformed + (1.0 - is_target) * user_emb

    def calculate_loss_for_0(self, interaction):
        users = interaction['users'] - 1
        pos_items = interaction['pos_items'] - 1
        neg_items = interaction['neg_items'] - 1

        u = self.user_emb_stage0(users)
        pos = self.item_emb_stage0(pos_items)
        neg = self.item_emb_stage0(neg_items)

        pos_scores = torch.sum(u * pos, dim=1)
        neg_scores = torch.sum(u * neg, dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss

    def calculate_loss_for_1(self, interaction):
        users_src = interaction['users_src'] - 1
        pos_items_src = interaction['pos_items_src'] - 1
        neg_items_src = interaction['neg_items_src'] - 1
        users_tgt = interaction['users_tgt'] - 1
        pos_items_tgt = interaction['pos_items_tgt'] - 1
        neg_items_tgt = interaction['neg_items_tgt'] - 1

        # L_s
        u_s = self.user_emb_all(users_src)
        pos_s = self.item_emb_all(pos_items_src)
        neg_s = self.item_emb_all(neg_items_src)
        pos_score_s = torch.sum(u_s * pos_s, dim=1)
        neg_score_s = torch.sum(u_s * neg_s, dim=1)
        loss_src = -torch.log(torch.sigmoid(pos_score_s - neg_score_s)).mean()

        # L_t
        Uo = self.num_users_overlap
        Us = self.num_users_src
        users_tgt_all = torch.where(
            users_tgt < Uo,
            users_tgt,
            users_tgt + (Us - Uo)
        )
        u_t = self.user_emb_all(users_tgt_all)
        u_t = self.transform_user(users_tgt_all, u_t)
        pos_t = self.item_emb_all(pos_items_tgt + self.num_items_src)
        neg_t = self.item_emb_all(neg_items_tgt + self.num_items_src)
        pos_score_t = torch.sum(u_t * pos_t, dim=1)
        neg_score_t = torch.sum(u_t * neg_t, dim=1)
        loss_tgt = -torch.log(torch.sigmoid(pos_score_t - neg_score_t)).mean()

        # L_c
        # without_contrastive_loss
        users_tgt_unique = torch.unique(users_tgt)
        users_tgt_all_unique = torch.where(
            users_tgt_unique < Uo,
            users_tgt_unique,
            users_tgt_unique + (Us - Uo)
        )
        with torch.no_grad():
            u_ref = self.user_emb_stage0_frozen[users_tgt_unique]
        u_cur = self.user_emb_all(users_tgt_all_unique)
        u_cur = self.transform_user(users_tgt_all_unique, u_cur)

        u_ref = F.normalize(u_ref, dim=1)
        u_cur = F.normalize(u_cur, dim=1)
        sim_ref = torch.matmul(u_ref, u_ref.T)
        sim_cur = torch.matmul(u_cur, u_cur.T)
        gamma = self.config['gamma']
        pos_mask = (sim_ref > gamma).float()
        pos_mask.fill_diagonal_(0)
        tau = self.config['tau']
        exp_sim = torch.exp(sim_cur / tau)
        log_prob = sim_cur / tau - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss_c = -(pos_mask * log_prob).sum() / (pos_mask.sum() + 1e-8)
        # without_contrastive_loss

        alpha = self.config['alpha']
        lambda_c = self.config['lambda']
        # without_contrastive_loss
        loss = alpha * loss_src + (1 - alpha) * loss_tgt + lambda_c * loss_c
        # loss = alpha * loss_src + (1 - alpha) * loss_tgt
        # without_contrastive_loss
        return loss

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self.calculate_loss_for_0(interaction)
        else:
            return self.calculate_loss_for_1(interaction)

    def full_sort_predict(self, interaction, is_warm):
        users_tgt = interaction[0].long() - 1

        Uo = self.num_users_overlap
        Us = self.num_users_src

        users_tgt_all = torch.where(
            users_tgt < Uo,
            users_tgt,
            users_tgt + (Us - Uo)
        )

        u = self.user_emb_all(users_tgt_all)
        u = self.transform_user(users_tgt_all, u)

        item_tgt = self.item_emb_all.weight[
            self.num_items_src: self.num_items_src + self.num_items_tgt
        ]

        scores = torch.matmul(u, item_tgt.T)

        padding = torch.zeros((scores.shape[0], 1), device=self.device)
        scores = torch.cat((padding, scores), dim=1)
        return scores

    def set_train_stage(self, stage_id):
        super().set_train_stage(stage_id)

        if stage_id == 0:
            self.user_emb_stage0.requires_grad_(True)
            self.item_emb_stage0.requires_grad_(True)
            self.user_emb_all.requires_grad_(False)
            self.item_emb_all.requires_grad_(False)
            self.user_transform_matrix_r.requires_grad_(False)

        elif stage_id == 1:
            self.user_emb_stage0.requires_grad_(False)
            self.item_emb_stage0.requires_grad_(False)
            self.user_emb_all.requires_grad_(True)
            self.item_emb_all.requires_grad_(True)
            self.user_transform_matrix_r.requires_grad_(True)

            if not hasattr(self, 'user_emb_stage0_frozen'):
                with torch.no_grad():
                    self.user_emb_stage0_frozen = self.user_emb_stage0.weight.detach().clone()
