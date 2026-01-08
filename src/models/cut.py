import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
import scipy.sparse as sp

class CUT(GeneralRecommender):
    """
    CUT with LightGCN backbone
    """
    def __init__(self, config, dataloader):
        super(CUT, self).__init__(config, dataloader)
        self.config=config
        self.feature_dim = config['feature_dim']

        self.user_emb_stage0 = nn.Embedding(self.num_users_tgt, self.feature_dim)
        self.item_emb_stage0 = nn.Embedding(self.num_items_tgt, self.feature_dim)

        self.user_emb_all = nn.Embedding(self.num_users_src + self.num_users_tgt - self.num_users_overlap, self.feature_dim)
        self.item_emb_all = nn.Embedding(self.num_items_src + self.num_items_tgt, self.feature_dim)

        src_mat = dataloader.inter_matrix(domain=0, form='coo')
        tgt_mat = dataloader.inter_matrix(domain=1, form='coo')
        all_mat = self.build_all_adj_from_src_tgt(src_mat, tgt_mat, self.num_users_src, self.num_users_tgt, self.num_items_src,
                                        self.num_items_tgt, self.num_users_overlap)
        tgt_mat = self.normalize(tgt_mat)
        self.tgt_adj = self._scipy_coo_to_torch(tgt_mat)
        all_mat = self.normalize(all_mat)
        self.all_adj = self._scipy_coo_to_torch(all_mat)

        self.transform_weight = config['transform_weight']
        self.user_transform_matrix_r = nn.Parameter(
            torch.zeros(self.feature_dim, self.feature_dim)
        )
        self.register_buffer(
            'user_transform_matrix',
            torch.eye(self.feature_dim)
        )

        self.apply(xavier_uniform_initialization)

    def build_all_adj_from_src_tgt(self, src_mat, tgt_mat, num_users_src, num_users_tgt, num_items_src, num_items_tgt,
                                   num_overlap_user):
        Us, Is = num_users_src, num_items_src
        Ut, It = num_users_tgt, num_items_tgt
        Uo = num_overlap_user

        U_all = Us + Ut - Uo
        I_all = Is + It

        src_mat = src_mat.tocoo()
        tgt_mat = tgt_mat.tocoo()

        R_all = sp.dok_matrix((U_all, I_all), dtype=np.float32)

        for u, i, v in zip(src_mat.row, src_mat.col, src_mat.data):
            R_all[u, i] = v

        user_shift = Us - Uo
        item_shift = Is

        for u, i, v in zip(tgt_mat.row, tgt_mat.col, tgt_mat.data):
            if u < Uo:
                u_all = u
            else:
                u_all = u + user_shift
            i_all = i + item_shift
            R_all[u_all, i_all] = v
        return R_all.tocoo()

    def normalize(self, mat):
        """
        Symmetric normalization for user-item bipartite graph
        \bar{R} = D_u^{-1/2} R D_v^{-1/2}
        """
        mat = mat.tocoo()

        row_sum = np.array(mat.sum(axis=1)).flatten()
        col_sum = np.array(mat.sum(axis=0)).flatten()
        with np.errstate(divide='ignore'):
            d_u_inv_sqrt = np.power(row_sum, -0.5)
            d_v_inv_sqrt = np.power(col_sum, -0.5)
        d_u_inv_sqrt[np.isinf(d_u_inv_sqrt)] = 0.
        d_v_inv_sqrt[np.isinf(d_v_inv_sqrt)] = 0.
        D_u_inv_sqrt = sp.diags(d_u_inv_sqrt)
        D_v_inv_sqrt = sp.diags(d_v_inv_sqrt)
        return D_u_inv_sqrt.dot(mat).dot(D_v_inv_sqrt)

    def _scipy_coo_to_torch(self, mat):
        mat = mat.tocoo()  # ensure COO format
        indices = torch.tensor(
            [mat.row, mat.col], dtype=torch.long, device=self.device
        )  # [2, nnz]
        values = torch.tensor(mat.data, dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(indices, values, mat.shape, device=self.device).coalesce()

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

        u0 = self.user_emb_stage0.weight
        v0 = self.item_emb_stage0.weight

        user_embs = [u0]
        item_embs = [v0]

        for _ in range(self.config['n_layers']):
            u_next = torch.sparse.mm(self.tgt_adj, item_embs[-1])
            v_next = torch.sparse.mm(self.tgt_adj.transpose(0, 1), user_embs[-1])
            user_embs.append(u_next)
            item_embs.append(v_next)

        user_final = torch.mean(torch.stack(user_embs, dim=0), dim=0)
        item_final = torch.mean(torch.stack(item_embs, dim=0), dim=0)

        u = user_final[users]
        pos = item_final[pos_items]
        neg = item_final[neg_items]

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

        u0 = self.user_emb_all.weight  # [U_all, d]
        v0 = self.item_emb_all.weight  # [I_all, d]
        user_embs = [u0]
        item_embs = [v0]
        for _ in range(self.config['n_layers']):
            u_next = torch.sparse.mm(self.all_adj, item_embs[-1])
            v_next = torch.sparse.mm(self.all_adj.transpose(0, 1), user_embs[-1])
            user_embs.append(u_next)
            item_embs.append(v_next)
        user_final = torch.mean(torch.stack(user_embs, dim=0), dim=0)
        item_final = torch.mean(torch.stack(item_embs, dim=0), dim=0)

        # L_s
        u_s = user_final[users_src]
        pos_s = item_final[pos_items_src]
        neg_s = item_final[neg_items_src]
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
        u_t = user_final[users_tgt_all]
        u_t = self.transform_user(users_tgt_all, u_t)
        pos_t = item_final[pos_items_tgt + self.num_items_src]
        neg_t = item_final[neg_items_tgt + self.num_items_src]
        pos_score_t = torch.sum(u_t * pos_t, dim=1)
        neg_score_t = torch.sum(u_t * neg_t, dim=1)
        loss_tgt = -torch.log(torch.sigmoid(pos_score_t - neg_score_t)).mean()

        # L_c
        users_tgt_unique = torch.unique(users_tgt)
        users_tgt_all_unique = torch.where(
            users_tgt_unique < Uo,
            users_tgt_unique,
            users_tgt_unique + (Us - Uo)
        )
        with torch.no_grad():
            u_ref = self.user_emb_stage0_frozen[users_tgt_unique]
        u_cur = user_final[users_tgt_all_unique]
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

        lambda_c = self.config['lambda']
        alpha = self.config['alpha']
        loss = alpha * loss_src + (1 - alpha) * loss_tgt + lambda_c * loss_c
        return loss

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self.calculate_loss_for_0(interaction)
        else:
            return self.calculate_loss_for_1(interaction)

    def full_sort_predict(self, interaction, is_warm):
        users_tgt = interaction[0].long() - 1

        u0 = self.user_emb_all.weight
        v0 = self.item_emb_all.weight
        user_embs = [u0]
        item_embs = [v0]
        for _ in range(self.config['n_layers']):
            u_next = torch.sparse.mm(self.all_adj, item_embs[-1])
            v_next = torch.sparse.mm(self.all_adj.transpose(0, 1), user_embs[-1])
            user_embs.append(u_next)
            item_embs.append(v_next)
        user_final = torch.mean(torch.stack(user_embs, dim=0), dim=0)
        item_final = torch.mean(torch.stack(item_embs, dim=0), dim=0)

        Uo = self.num_users_overlap
        Us = self.num_users_src

        users_tgt_all = torch.where(
            users_tgt < Uo,
            users_tgt,
            users_tgt + (Us - Uo)
        )

        u = user_final[users_tgt_all]
        u = self.transform_user(users_tgt_all, u)

        item_tgt = item_final[
            self.num_items_src: self.num_items_src + self.num_items_tgt
        ]

        scores_tgt = torch.matmul(u, item_tgt.transpose(0, 1))  # [B, num_item_tgt]
        padding = torch.zeros((scores_tgt.shape[0], 1), device=self.device) # [B, 1]
        scores_tgt = torch.concat((padding, scores_tgt), dim=1)
        return scores_tgt

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





