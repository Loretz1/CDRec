import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss


class LightGCN(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LightGCN, self).__init__(config, dataloader)
        self.config = config

        self.feature_dim = config["feature_dim"]
        self.n_layers = config["n_layers"]
        self.loss_type = config["loss_type"].lower()

        self.user_embedding = nn.Embedding(self.num_users_tgt, self.feature_dim)
        self.item_embedding = nn.Embedding(self.num_items_tgt, self.feature_dim)

        tgt_mat = dataloader.inter_matrix(domain=1, form="coo")
        tgt_mat = self.normalize(tgt_mat)
        self.tgt_adj = self._scipy_coo_to_torch(tgt_mat)

        if self.loss_type == "bpr":
            self.criterion = BPRLoss()
        elif self.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        self.apply(xavier_uniform_initialization)

    def normalize(self, mat):
        """
        Symmetric normalization for user-item bipartite graph
        \bar{R} = D_u^{-1/2} R D_v^{-1/2}
        """
        mat = mat.tocoo()
        row_sum = np.array(mat.sum(axis=1)).flatten()
        col_sum = np.array(mat.sum(axis=0)).flatten()
        with np.errstate(divide="ignore"):
            d_u_inv_sqrt = np.power(row_sum, -0.5)
            d_v_inv_sqrt = np.power(col_sum, -0.5)
        d_u_inv_sqrt[np.isinf(d_u_inv_sqrt)] = 0.0
        d_v_inv_sqrt[np.isinf(d_v_inv_sqrt)] = 0.0
        D_u_inv_sqrt = sp.diags(d_u_inv_sqrt)
        D_v_inv_sqrt = sp.diags(d_v_inv_sqrt)
        return D_u_inv_sqrt.dot(mat).dot(D_v_inv_sqrt)

    def _scipy_coo_to_torch(self, mat):
        mat = mat.tocoo()
        indices = torch.tensor(
            [mat.row, mat.col], dtype=torch.long, device=self.device
        )
        values = torch.tensor(mat.data, dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(
            indices, values, mat.shape, device=self.device
        ).coalesce()

    def compute_embeddings(self):
        u0 = self.user_embedding.weight
        v0 = self.item_embedding.weight

        user_embs = [u0]
        item_embs = [v0]

        for _ in range(self.n_layers):
            u_next = torch.sparse.mm(self.tgt_adj, item_embs[-1])
            v_next = torch.sparse.mm(self.tgt_adj.transpose(0, 1), user_embs[-1])
            user_embs.append(u_next)
            item_embs.append(v_next)

        user_final = torch.mean(torch.stack(user_embs, dim=0), dim=0)
        item_final = torch.mean(torch.stack(item_embs, dim=0), dim=0)

        return user_final, item_final

    def forward(self, users, pos_items, neg_items):
        user_final, item_final = self.compute_embeddings()

        u = user_final[users]
        pos = item_final[pos_items]
        neg = item_final[neg_items]

        return u, pos, neg

    def calculate_loss_for_0(self, interaction):
        users = interaction["users"] - 1
        pos_items = interaction["pos_items"] - 1
        neg_items = interaction["neg_items"] - 1

        u, pos, neg = self.forward(users, pos_items, neg_items)

        pos_scores = torch.sum(u * pos, dim=1)
        neg_scores = torch.sum(u * neg, dim=1)

        if self.loss_type == "bce":
            pos_label = torch.ones_like(pos_scores)
            neg_label = torch.zeros_like(neg_scores)
            loss = (
                self.criterion(pos_scores, pos_label)
                + self.criterion(neg_scores, neg_label)
            )
        else:  # BPR
            loss = self.criterion(pos_scores, neg_scores)

        return loss

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self.calculate_loss_for_0(interaction)

    @torch.no_grad()
    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long() - 1

        user_final, item_final = self.compute_embeddings()
        u = user_final[users]

        scores = torch.matmul(u, item_final.transpose(0, 1))  # [B, num_items]
        padding = torch.zeros((scores.shape[0], 1), device=self.device)
        scores = torch.cat([padding, scores], dim=1)

        return scores

    def set_train_stage(self, stage_id):
        super().set_train_stage(stage_id)
