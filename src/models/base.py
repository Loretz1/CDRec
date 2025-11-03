import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss


class Base(GeneralRecommender):
    def __init__(self, config, dataset):
        super(Base, self).__init__(config, dataset)

        self.feature_dim = config["feature_dim"]
        self.mlp_hidden_dim = config["mlp_hidden_dim"]
        self.mlp_layers = config["mlp_layers"]
        self.dropout = config["dropout"]
        self.loss_type = config["loss_type"].lower()

        self.source_user_embedding = nn.Embedding(self.n_users, self.feature_dim)
        self.target_user_embedding = nn.Embedding(self.n_users, self.feature_dim)
        self.source_item_embedding = nn.Embedding(self.n_source_items, self.feature_dim)
        self.target_item_embedding = nn.Embedding(self.n_target_items, self.feature_dim)

        def make_mlp(in_dim, hidden_dim, layers):
            mlp = []
            last_dim = in_dim
            for _ in range(layers):
                mlp += [
                    nn.Linear(last_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
                last_dim = hidden_dim
            return nn.Sequential(*mlp)

        self.source_mlp = make_mlp(self.feature_dim, self.mlp_hidden_dim, self.mlp_layers)
        self.target_mlp = make_mlp(self.feature_dim, self.mlp_hidden_dim, self.mlp_layers)

        if self.loss_type == "bpr":
            self.criterion = BPRLoss()
        elif self.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        self.apply(xavier_uniform_initialization)

    def forward(self, user_ids, pos_ids, neg_ids, domain):
        if domain == 0:
            user_emb = self.source_user_embedding(user_ids)
            pos_emb = self.source_item_embedding(pos_ids)
            neg_emb = self.source_item_embedding(neg_ids)
            user_emb = self.source_mlp(user_emb)
            pos_emb = self.source_mlp(pos_emb)
            neg_emb = self.source_mlp(neg_emb)
        else:
            user_emb = self.target_user_embedding(user_ids)
            pos_emb = self.target_item_embedding(pos_ids)
            neg_emb = self.target_item_embedding(neg_ids)
            user_emb = self.target_mlp(user_emb)
            pos_emb = self.target_mlp(pos_emb)
            neg_emb = self.target_mlp(neg_emb)

        return user_emb, pos_emb, neg_emb

    def calculate_loss(self, interaction, epoch_idx):
        user, src_pos, src_neg, tgt_pos, tgt_neg = interaction

        src_user, src_pos_emb, src_neg_emb = self.forward(user, src_pos, src_neg, domain=0)
        tgt_user, tgt_pos_emb, tgt_neg_emb = self.forward(user, tgt_pos, tgt_neg, domain=1)

        pos_src = (src_user * src_pos_emb).sum(dim=-1)
        neg_src = (src_user * src_neg_emb).sum(dim=-1)
        pos_tgt = (tgt_user * tgt_pos_emb).sum(dim=-1)
        neg_tgt = (tgt_user * tgt_neg_emb).sum(dim=-1)

        if self.loss_type == "bce":
            pos_label = torch.ones_like(pos_src)
            neg_label = torch.zeros_like(neg_src)
            loss = (
                    self.criterion(pos_src, pos_label)
                    + self.criterion(neg_src, neg_label)
                    + self.criterion(pos_tgt, pos_label)
                    + self.criterion(neg_tgt, neg_label)
            )
        elif self.loss_type == "bpr":
            loss = self.criterion(pos_src, neg_src) + self.criterion(pos_tgt, neg_tgt)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0].long()

        src_user = self.source_user_embedding(user)
        tgt_user = self.target_user_embedding(user)

        all_src_items = self.source_item_embedding.weight
        all_tgt_items = self.target_item_embedding.weight

        src_user = self.source_mlp(src_user)
        tgt_user = self.target_mlp(tgt_user)
        src_items = self.source_mlp(all_src_items)
        tgt_items = self.target_mlp(all_tgt_items)

        scores_src = torch.matmul(src_user, src_items.T)  # [B, n_source_items]
        scores_tgt = torch.matmul(tgt_user, tgt_items.T)  # [B, n_target_items]

        return scores_src, scores_tgt
