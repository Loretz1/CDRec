import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss


class Base(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(Base, self).__init__(config, dataloader)

        self.feature_dim = config["feature_dim"]
        self.mlp_hidden_dim = config["mlp_hidden_dim"]
        self.mlp_layers = config["mlp_layers"]
        self.dropout = config["dropout"]
        self.loss_type = config["loss_type"].lower()

        self.target_user_embedding = nn.Embedding(self.num_users_tgt + 1, self.feature_dim, padding_idx=0)
        self.target_item_embedding = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)

        def make_mlp(in_dim, hidden_dim, layers):
            if layers == 0:
                return nn.Identity()
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

        self.target_mlp_user = make_mlp(self.feature_dim, self.mlp_hidden_dim, self.mlp_layers)
        self.target_mlp_item = make_mlp(self.feature_dim, self.mlp_hidden_dim, self.mlp_layers)

        if self.loss_type == "bpr":
            self.criterion = BPRLoss()
        elif self.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        self.apply(xavier_uniform_initialization)

    def forward(self, user_ids, pos_ids, neg_ids):
        user_emb = self.target_user_embedding(user_ids)
        pos_emb = self.target_item_embedding(pos_ids)
        neg_emb = self.target_item_embedding(neg_ids)
        user_emb = self.target_mlp_user(user_emb)
        pos_emb = self.target_mlp_item(pos_emb)
        neg_emb = self.target_mlp_item(neg_emb)

        return user_emb, pos_emb, neg_emb

    def calculate_loss_for_0(self, interaction):
        user_tgt = interaction['users']
        tgt_pos = interaction['pos_items']
        tgt_neg = interaction['neg_items']

        tgt_user, tgt_pos_emb, tgt_neg_emb = self.forward(user_tgt, tgt_pos, tgt_neg)

        pos_tgt = (tgt_user * tgt_pos_emb).sum(dim=-1)
        neg_tgt = (tgt_user * tgt_neg_emb).sum(dim=-1)

        if self.loss_type == "bce":
            pos_label = torch.ones_like(pos_tgt)
            neg_label = torch.zeros_like(neg_tgt)
            loss = (
                    self.criterion(pos_tgt, pos_label)
                    + self.criterion(neg_tgt, neg_label)
            )
        elif self.loss_type == "bpr":
            loss = self.criterion(pos_tgt, neg_tgt)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        return loss

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self.calculate_loss_for_0(interaction)

    def full_sort_predict(self, interaction, is_warm):
        user = interaction[0].long()
        user_emb = self.target_user_embedding(user)
        user_emb = self.target_mlp_user(user_emb)
        all_tgt_items_emb = self.target_item_embedding.weight
        all_tgt_items_emb = self.target_mlp_item(all_tgt_items_emb)
        scores_tgt = torch.matmul(user_emb, all_tgt_items_emb.T)  # [B, n_target_items + 1]
        return scores_tgt

    def set_train_stage(self, stage_id):
        super(Base, self).set_train_stage(stage_id)
