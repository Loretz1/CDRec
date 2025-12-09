import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss

class EMCDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(EMCDR, self).__init__(config, dataloader)

        self.feature_dim = config.get("feature_dim", 64)
        self.mlp_hidden_dim = config.get("mlp_hidden_dim", 128)
        self.mlp_layers = config.get("mlp_layers", 2)
        self.dropout = config.get("dropout", 0.0)
        self.loss_type = config.get("loss_type", "bpr").lower()

        # ====== Embedding ======
        self.source_user_embedding = nn.Embedding(self.num_users_src + 1, self.feature_dim, padding_idx=0)
        self.source_item_embedding = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)
        self.target_user_embedding = nn.Embedding(self.num_users_tgt + 1, self.feature_dim, padding_idx=0)
        self.target_item_embedding = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)

        # ====== User tower MLPs ======
        self.source_user_mlp = self._make_mlp()
        self.target_user_mlp = self._make_mlp()

        # ====== Mapping MLP g(u_src) ======
        hid = 2 * self.feature_dim
        self.mapping_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hid),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(hid, self.feature_dim),
        )

        # ====== Loss ======
        self.bpr_loss = BPRLoss()
        self.map_loss_fn = nn.MSELoss()

        self.apply(xavier_uniform_initialization)

    def _make_mlp(self):
        layers = []
        in_dim = self.feature_dim
        for _ in range(self.mlp_layers):
            layers.append(nn.Linear(in_dim, self.mlp_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_dim = self.mlp_hidden_dim
        layers.append(nn.Linear(in_dim, self.feature_dim))  # back to dim K
        return nn.Sequential(*layers)

    @staticmethod
    def _set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def _bpr(self, u, p, n):
        pos = (u * p).sum(dim=-1)
        neg = (u * n).sum(dim=-1)
        return self.bpr_loss(pos, neg)

    def _source_loss(self, inter):
        user = self.source_user_embedding(inter["users"])
        user = self.source_user_mlp(user)
        pos = self.source_item_embedding(inter["pos_items"])
        pos = self.source_user_mlp(pos)
        neg = self.source_item_embedding(inter["neg_items"])
        neg = self.source_user_mlp(neg)
        return self._bpr(user, pos, neg)

    def _target_loss(self, inter):
        user = self.target_user_embedding(inter["users"])
        user = self.target_user_mlp(user)
        pos = self.target_item_embedding(inter["pos_items"])
        pos = self.target_user_mlp(pos)
        neg = self.target_item_embedding(inter["neg_items"])
        neg = self.target_user_mlp(neg)
        return self._bpr(user, pos, neg)

    def _mapping_loss(self, inter):
        user_src = self.source_user_embedding(inter["users_overlapped"])
        user_src = self.source_user_mlp(user_src)

        user_tgt = self.target_user_embedding(inter["users_overlapped"])
        user_tgt = self.target_user_mlp(user_tgt)

        mapped = self.mapping_mlp(user_src)
        return self.map_loss_fn(mapped, user_tgt)

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self._source_loss(interaction)
        elif self.stage_id == 1:
            return self._target_loss(interaction)
        elif self.stage_id == 2:
            return self._mapping_loss(interaction)

    def full_sort_predict(self, interaction, is_warm):
        user = interaction[0].long()
        assert is_warm == False

        u = self.source_user_embedding(user)
        u = self.source_user_mlp(u)
        u = self.mapping_mlp(u)

        all_items = self.target_item_embedding.weight
        all_items = self.target_user_mlp(all_items)
        return torch.matmul(u, all_items.T)

    def set_train_stage(self, stage_id):
        super(EMCDR, self).set_train_stage(stage_id)

        # freeze all first
        for m in [
            self.source_user_embedding, self.source_item_embedding, self.source_user_mlp,
            self.target_user_embedding, self.target_item_embedding, self.target_user_mlp,
            self.mapping_mlp]:
            self._set_requires_grad(m, False)

        if stage_id == 0:  # source
            self._set_requires_grad(self.source_user_embedding, True)
            self._set_requires_grad(self.source_item_embedding, True)
            self._set_requires_grad(self.source_user_mlp, True)
        elif stage_id == 1:  # target
            self._set_requires_grad(self.target_user_embedding, True)
            self._set_requires_grad(self.target_item_embedding, True)
            self._set_requires_grad(self.target_user_mlp, True)
        elif stage_id == 2:  # mapping
            self._set_requires_grad(self.mapping_mlp, True)

    def show_all_params(self):
        print("\n===== All Parameters in Model =====")
        for name, p in self.named_parameters():
            print(f"{name:50s} | size={tuple(p.shape)} | requires_grad={p.requires_grad}")
        print("===================================\n")

    def show_params_and_grads(self):
        print("\n===== PARAM & GRAD CHECK =====")
        for name, p in self.named_parameters():
            req = p.requires_grad
            if p.grad is None:
                grad_info = "grad=None"
            else:
                grad_info = f"grad_mean={p.grad.mean().item():.6f}, grad_std={p.grad.std().item():.6f}"
            print(f"{name:50s} | train={req} | {grad_info}")
        print("===================================\n")
