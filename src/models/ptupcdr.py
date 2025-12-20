import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss


class PTUPCDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(PTUPCDR, self).__init__(config, dataloader)
        self.config = config
        self.feature_dim = config["feature_dim"]
        self.max_len = config["history_len"]
        self.bpr_loss = BPRLoss()
        self.history_items_src, self.history_len_src = self._build_history(dataloader)

        self.emb_user_src = nn.Embedding(self.num_users_src + 1, self.feature_dim, padding_idx=0)
        self.emb_item_src = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)

        self.emb_user_tgt = nn.Embedding(self.num_users_tgt + 1, self.feature_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)

        self.meta_net = MetaNet(self.feature_dim, self.config['meta_dim'])

        self.apply(xavier_uniform_initialization)
        self.emb_user_src.weight.data[0, :] = 0
        self.emb_item_src.weight.data[0, :] = 0
        self.emb_user_tgt.weight.data[0, :] = 0
        self.emb_item_tgt.weight.data[0, :] = 0

    def _build_history(self, dataloader):
        device = self.device
        max_len = self.max_len

        H_src = torch.zeros((self.num_users_src + 1, max_len), dtype=torch.long, device=device)
        L_src = torch.zeros(self.num_users_src + 1, dtype=torch.long, device=device)
        for u, items in dataloader.dataset.positive_items_src.items():
            items = list(items)
            n = len(items)
            if n == 0:
                continue
            if n > max_len:
                rand_idx = torch.randperm(n)[:max_len]
                sampled = torch.tensor([items[i] for i in rand_idx], dtype=torch.long, device=device)
                H_src[u] = sampled
                L_src[u] = max_len
            else:
                H_src[u, :n] = torch.tensor(items, dtype=torch.long, device=device)
                L_src[u] = n
        return H_src, L_src

    def _bpr(self, u, p, n):
        pos = (u * p).sum(dim=-1)
        neg = (u * n).sum(dim=-1)
        return self.bpr_loss(pos, neg)

    def _source_loss(self, inter):
        user = self.emb_user_src(inter["users"])
        pos = self.emb_item_src(inter["pos_items"])
        neg = self.emb_item_src(inter["neg_items"])
        return self._bpr(user, pos, neg)

    def _target_loss(self, inter):
        user = self.emb_user_tgt(inter["users"])
        pos = self.emb_item_tgt(inter["pos_items"])
        neg = self.emb_item_tgt(inter["neg_items"])
        return self._bpr(user, pos, neg)

    def _meta_loss(self, inter):
        users = inter['users']
        pos_tgt = inter['pos_items_tgt']
        neg_tgt = inter['neg_items_tgt']

        u_src = self.emb_user_src(users).detach()
        hist_items = self.history_items_src[users]
        hist_emb = self.emb_item_src(hist_items).detach()
        mapping = self.meta_net(hist_emb, hist_items)
        mapping = mapping.view(-1, self.feature_dim, self.feature_dim)
        u_tgt_hat = torch.bmm(u_src.unsqueeze(1), mapping).squeeze(1)

        pos_emb = self.emb_item_tgt(pos_tgt)
        neg_emb = self.emb_item_tgt(neg_tgt)

        loss = self._bpr(u_tgt_hat, pos_emb, neg_emb)
        return loss

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self._source_loss(interaction)
        elif self.stage_id == 1:
            return self._target_loss(interaction)
        elif self.stage_id == 2:
            return self._meta_loss(interaction)

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        u_src = self.emb_user_src(users).detach()
        hist_items = self.history_items_src[users]
        hist_emb = self.emb_item_src(hist_items).detach()
        mapping = self.meta_net(hist_emb, hist_items)  # [B, d*d]
        mapping = mapping.view(-1, self.feature_dim, self.feature_dim)
        user_emb = torch.bmm(u_src.unsqueeze(1), mapping).squeeze(1)  # [B, d]
        item_emb = self.emb_item_tgt.weight
        scores = torch.matmul(user_emb, item_emb.t())
        scores[:, 0] = 0
        return scores

    def set_train_stage(self, stage_id):
        super(PTUPCDR, self).set_train_stage(stage_id)

        # freeze all first
        for m in [self.emb_user_src, self.emb_item_src, self.emb_user_tgt, self.emb_item_tgt, self.meta_net]:
            self._set_requires_grad(m, False)

        if stage_id == 0:  # source
            self._set_requires_grad(self.emb_user_src, True)
            self._set_requires_grad(self.emb_item_src, True)
        elif stage_id == 1:  # target
            self._set_requires_grad(self.emb_user_tgt, True)
            self._set_requires_grad(self.emb_item_tgt, True)
        elif stage_id == 2:  # meta
            self._set_requires_grad(self.meta_net, True)

    @staticmethod
    def _set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag



class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)

