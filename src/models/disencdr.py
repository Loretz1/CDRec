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

class DisenCDR(GeneralRecommender):
    def __init__(self, opt, dataloader):
        super(DisenCDR, self).__init__(opt, dataloader)
        self.opt=opt
        self.feature_dim = opt['feature_dim']
        self.rate = {}
        for u in dataloader.dataset.positive_items_src:
            if u > self.num_users_overlap:
                continue
            src_cnt = len(dataloader.dataset.positive_items_src.get(u, set()))
            tgt_cnt = len(dataloader.dataset.positive_items_tgt.get(u, set()))
            self.rate[u] = src_cnt / (src_cnt + tgt_cnt)
        self.rate = [self.rate[u] for u in sorted(self.rate.keys())]
        self.criterion = nn.BCEWithLogitsLoss()

        self.source_specific_GNN = singleVBGE(opt)
        self.source_share_GNN = singleVBGE(opt)

        self.target_specific_GNN = singleVBGE(opt)
        self.target_share_GNN = singleVBGE(opt)

        self.share_GNN = crossVBGE(opt)

        self.dropout = opt["dropout"]

        self.source_user_embedding = nn.Embedding(self.num_users_overlap + 1, self.feature_dim, padding_idx=0)
        self.target_user_embedding = nn.Embedding(self.num_users_overlap + 1, self.feature_dim, padding_idx=0)
        self.source_item_embedding = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)
        self.target_item_embedding = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)
        self.source_user_embedding_share = nn.Embedding(self.num_users_overlap + 1, self.feature_dim, padding_idx=0)
        self.target_user_embedding_share = nn.Embedding(self.num_users_overlap + 1, self.feature_dim, padding_idx=0)

        self.share_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.share_sigma = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        src_mat = dataloader.inter_matrix(domain=0, form='coo')
        tgt_mat = dataloader.inter_matrix(domain=1, form='coo')
        src_mat = self.filter_overlap_users(src_mat, self.num_users_overlap)
        tgt_mat = self.filter_overlap_users(tgt_mat, self.num_users_overlap)
        src_mat = self.normalize(src_mat)
        tgt_mat = self.normalize(tgt_mat)
        self.source_UV = self._scipy_coo_to_torch(src_mat).coalesce()
        self.source_VU = self.source_UV.transpose(0, 1).coalesce()
        self.target_UV = self._scipy_coo_to_torch(tgt_mat).coalesce()
        self.target_VU = self.target_UV.transpose(0, 1).coalesce()

        self.apply(xavier_uniform_initialization)

    def _scipy_coo_to_torch(self, mat):
        mat = mat.tocoo()  # ensure COO format
        indices = torch.tensor(
            [mat.row, mat.col], dtype=torch.long, device=self.device
        )  # [2, nnz]
        values = torch.tensor(mat.data, dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(indices, values, mat.shape, device=self.device).coalesce()

    def filter_overlap_users(self, mat: sp.coo_matrix, num_overlap_user: int):
        mask = mat.row < num_overlap_user

        new_row = mat.row[mask]
        new_col = mat.col[mask]
        new_data = mat.data[mask]

        new_shape = (num_overlap_user, mat.shape[1])

        return sp.coo_matrix(
            (new_data, (new_row, new_col)),
            shape=new_shape
        )

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        # sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logstd, 0.4)))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.share_mean.training:
            sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.zeros_like(logstd))
        return sampled_z, (1 - self.opt["beta"]) * kld_loss

    def calculate_loss(self, interaction, epoch_idx):
        user = interaction['users']
        src_pos = interaction['pos_items_src']
        src_neg = interaction['neg_items_src']
        tgt_pos = interaction['pos_items_tgt']
        tgt_neg = interaction['neg_items_tgt']
        if epoch_idx < 10:
            source_user = self.source_user_embedding.weight[1:]
            target_user = self.target_user_embedding.weight[1:]
            source_item = self.source_item_embedding.weight[1:]
            target_item = self.target_item_embedding.weight[1:]
            source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,
                                                                                              self.source_UV,  self.source_VU)
            target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,
                                                                                              self.target_UV, self.target_VU)
            self.kld_loss = 0
            # add 0-padding row at top
            zero_pad_s = torch.zeros(1, source_learn_specific_user.size(1), device=self.device)
            zero_pad_t = torch.zeros(1, source_learn_specific_user.size(1), device=self.device)

            source_learn_user = torch.cat([zero_pad_s, source_learn_specific_user], dim=0)
            target_learn_user = torch.cat([zero_pad_t, target_learn_specific_user], dim=0)
            source_learn_specific_item = torch.cat([zero_pad_s, source_learn_specific_item], dim=0)
            target_learn_specific_item = torch.cat([zero_pad_t, target_learn_specific_item], dim=0)

            user_src_emb = source_learn_user[user]
            src_pos_emb = source_learn_specific_item[src_pos]
            src_neg_emb = source_learn_specific_item[src_neg]
            user_tgt_emb = target_learn_user[user]
            tgt_pos_emb = target_learn_specific_item[tgt_pos]
            tgt_neg_emb = target_learn_specific_item[tgt_neg]

            pos_source_score = (user_src_emb * src_pos_emb).sum(dim=-1)
            neg_source_score = (user_src_emb * src_neg_emb).sum(dim=-1)
            pos_target_score = (user_tgt_emb * tgt_pos_emb).sum(dim=-1)
            neg_target_score = (user_tgt_emb * tgt_neg_emb).sum(dim=-1)

            pos_labels, neg_labels = torch.ones(pos_source_score.size()).to(self.device), torch.zeros(
                pos_source_score.size()).to(self.device)

            loss = self.criterion(pos_source_score, pos_labels) + \
                   self.criterion(neg_source_score, neg_labels) + \
                   self.criterion(pos_target_score, pos_labels) + \
                   self.criterion(neg_target_score, neg_labels) + \
                   self.source_specific_GNN.encoder[-1].kld_loss + \
                   self.target_specific_GNN.encoder[-1].kld_loss + self.kld_loss

            return loss

        source_user = self.source_user_embedding.weight[1:]
        target_user = self.target_user_embedding.weight[1:]
        source_item = self.source_item_embedding.weight[1:]
        target_item = self.target_item_embedding.weight[1:]
        source_user_share = self.source_user_embedding_share.weight[1:]
        target_user_share = self.target_user_embedding_share.weight[1:]

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, self.source_UV,  self.source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, self.target_UV, self.target_VU)

        source_user_mean, source_user_sigma = self.source_share_GNN.forward_user_share(source_user, self.source_UV,  self.source_VU)
        target_user_mean, target_user_sigma = self.target_share_GNN.forward_user_share(target_user, self.target_UV, self.target_VU)

        rate = torch.tensor(self.rate, dtype=torch.float32, device=self.device).unsqueeze(1)
        mean, sigma, = self.share_GNN(source_user_share, target_user_share,self.source_UV,  self.source_VU, self.target_UV, self.target_VU, rate)

        user_share, share_kld_loss = self.reparameters(mean, sigma)

        source_share_kld = self._kld_gauss(mean, sigma, source_user_mean, source_user_sigma)
        target_share_kld = self._kld_gauss(mean, sigma, target_user_mean, target_user_sigma)

        self.kld_loss =  share_kld_loss + self.opt["beta"] * source_share_kld + self.opt[
            "beta"] * target_share_kld

        # source_learn_user = self.source_merge(torch.cat((user_share, source_learn_specific_user), dim = -1))
        # target_learn_user = self.target_merge(torch.cat((user_share, target_learn_specific_user), dim = -1))
        source_learn_user = user_share + source_learn_specific_user
        target_learn_user = user_share + target_learn_specific_user

        # add 0-padding row at top
        zero_pad_s = torch.zeros(1, source_learn_user.size(1), device=self.device)
        zero_pad_t = torch.zeros(1, target_learn_user.size(1), device=self.device)

        source_learn_user = torch.cat([zero_pad_s, source_learn_user], dim=0)
        target_learn_user = torch.cat([zero_pad_t, target_learn_user], dim=0)
        source_learn_specific_item = torch.cat([zero_pad_s, source_learn_specific_item], dim=0)
        target_learn_specific_item = torch.cat([zero_pad_t, target_learn_specific_item], dim=0)

        user_src_emb = source_learn_user[user]
        src_pos_emb = source_learn_specific_item[src_pos]
        src_neg_emb = source_learn_specific_item[src_neg]
        user_tgt_emb = target_learn_user[user]
        tgt_pos_emb = target_learn_specific_item[tgt_pos]
        tgt_neg_emb = target_learn_specific_item[tgt_neg]

        pos_source_score = (user_src_emb * src_pos_emb).sum(dim=-1)
        neg_source_score = (user_src_emb * src_neg_emb).sum(dim=-1)
        pos_target_score = (user_tgt_emb * tgt_pos_emb).sum(dim=-1)
        neg_target_score =  (user_tgt_emb * tgt_neg_emb).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_source_score.size()).to(self.device), torch.zeros(
            pos_source_score.size()).to(self.device)

        loss = self.criterion(pos_source_score, pos_labels) + \
               self.criterion(neg_source_score, neg_labels) + \
               self.criterion(pos_target_score, pos_labels) + \
               self.criterion(neg_target_score, neg_labels) + \
               self.source_specific_GNN.encoder[-1].kld_loss + \
               self.target_specific_GNN.encoder[-1].kld_loss + self.kld_loss

        return loss

    def full_sort_predict(self, interaction, is_warm):
        user = interaction[0].long()
        source_user = self.source_user_embedding.weight[1:]
        target_user = self.target_user_embedding.weight[1:]
        source_item = self.source_item_embedding.weight[1:]
        target_item = self.target_item_embedding.weight[1:]
        source_user_share = self.source_user_embedding_share.weight[1:]
        target_user_share = self.target_user_embedding_share.weight[1:]

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,
                                                                                          self.source_UV,
                                                                                          self.source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,
                                                                                          self.target_UV,
                                                                                          self.target_VU)

        source_user_mean, source_user_sigma = self.source_share_GNN.forward_user_share(source_user, self.source_UV,
                                                                                       self.source_VU)
        target_user_mean, target_user_sigma = self.target_share_GNN.forward_user_share(target_user, self.target_UV,
                                                                                       self.target_VU)

        rate = torch.tensor(self.rate, dtype=torch.float32, device=self.device).unsqueeze(1)
        mean, sigma, = self.share_GNN(source_user_share, target_user_share, self.source_UV, self.source_VU,
                                      self.target_UV, self.target_VU, rate)

        user_share, share_kld_loss = self.reparameters(mean, sigma)

        source_share_kld = self._kld_gauss(mean, sigma, source_user_mean, source_user_sigma)
        target_share_kld = self._kld_gauss(mean, sigma, target_user_mean, target_user_sigma)

        self.kld_loss = share_kld_loss + self.opt["beta"] * source_share_kld + self.opt[
            "beta"] * target_share_kld

        # source_learn_user = self.source_merge(torch.cat((user_share, source_learn_specific_user), dim = -1))
        # target_learn_user = self.target_merge(torch.cat((user_share, target_learn_specific_user), dim = -1))
        source_learn_user = user_share + source_learn_specific_user
        target_learn_user = user_share + target_learn_specific_user

        zero_pad_t = torch.zeros(1, target_learn_user.size(1), device=self.device)
        target_learn_user = torch.cat([zero_pad_t, target_learn_user], dim=0)

        user_emb = target_learn_user[user]
        all_tgt_items_emb = torch.cat(
            [torch.zeros(1, target_learn_specific_item.size(1), device=self.device), target_learn_specific_item], dim=0)
        scores_tgt = torch.matmul(user_emb, all_tgt_items_emb.T)  # [B, n_target_items + 1]
        return scores_tgt


class singleVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(singleVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer(opt))
        self.encoder.append(LastLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
        return learn_user, learn_item

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        learn_user = ufea
        for layer in self.encoder[:-1]:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_user = layer.forward_user_share(learn_user, UV_adj, VU_adj)
        mean, sigma = self.encoder[-1].forward_user_share(learn_user, UV_adj, VU_adj)
        return mean, sigma

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        Item = torch.cat((Item_ho, vfea), dim=1)
        Item = self.item_union(Item)
        return F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

class LastLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        # sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logstd, 0.4)))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.zeros_like(logstd))
        return sampled_z, kld_loss

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        user, user_kld = self.forward_user(ufea, vfea, UV_adj,VU_adj)
        item, item_kld = self.forward_item(ufea, vfea, UV_adj, VU_adj)

        self.kld_loss = user_kld + item_kld

        return user, item


    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return user, kld_loss

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)

        Item_ho_mean = self.gc4_mean(Item_ho, VU_adj)
        Item_ho_logstd = self.gc4_logstd(Item_ho, VU_adj)
        Item_ho_mean = torch.cat((Item_ho_mean, vfea), dim=1)
        Item_ho_mean = self.item_union_mean(Item_ho_mean)

        Item_ho_logstd = torch.cat((Item_ho_logstd, vfea), dim=1)
        Item_ho_logstd = self.item_union_logstd(Item_ho_logstd)

        item, kld_loss = self.reparameters(Item_ho_mean, Item_ho_logstd)
        return item, kld_loss

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        # user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return User_ho_mean, User_ho_logstd
        # return user, kld_loss

class crossVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(crossVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer2(opt))
        self.encoder.append(LastLayer2(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj, source_rate):
        learn_user_source = source_ufea
        learn_user_target = target_ufea
        for layer in self.encoder[:-1]:
            learn_user_source = F.dropout(learn_user_source, self.dropout, training=self.training)
            learn_user_target = F.dropout(learn_user_target, self.dropout, training=self.training)
            learn_user_source, learn_user_target = layer(learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj, source_rate)

        mean, sigma, = self.encoder[-1](learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj, source_rate)
        return mean, sigma

class DGCNLayer2(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer2, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj, source_rate):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho = self.gc3(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho = self.gc4(target_User_ho, target_UV_adj)

        source_User = torch.cat((source_User_ho , source_ufea), dim=1)
        source_User = self.source_user_union(source_User)
        target_User = torch.cat((target_User_ho, target_ufea), dim=1)
        target_User = self.target_user_union(target_User)

        return source_rate * F.relu(source_User) +  (1 - source_rate) * F.relu(target_User), source_rate * F.relu(source_User) + (1 - source_rate) * F.relu(target_User)

class LastLayer2(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(LastLayer2, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.source_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])


    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        # sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logstd, 0.4)))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.zeros_like(logstd))
        return sampled_z, kld_loss

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj, source_rate):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho_mean = self.gc3_mean(source_User_ho, source_UV_adj)
        source_User_ho_logstd = self.gc3_logstd(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho_mean = self.gc4_mean(target_User_ho, target_UV_adj)
        target_User_ho_logstd = self.gc4_logstd(target_User_ho, target_UV_adj)

        source_User_mean = torch.cat(
            (source_User_ho_mean, source_ufea), dim=1)
        source_User_mean = self.source_user_union_mean(source_User_mean)

        source_User_logstd = torch.cat((source_User_ho_logstd, source_ufea), dim=1)
        source_User_logstd = self.source_user_union_logstd(source_User_logstd)

        target_User_mean = torch.cat(
            (target_User_ho_mean, target_ufea), dim=1)
        target_User_mean = self.target_user_union_mean(target_User_mean)

        target_User_logstd = torch.cat(
            (target_User_ho_logstd, target_ufea),
            dim=1)
        target_User_logstd = self.target_user_union_logstd(target_User_logstd)

        return source_rate * source_User_mean + (1 - source_rate) * target_User_mean, source_rate * source_User_logstd + (1 - source_rate) * target_User_logstd

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.leakyrelu(self.gc1(x, adj))
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = self.glorot_init(in_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial / 2)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'