import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from common.abstract_recommender import GeneralRecommender
import numpy as np
import scipy.sparse as sp

def scipy_coo_to_torch_sparse(coo):
    coo = coo.tocoo().astype(np.float32)

    rowsum = np.array(coo.sum(1)).flatten()
    r_inv = np.power(rowsum, -1, where=rowsum != 0)
    r_inv[np.isinf(r_inv)] = 0.

    r_mat_inv = sp.diags(r_inv)
    coo_norm = r_mat_inv.dot(coo)

    coo_norm = coo_norm.tocoo().astype(np.float32)

    row = torch.from_numpy(coo_norm.row).long()
    col = torch.from_numpy(coo_norm.col).long()
    data = torch.from_numpy(coo_norm.data).float()
    indices = torch.stack([row, col])
    shape = torch.Size(coo_norm.shape)

    return torch.sparse_coo_tensor(indices, data, shape).coalesce()


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias if self.bias is not None else output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.leakyrelu(self.gc(x, adj))
        return x

class DGCNLayer(nn.Module):
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.gc1 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc2 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc3 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.user_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        user_ho = self.gc1(ufea, VU_adj)
        item_ho = self.gc2(vfea, UV_adj)
        user_ho = self.gc3(user_ho, UV_adj)
        item_ho = self.gc4(item_ho, VU_adj)
        user = self.user_union(torch.cat((user_ho, ufea), dim=1))
        item = self.item_union(torch.cat((item_ho, vfea), dim=1))
        return F.relu(user), F.relu(item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        user_ho = self.gc1(ufea, VU_adj)
        user_ho = self.gc3(user_ho, UV_adj)
        user = self.user_union(torch.cat((user_ho, ufea), dim=1))
        return F.relu(user)

class LastLayer(nn.Module):
    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt = opt
        in_dim = opt["feature_dim"]
        hid_dim = opt["hidden_dim"]
        dropout = opt["dropout"]
        alpha = opt["leakey"]

        self.gc1 = GCN(in_dim, hid_dim, dropout, alpha)
        self.gc3_mean = GCN(in_dim, hid_dim, dropout, alpha)
        self.gc3_logstd = GCN(in_dim, hid_dim, dropout, alpha)

        self.user_union_mean = nn.Linear(hid_dim * 2, hid_dim)
        self.user_union_logstd = nn.Linear(hid_dim * 2, hid_dim)
        self.item_union_mean = nn.Linear(hid_dim * 2, hid_dim)
        self.item_union_logstd = nn.Linear(hid_dim * 2, hid_dim)

    def _kld_gauss(self, mu1, logstd1, mu2, logstd2):
        return torch.mean(
            0.5 * (
                2 * (logstd2 - logstd1)
                + (torch.exp(2 * logstd1) + (mu1 - mu2) ** 2) / torch.exp(2 * logstd2)
                - 1
            )
        )

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        noise = torch.randn_like(mean)
        sampled = noise * sigma + mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled, kld_loss


    def forward(self, ufea, vfea, UV_adj, VU_adj):
        user_ho = self.gc1(ufea, VU_adj)               # [item_num, hid]
        user_ho_mean = self.gc3_mean(user_ho, UV_adj)  # [user_num, hid]
        user_ho_logstd = self.gc3_logstd(user_ho, UV_adj)

        user_mean_input = torch.cat((user_ho_mean, ufea), dim=1)      # [user_num, hid+feat]
        user_logstd_input = torch.cat((user_ho_logstd, ufea), dim=1)

        user_mean = self.user_union_mean(user_mean_input)
        user_logstd = self.user_union_logstd(user_logstd_input)

        user_z, user_kld = self.reparameters(user_mean, user_logstd)

        item_ho = self.gc1(vfea, UV_adj)               # [user_num, hid]
        item_ho_mean = self.gc3_mean(item_ho, VU_adj)  # [item_num, hid]
        item_ho_logstd = self.gc3_logstd(item_ho, VU_adj)

        item_mean_input = torch.cat((item_ho_mean, vfea), dim=1)      # [item_num, hid+feat]
        item_logstd_input = torch.cat((item_ho_logstd, vfea), dim=1)

        item_mean = self.item_union_mean(item_mean_input)
        item_logstd = self.item_union_logstd(item_logstd_input)

        item_z, item_kld = self.reparameters(item_mean, item_logstd)

        self.kld_loss = user_kld + item_kld
        return user_z, item_z

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)
        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)
        return User_ho_mean, User_ho_logstd


class CrossDGCNLayer(nn.Module):
    def __init__(self, opt):
        super(CrossDGCNLayer, self).__init__()
        nfeat = opt["feature_dim"]
        nhid = opt["hidden_dim"]
        dropout = opt["dropout"]
        alpha = opt["leakey"]

        self.gc1 = GCN(nfeat, nhid, dropout, alpha)  # source U<-V
        self.gc2 = GCN(nfeat, nhid, dropout, alpha)  # target U<-V
        self.gc3 = GCN(nhid, nhid, dropout, alpha)   # source U->V
        self.gc4 = GCN(nhid, nhid, dropout, alpha)   # target U->V

        self.source_user_union = nn.Linear(nhid * 2, nhid)
        self.target_user_union = nn.Linear(nhid * 2, nhid)
        self.source_rate = opt["rate"]

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho = self.gc3(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho = self.gc4(target_User_ho, target_UV_adj)

        source_User = torch.cat((source_User_ho, source_ufea), dim=1)
        source_User = self.source_user_union(source_User)

        target_User = torch.cat((target_User_ho, target_ufea), dim=1)
        target_User = self.target_user_union(target_User)

        return self.source_rate * F.relu(source_User) + (1 - self.source_rate) * F.relu(target_User), \
               self.source_rate * F.relu(source_User) + (1 - self.source_rate) * F.relu(target_User)

class CrossLastLayer(nn.Module):
    def __init__(self, opt):
        super(CrossLastLayer, self).__init__()
        nfeat = opt["feature_dim"]
        nhid = opt["hidden_dim"]
        dropout = opt["dropout"]
        alpha = opt["leakey"]
        self.source_rate = opt["rate"]

        self.gc1 = GCN(nfeat, nhid, dropout, alpha)
        self.gc3_mean = GCN(nhid, nhid, dropout, alpha)
        self.gc3_logstd = GCN(nhid, nhid, dropout, alpha)

        self.gc2 = GCN(nfeat, nhid, dropout, alpha)
        self.gc4_mean = GCN(nhid, nhid, dropout, alpha)
        self.gc4_logstd = GCN(nhid, nhid, dropout, alpha)

        self.source_user_union_mean = nn.Linear(nhid * 2, nhid)
        self.source_user_union_logstd = nn.Linear(nhid * 2, nhid)
        self.target_user_union_mean = nn.Linear(nhid * 2, nhid)
        self.target_user_union_logstd = nn.Linear(nhid * 2, nhid)

    def forward(self, source_ufea, target_ufea,
                source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho_mean = self.gc3_mean(source_User_ho, source_UV_adj)
        source_User_ho_logstd = self.gc3_logstd(source_User_ho, source_UV_adj)

        source_User_mean = torch.cat((source_User_ho_mean, source_ufea), dim=1)
        source_User_mean = self.source_user_union_mean(source_User_mean)

        source_User_logstd = torch.cat((source_User_ho_logstd, source_ufea), dim=1)
        source_User_logstd = self.source_user_union_logstd(source_User_logstd)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho_mean = self.gc4_mean(target_User_ho, target_UV_adj)
        target_User_ho_logstd = self.gc4_logstd(target_User_ho, target_UV_adj)

        target_User_mean = torch.cat((target_User_ho_mean, target_ufea), dim=1)
        target_User_mean = self.target_user_union_mean(target_User_mean)

        target_User_logstd = torch.cat((target_User_ho_logstd, target_ufea), dim=1)
        target_User_logstd = self.target_user_union_logstd(target_User_logstd)

        mean = self.source_rate * source_User_mean + (1 - self.source_rate) * target_User_mean
        logstd = self.source_rate * source_User_logstd + (1 - self.source_rate) * target_User_logstd
        return mean, logstd


class singleVBGE(nn.Module):
    def __init__(self, opt):
        super(singleVBGE, self).__init__()
        self.opt = opt
        self.encoder = nn.ModuleList(
            [DGCNLayer(opt) for _ in range(opt["GNN"] - 1)] + [LastLayer(opt)]
        )
        self.dropout = opt["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user, learn_item = ufea, vfea
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

class crossVBGE(nn.Module):
    def __init__(self, opt):
        super(crossVBGE, self).__init__()
        self.encoder = nn.ModuleList(
            [CrossDGCNLayer(opt) for _ in range(opt["GNN"] - 1)] +
            [CrossLastLayer(opt)]
        )
        self.dropout = opt["dropout"]

    def forward(self, src_ufea, tgt_ufea, src_UV, src_VU, tgt_UV, tgt_VU):
        learn_src, learn_tgt = src_ufea, tgt_ufea
        for layer in self.encoder[:-1]:
            learn_src = F.dropout(learn_src, self.dropout, training=self.training)
            learn_tgt = F.dropout(learn_tgt, self.dropout, training=self.training)
            learn_src, learn_tgt = layer(learn_src, learn_tgt, src_UV, src_VU, tgt_UV, tgt_VU)

        mean, sigma = self.encoder[-1](learn_src, learn_tgt, src_UV, src_VU, tgt_UV, tgt_VU)
        return mean, sigma


class DisenCDR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DisenCDR, self).__init__(config, dataset)
        self.opt = config
        self.beta = config["beta"]
        self.dropout = config["dropout"]

        mat_source, mat_target = dataset.inter_matrix(form="coo")
        self.source_VU = scipy_coo_to_torch_sparse(mat_source.T).to(config["device"])
        self.source_UV = scipy_coo_to_torch_sparse(mat_source).to(config["device"])
        self.target_VU = scipy_coo_to_torch_sparse(mat_target.T).to(config["device"])
        self.target_UV = scipy_coo_to_torch_sparse(mat_target).to(config["device"])

        self.source_specific_GNN = singleVBGE(config)
        self.source_share_GNN = singleVBGE(config)
        self.target_specific_GNN = singleVBGE(config)
        self.target_share_GNN = singleVBGE(config)
        self.share_GNN = crossVBGE(config)

        self.source_user_embedding = nn.Embedding(self.n_users, config["feature_dim"])
        self.target_user_embedding = nn.Embedding(self.n_users, config["feature_dim"])
        self.source_item_embedding = nn.Embedding(self.n_source_items, config["feature_dim"])
        self.target_item_embedding = nn.Embedding(self.n_target_items, config["feature_dim"])
        self.source_user_embedding_share = nn.Embedding(self.n_users, config["feature_dim"])
        self.target_user_embedding_share = nn.Embedding(self.n_users, config["feature_dim"])

        self.criterion = nn.BCEWithLogitsLoss()

    def _kld_gauss(self, mu1, log1, mu2, log2):
        s1 = torch.exp(0.1 + 0.9 * F.softplus(log1))
        s2 = torch.exp(0.1 + 0.9 * F.softplus(log2))
        return kl_divergence(Normal(mu1, s1), Normal(mu2, s2)).mean(dim=0).sum()

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).to(mean.device)
        z = noise * torch.exp(sigma) + mean if self.training else mean
        kld = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return z, (1 - self.beta) * kld

    def forward(self):
        src_user = self.source_user_embedding.weight
        tgt_user = self.target_user_embedding.weight
        src_item = self.source_item_embedding.weight
        tgt_item = self.target_item_embedding.weight
        src_user_share = self.source_user_embedding_share.weight
        tgt_user_share = self.target_user_embedding_share.weight

        src_spec_u, src_spec_i = self.source_specific_GNN(src_user, src_item, self.source_UV, self.source_VU)
        tgt_spec_u, tgt_spec_i = self.target_specific_GNN(tgt_user, tgt_item, self.target_UV, self.target_VU)
        src_mean, src_log = self.source_share_GNN.forward_user_share(src_user, self.source_UV, self.source_VU)
        tgt_mean, tgt_log = self.target_share_GNN.forward_user_share(tgt_user, self.target_UV, self.target_VU)
        mean, sigma = self.share_GNN(src_user_share, tgt_user_share, self.source_UV, self.source_VU, self.target_UV, self.target_VU)

        user_share, share_kld = self.reparameters(mean, sigma)
        src_share_kld = self._kld_gauss(mean, sigma, src_mean, src_log)
        tgt_share_kld = self._kld_gauss(mean, sigma, tgt_mean, tgt_log)
        self.kld_loss = share_kld + self.beta * (src_share_kld + tgt_share_kld)

        src_learn_u = user_share + src_spec_u
        tgt_learn_u = user_share + tgt_spec_u
        return src_learn_u, src_spec_i, tgt_learn_u, tgt_spec_i

    def wramup(self):
        src_user = self.source_user_embedding.weight
        tgt_user = self.target_user_embedding.weight
        src_item = self.source_item_embedding.weight
        tgt_item = self.target_item_embedding.weight

        src_user_fea, src_item_fea = self.source_specific_GNN(src_user, src_item, self.source_UV, self.source_VU)
        tgt_user_fea, tgt_item_fea = self.target_specific_GNN(tgt_user, tgt_item, self.target_UV, self.target_VU)

        self.kld_loss = 0
        return src_user_fea, src_item_fea, tgt_user_fea, tgt_item_fea

    def calculate_loss(self, interaction, epoch_idx):
        user, src_pos, src_neg, tgt_pos, tgt_neg = interaction
        if epoch_idx < 10:
            src_user, src_item, tgt_user, tgt_item = self.wramup()
        else:
            src_user, src_item, tgt_user, tgt_item = self.forward()

        src_user_fea = torch.index_select(src_user, 0, user)
        tgt_user_fea = torch.index_select(tgt_user, 0, user)
        src_pos_fea = torch.index_select(src_item, 0, src_pos)
        src_neg_fea = torch.index_select(src_item, 0, src_neg)
        tgt_pos_fea = torch.index_select(tgt_item, 0, tgt_pos)
        tgt_neg_fea = torch.index_select(tgt_item, 0, tgt_neg)

        pos_src = (src_user_fea * src_pos_fea).sum(dim=-1)
        neg_src = (src_user_fea * src_neg_fea).sum(dim=-1)
        pos_tgt = (tgt_user_fea * tgt_pos_fea).sum(dim=-1)
        neg_tgt = (tgt_user_fea * tgt_neg_fea).sum(dim=-1)

        pos_label = torch.ones_like(pos_src)
        neg_label = torch.zeros_like(neg_src)

        loss = (
            self.criterion(pos_src, pos_label)
            + self.criterion(neg_src, neg_label)
            + self.criterion(pos_tgt, pos_label)
            + self.criterion(neg_tgt, neg_label)
            + self.source_specific_GNN.encoder[-1].kld_loss
            + self.target_specific_GNN.encoder[-1].kld_loss
            + self.kld_loss
        )
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0].long()  # [B]

        src_user, src_item, tgt_user, tgt_item = self.forward()

        src_user_batch = torch.index_select(src_user, 0, user)
        tgt_user_batch = torch.index_select(tgt_user, 0, user)

        scores_src = torch.matmul(src_user_batch, src_item.T)
        scores_tgt = torch.matmul(tgt_user_batch, tgt_item.T)

        return scores_src, scores_tgt

