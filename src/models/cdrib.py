import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
import numpy as np
import math
import torch.nn.functional as F
import scipy.sparse as sp

class CDRIB(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(CDRIB, self).__init__(config, dataloader)
        self.config =config
        self.feature_dim = config['feature_dim']

        self.source_GNN = VBGE(config)
        self.target_GNN = VBGE(config)
        self.criterion = nn.BCEWithLogitsLoss()
        self.discri = nn.Sequential(
            nn.Linear(config["feature_dim"]*2 * config["GNN"], config["feature_dim"]),
            nn.ReLU(),
            nn.Linear(config["feature_dim"], 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        self.dropout = config["dropout"]

        self.source_user_embedding = nn.Embedding(self.num_users_src + 1, self.feature_dim, padding_idx=0)
        self.target_user_embedding = nn.Embedding(self.num_users_tgt + 1, self.feature_dim, padding_idx=0)
        self.source_item_embedding = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)
        self.target_item_embedding = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)

        src_mat = dataloader.inter_matrix(domain=0, form='coo')
        tgt_mat = dataloader.inter_matrix(domain=1, form='coo')
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

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def dis(self, A, B):
        C = torch.cat((A,B), dim = 1)
        return self.discri(C)

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def HingeLoss(self, pos, neg):
        pos = torch.sigmoid(pos)
        neg = torch.sigmoid(neg)
        gamma = torch.tensor(self.config["margin"], device=pos.device)
        return torch.relu(gamma - pos + neg).mean()

    def calculate_loss(self, interaction, epoch_idx):
        user_src = interaction['users_src']
        src_pos = interaction['pos_items_src']
        src_neg = interaction['neg_items_src']
        user_tgt = interaction['users_tgt']
        tgt_pos = interaction['pos_items_tgt']
        tgt_neg = interaction['neg_items_tgt']

        source_user = self.source_user_embedding.weight[1:]
        target_user = self.target_user_embedding.weight[1:]
        source_item = self.source_item_embedding.weight[1:]
        target_item = self.target_item_embedding.weight[1:]
        source_learn_user, source_learn_item = self.source_GNN(source_user, source_item, self.source_UV, self.source_VU)  # no padding of index = 0
        target_learn_user, target_learn_item = self.target_GNN(target_user, target_item, self.target_UV, self.target_VU)  # no padding of index = 0

        per_stable = torch.randperm(self.num_users_overlap)[:self.config["user_batch_size"]].to(self.device)
        pos = self.dis(self.my_index_select(source_learn_user, per_stable),self.my_index_select(target_learn_user, per_stable)).view(-1)
        per = torch.randperm(self.num_users_tgt)[:self.config["user_batch_size"]].to(self.device)
        neg_share = self.my_index_select(target_learn_user, per)
        neg_1 = self.dis(self.my_index_select(source_learn_user, per_stable), neg_share).view(-1)
        per = torch.randperm(self.num_users_src)[:self.config["user_batch_size"]].to(self.device)
        neg_share = self.my_index_select(source_learn_user, per)
        neg_2 = self.dis(neg_share, self.my_index_select(target_learn_user, per_stable)).view(-1)

        if self.config['bce']:
            pos_label = torch.ones_like(pos, device=pos.device)
            neg_label = torch.zeros_like(neg_1, device=neg_1.device)
            self.critic_loss = (
                    self.criterion(pos, pos_label)
                    + self.criterion(neg_1, neg_label)
                    + self.criterion(neg_2, neg_label)
            )
        else:
            self.critic_loss = self.HingeLoss(pos, neg_1) + self.HingeLoss(pos, neg_2)

        source_learn_user_concat = torch.cat(
            (target_learn_user[:self.num_users_overlap], source_learn_user[self.num_users_overlap:]), dim=0)
        target_learn_user_concat = torch.cat(
            (source_learn_user[:self.num_users_overlap], target_learn_user[self.num_users_overlap:]), dim=0)
        # add 0-padding row at top
        zero_pad_s = torch.zeros(1, source_learn_user_concat.size(1), device=self.device)
        zero_pad_t = torch.zeros(1, target_learn_user_concat.size(1), device=self.device)

        source_learn_user_concat = torch.cat([zero_pad_s, source_learn_user_concat], dim=0)
        target_learn_user_concat = torch.cat([zero_pad_t, target_learn_user_concat], dim=0)
        source_learn_item = torch.cat([zero_pad_s, source_learn_item], dim=0)
        target_learn_item = torch.cat([zero_pad_t, target_learn_item], dim=0)

        user_src_emb = source_learn_user_concat[user_src]
        src_pos_emb = source_learn_item[src_pos]
        src_neg_emb = source_learn_item[src_neg]
        user_tgt_emb = target_learn_user_concat[user_tgt]
        tgt_pos_emb = target_learn_item[tgt_pos]
        tgt_neg_emb = target_learn_item[tgt_neg]

        pos_source_score = (user_src_emb * src_pos_emb).sum(dim=-1)
        neg_source_score = (user_src_emb * src_neg_emb).sum(dim=-1)
        pos_target_score = (user_tgt_emb * tgt_pos_emb).sum(dim=-1)
        neg_target_score =  (user_tgt_emb * tgt_neg_emb).sum(dim=-1)

        source_pos_labels, source_neg_labels = torch.ones(pos_source_score.size()).to(self.device), torch.zeros(pos_source_score.size()).to(self.device)
        target_pos_labels, target_neg_labels = torch.ones(pos_target_score.size()).to(self.device), torch.zeros(pos_target_score.size()).to(self.device)

        loss = (self.criterion(pos_source_score, source_pos_labels)
                + self.criterion(neg_source_score, source_neg_labels)
                + self.criterion(pos_target_score, target_pos_labels)
                + self.criterion(neg_target_score, target_neg_labels) + \
               self.source_GNN.encoder[-1].kld_loss + self.target_GNN.encoder[-1].kld_loss
                + self.critic_loss)

        # without_contrastive variant
        # loss = (self.criterion(pos_source_score, source_pos_labels)
        #         + self.criterion(neg_source_score, source_neg_labels)
        #         + self.criterion(pos_target_score, target_pos_labels)
        #         + self.criterion(neg_target_score, target_neg_labels) + \
        #        self.source_GNN.encoder[-1].kld_loss + self.target_GNN.encoder[-1].kld_loss)

        return loss

    def full_sort_predict(self, interaction, is_warm):
        user = interaction[0].long()
        source_user = self.source_user_embedding.weight[1:]
        target_user = self.target_user_embedding.weight[1:]
        source_item = self.source_item_embedding.weight[1:]
        target_item = self.target_item_embedding.weight[1:]
        source_learn_user, source_learn_item = self.source_GNN(source_user, source_item, self.source_UV, self.source_VU)  # no padding of index = 0
        target_learn_user, target_learn_item = self.target_GNN(target_user, target_item, self.target_UV, self.target_VU)  # no padding of index = 0
        if not is_warm:
            zero_pad = torch.zeros(1, source_learn_user.size(1), device=self.device)
            source_learn_user = torch.cat([zero_pad, source_learn_user], dim=0)
            user_emb = source_learn_user[user]
            all_tgt_items_emb = torch.cat(
                [torch.zeros(1, target_learn_item.size(1), device=self.device), target_learn_item], dim=0)
            scores_tgt = torch.matmul(user_emb, all_tgt_items_emb.T)  # [B, n_target_items + 1]
            return scores_tgt
        else:
            zero_pad = torch.zeros(1, target_learn_user.size(1), device=self.device)
            target_learn_user = torch.cat([zero_pad, target_learn_user], dim=0)
            user_emb = target_learn_user[user]
            all_tgt_items_emb = torch.cat(
                [torch.zeros(1, target_learn_item.size(1), device=self.device), target_learn_item], dim=0)
            scores_tgt = torch.matmul(user_emb, all_tgt_items_emb.T)  # [B, n_target_items + 1]
            return scores_tgt

    def set_train_stage(self, stage_id):
        super(CDRIB, self).set_train_stage(stage_id)



class VBGE(nn.Module):
    """
        VBGE Module layer
    """

    def __init__(self, config):
        super(VBGE, self).__init__()
        self.config = config
        self.layer_number = config["GNN"]
        self.encoder = []
        for i in range(self.layer_number - 1):
            self.encoder.append(DGCNLayer(config))
        self.encoder.append(LastLayer(config))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = config["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user = ufea
        learn_item = vfea
        user_ret = None
        item_ret = None
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
            if user_ret is None:
                user_ret = learn_user
                item_ret = learn_item
            else :
                user_ret = torch.cat((user_ret, learn_user), dim = -1)
                item_ret = torch.cat((item_ret, learn_item), dim = -1)
        return user_ret, item_ret

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

    def __init__(self, config):
        super(DGCNLayer, self).__init__()
        self.opt = config
        self.dropout = config["dropout"]
        self.gc1 = GCN(
            nfeat=config["feature_dim"],
            nhid=config["hidden_dim"],
            dropout=config["dropout"],
            alpha=config["leakey"]
        )

        self.gc2 = GCN(
            nfeat=config["feature_dim"],
            nhid=config["hidden_dim"],
            dropout=config["dropout"],
            alpha=config["leakey"]
        )
        self.gc3 = GCN(
            nfeat=config["hidden_dim"],  # change
            nhid=config["feature_dim"],
            dropout=config["dropout"],
            alpha=config["leakey"]
        )

        self.gc4 = GCN(
            nfeat=config["hidden_dim"],  # change
            nhid=config["feature_dim"],
            dropout=config["dropout"],
            alpha=config["leakey"]
        )
        self.user_union = nn.Linear(config["feature_dim"] + config["feature_dim"], config["feature_dim"])
        self.item_union = nn.Linear(config["feature_dim"] + config["feature_dim"], config["feature_dim"])

    def forward(self, ufea, vfea, UV_adj, VU_adj):
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

    def forward_item(self, ufea, vfea, UV_adj, VU_adj):
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
        self.opt = opt
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
            nfeat=opt["hidden_dim"],  # change
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
            nfeat=opt["hidden_dim"],  # change
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
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logstd, 0.4)))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            self.sigma = sigma
            sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        item, item_kld = self.forward_item(ufea, vfea, UV_adj, VU_adj)
        user, user_kld = self.forward_user(ufea, vfea, UV_adj, VU_adj)

        self.kld_loss = self.opt["beta"] * user_kld + item_kld

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

    def forward_item(self, ufea, vfea, UV_adj, VU_adj):
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

        return User_ho_mean, User_ho_logstd

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.leakyrelu(self.gc1(x, adj))
        return x

class GraphConvolution(Module):
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