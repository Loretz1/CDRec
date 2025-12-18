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

class GDCCDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(GDCCDR, self).__init__(config, dataloader)
        self.config = config
        self.feature_dim = config['feature_dim']
        self.rank_k = config['rank_k']
        self.beta = config['beta']
        self.meta_hidden = config['meta_hidden']

        self.user_emb_src = nn.Embedding(self.num_users_src + 1, self.feature_dim, padding_idx=0)
        self.user_emb_tgt = nn.Embedding(self.num_users_tgt + 1, self.feature_dim, padding_idx=0)
        self.item_emb_src = nn.Embedding(self.num_items_src + 1, self.feature_dim, padding_idx=0)
        self.item_emb_tgt = nn.Embedding(self.num_items_tgt + 1, self.feature_dim, padding_idx=0)

        self.src_user_I_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.src_user_S_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.tgt_user_I_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.tgt_user_S_proj = nn.Linear(self.feature_dim, self.feature_dim)

        src_raw = dataloader.inter_matrix(domain=0, form='coo')
        tgt_raw = dataloader.inter_matrix(domain=1, form='coo')
        self.src_R = self._scipy_coo_to_torch(src_raw)
        self.tgt_R = self._scipy_coo_to_torch(tgt_raw)

        src_mat = dataloader.inter_matrix(domain=0, form='coo')
        tgt_mat = dataloader.inter_matrix(domain=1, form='coo')
        src_mat = self.normalize(src_mat)
        tgt_mat = self.normalize(tgt_mat)
        self.src_adj = self._scipy_coo_to_torch(src_mat)
        self.tgt_adj = self._scipy_coo_to_torch(tgt_mat)

        # H_U^A: 4d -> (d*k)
        self.meta_user_src = nn.Sequential(
            nn.Linear(4 * self.feature_dim, self.meta_hidden),
            nn.Tanh(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.meta_hidden, self.feature_dim * self.rank_k),
            nn.Tanh()
        )

        # H_V^A: 2d -> (k*d)
        self.meta_item_src = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.meta_hidden),
            nn.Tanh(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.meta_hidden, self.rank_k * self.feature_dim),
            nn.Tanh()
        )

        # H_U^A: 4d -> (d*k)
        self.meta_user_tgt = nn.Sequential(
            nn.Linear(4 * self.feature_dim, self.meta_hidden),
            nn.Tanh(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.meta_hidden, self.feature_dim * self.rank_k),
            nn.Tanh()
        )

        # H_V^A: 2d -> (k*d)
        self.meta_item_tgt = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.meta_hidden),
            nn.Tanh(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.meta_hidden, self.rank_k * self.feature_dim),
            nn.Tanh()
        )

        self.apply(xavier_uniform_initialization)

    def _scipy_coo_to_torch(self, mat):
        mat = mat.tocoo()  # ensure COO format
        indices = torch.tensor(
            [mat.row, mat.col], dtype=torch.long, device=self.device
        )  # [2, nnz]
        values = torch.tensor(mat.data, dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(indices, values, mat.shape, device=self.device).coalesce()

    def normalize(self, mat):
        """
        Symmetric normalization for user-item bipartite graph
        \bar{R} = D_u^{-1/2} R D_v^{-1/2}
        """
        mat = mat.tocoo()

        # user degree
        row_sum = np.array(mat.sum(axis=1)).flatten()
        d_u_inv_sqrt = np.power(row_sum, -0.5)
        d_u_inv_sqrt[np.isinf(d_u_inv_sqrt)] = 0.
        D_u_inv_sqrt = sp.diags(d_u_inv_sqrt)

        # item degree
        col_sum = np.array(mat.sum(axis=0)).flatten()
        d_v_inv_sqrt = np.power(col_sum, -0.5)
        d_v_inv_sqrt[np.isinf(d_v_inv_sqrt)] = 0.
        D_v_inv_sqrt = sp.diags(d_v_inv_sqrt)

        return D_u_inv_sqrt.dot(mat).dot(D_v_inv_sqrt)

    def disentangle_user_embedding(self, u_emb, domain):
        if domain == 'src':
            gate_I = torch.sigmoid(self.src_user_I_proj(u_emb))
            gate_S = torch.sigmoid(self.src_user_S_proj(u_emb))
        elif domain == 'tgt':
            gate_I = torch.sigmoid(self.tgt_user_I_proj(u_emb))
            gate_S = torch.sigmoid(self.tgt_user_S_proj(u_emb))
        else:
            raise ValueError(f"Unknown domain: {domain}")
        u_I = u_emb * gate_I
        u_S = u_emb * gate_S

        return u_I, u_S

    def graph_forward(self, domain: str):
        assert domain in ['src', 'tgt']
        if domain == 'src':
            u_emb = self.user_emb_src.weight[1:]  # [U, d]
            v_emb = self.item_emb_src.weight[1:] # [V, d]
            adj = self.src_adj
        else:
            u_emb = self.user_emb_tgt.weight[1:]
            v_emb = self.item_emb_tgt.weight[1:]
            adj = self.tgt_adj

        u_I, u_S = self.disentangle_user_embedding(u_emb, domain)
        v_I = v_emb
        v_S = v_emb

        u_I_layers = [u_I]
        u_S_layers = [u_S]
        v_I_layers = [v_I]
        v_S_layers = [v_S]

        edge_index = adj.indices()
        row, col = edge_index[0], edge_index[1]

        for _ in range(self.config['GNN']):
            u_I_base = torch.sparse.mm(adj, v_I_layers[-1])
            v_I_base = torch.sparse.mm(adj.t(), u_I_layers[-1])
            u_S_base = torch.sparse.mm(adj, v_S_layers[-1])
            v_S_base = torch.sparse.mm(adj.t(), u_S_layers[-1])

            score_I = (u_I_layers[-1][row] * v_I_layers[-1][col]).sum(dim=1)
            score_S = (u_S_layers[-1][row] * v_S_layers[-1][col]).sum(dim=1)
            F_I = torch.sigmoid(score_I)
            F_S = torch.sigmoid(score_S)
            G_I = torch.sparse_coo_tensor(
                edge_index,
                adj.values() * F_I,
                adj.shape,
                device=adj.device
            )
            G_S = torch.sparse_coo_tensor(
                edge_index,
                adj.values() * F_S,
                adj.shape,
                device=adj.device
            )
            u_I_adp = torch.sparse.mm(G_I, v_I_layers[-1])
            v_I_adp = torch.sparse.mm(G_I.t(), u_I_layers[-1])
            u_S_adp = torch.sparse.mm(G_S, v_S_layers[-1])
            v_S_adp = torch.sparse.mm(G_S.t(), u_S_layers[-1])

            u_I_next = u_I_base + self.config['alpha'] * u_I_adp
            v_I_next = v_I_base + self.config['alpha'] * v_I_adp
            u_S_next = u_S_base + self.config['alpha'] * u_S_adp
            v_S_next = v_S_base + self.config['alpha'] * v_S_adp

            u_I_layers.append(u_I_next)
            v_I_layers.append(v_I_next)
            u_S_layers.append(u_S_next)
            v_S_layers.append(v_S_next)

        u_I_final = torch.mean(torch.stack(u_I_layers, dim=0), dim=0)
        u_S_final = torch.mean(torch.stack(u_S_layers, dim=0), dim=0)
        v_I_final = torch.mean(torch.stack(v_I_layers, dim=0), dim=0)
        v_S_final = torch.mean(torch.stack(v_S_layers, dim=0), dim=0)
        v_final = (v_I_final + v_S_final) / 2
        return u_I_final, u_S_final, v_final, u_I_layers, u_S_layers

    def ecl_one_domain(self, u_S_dom, v_dom, u_S_other_layers, users, pos_items, tau_ecl):
        """
        u_S_dom: [U, d]   (final pooled)
        v_dom:   [V, d]
        u_S_other_layers: list of tensors, each [U, d], length L+1
        users, pos_items: [B]
        """
        # positive score: s(u^{A,S}, v^A)
        pos = torch.sum(u_S_dom[users] * v_dom[pos_items], dim=1)  # [B]

        # negative scores: s(u^{B,S}_{l}, v^A) for all layers l, then sum exp over l
        # Build [B, L+1]
        neg_scores = []
        vj = v_dom[pos_items]  # [B, d]
        for uS_l in u_S_other_layers:  # each [U, d]
            neg_scores.append(torch.sum(uS_l[users] * vj, dim=1))  # [B]
        neg = torch.stack(neg_scores, dim=1)  # [B, L+1]

        # denominator = log(exp(pos) + sum_l exp(neg_l))
        # use logsumexp over concatenated logits
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1) / tau_ecl  # [B, 1+L+1]
        # -log exp(pos)/sum exp = -(pos - logsumexp)
        loss = -(logits[:, 0] - torch.logsumexp(logits, dim=1)).mean()
        return loss

    def build_meta_matrices(self, u_I_dom, u_S_dom, v_dom, u_I_other, domain: str):
        """
        Build personalized low-rank transfer matrices for a given domain.

        Args:
            u_I_dom:  [U, d]   current domain invariant users (e.g., U^{A,I})
            u_S_dom:  [U, d]   current domain specific users   (e.g., U^{A,S})
            v_dom:    [V, d]   current domain items (V^A)
            u_I_other:[U, d]   other domain invariant users (U^{B,I})
            domain: 'src' or 'tgt'  (current domain)

        Returns:
            WeU: [U, d, k]
            WeV: [V, k, d]
        """
        if domain == 'src':
            R = self.src_R
            meta_user = self.meta_user_src
            meta_item = self.meta_item_src
        else:
            R = self.tgt_R
            meta_user = self.meta_user_tgt
            meta_item = self.meta_item_tgt

        # Eq.(9): neighbor sums (use raw R)
        # sum_{j in N_i} v_j  -> [U, d]
        neigh_item_sum = torch.sparse.mm(R, v_dom)

        # sum_{i in N_j} (u_I + u_S) -> [V, d]
        neigh_user_sum = torch.sparse.mm(R.t(), (u_I_dom + u_S_dom))

        # H_U: [U, 4d] = U^{dom,I} || U^{other,I} || U^{dom,S} || neigh_item_sum
        H_U = torch.cat([u_I_dom, u_I_other, u_S_dom, neigh_item_sum], dim=1)

        # H_V: [V, 2d] = V^{dom} || neigh_user_sum
        H_V = torch.cat([v_dom, neigh_user_sum], dim=1)

        # Eq.(10): meta nets -> low-rank matrices
        WeU = meta_user(H_U).view(-1, self.feature_dim, self.rank_k)  # [U, d, k]
        WeV = meta_item(H_V).view(-1, self.rank_k, self.feature_dim)  # [V, k, d]
        return WeU, WeV

    def personalized_transfer(self, domain, users, items, u_I_dom, u_S_dom, v_dom, u_I_other):
        """
        Compute u^{dom,F}_{i,j} for a batch of (users, items).

        Returns:
            u_F_batch: [B, d]
        """
        WeU, WeV = self.build_meta_matrices(u_I_dom, u_S_dom, v_dom, u_I_other, domain)

        Wu = WeU[users]  # [B, d, k]
        Wv = WeV[items]  # [B, k, d]
        u_other = u_I_other[users].unsqueeze(-1)  # [B, d, 1]

        # Eq.(11): u_T = Wu * Wv * u_other + u_other
        tmp = torch.bmm(Wv, u_other)  # [B, k, 1]
        u_T = torch.bmm(Wu, tmp).squeeze(-1)  # [B, d]
        u_T = u_T + u_other.squeeze(-1)  # resi
        # dual + u^{other,I}_i

        # Eq.(12): u_F = u^{dom,I}_i + beta * u_T
        u_F = u_I_dom[users] + self.beta * u_T
        return u_F, u_T

    def calculate_loss(self, interaction, epoch_idx):
        user = interaction['users'] - 1
        src_pos = interaction['pos_items_src'] - 1
        src_neg = interaction['neg_items_src'] - 1
        tgt_pos = interaction['pos_items_tgt'] - 1
        tgt_neg = interaction['neg_items_tgt'] - 1

        u_I_src, u_S_src, v_src, uI_src_layers, uS_src_layers = self.graph_forward(domain='src')
        u_I_tgt, u_S_tgt, v_tgt, uI_tgt_layers, uS_tgt_layers = self.graph_forward(domain='tgt')

        loss_ecl_src = self.ecl_one_domain(
            u_S_dom=u_S_src, v_dom=v_src,
            u_S_other_layers=uS_tgt_layers,
            users=user, pos_items=src_pos,
            tau_ecl=self.config['tau_ecl']
        )

        loss_ecl_tgt = self.ecl_one_domain(
            u_S_dom=u_S_tgt, v_dom=v_tgt,
            u_S_other_layers=uS_src_layers,
            users=user, pos_items=tgt_pos,
            tau_ecl=self.config['tau_ecl']
        )

        uF_src_pos, uT_src_pos = self.personalized_transfer(
            domain='src',
            users=user,
            items=src_pos,
            u_I_dom=u_I_src, u_S_dom=u_S_src, v_dom=v_src,
            u_I_other=u_I_tgt
        )  # [B, d]
        uF_src_neg, _ = self.personalized_transfer(
            domain='src',
            users=user,
            items=src_neg,
            u_I_dom=u_I_src, u_S_dom=u_S_src, v_dom=v_src,
            u_I_other=u_I_tgt
        )

        src_pos_score = ((uF_src_pos * v_src[src_pos]).sum(dim=1) + (u_S_src[user] * v_src[src_pos]).sum(dim=1))
        src_neg_score = ((uF_src_neg * v_src[src_neg]).sum(dim=1) + (u_S_src[user] * v_src[src_neg]).sum(dim=1))
        loss_bpr_src = -F.logsigmoid(src_pos_score - src_neg_score).mean()

        z_I_src = F.normalize(u_I_src[user], dim=1)
        z_T_src = F.normalize(uT_src_pos, dim=1)
        logits_src = torch.matmul(z_I_src, z_T_src.T) / self.config['tau_pcl']
        labels = torch.arange(logits_src.size(0), device=logits_src.device)
        loss_pcl_src = F.cross_entropy(logits_src, labels)

        uF_tgt_pos, uT_tgt_pos = self.personalized_transfer(
            domain='tgt',
            users=user,
            items=tgt_pos,
            u_I_dom=u_I_tgt, u_S_dom=u_S_tgt, v_dom=v_tgt,
            u_I_other=u_I_src
        )
        uF_tgt_neg, _ = self.personalized_transfer(
            domain='tgt',
            users=user,
            items=tgt_neg,
            u_I_dom=u_I_tgt, u_S_dom=u_S_tgt, v_dom=v_tgt,
            u_I_other=u_I_src
        )

        tgt_pos_score = ((uF_tgt_pos * v_tgt[tgt_pos]).sum(dim=1) + (u_S_tgt[user] * v_tgt[tgt_pos]).sum(dim=1))
        tgt_neg_score = ((uF_tgt_neg * v_tgt[tgt_neg]).sum(dim=1) + (u_S_tgt[user] * v_tgt[tgt_neg]).sum(dim=1))
        loss_bpr_tgt = -F.logsigmoid(tgt_pos_score - tgt_neg_score).mean()

        z_I_tgt = F.normalize(u_I_tgt[user], dim=1)
        z_T_tgt = F.normalize(uT_tgt_pos, dim=1)
        logits_tgt = torch.matmul(z_I_tgt, z_T_tgt.T) / self.config['tau_pcl']
        loss_pcl_tgt = F.cross_entropy(logits_tgt, labels)

        loss_bpr = loss_bpr_src + loss_bpr_tgt
        loss_pcl = loss_pcl_src + loss_pcl_tgt
        loss_ecl = loss_ecl_src + loss_ecl_tgt

        loss = loss_bpr + self.config['lamda_p'] * loss_pcl + self.config['lamda_e'] * loss_ecl
        return loss

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0] - 1  # [B]

        B = users.size(0)
        num_items = self.num_items_tgt

        u_I_src, u_S_src, v_src, _, _ = self.graph_forward(domain='src')
        u_I_tgt, u_S_tgt, v_tgt, _, _ = self.graph_forward(domain='tgt')

        all_items = torch.arange(num_items, device=self.device)

        users_exp = users.unsqueeze(1).expand(-1, num_items)
        items_exp = all_items.unsqueeze(0).expand(B, -1)

        users_flat = users_exp.reshape(-1)
        items_flat = items_exp.reshape(-1)

        uF_flat, _ = self.personalized_transfer(
            domain='tgt',
            users=users_flat,
            items=items_flat,
            u_I_dom=u_I_tgt,
            u_S_dom=u_S_tgt,
            v_dom=v_tgt,
            u_I_other=u_I_src
        )

        v_flat = v_tgt[items_flat]
        uS_flat = u_S_tgt[users_flat]

        scores_flat = (
                (uF_flat * v_flat).sum(dim=1)
                + (uS_flat * v_flat).sum(dim=1)
        )

        scores = scores_flat.view(B, num_items)
        padding = torch.zeros((B,1), device=self.device)
        scores = torch.concat((padding, scores), dim=1)
        return scores
