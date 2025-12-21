import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
import copy

class Disco(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(Disco, self).__init__(config, dataloader)
        self.config = config
        self.feature_dim = config["feature_dim"]
        self.num_intents = config["num_intents"]
        self.GNN = config["GNN"]
        self.dropout = config["dropout"]
        self.leaky = config["leaky"]
        self.momentum = config["momentum"]
        self.inter_batch_size = config['inter_batch_size']

        src_mat = dataloader.inter_matrix(domain=0, form='coo')
        tgt_mat = dataloader.inter_matrix(domain=1, form='coo')
        self.src_edge_u, self.src_edge_v = self._coo_to_edges(src_mat)
        self.src_deg_u = self._degree(self.src_edge_u, self.num_users_src)
        self.src_deg_v = self._degree(self.src_edge_v, self.num_items_src)
        self.tgt_edge_u, self.tgt_edge_v = self._coo_to_edges(tgt_mat)
        self.tgt_deg_u = self._degree(self.tgt_edge_u, self.num_users_tgt)
        self.tgt_deg_v = self._degree(self.tgt_edge_v, self.num_items_tgt)

        self.emb_user_src = nn.Embedding(self.num_users_src, self.feature_dim)
        self.emb_user_tgt = nn.Embedding(self.num_users_tgt, self.feature_dim)
        self.emb_item_src = nn.Embedding(self.num_items_src, self.feature_dim)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt, self.feature_dim)

        self.base_gnn_src = nn.ModuleList([
            BipartiteGraphConv(self.feature_dim, self.leaky, self.dropout)
            for _ in range(self.GNN)])
        self.base_gnn_tgt = nn.ModuleList([
            BipartiteGraphConv(self.feature_dim, self.leaky, self.dropout)
            for _ in range(self.GNN)])
        self.intent_gnn_src = nn.ModuleList([
            BipartiteGraphConv(self.feature_dim, self.leaky, self.dropout)
            for _ in range(self.num_intents)])
        self.intent_gnn_tgt = nn.ModuleList([
            BipartiteGraphConv(self.feature_dim, self.leaky, self.dropout)
            for _ in range(self.num_intents)])

        self.base_gnn_src_t = copy.deepcopy(self.base_gnn_src)
        self.base_gnn_tgt_t = copy.deepcopy(self.base_gnn_tgt)
        self.intent_gnn_src_t = copy.deepcopy(self.intent_gnn_src)
        self.intent_gnn_tgt_t = copy.deepcopy(self.intent_gnn_tgt)
        for p in self.base_gnn_src_t.parameters():
            p.requires_grad = False
        for p in self.base_gnn_tgt_t.parameters():
            p.requires_grad = False
        for p in self.intent_gnn_src_t.parameters():
            p.requires_grad = False
        for p in self.intent_gnn_tgt_t.parameters():
            p.requires_grad = False

        self.decoder_s2t = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.apply(xavier_uniform_initialization)

    def _coo_to_edges(self, mat):
        mat = mat.tocoo()
        edge_u = torch.from_numpy(mat.row).long()
        edge_v = torch.from_numpy(mat.col).long()
        return edge_u.to(self.device), edge_v.to(self.device)

    def _degree(self, edge_idx, num_nodes):
        deg = torch.zeros(num_nodes).to(self.device)
        deg.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))
        return deg

    @torch.no_grad()
    def _momentum_update(self):
        m = self.momentum

        def _ema_update(online, target):
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data = p_t.data * m + p_o.data * (1.0 - m)

        _ema_update(self.base_gnn_src, self.base_gnn_src_t)
        _ema_update(self.base_gnn_tgt, self.base_gnn_tgt_t)
        _ema_update(self.intent_gnn_src, self.intent_gnn_src_t)
        _ema_update(self.intent_gnn_tgt, self.intent_gnn_tgt_t)

    def post_batch_processing(self):
        self._momentum_update()

    def encode_intents(self, emb_user_src, emb_item_src, emb_user_tgt, emb_item_tgt, base_gnn_src, base_gnn_tgt,
                        intent_gnn_src, intent_gnn_tgt):
        U_src, V_src = emb_user_src, emb_item_src
        U_tgt, V_tgt = emb_user_tgt, emb_item_tgt

        for gnn in base_gnn_src:
            U_src, V_src = gnn(U_src, V_src, self.src_edge_u, self.src_edge_v, self.src_deg_u, self.src_deg_v)
        for gnn in base_gnn_tgt:
            U_tgt, V_tgt = gnn(U_tgt, V_tgt, self.tgt_edge_u, self.tgt_edge_v, self.tgt_deg_u, self.tgt_deg_v)

        Zk_u_src, Zk_v_src = [], []
        for gnn_k in intent_gnn_src:
            U_k, V_k = gnn_k(U_src, V_src, self.src_edge_u, self.src_edge_v, self.src_deg_u, self.src_deg_v)
            Zk_u_src.append(U_k)
            Zk_v_src.append(V_k)
        Zk_u_tgt, Zk_v_tgt = [], []
        for gnn_k in intent_gnn_tgt:
            U_k, V_k = gnn_k(U_tgt, V_tgt, self.tgt_edge_u, self.tgt_edge_v, self.tgt_deg_u, self.tgt_deg_v)
            Zk_u_tgt.append(U_k)
            Zk_v_tgt.append(V_k)
        Zk_u_src = torch.stack(Zk_u_src, dim=1)
        Zk_v_src = torch.stack(Zk_v_src, dim=1)
        Zk_u_tgt = torch.stack(Zk_u_tgt, dim=1)
        Zk_v_tgt = torch.stack(Zk_v_tgt, dim=1)
        return Zk_u_src, Zk_v_src, Zk_u_tgt, Zk_v_tgt

    def weighted_intent_score(self, u, v, p_ku):
        sim = torch.sum(u * v, dim=-1)  # [B, K]
        return torch.sum(p_ku * sim, dim=-1)  # [B]

    def row_normalize(self, mat, eps=1e-12):
        return mat / (mat.sum(dim=1, keepdim=True) + eps)

    def build_Tk_from_random_walk(self, Z_hat_k, tau, alpha, walk_steps):
        """
        Z_hat_k: [B, d]  target encoder 下 intent k 的用户表示（batch用户）
        return: Tk [B, B]
        """
        # (3) R_ij = exp(-||z_i - z_j|| / tau)
        # pairwise L2 distance
        dist = torch.cdist(Z_hat_k, Z_hat_k, p=2)  # [B, B]
        R = torch.exp(-dist / tau)  # [B, B]

        # row-wise normalize -> transition matrix
        R_tilde = self.row_normalize(R)

        # d-step random walk
        R_high = torch.linalg.matrix_power(R_tilde, walk_steps)  # [B, B]

        I = torch.eye(R_high.size(0), device=R_high.device, dtype=R_high.dtype)
        Tk = alpha * I + (1.0 - alpha) * R_high
        Tk = self.row_normalize(Tk)
        return Tk

    def rho_online_to_target(self, Z_k, Z_hat_k, tau, sim='dot'):
        """
        Z_k:     [B, d]  online intent k
        Z_hat_k: [B, d]  target intent k
        return: rho [B, B] row-normalized
        """
        if sim == 'cos':
            Z_k = F.normalize(Z_k, dim=1)
            Z_hat_k = F.normalize(Z_hat_k, dim=1)

        # phi(z_i, zhat_j): dot-product similarity
        logits = (Z_k @ Z_hat_k.t()) / tau  # [B, B]
        rho = F.softmax(logits, dim=1)
        return rho

    def intra_domain_loss(self, Z_users, Zhat_users, tau, alpha, walk_steps, sim='dot', eps=1e-12):
        """
        Z_users:    [B, K, d] online
        Zhat_users: [B, K, d] target (no_grad)
        return scalar loss_intra for this domain
        """
        B, K, d = Z_users.shape
        loss = 0.0
        for k in range(K):
            Z_k = Z_users[:, k, :]  # [B, d]
            Zhat_k = Zhat_users[:, k, :]  # [B, d]

            Tk = self.build_Tk_from_random_walk(Zhat_k, tau=tau, alpha=alpha, walk_steps=walk_steps)  # [B,B]
            rho = self.rho_online_to_target(Z_k, Zhat_k, tau=tau, sim=sim)  # [B,B]

            loss_k = -(Tk * torch.log(rho + eps)).sum(dim=1).mean()
            loss = loss + loss_k
        return loss

    def orthogonal_loss(self, Z, Zhat):
        """
        Z:    [B, K, d] online intent embeddings
        Zhat: [B, K, d] target intent embeddings (optional)
        """
        B, K, d = Z.shape

        def _orth(Z_):
            Z_ = F.normalize(Z_, dim=-1)  # [B, K, d]
            G = torch.bmm(Z_, Z_.transpose(1, 2))  # [B, K, K]  -> Z Z^T
            I = torch.eye(K, device=Z_.device).unsqueeze(0)  # [1, K, K]
            return ((G - I) ** 2).sum(dim=(1, 2)).mean()

        loss = _orth(Z)
        loss = loss + _orth(Zhat.detach())
        return loss

    def inter_domain_loss(
            self,
            e_s2t,  # [B, K, d]   e^{(s->t)}_{i,k}  (Eq.8)
            u_hat_tgt,  # [B, K, d]   z^t_{j,k} (target encoder, stop-grad)
            tau,
            alpha,
            walk_steps,
            sim='dot',
            eps=1e-12
    ):
        """
        Implement Eq.(9)-(15) with a variational EM style objective (ELBO).
        """

        B, K, d = e_s2t.shape

        # ---------- Build T^t by integrating all intents:  T^t = (1/K) Σ_k T_k^t ----------
        # Each T_k^t is built from target-domain embeddings (swap) via random walk (Eq.6 style)
        Tt = 0.0
        for k in range(K):
            Zhat_k = u_hat_tgt[:, k, :]  # [B, d]
            Tk = self.build_Tk_from_random_walk(Zhat_k, tau=tau, alpha=alpha, walk_steps=walk_steps)  # [B,B]
            Tt = Tt + Tk
        Tt = Tt / K
        Tt = self.row_normalize(Tt)  # ensure row-stochastic (safe)

        # ---------- Prior over intents p(k|u_i) (Eq.11) ----------
        # intent prototypes c_k: your code uses C = mean over batch, shape [K, d]
        C = e_s2t.mean(dim=0)  # [K, d]  prototypes {c_k}

        # phi(e_{i,k}, c_k): dot similarity; produce logits over k for each i
        prior_logits = torch.sum(e_s2t * C.unsqueeze(0), dim=-1)  # [B, K]
        p_ku = F.softmax(prior_logits, dim=1)  # p(k|u_i)

        # ---------- Likelihood under each intent: p_hat(u_j|u_i,k) in a mini-batch (Eq.15) ----------
        # For each k, compute softmax over j with similarity between e_{i,k} and z^t_{j,k}
        # We'll store log p_hat for numerical stability
        log_phat = []
        for k in range(K):
            Ei = e_s2t[:, k, :]  # [B, d]
            Zj = u_hat_tgt[:, k, :]  # [B, d]

            if sim == 'cos':
                Ei = F.normalize(Ei, dim=1)
                Zj = F.normalize(Zj, dim=1)

            logits_ij = (Ei @ Zj.t()) / tau  # [B, B]
            log_p = F.log_softmax(logits_ij, dim=1)  # log p_hat(u_j|u_i,k)
            log_phat.append(log_p)
        # [K, B, B]
        log_phat = torch.stack(log_phat, dim=0)

        # ---------- E-step: posterior q(k|u_j,u_i) (Eq.14) ----------
        # q ∝ p(k|u_i) * p_hat(u_j|u_i,k)
        # Work in log-space: log q ∝ log p(k|u_i) + log p_hat
        log_pku = torch.log(p_ku + eps)  # [B, K]
        log_pku = log_pku.transpose(0, 1)  # [K, B]  -> align with [K,B,B]
        log_pku = log_pku.unsqueeze(-1)  # [K, B, 1]

        log_unnorm_q = log_pku + log_phat  # [K, B, B]
        # normalize over k
        log_q = log_unnorm_q - torch.logsumexp(log_unnorm_q, dim=0, keepdim=True)  # [K,B,B]
        q = torch.exp(log_q)  # [K,B,B]

        # ---------- M-step (optimize ELBO) ----------
        # Eq.(13): log p(u_j|u_i) >= E_q[log p_hat(u_j|u_i,k)] - KL(q || p(k|u_i))
        # We minimize negative ELBO weighted by T^t_ij (Eq.9)
        # E_q term:
        Eq_logp = (q * log_phat).sum(dim=0)  # [B,B]

        # KL term: Σ_k q_kij (log q_kij - log p_ki)
        # log p_ki needs broadcast to [K,B,B]
        log_pku_full = log_pku.expand(-1, -1, B)  # [K,B,B]
        KL = (q * (log_q - log_pku_full)).sum(dim=0)  # [B,B]

        elbo = Eq_logp - KL  # [B,B]
        loss = -(Tt * elbo).sum(dim=1).mean()  # Eq.(9) weighting

        return loss

    def calculate_loss(self, interaction, epoch_idx):
        user_src = interaction['users_src'] - 1
        src_pos = interaction['pos_items_src'] - 1
        src_neg = interaction['neg_items_src'] - 1
        user_tgt = interaction['users_tgt'] - 1
        tgt_pos = interaction['pos_items_tgt'] - 1
        tgt_neg = interaction['neg_items_tgt'] - 1

        Zk_u_src, Zk_v_src, Zk_u_tgt, Zk_v_tgt = self.encode_intents(
            self.emb_user_src.weight, self.emb_item_src.weight, self.emb_user_tgt.weight, self.emb_item_tgt.weight,
            self.base_gnn_src, self.base_gnn_tgt, self.intent_gnn_src, self.intent_gnn_tgt)

        with torch.no_grad():
            Zk_hat_u_src, Zk_hat_v_src, Zk_hat_u_tgt, Zk_hat_v_tgt = self.encode_intents(
                self.emb_user_src.weight, self.emb_item_src.weight, self.emb_user_tgt.weight, self.emb_item_tgt.weight,
                self.base_gnn_src_t, self.base_gnn_tgt_t, self.intent_gnn_src_t, self.intent_gnn_tgt_t)

        # Calculate Loss_rec
        u_src = Zk_u_src[user_src]
        i_pos_src = Zk_v_src[src_pos]
        i_neg_src = Zk_v_src[src_neg]
        C_src = Zk_u_src.mean(dim=0)
        logits_src = torch.sum(u_src * C_src.unsqueeze(0), dim=-1)
        p_ku_src = F.softmax(logits_src, dim=1)
        pos_score_src = self.weighted_intent_score(u_src, i_pos_src, p_ku_src)
        neg_score_src = self.weighted_intent_score(u_src, i_neg_src, p_ku_src)
        loss_rec_src = -torch.log(torch.sigmoid(pos_score_src - neg_score_src) + 1e-8).mean()
        u_tgt = Zk_u_tgt[user_tgt]
        i_pos_tgt = Zk_v_tgt[tgt_pos]
        i_neg_tgt = Zk_v_tgt[tgt_neg]
        C_tgt = Zk_u_tgt.mean(dim=0)
        logits_tgt = torch.sum(u_tgt * C_tgt.unsqueeze(0), dim=-1)
        p_ku_tgt = F.softmax(logits_tgt, dim=1)
        pos_score_tgt = self.weighted_intent_score(u_tgt, i_pos_tgt, p_ku_tgt)
        neg_score_tgt = self.weighted_intent_score(u_tgt, i_neg_tgt, p_ku_tgt)
        loss_rec_tgt = -torch.log(torch.sigmoid(pos_score_tgt - neg_score_tgt) + 1e-8).mean()
        loss_rec = loss_rec_src + loss_rec_tgt

        # Calculate Loss_intra
        u_src = Zk_u_src[user_src]  # [B, K, d]
        u_hat_src = Zk_hat_u_src[user_src]  # [B, K, d]
        u_tgt = Zk_u_tgt[user_tgt]
        u_hat_tgt = Zk_hat_u_tgt[user_tgt]
        tau = self.config['tau']
        alpha = self.config['alpha']
        d = self.config['walk_steps']
        loss_intra_src = self.intra_domain_loss(u_src, u_hat_src, tau=tau, alpha=alpha, walk_steps=d, sim='dot')
        loss_intra_tgt = self.intra_domain_loss(u_tgt, u_hat_tgt, tau=tau, alpha=alpha, walk_steps=d, sim='dot')
        loss_intra = loss_intra_src + loss_intra_tgt

        # Calculate Loss_orth
        loss_orth_src = self.orthogonal_loss(u_src, u_hat_src)
        loss_orth_tgt = self.orthogonal_loss(u_tgt, u_hat_tgt)
        loss_orth = loss_orth_src + loss_orth_tgt

        # Calculate Loss_inter
        perm = torch.randperm(self.num_users_overlap, device=Zk_u_src.device)
        overlap_users = perm[:self.inter_batch_size]
        u_src_ol = Zk_u_src[overlap_users]  # [Bo, K, d]
        u_hat_tgt_ol = Zk_hat_u_tgt[overlap_users]  # [Bo, K, d]
        e_s2t = self.decoder_s2t(u_src_ol)  # [Bo, K, d]
        loss_inter = self.inter_domain_loss(
            e_s2t=e_s2t,
            u_hat_tgt=u_hat_tgt_ol,
            tau=tau,
            alpha=alpha,
            walk_steps=d,
            sim='dot'
        )

        loss_contra = (self.config['beta'] * loss_inter +
                       (1 - self.config['beta']) * (loss_intra + self.config['gamma'] * loss_orth))
        loss = self.config['lamda'] * loss_contra + (1 - self.config['lamda']) * loss_rec
        return loss

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0] - 1

        Zk_u_src, _, _, Zk_v_tgt = self.encode_intents(
            self.emb_user_src.weight,
            self.emb_item_src.weight,
            self.emb_user_tgt.weight,
            self.emb_item_tgt.weight,
            self.base_gnn_src,
            self.base_gnn_tgt,
            self.intent_gnn_src,
            self.intent_gnn_tgt
        )
        u_src = Zk_u_src[users]  # [B, K, d]
        e_s2t = self.decoder_s2t(u_src)  # [B, K, d]

        C = e_s2t.mean(dim=0)  # [K, d]
        p_ku = F.softmax((e_s2t * C).sum(-1), dim=1)  # [B,K]

        chunk = 512
        scores_all = []
        for s in range(0, self.num_items_tgt, chunk):
            v = Zk_v_tgt[s:s + chunk]  # [c,K,d]
            sim = torch.einsum('bkd,ikd->bik', e_s2t, v)  # [B,c,K]
            score = (sim * p_ku.unsqueeze(1)).sum(-1)  # [B,c]
            scores_all.append(score)

        scores = torch.cat(scores_all, dim=1)  # [B,I]
        pad = torch.zeros(users.size(0), 1, device=self.device, dtype=scores.dtype)
        scores = torch.cat([pad, scores], dim=1)  # [B, I+1]
        return scores


class BipartiteGraphConv(nn.Module):
    def __init__(self, dim, leaky=0.1, dropout=0.2):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.act = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, U, V, edge_u, edge_v, deg_u, deg_v):
        u_e = U[edge_u]
        v_e = V[edge_v]

        m_uv = self.W1(v_e) + self.W2(v_e * u_e)
        norm_uv = 1.0 / torch.sqrt(deg_u[edge_u] * deg_v[edge_v] + 1e-8)
        m_uv = m_uv * norm_uv.unsqueeze(-1)
        U_msg = torch.zeros_like(U)
        U_msg.index_add_(0, edge_u, m_uv)

        m_vu = self.W1(u_e) + self.W2(u_e * v_e)
        norm_vu = 1.0 / torch.sqrt(deg_v[edge_v] * deg_u[edge_u] + 1e-8)
        m_vu = m_vu * norm_vu.unsqueeze(-1)
        V_msg = torch.zeros_like(V)
        V_msg.index_add_(0, edge_v, m_vu)

        U_new = self.W1(U) + U_msg
        V_new = self.W1(V) + V_msg
        U_new = self.act(U_new)
        V_new = self.act(V_new)
        if self.dropout is not None:
            U_new = self.dropout(U_new)
            V_new = self.dropout(V_new)
        return U_new, V_new


