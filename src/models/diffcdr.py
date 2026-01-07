import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

class DiffCDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(DiffCDR, self).__init__(config, dataloader)
        self.config = config
        self.emb_dim = config.get("emb_dim", 64)
        self.bpr_loss = BPRLoss()

        self.source_user_embedding = nn.Embedding(self.num_users_src + 1, self.emb_dim, padding_idx=0)
        self.source_item_embedding = nn.Embedding(self.num_items_src + 1, self.emb_dim, padding_idx=0)
        self.target_user_embedding = nn.Embedding(self.num_users_tgt + 1, self.emb_dim, padding_idx=0)
        self.target_item_embedding = nn.Embedding(self.num_items_tgt + 1, self.emb_dim, padding_idx=0)

        # without_diffusion
        self.diff_model = DiffModel(config['diff_steps'], config['diff_dim'], config['emb_dim'], config['diff_scale'], config['diff_sample_steps'],
                                  config['diff_task_lambda'], config['diff_mask_rate'])
        # self.mapper = MLPMapper(self.emb_dim)
        # self.map_align_loss = config['map_align_loss']
        # if self.map_align_loss in ["l1", "smooth_l1"]:
        #     self.align_loss_fn = F.smooth_l1_loss
        # elif self.map_align_loss in ["l2", "mse"]:
        #     self.align_loss_fn = F.mse_loss
        # else:
        #     raise ValueError(f"Unknown map_align_loss: {self.map_align_loss}")

        self.apply(xavier_uniform_initialization)

    def _bpr(self, u, p, n):
        pos = (u * p).sum(dim=-1)
        neg = (u * n).sum(dim=-1)
        return self.bpr_loss(pos, neg)

    @staticmethod
    def _set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def _source_loss(self, inter):
        user = self.source_user_embedding(inter["users"])
        pos = self.source_item_embedding(inter["pos_items"])
        neg = self.source_item_embedding(inter["neg_items"])
        return self._bpr(user, pos, neg)

    def _target_loss(self, inter):
        user = self.target_user_embedding(inter["users"])
        pos = self.target_item_embedding(inter["pos_items"])
        neg = self.target_item_embedding(inter["neg_items"])
        return self._bpr(user, pos, neg)

    def _mapping_loss(self, inter):
        users = inter['users']
        pos_tgt = inter['pos_items_tgt']
        neg_tgt = inter['neg_items_tgt']

        # without_diffusion
        # src_emb = self.source_user_embedding(users)
        # tgt_emb = self.target_user_embedding(users).detach()
        # pred_tgt = self.mapper(src_emb)
        # loss_align = self.align_loss_fn(pred_tgt, tgt_emb)
        # pos_item = self.target_item_embedding(pos_tgt)
        # neg_item = self.target_item_embedding(neg_tgt)
        # pos_score = (pred_tgt * pos_item).sum(dim=-1)
        # neg_score = (pred_tgt * neg_item).sum(dim=-1)
        # loss_bpr = self.bpr_loss(pos_score, neg_score)
        # return loss_align + self.config['map_bpr_lambda'] * loss_bpr
        # return loss_bpr

        # self.optimizer.zero_grad()
        tgt_emb = self.target_user_embedding(users)
        cond_emb = self.source_user_embedding(users)
        iid_pos_emb = self.target_item_embedding(pos_tgt)
        iid_neg_emb = self.target_item_embedding(neg_tgt)
        loss_diff = diffusion_loss_fn(self.diff_model, tgt_emb, cond_emb,iid_pos_emb, iid_neg_emb, self.device, is_task=False)
        # loss_diff.backward()
        # if self.config['training_stages'][2]['clip_grad_norm']:
        #     clip_grad_norm_(self.parameters(), **self.config['training_stages'][2]['clip_grad_norm'])
        # self.optimizer.step()

        # tgt_emb = self.target_user_embedding(users)
        # cond_emb = self.source_user_embedding(users)
        # iid_pos_emb = self.target_item_embedding(pos_tgt)
        # iid_neg_emb = self.target_item_embedding(neg_tgt)
        loss_task = diffusion_loss_fn(self.diff_model, tgt_emb, cond_emb,iid_pos_emb, iid_neg_emb, self.device, is_task=True)
        # return loss_task

        loss = loss_diff + loss_task
        return loss

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self._source_loss(interaction)
        elif self.stage_id == 1:
            return self._target_loss(interaction)
        elif self.stage_id == 2:
            return self._mapping_loss(interaction)

    def full_sort_predict(self, interaction, is_warm):
        assert is_warm == False
        user = interaction[0].long()
        all_items = self.target_item_embedding.weight

        # without_diffusion
        # src_emb = self.source_user_embedding(user)
        # trans_emb = self.mapper(src_emb)
        # scores = torch.matmul(trans_emb, all_items.T)
        # scores[:, 0] = -1e9
        # return scores

        cond_emb = self.source_user_embedding(user)
        trans_emb = p_sample_loop(self.diff_model, cond_emb, self.device)

        scores = torch.matmul(trans_emb, all_items.T)
        scores[:, 0] = -1e9
        return scores

    def set_train_stage(self, stage_id):
        super(DiffCDR, self).set_train_stage(stage_id)

        # freeze all first
        for m in [
            self.source_user_embedding, self.source_item_embedding,
            self.target_user_embedding, self.target_item_embedding,
            # without_diffusion
            self.diff_model]:
            # self.mapper]:
            # without_diffusion
            self._set_requires_grad(m, False)

        if stage_id == 0:  # source
            self._set_requires_grad(self.source_user_embedding, True)
            self._set_requires_grad(self.source_item_embedding, True)
        elif stage_id == 1:  # target
            self._set_requires_grad(self.target_user_embedding, True)
            self._set_requires_grad(self.target_item_embedding, True)
        elif stage_id == 2:  # mapping
            # without_diffusion
            # self._set_requires_grad(self.mapper, True)
            self._set_requires_grad(self.diff_model, True)
            # without_diffusion
            # two_step_optimizing
            # self.optimizer = self._build_optimizer(params=filter(lambda p: p.requires_grad, self.parameters()))
            # two_step_optimizing

    # def _build_optimizer(self, params):
    #     learner = self.config['training_stages'][2]['learner']
    #     lr = self.config['training_stages'][2]['learning_rate']
    #     weight_decay = self.config['training_stages'][2]['weight_decay']
    #     if learner.lower() == 'adam':
    #         optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    #     elif learner.lower() == 'sgd':
    #         optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
    #     elif learner.lower() == 'adagrad':
    #         optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    #     elif learner.lower() == 'rmsprop':
    #         optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    #     else:
    #         self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
    #         optimizer = optim.Adam(params, lr=lr)
    #     return optimizer

class DiffModel(nn.Module):
    def __init__(self, num_steps=200, diff_dim=32, input_dim=32, c_scale=0.1, diff_sample_steps=30,
                 diff_task_lambda=0.1, diff_mask_rate=0.1):
        super(DiffModel, self).__init__()

        # -------------------------------------------
        # define params
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape == self.alphas_prod.shape == self.alphas_prod_p.shape == \
               self.alphas_bar_sqrt.shape == self.one_minus_alphas_bar_log.shape \
               == self.one_minus_alphas_bar_sqrt.shape

        # -----------------------------------------------
        self.diff_dim = diff_dim
        self.input_dim = input_dim
        self.task_lambda = diff_task_lambda
        self.sample_steps = diff_sample_steps
        self.c_scale = c_scale
        self.mask_rate = diff_mask_rate
        # -----------------------------------------------

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, diff_dim),
                nn.Linear(diff_dim, diff_dim),
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.step_emb_linear = nn.ModuleList(
            [
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.cond_emb_linear = nn.ModuleList(
            [
                nn.Linear(input_dim, input_dim),
            ]
        )

        self.num_layers = 1

        # linear for alm
        self.al_linear = nn.Linear(input_dim, input_dim, False)

    def forward(self, x, t, cond_emb, cond_mask):
        for idx in range(self.num_layers):
            t_embedding = get_timestep_embedding(t, self.diff_dim)
            t_embedding = self.step_emb_linear[idx](t_embedding)

            cond_embedding = self.cond_emb_linear[idx](cond_emb)

            t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            x = x + t_c_emb
            # x= torch.cat([t_embedding,cond_embedding * cond_mask.unsqueeze(-1),x],axis=1)

            x = self.linears[0](x)
            x = self.linears[1](x)
            x = self.linears[2](x)

        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)

def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps = timesteps.to(dtype=torch.float32)

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
    #emb = tf.cast(timesteps, dtype=torch.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    #if embedding_dim % 2 == 1:  # zero pad
    #    emb = torch.pad(emb, [0,1])
    assert emb.shape == torch.Size([timesteps.shape[0], embedding_dim])
    return emb

def q_x_fn(model, x_0, t, device):
    # eq(4)
    noise = torch.normal(0, 1, size=x_0.size(), device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise), noise


def diffusion_loss_fn(model, x_0, cond_emb, iid_pos_emb, iid_neg_emb, device, is_task):
    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:

        # ------------------------
        # sampling
        # ------------------------
        batch_size = x_0.shape[0]
        # sample t
        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)

        x, e = q_x_fn(model, x_0, t, device)

        # random mask
        cond_mask = 1 * (torch.rand(cond_emb.shape[0], device=device) <= mask_rate)
        cond_mask = 1 - cond_mask.int()

        # pred noise
        output = model(x, t.squeeze(-1), cond_emb, cond_mask)

        return F.smooth_l1_loss(e, output)

    elif is_task:
        final_output = p_sample_loop(model, cond_emb, device)
        pos_pred = torch.sum(final_output * iid_pos_emb, dim=1)
        neg_pred = torch.sum(final_output * iid_neg_emb, dim=1)
        task_loss = BPRLoss()(pos_pred, neg_pred)
        return F.smooth_l1_loss(x_0, final_output) + model.task_lambda * task_loss

# generation fun
def p_sample(model, cond_emb, x, device):
    # wrap for dpm_solver
    classifier_scale_para = model.c_scale
    dmp_sample_steps = model.sample_steps
    num_steps = model.num_steps

    model_kwargs = {'cond_emb': cond_emb,
                    'cond_mask': torch.zeros(cond_emb.size()[0], device=device),
                    }
    noise_schedule = NoiseScheduleVP(schedule='linear')
    model_fn = model_wrapper(
        model,
        noise_schedule,
        is_cond_classifier=True,
        classifier_scale=classifier_scale_para,
        time_input_type="1",
        total_N=num_steps,
        model_kwargs=model_kwargs
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule)

    sample = dpm_solver.sample(
        x,
        steps=dmp_sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True,
    )

    return model.get_al_emb(sample).to(device)


def p_sample_loop(model, cond_emb, device):
    # source emb input
    cur_x = cond_emb
    # noise input
    # cur_x = torch.normal(0,1,size = cond_emb.size() ,device=device)

    # reversing
    cur_x = p_sample(model, cond_emb, cur_x, device)

    return cur_x


class NoiseScheduleVP:
    def __init__(self, schedule='linear'):
        """Create a wrapper class for the forward SDE (VP type).

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
        schedule are the default settings in DDPM and improved-DDPM:

            beta_min: A `float` number. The smallest beta for the linear schedule.
            beta_max: A `float` number. The largest beta for the linear schedule.
            cosine_s: A `float` number. The hyperparameter in the cosine schedule.
            cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
            T: A `float` number. The ending time of the forward process.

        Note that the original DDPM (linear schedule) used the discrete-time label (0 to 999). We convert the discrete-time
        label to the continuous-time time (followed Song et al., 2021), so the beta here is 1000x larger than those in DDPM.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE ('linear' or 'cosine').

        Returns:
            A wrapper object of the forward SDE (VP type).
        """
        if schedule not in ['linear', 'cosine']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'linear' or 'cosine'".format(schedule))
        self.beta_0 = 0.1
        self.beta_1 = 20
        self.cosine_s = 0.008
        self.cosine_beta_max = 999.
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        self.schedule = schedule
        if schedule == 'cosine':
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = 0.9946
        else:
            self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t
        else:
            raise ValueError("Unsupported ")

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


def model_wrapper(model, noise_schedule=None, is_cond_classifier=False, classifier_fn=None, classifier_scale=1.,
                  time_input_type='1', total_N=1000, model_kwargs={}):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a function that accepts the continuous time as the input.

    The input `model` has the following format:

    ``
        model(x, t_input, **model_kwargs) -> noise
    ``

    where `x` and `noise` have the same shape, and `t_input` is the time label of the model.
    (may be discrete-time labels (i.e. 0 to 999) or continuous-time labels (i.e. epsilon to T).)

    We wrap the model function to the following format:

    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return model(x, t_input, **model_kwargs)
    ``

    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    For DPMs with classifier guidance, we also combine the model output with the classifier gradient as used in [1].

    [1] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis," in Advances in Neural
    Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

    ===============================================================

    Args:
        model: A noise prediction model with the following format:
            ``
                def model(x, t_input, **model_kwargs):
                    return noise
            ``
        noise_schedule: A noise schedule object, such as NoiseScheduleVP. Only used for the classifier guidance.
        is_cond_classifier: A `bool`. Whether to use the classifier guidance.
        classifier_fn: A classifier function. Only used for the classifier guidance. The format is:
            ``
                def classifier_fn(x, t_input):
                    return logits
            ``
        classifier_scale: A `float`. The scale for the classifier guidance.
        time_input_type: A `str`. The type for the time input of the model. We support three types:
            - '0': The continuous-time type. In this case, the model is trained on the continuous time,
                so `t_input` = `t_continuous`.
            - '1': The Type-1 discrete type described in the Appendix of DPM-Solver paper.
                **For discrete-time DPMs, we recommend to use this type for DPM-Solver**.
            - '2': The Type-2 discrete type described in the Appendix of DPM-Solver paper.
        total_N: A `int`. The total number of the discrete-time DPMs (default is 1000), used when `time_input_type`
            is '1' or '2'.
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
    Returns:
        A function that accepts the continuous time as the input, with the following format:
            ``
                def model_fn(x, t_continuous):
                    t_input = get_model_input_time(t_continuous)
                    return model(x, t_input, **model_kwargs)
            ``
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        """
        if time_input_type == '0':
            # discrete_type == '0' means that the model is continuous-time model.
            # For continuous-time DPMs, the continuous time equals to the discrete time.
            return t_continuous
        elif time_input_type == '1':
            # Type-1 discrete label, as detailed in the Appendix of DPM-Solver.
            return 1000. * torch.max(t_continuous - 1. / total_N, torch.zeros_like(t_continuous).to(t_continuous))
        elif time_input_type == '2':
            # Type-2 discrete label, as detailed in the Appendix of DPM-Solver.
            max_N = (total_N - 1) / total_N * 1000.
            return max_N * t_continuous
        else:
            raise ValueError("Unsupported time input type {}, must be '0' or '1' or '2'".format(time_input_type))

    # def cond_fn(x, t_discrete, y):
    #    """
    #    Compute the gradient of the classifier, multiplied with the sclae of the classifier guidance.
    #    """
    #    assert y is not None
    #    with torch.enable_grad():
    #        x_in = x.detach().requires_grad_(True)
    #        logits = classifier_fn(x_in, t_discrete)
    #        log_probs = F.log_softmax(logits, dim=-1)
    #        selected = log_probs[range(len(logits)), y.view(-1)]
    #        return classifier_scale * torch.autograd.grad(selected.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if is_cond_classifier:
            # y = model_kwargs.get("y", None)
            # if y is None:
            #    raise ValueError("For classifier guidance, the label y has to be in the input.")
            # t_discrete = get_model_input_time(t_continuous)
            # noise_uncond = model(x, t_discrete, **model_kwargs)
            # cond_grad = cond_fn(x, t_discrete, y)
            # sigma_t = noise_schedule.marginal_std(t_continuous)
            # dims = len(cond_grad.shape) - 1
            # return noise_uncond - sigma_t[(...,) + (None,)*dims] * cond_grad

            # classifier free

            cond_emb = model_kwargs.get("cond_emb", None)
            cond_mask = model_kwargs.get("cond_mask", None)

            t_discrete = get_model_input_time(t_continuous)
            noise_uncond = model(x, t_discrete, cond_emb, cond_mask)

            cond_mask_c = torch.ones_like(cond_mask, device=cond_mask.device)
            cond_grad = model(x, t_discrete, cond_emb, cond_mask_c)

            # sigma_t = noise_schedule.marginal_std(t_continuous)
            # dims = len(cond_grad.shape) - 1
            # sigma_t
            # return noise_uncond + sigma_t[(...,) + (None,)*dims] * classifier_scale * (cond_grad - noise_uncond  )
            # no sigma_t , compatible with paper
            return noise_uncond + classifier_scale * (cond_grad - noise_uncond)

        else:
            t_discrete = get_model_input_time(t_continuous)
            return model(x, t_discrete, **model_kwargs)

    return model_fn


class DPM_Solver:
    def __init__(self, model_fn, noise_schedule):
        """Construct a DPM-Solver.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input
                (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        """
        self.model_fn = model_fn
        self.noise_schedule = noise_schedule

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t = torch.linspace(t_0, t_T, 10000000).to(device)
            quadratic_t = torch.sqrt(t)
            quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1).to(device)
            return torch.flip(
                torch.cat([t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]], t_T * torch.ones((1,)).to(device)],
                          dim=0), dims=[0])
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_time_steps_for_dpm_solver_fast(self, t_T, t_0, steps, device):
        """
        Compute the intermediate time steps and the order of each step for sampling by DPM-Solver-fast.

        We recommend DPM-Solver-fast for fast sampling of DPMs. Given a fixed number of function evaluations by `steps`,
        the sampling procedure by DPM-Solver-fast is:
            - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
            - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
            - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            steps: A `int`. The total number of function evaluations (NFE).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
            timesteps: A pytorch tensor of the time steps, with the shape of (K + 1,).
        """
        K = steps // 3 + 1
        if steps % 3 == 0:
            orders = [3, ] * (K - 2) + [2, 1]
        elif steps % 3 == 1:
            orders = [3, ] * (K - 1) + [1]
        else:
            orders = [3, ] * (K - 1) + [2]
        timesteps = self.get_time_steps('logSNR', t_T, t_0, K, device)
        return orders, timesteps

    def dpm_solver_first_update(self, x, s, t, return_noise=False):
        """
        A single step for DPM-Solver-1.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)

        phi_1 = torch.expm1(h)

        noise_s = self.model_fn(x, s)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
        )
        if return_noise:
            return x_t, {'noise_s': noise_s}
        else:
            return x_t

    def dpm_solver_second_update(self, x, s, t, r1=0.5, noise_s=None, return_noise=False):
        """
        A single step for DPM-Solver-2.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver. We recommend the default setting `0.5`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            return_noise: A `bool`. If true, also return the predicted noise at time `s` and `s1` (the intermediate time).
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(
            s1), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
        )
        noise_s1 = self.model_fn(x_s1, s1)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                - (0.5 / r1) * (sigma_t * phi_1)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
        )
        if return_noise:
            return x_t, {'noise_s': noise_s, 'noise_s1': noise_s1}
        else:
            return x_t

    def dpm_solver_third_update(self, x, s, t, r1=1. / 3., r2=2. / 3., noise_s=None, noise_s1=None, noise_s2=None):
        """
        A single step for DPM-Solver-3.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `1 / 3`.
            r2: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `2 / 3`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            noise_s1: A pytorch tensor. The predicted noise at time `s1` (the intermediate time given by `r1`).
                If `noise_s1` is None, we compute the predicted noise by `s1`; otherwise we directly use it.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
        phi_2 = torch.expm1(h) / h - 1.

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        if noise_s1 is None:
            x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
            )
            noise_s1 = self.model_fn(x_s1, s1)
        if noise_s2 is None:
            x_s2 = (
                    torch.exp(log_alpha_s2 - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_s2 * phi_12)[(...,) + (None,) * dims] * noise_s
                    - r2 / r1 * (sigma_s2 * phi_22)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
            )
            noise_s2 = self.model_fn(x_s2, s2)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                - (1. / r2) * (sigma_t * phi_2)[(...,) + (None,) * dims] * (noise_s2 - noise_s)
        )
        return x_t

    def dpm_solver_update(self, x, s, t, order):
        """
        A single step for DPM-Solver of the given order `order`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t)
        elif order == 2:
            return self.dpm_solver_second_update(x, s, t)
        elif order == 3:
            return self.dpm_solver_third_update(x, s, t)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5):
        """
        The adaptive step size solver based on DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((x.shape[0],)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_noise=True)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_second_update(x, s, t, r1=r1, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.dpm_solver_second_update(x, s, t, r1=r1, return_noise=True)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_third_update(x, s, t, r1=r1, r2=r2, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def sample(self, x, steps=10, eps=1e-4, T=None, order=3, skip_type='logSNR',
               adaptive_step_size=False, fast_version=True, atol=0.0078, rtol=0.05,
               ):
        """
        Compute the sample at time `eps` by DPM-Solver, given the initial `x` at time `T`.

        We support the following algorithms:

            - Adaptive step size DPM-Solver (i.e. DPM-Solver-12 and DPM-Solver-23)

            - Fixed order DPM-Solver (i.e. DPM-Solver-1, DPM-Solver-2 and DPM-Solver-3).

            - Fast version of DPM-Solver (i.e. DPM-Solver-fast), which uses uniform logSNR steps and combine
                different orders of DPM-Solver.

        **We recommend DPM-Solver-fast for both fast sampling in few steps (<=20) and fast convergence in many steps (50 to 100).**

        Choosing the algorithms:

            - If `adaptive_step_size` is True:
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                If `order`=2, we use DPM-Solver-12 which combines DPM-Solver-1 and DPM-Solver-2.
                If `order`=3, we use DPM-Solver-23 which combines DPM-Solver-2 and DPM-Solver-3.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.

            - If `adaptive_step_size` is False and `fast_version` is True:
                We ignore `order` and use DPM-Solver-fast with number of function evaluations (NFE) = `steps`.
                We ignore `skip_type` and use uniform logSNR steps for DPM-Solver-fast.
                Given a fixed NFE=`steps`, the sampling procedure by DPM-Solver-fast is:
                    - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                    - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

            - If `adaptive_step_size` is False and `fast_version` is False:
                We use DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
                We support three types of `skip_type`:
                    - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                    - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                    - 'time_quadratic': quadratic time for the time steps. (Used in DDIM.)

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `T` (a sample from the normal distribution).
            steps: A `int`. The total number of function evaluations (NFE).
            eps: A `float`. The ending time of the sampling.
                We recommend `eps`=1e-3 when `steps` <= 15; and `eps`=1e-4 when `steps` > 15.
            T: A `float`. The starting time of the sampling. Default is `None`.
                If `T` is None, we use self.noise_schedule.T.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. Default is 'logSNR'.
            adaptive_step_size: A `bool`. If true, use the adaptive step size DPM-Solver.
            fast_version: A `bool`. If true, use DPM-Solver-fast (recommended).
            atol: A `float`. The absolute tolerance of the adaptive step size solver.
            rtol: A `float`. The relative tolerance of the adaptive step size solver.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        t_0 = eps
        t_T = self.noise_schedule.T if T is None else T
        device = x.device
        if adaptive_step_size:
            with torch.no_grad():
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol)
        else:
            if fast_version:
                orders, timesteps = self.get_time_steps_for_dpm_solver_fast(t_T=t_T, t_0=t_0, steps=steps,
                                                                            device=device)
            else:
                N_steps = steps // order
                orders = [order, ] * N_steps
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=N_steps, device=device)
            with torch.no_grad():
                for i, order in enumerate(orders):
                    vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(
                        device) * timesteps[i + 1]
                    x = self.dpm_solver_update(x, vec_s, vec_t, order)
        return x


class MLPMapper(nn.Module):
    def __init__(self, emb_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        in_dim = emb_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, emb_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, src_user_emb):
        return self.mlp(src_user_emb)