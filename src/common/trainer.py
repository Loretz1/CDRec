import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.enum_type import EvalDataLoaderState
from utils.utils import early_stopping, metrics_dict2str
from utils.topk_evaluator import TopKEvaluator

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, stage_id, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = None
        self.learning_rate = None
        self.epochs = None
        self.eval_step = config['eval_step']
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = None
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = None

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step_warm = 0
        self.cur_step_cold = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score_warm = -1
        self.best_valid_score_cold = -1
        self.best_valid_result_warm= tmp_dd.copy()
        self.best_valid_result_cold = tmp_dd.copy()
        self.best_test_upon_valid_warm = tmp_dd.copy()
        self.best_test_upon_valid_cold = tmp_dd.copy()
        self.train_loss_dict = dict()
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.lr_scheduler = None

        self.eval = None
        self.evaluator = TopKEvaluator(config)

    def _build_optimizer(self, params):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def _build_lr_scheduler(self):
        fac = lambda epoch: self.learning_rate_scheduler[0] ** (epoch / self.learning_rate_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction, epoch_idx)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)

            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            # if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result_warm, valid_result_cold = self.evaluate(valid_data)
        valid_score_warm = None if not valid_data.warm else valid_result_warm[self.valid_metric] if self.valid_metric else valid_result_warm['NDCG@20']
        valid_score_cold = None if not valid_data.cold else valid_result_cold[self.valid_metric] if self.valid_metric else valid_result_cold['NDCG@20']
        return valid_score_warm, valid_result_warm, valid_score_cold, valid_result_cold

    def _check_nan(self, loss):
        if torch.isnan(loss):
            # raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, stage_id, epoch_idx, s_time, e_time, losses):
        duration = e_time - s_time
        if isinstance(losses, tuple):
            loss_str = ' | '.join(f"Loss{i + 1}: {loss:.4f}" for i, loss in enumerate(losses))
        else:
            loss_str = f"Total Loss: {losses:.4f}"
        train_output = (
            f"\n{'=' * 30} [Stage {stage_id}, Epoch {epoch_idx}] Training Summary {'=' * 30}\n"
            f"⏱  Time used: {duration:.2f}s\n"
            f"📉 {loss_str}\n"
            f"{'=' * 90}"
        )
        return train_output

    def _generate_eval_output(self, epoch_idx, mode,
                              results_warm, results_cold,
                              score_warm=None, score_cold=None,
                              best_warm=None, best_cold=None,
                              update_warm=False, update_cold=False,
                              stop_warm=False, stop_cold=False,
                              elapsed_time=None):
        eval_output = ""

        header = f"\n{'=' * 30} [Epoch {epoch_idx}] {mode} Summary {'=' * 30}\n"
        eval_output += header

        time_str = f"⏱  Time used: {elapsed_time:.2f}s\n\n" if elapsed_time else ""
        eval_output += time_str

        if self.config['warm_eval']:
            warm_block = (
                    f"🔥 Warm-start Users:\n"
                    + (f"   {mode} Score: {score_warm:.6f} | Best: {best_warm:.6f} "
                       f"| {'✅ Updated' if update_warm else '❌ No Update'} "
                       f"| {'🛑 Early Stop' if stop_warm else ''}\n" if score_warm is not None else "")
                    + f"   Metrics:\n{metrics_dict2str(results_warm)}\n\n"
            )
            eval_output += warm_block

        if self.config['cold_start_eval']:
            cold_block = (
                    f"🎯 Cold-start Users:\n"
                    + (f"   {mode} Score: {score_cold:.6f} | Best: {best_cold:.6f} "
                       f"| {'✅ Updated' if update_cold else '❌ No Update'} "
                       f"| {'🛑 Early Stop' if stop_cold else ''}\n" if score_cold else "")
                    + f"   Metrics:\n{metrics_dict2str(results_cold)}\n"
            )
            eval_output += cold_block

        eval_output += f"{'=' * 90}"
        return eval_output

    def fit(self, stage_id, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            # for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[(stage_id, epoch_idx)] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(stage_id, epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if not self.eval:
                continue

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score_warm, valid_result_warm, valid_score_cold, valid_result_cold = self._valid_epoch(valid_data)

                self.best_valid_score_warm, self.cur_step_warm, stop_flag_warm, update_flag_warm = early_stopping(
                    valid_score_warm, self.best_valid_score_warm, self.cur_step_warm,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                self.best_valid_score_cold, self.cur_step_cold, stop_flag_cold, update_flag_cold = early_stopping(
                    valid_score_cold, self.best_valid_score_cold, self.cur_step_cold,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_output = self._generate_eval_output(epoch_idx, "Validation",
                                                          valid_result_warm, valid_result_cold,
                                                          valid_score_warm, valid_score_cold,
                                                          self.best_valid_score_warm, self.best_valid_score_cold,
                                                          update_flag_warm, update_flag_cold,
                                                          stop_flag_warm, stop_flag_cold,
                                                          valid_end_time - valid_start_time)
                # test
                _, test_result_warm, _, test_result_cold = self._valid_epoch(test_data)
                test_output = self._generate_eval_output(epoch_idx, "Test",
                                                         test_result_warm, test_result_cold)
                if verbose:
                    self.logger.info(valid_output)
                    self.logger.info(test_output)
                if update_flag_warm:
                    update_output_warm = f"██ {self.config['model']} -- 🌐 Warm-start validation result improved — best score updated!!!"
                    if verbose:
                        self.logger.info(update_output_warm)
                    self.best_valid_result_warm = valid_result_warm
                    self.best_test_upon_valid_warm = test_result_warm
                if update_flag_cold:
                    update_output_cold = f"██ {self.config['model']} -- 🎯 Cold-start validation result improved — best score updated!!!"
                    if verbose:
                        self.logger.info(update_output_cold)
                    self.best_valid_result_cold = valid_result_cold
                    self.best_test_upon_valid_cold = test_result_cold

                if stop_flag_warm and stop_flag_cold:
                    stop_msg = f"+++++ Finished training at epoch {epoch_idx}, best eval results:"
                    if verbose:
                        self.logger.info(stop_msg)
                        if self.config['warm_eval']:
                            stop_output_src = (
                                f"🛑 Early stopping triggered for 🌐 Warm-Start Evaluation "
                                f"(best epoch: {epoch_idx - self.cur_step_warm * self.eval_step})"
                            )
                            self.logger.info(stop_output_src)
                        if self.config['cold_start_eval']:
                            stop_output_tgt = (
                                f"🛑 Early stopping triggered for 🎯 Cold-Start Evaluation "
                                f"(best epoch: {epoch_idx - self.cur_step_cold * self.eval_step})"
                            )
                            self.logger.info(stop_output_tgt)
                    break

        if not self.eval:
            return (None, None, None, None, None, None)
        return (self.best_valid_score_warm, self.best_valid_result_warm, self.best_test_upon_valid_warm,
                self.best_valid_score_cold, self.best_valid_result_cold, self.best_test_upon_valid_cold)

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()
        result_warm = None
        result_cold = None

        # batch full users
        batch_matrix_list_warm, batch_matrix_list_cold = None, None
        # warm eval
        if eval_data.warm:
            batch_matrix_list_warm = []
            eval_data.set_state_for_eval(EvalDataLoaderState.WARM)
            for batch_idx, batched_data in enumerate(eval_data):
                # predict: interaction without item ids
                scores = self.model.full_sort_predict(batched_data, is_warm=True)
                masked_items = batched_data[1]
                # mask out pos items
                scores[masked_items[0], masked_items[1]] = -1e10
                # mask the item 0 which is PAD
                scores[:, 0] = -1e10
                # rank and get top-k
                _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
                batch_matrix_list_warm.append(topk_index)
            result_warm = self.evaluator.evaluate(batch_matrix_list_warm, eval_data, is_test=is_test, idx=idx,is_warm=True)
        # cold eval
        if eval_data.cold:
            batch_matrix_list_cold = []
            eval_data.set_state_for_eval(EvalDataLoaderState.COLD)
            for batch_idx, batched_data in enumerate(eval_data):
                # predict: interaction without item ids
                scores = self.model.full_sort_predict(batched_data, is_warm=False)
                masked_items = batched_data[1]
                # mask out pos items
                scores[masked_items[0], masked_items[1]] = -1e10
                # rank and get top-k
                _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
                batch_matrix_list_cold.append(topk_index)
            result_cold = self.evaluator.evaluate(batch_matrix_list_cold, eval_data, is_test=is_test, idx=idx,is_warm=False)

        return result_warm, result_cold

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def set_train_stage(self, stage_id, stage_config, eval = False):
        train_keys = ["epochs", "learner", "learning_rate", "learning_rate_scheduler", "weight_decay", "clip_grad_norm"]

        for key in train_keys:
            setattr(self, key, None)
            if key in stage_config:
                setattr(self, key, stage_config[key])

        self.model.set_train_stage(stage_id)
        self.optimizer = self._build_optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters()))
        self.lr_scheduler = self._build_lr_scheduler()
        self.eval = eval
