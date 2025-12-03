import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

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

    def fit(self, train_data):
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
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result= tmp_dd.copy()
        self.best_test_upon_valid = tmp_dd.copy()
        self.train_loss_dict = dict()
        self.optimizer = None

        # fac = lambda epoch: 0.96 ** (epoch / 50)
        # lr_scheduler = config['learning_rate_scheduler']  # check zero?
        # fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        # scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        # self.lr_scheduler = scheduler
        self.lr_scheduler = None

        self.evaluator = TopKEvaluator(config)

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

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
            second_inter = interaction.clone()
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
        valid_result_src, valid_result_tgt = self.evaluate(valid_data)
        valid_score_src = valid_result_src[self.valid_metric] if self.valid_metric else valid_result_src['NDCG@20']
        valid_score_tgt = valid_result_tgt[self.valid_metric] if self.valid_metric else valid_result_tgt['NDCG@20']
        return valid_score_src, valid_result_src, valid_score_tgt, valid_result_tgt

    def _check_nan(self, loss):
        if torch.isnan(loss):
            # raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        duration = e_time - s_time
        if isinstance(losses, tuple):
            loss_str = ' | '.join(f"Loss{i + 1}: {loss:.4f}" for i, loss in enumerate(losses))
        else:
            loss_str = f"Total Loss: {losses:.4f}"
        train_output = (
            f"\n{'=' * 30} [Epoch {epoch_idx}] Training Summary {'=' * 30}\n"
            f"⏱  Time used: {duration:.2f}s\n"
            f"📉 {loss_str}\n"
            f"{'=' * 90}"
        )
        return train_output

    def _generate_eval_output(self, epoch_idx, mode, results_src, results_tgt,
                              score_src=None, score_tgt=None,
                              best_src=None, best_tgt=None,
                              update_src=False, update_tgt=False,
                              stop_src=False, stop_tgt=False,
                              elapsed_time=None):
        header = f"\n{'=' * 30} [Epoch {epoch_idx}] {mode} Summary {'=' * 30}\n"
        time_str = f"⏱  Time used: {elapsed_time:.2f}s\n\n" if elapsed_time else ""
        src_block = (
                f"🌐 Source Domain:\n"
                + (f"   {mode} Score: {score_src:.6f} | Best: {best_src:.6f} "
                   f"| {'✅ Updated' if update_src else '❌ No Update'} "
                   f"| {'🛑 Early Stop' if stop_src else ''}\n" if score_src else "")
                + f"   Metrics:\n{metrics_dict2str(results_src)}\n\n"
        )
        tgt_block = (
                f"🎯 Target Domain:\n"
                + (f"   {mode} Score: {score_tgt:.6f} | Best: {best_tgt:.6f} "
                   f"| {'✅ Updated' if update_tgt else '❌ No Update'} "
                   f"| {'🛑 Early Stop' if stop_tgt else ''}\n" if score_tgt else "")
                + f"   Metrics:\n{metrics_dict2str(results_tgt)}\n"
        )
        return header + time_str + src_block + tgt_block + f"{'=' * 90}"

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
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

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score_src, valid_result_src, valid_score_tgt, valid_result_tgt = self._valid_epoch(valid_data)

                self.best_valid_score_src, self.cur_step_src, stop_flag_src, update_flag_src = early_stopping(
                    valid_score_src, self.best_valid_score_src, self.cur_step_src,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                self.best_valid_score_tgt, self.cur_step_tgt, stop_flag_tgt, update_flag_tgt = early_stopping(
                    valid_score_tgt, self.best_valid_score_tgt, self.cur_step_tgt,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_output = self._generate_eval_output(epoch_idx, "Validation",
                                                          valid_result_src, valid_result_tgt,
                                                          valid_score_src, valid_score_tgt,
                                                          self.best_valid_score_src, self.best_valid_score_tgt,
                                                          update_flag_src, update_flag_tgt,
                                                          stop_flag_src, stop_flag_tgt,
                                                          valid_end_time - valid_start_time)
                # test
                _, test_result_src, _, test_result_tgt = self._valid_epoch(test_data)
                test_output = self._generate_eval_output(epoch_idx, "Test",
                                                         test_result_src, test_result_tgt)
                if verbose:
                    self.logger.info(valid_output)
                    self.logger.info(test_output)
                if update_flag_src:
                    update_output_src = f"██ {self.config['model']} -- 🌐 Source Domain best validation results updated!!!"
                    if verbose:
                        self.logger.info(update_output_src)
                    self.best_valid_result_src = valid_result_src
                    self.best_test_upon_valid_src = test_result_src
                if update_flag_tgt:
                    update_output_tgt = f"██ {self.config['model']} -- 🎯 Target Domain best validation results updated!!!"
                    if verbose:
                        self.logger.info(update_output_tgt)
                    self.best_valid_result_tgt = valid_result_tgt
                    self.best_test_upon_valid_tgt = test_result_tgt

                if stop_flag_src and stop_flag_tgt:
                    stop_msg = f"+++++ Finished training at epoch {epoch_idx}, best eval results:"
                    if verbose:
                        self.logger.info(stop_msg)
                        stop_output_src = (
                            f"🛑 Early stopping triggered for 🌐 Source Domain "
                            f"(best epoch: {epoch_idx - self.cur_step_src * self.eval_step})"
                        )
                        stop_output_tgt = (
                            f"🛑 Early stopping triggered for 🎯 Target Domain "
                            f"(best epoch: {epoch_idx - self.cur_step_tgt * self.eval_step})"
                        )
                        self.logger.info(stop_output_src)
                        self.logger.info(stop_output_tgt)
                    break
        return (self.best_valid_score_src, self.best_valid_result_src, self.best_test_upon_valid_src,
                self.best_valid_score_tgt, self.best_valid_result_tgt, self.best_test_upon_valid_tgt)

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        src_batch_matrix_list, tgt_batch_matrix_list = [],[]
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            src_scores, tgt_scores = self.model.full_sort_predict(batched_data)
            src_masked_items = batched_data[1]
            tgt_masked_items = batched_data[2]
            # mask out pos items
            src_scores[src_masked_items[0], src_masked_items[1]] = -1e10
            tgt_scores[tgt_masked_items[0], tgt_masked_items[1]] = -1e10
            # rank and get top-k
            _, src_topk_index = torch.topk(src_scores, max(self.config['topk']), dim=-1)
            _, tgt_topk_index = torch.topk(tgt_scores, max(self.config['topk']), dim=-1)
            src_batch_matrix_list.append(src_topk_index)
            tgt_batch_matrix_list.append(tgt_topk_index)
        src_result = self.evaluator.evaluate(src_batch_matrix_list, eval_data, domain = 0, is_test=is_test, idx=idx)
        tgt_result = self.evaluator.evaluate(tgt_batch_matrix_list, eval_data, domain = 1, is_test=is_test, idx=idx)
        return src_result, tgt_result

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

