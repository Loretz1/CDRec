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

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, writer=None):
        """
        功能：
            执行一个 epoch 的模型训练，遍历训练数据并完成前向、反向与参数更新。

        数据来源：
            - train_data：TrainDataLoader
                提供当前训练阶段的 batch 数据
            - loss_func：
                若为 None，默认使用 model.calculate_loss

        处理逻辑：
            - 若 req_training 为 False，直接返回空损失
            - 将模型切换为训练模式（model.train()）
            - 对每个 batch：
                1) 调用 model.pre_batch_processing()
                2) 前向计算损失（loss_func）
                3) 累计损失（支持单损失或多损失 tuple）
                4) 反向传播（loss.backward()）
                5) 可选梯度裁剪（clip_grad_norm）
                6) 优化器更新参数（optimizer.step()）
                7) 调用 model.post_batch_processing()
            - 若检测到 NaN 损失，提前终止该 epoch

        输入：
            train_data: TrainDataLoader
                训练数据加载器
            epoch_idx: int
                当前 epoch 编号
            loss_func: callable, optional
                自定义损失函数，输入为 (interaction, epoch_idx)

        输出：
            total_loss:
                - float：单损失情况下为该 epoch 的累计损失
                - tuple：多损失情况下为各损失分量的累计值
            loss_batches: List[Tensor]
                每个 batch 的损失值（detach 后）
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            # check the valid of the interaction
            # self._check_interaction_valid(train_data, interaction)
            self.model.pre_batch_processing(epoch_idx,batch_idx)

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
            self.model.post_batch_processing(epoch_idx,batch_idx)
            # for test
            # if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        """
        功能：
            在验证阶段对模型进行评估，
            分别返回 warm 和 cold 用户的评估结果与主评估指标。

        输入：
            valid_data: EvalDataLoader
                验证数据加载器

        输出：
            Tuple:
                (
                    valid_score_warm,    # float or None
                    valid_result_warm,   # dict
                    valid_score_cold,    # float or None
                    valid_result_cold    # dict
                )

            说明：
                - 若对应评估模式未启用（warm / cold），
                  该部分返回值为 None
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
        """
            根据单个 epoch 的训练结果，生成格式化的训练日志字符串。
        """
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
        """
            根据当前 epoch 的评估结果，生成格式化的评估日志字符串，
            用于输出验证或测试阶段的 warm / cold 评估信息。
        """
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
                       f"| {'🛑 Early Stop' if stop_cold else ''}\n" if score_cold is not None else "")
                    + f"   Metrics:\n{metrics_dict2str(results_cold)}\n"
            )
            eval_output += cold_block

        eval_output += f"{'=' * 90}"
        return eval_output

    def _check_interaction_valid(self, train_data, interaction):
        from utils.utils import get_dict_from_raw_data_for_Amazon2014
        import os
        dataset_path0 = os.path.join(self.config['data_path'], self.config['dataset'])
        dataset_path1 = os.path.join(self.config['data_path'], self.config['dataset'])
        domain0 = self.config['domains'][0]
        domain1 = self.config['domains'][1]
        review0 = get_dict_from_raw_data_for_Amazon2014(dataset_path0, domain0, True, ['reviewerID', 'asin'],
                                                        ['reviewerID', 'asin', "reviewText"])
        review1 = get_dict_from_raw_data_for_Amazon2014(dataset_path1, domain1, True, ['reviewerID', 'asin'],
                                                        ['reviewerID', 'asin', "reviewText"])
        for i in range(len(interaction['users_src'])):
            user_src = train_data.dataset.id_mapping['src']['id2user'][interaction['users_src'][i]]
            pos_items_src = train_data.dataset.id_mapping['src']['id2item'][interaction['pos_items_src'][i]]
            neg_items_src = train_data.dataset.id_mapping['src']['id2item'][interaction['neg_items_src'][i]]
            user_tgt = train_data.dataset.id_mapping['tgt']['id2user'][interaction['users_tgt'][i]]
            pos_items_tgt = train_data.dataset.id_mapping['tgt']['id2item'][interaction['pos_items_tgt'][i]]
            neg_items_tgt = train_data.dataset.id_mapping['tgt']['id2item'][interaction['neg_items_tgt'][i]]
            assert (user_src, pos_items_src) in review0 and (interaction['neg_items_src'][i] not in train_data.dataset.positive_items_src[interaction['users_src'][i].item()])
            assert (user_tgt, pos_items_tgt) in review1 and (interaction['neg_items_tgt'][i] not in train_data.dataset.positive_items_tgt[interaction['users_tgt'][i].item()])

    def fit(self, stage_id, train_data, valid_data=None, test_data=None, saved=False, verbose=True, writer=None):
        stage_train_time_total = 0
        stop_epoch = 0
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing(epoch_idx)
            train_loss, _ = self._train_epoch(train_data, epoch_idx, writer=writer)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            # for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[(stage_id, epoch_idx)] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            stage_train_time_total += training_end_time - training_start_time
            stop_epoch = epoch_idx
            writer.add_scalar(f"Stage{stage_id} training Loss", self.train_loss_dict[(stage_id, epoch_idx)], epoch_idx)  # tb
            train_loss_output = \
                self._generate_train_loss_output(stage_id, epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing(epoch_idx)
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
                test_score_warm, test_result_warm, test_score_cold, test_result_cold = self._valid_epoch(test_data)
                if test_score_warm is not None:
                    writer.add_scalar("Warm testing acc:", test_score_warm, epoch_idx)  # tb
                if test_score_cold is not None:
                    writer.add_scalar("Cold testing acc:", test_score_cold, epoch_idx)  # tb
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
                # 早停的逻辑是Cold和Warm都满足了早停条件，才早停。
                # 分三种情况
                # 1. Yaml中warm_eval=true，cold_eval=false，
                # 即评估热启动效果，那么这里的返回的stop_flag_cold恒为True，
                # 只根据stop_flag_warm判断是否早停，stop_flag_warm=True的条件就是热评估连续stopping_step个epoch在valid上无提升
                # 2. Yaml中warm_eval=false，cold_eval=true，
                # 即评估冷启动效果，和上面的热启动逻辑差不多，就是stop_flag_warm和stop_flag_cold反过来，
                # stop_flag_warm恒为True，只看stop_flag_cold
                # 3. Yaml中warm_eval=true，cold_eval=true，
                # 保留了同时评估冷/热的能力，stop_flag_warm和stop_flag_cold同时为True，
                # 即冷和热评估都连续stopping_step个epoch在valid上无提升时，才早停。这个场景在目前的冷热分离评估场景下是用不到的。
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
            return (None, None, None, None, None, None), (stage_train_time_total, stop_epoch)
        return (self.best_valid_score_warm, self.best_valid_result_warm, self.best_test_upon_valid_warm,
                self.best_valid_score_cold, self.best_valid_result_cold, self.best_test_upon_valid_cold), (stage_train_time_total, stop_epoch)

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        """
        功能：
            在评估阶段对模型进行 full-sort 推荐评估，
            分别计算 warm 用户和 cold 用户的 Top-K 推荐指标。

        数据来源：
            - eval_data：EvalDataLoader
                提供评估阶段的用户、正样本 mask 和评估物品信息

        评估流程：
            - 将模型切换为 eval 模式（model.eval()）
            - 对 warm 用户评估（若启用）：
                1) 设置评估状态为 WARM（eval_data.set_state_for_eval）
                2) 按 batch 遍历评估用户
                3) 调用 model.full_sort_predict(...) 计算用户对所有目标域物品的评分
                4) 使用训练阶段正样本 mask 屏蔽已交互物品
                5) 屏蔽 padding item（item 0）
                6) 对评分结果进行 Top-K 排序
                7) 调用 TopKEvaluator.evaluate 计算评估指标
            - 对 cold 用户评估（若启用）：
                * 流程与 warm 用户一致，但使用 cold 用户评估数据
                * 调用 full_sort_predict(..., is_warm=False)

        输入：
            eval_data: EvalDataLoader
                验证或测试阶段的数据加载器
            is_test: bool
                是否为测试阶段（传递给 evaluator，用于结果区分）
            idx: int
                当前评估编号（用于多次评估区分）

        输出：
            Tuple:
                (
                    result_warm,   # dict or None
                    result_cold    # dict or None
                )

            说明：
                - result_warm / result_cold 为 Top-K 指标字典
                - 若对应评估模式未启用，则返回 None
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
                # mask the item 0 which is PAD
                scores[:, 0] = -1e10
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

        self.model.set_train_stage(stage_id) # 调用模型set_train_stage方法，让模型可以冻结参数，或者保存之前阶段的结果
        self.optimizer = self._build_optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters()))
        self.lr_scheduler = self._build_lr_scheduler()
        self.eval = eval
