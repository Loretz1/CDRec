import math
import torch
import random
import numpy as np
import pandas as pd
from logging import getLogger
from pandas.io.formats.format import return_docstring
from utils.enum_type import TrainDataLoaderState, EvalDataLoaderState
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.BASIC_DATA_FIELDS)
        for key in self.dataset.df:
            df = self.dataset.df[key]
            self.dataset_bk.df[key] = pd.DataFrame(
                df.values.copy(), columns=df.columns
            ).reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = config['device']

        self.pr = 0
        self.inter_pr_src = 0

    def pretrain_setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def data_preprocess(self):
        """This function is used to do some data preprocess, such as pre-neg-sampling and pre-data-augmentation.
        By default, it will do nothing.
        """
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')



class TrainDataLoader(AbstractDataLoader):
    """
    功能：
        基于 RecDataset 的训练数据加载器，
        按训练阶段（TrainDataLoaderState）生成对应形式的训练 batch。

    数据来源：
        - 输入为 split() 后得到的 train_dataset（RecDataset）
        - 使用其中的：
            * train_src / train_tgt / train_both / train_overlap / train_overlap_user
            * positive_items_src / positive_items_tgt
            * num_users / num_items 统计信息

    核心处理：
        - 通过 set_state_for_train(state) 指定当前训练阶段
        - 根据 state 选择对应的采样函数：
            * SOURCE / TARGET       ：单域 (u, i_pos, i_neg)
            * BOTH                  ：src + tgt 双域同时采样
            * OVERLAP               ：overlap 用户的双域正负样本
            * OVERLAP_USER          ：仅 overlap 用户 ID
        - 在每个 batch 中执行：
            * 正样本读取（来自当前训练交互）
            * 负样本随机采样（基于正样本集合排除）

    输出形式：
        - 每次迭代返回一个 dict，
          key 和 tensor 结构由当前 TrainDataLoaderState 决定，
          可直接作为模型 forward / loss 的输入

    说明：
        - 仅用于训练阶段（不用于 valid / test）
        - 不负责评估逻辑，仅负责 batch 构造与负采样
    """
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, shuffle=shuffle)
        self.state = None

    def set_state_for_train(self, state):
        """
        功能：
            在训练阶段设置当前 DataLoader 的采样状态，
            并同步切换 RecDataset 中对应的训练数据视图。
        处理逻辑：
            - 校验 state 类型为 TrainDataLoaderState
            - 将 state 同步设置到 dataset（dataset.set_state_for_train）
            - 根据 state 选择对应的 batch 采样函数：
                * SOURCE / TARGET       → _sample_single_domain
                * BOTH                  → _sample_both
                * OVERLAP               → _sample_overlap
                * OVERLAP_USER          → _sample_overlap_user
            - 重置 batch 指针 pr
        输入：
            state: TrainDataLoaderState
                指定当前训练阶段的数据采样方式
        """
        if not isinstance(state, TrainDataLoaderState):
            raise TypeError("state must be an instance of TrainDataLoaderState")

        self.state = state
        self.dataset.set_state_for_train(state)
        self.sample_func = self._get_sample_func()
        self.pr = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def pretrain_setup(self):
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.BASIC_DATA_FIELDS)
            for key in self.dataset_bk.df:
                df = self.dataset_bk.df[key]
                self.dataset.df[key] = pd.DataFrame(
                    df.values.copy(), columns=df.columns
                ).reset_index(drop=True)

    def get_positive_items_for_u_in_domain(self, user, domain):
        """
        功能：
            返回指定用户在指定域中的正样本物品集合。

        输入：
            user: int
                重编号后的 user ID（从 1 开始）
            domain: int
                0 表示源域（src），1 表示目标域（tgt）

        输出：
            Set[int]
                该用户在指定域中的正样本 item ID 集合
                （item ID 从 1 开始，不包含 padding）
        """
        if domain == 0:
            return self.dataset.positive_items_src[user]
        else:
            return self.dataset.positive_items_tgt[user]

    def inter_matrix(self, domain, form='coo'):
        """
        功能：
            根据训练数据构建指定域的用户-物品交互稀疏矩阵。
        编号约定：
            - 矩阵中的 user 和 item 均采用 0-based 编号
            - 不包含 padding（原始 ID 中的 0 号 padding 已被移除）
            - user ∈ [0, num_users - 1]，item ∈ [0, num_items - 1]
        矩阵规模：
            - 行数：对应域的用户数
                * src → num_users_src
                * tgt → num_users_tgt
            - 列数：对应域的物品数
                * src → num_items_src
                * tgt → num_items_tgt
        """
        all_inter = self.dataset.df['train_src'] if domain == 0 else self.dataset.df['train_tgt']
        num_users = self.dataset.num_users_src if domain == 0 else self.dataset.num_users_tgt
        num_items = self.dataset.num_items_src if domain == 0 else self.dataset.num_items_tgt

        # ID start from 1 → shift to 0-based index
        users = all_inter['user'].values - 1
        items = all_inter['item'].values - 1
        data = np.ones_like(users, dtype=np.float32)
        mat = coo_matrix((data, (users, items)), shape=(num_users, num_items))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f"sparse matrix format [{form}] not implemented.")

    def _get_sample_func(self):
        """
        功能：
            根据当前训练状态（TrainDataLoaderState）
            选择并返回对应的 batch 采样函数。

        状态与采样函数映射：
            - SOURCE / TARGET       → _sample_single_domain
            - BOTH                  → _sample_both
            - OVERLAP               → _sample_overlap
            - OVERLAP_USER          → _sample_overlap_user
        """
        if self.state == TrainDataLoaderState.SOURCE or self.state == TrainDataLoaderState.TARGET:
            return self._sample_single_domain
        elif self.state == TrainDataLoaderState.BOTH:
            return self._sample_both
        elif self.state == TrainDataLoaderState.OVERLAP:
            return self._sample_overlap
        elif self.state == TrainDataLoaderState.OVERLAP_USER:
            return self._sample_overlap_user
        return None

    def _sample_single_domain(self):
        """
        功能：
            在单域训练状态下（SOURCE 或 TARGET），
            从当前训练交互中构造一个 batch 的 (user, pos_item, neg_item)。

        数据来源：
            - 根据当前 state 决定域：
                * SOURCE → 源域（src）
                * TARGET → 目标域（tgt）
            - 使用 dataset 中已选定的活跃训练 DataFrame（_active_df）

        处理逻辑：
            - 顺序读取 batch_size 条训练交互 (u, i_pos)
            - 对每个用户 u：
                * 从对应域的正样本集合中确定正样本 i_pos
                * 从对应域中随机采样一个负样本 i_neg（不在正样本集合中）

        输出：
            dict:
                {
                    "users":     LongTensor [batch_size],
                    "pos_items": LongTensor [batch_size],
                    "neg_items": LongTensor [batch_size]
                }
            所有 ID 均为重编号后的 ID（从 1 开始，无 padding）
        """
        domain = 0 if self.state == TrainDataLoaderState.SOURCE else 1
        cur_data = self.dataset[self.pr: self.pr + self.batch_size]
        self.pr += self.batch_size

        users, pos_items, neg_items = [], [], []

        for _, row in cur_data.iterrows():
            u, i_pos= row['user'], row['item']
            i_neg = self._sample_neg_item_from_domain_for_u(u, domain)
            users.append(u)
            pos_items.append(i_pos)
            neg_items.append(i_neg)

        return {
            "users": torch.tensor(users, dtype=torch.long, device=self.device),
            "pos_items": torch.tensor(pos_items, dtype=torch.long, device=self.device),
            "neg_items": torch.tensor(neg_items, dtype=torch.long, device=self.device)
        }

    def _sample_both(self):
        """
        功能：
            在 BOTH 训练状态下，从源域（src）和目标域（tgt）
            同时构造一个 batch 的双域训练样本。

        数据来源：
            - 使用 dataset 中的活跃训练 DataFrame（train_both）
            - 每条记录包含 (user, item, domain)

        处理逻辑：
            - 顺序读取 batch_size 条训练交互
            - 对每条交互：
                * 若来自 src 域：
                    - 构造一条 src 正负样本 (u_src, i_pos_src, i_neg_src)
                    - 同时从 tgt 域随机采样一条交互，构造 tgt 正负样本
                * 若来自 tgt 域：
                    - 构造一条 tgt 正负样本 (u_tgt, i_pos_tgt, i_neg_tgt)
                    - 同时从 src 域随机采样一条交互，构造 src 正负样本
            - 保证每个 batch 同时包含 src 和 tgt 的训练信号

        输出：
            dict:
                {
                    "users_src":     LongTensor [batch_size],
                    "pos_items_src": LongTensor [batch_size],
                    "neg_items_src": LongTensor [batch_size],
                    "users_tgt":     LongTensor [batch_size],
                    "pos_items_tgt": LongTensor [batch_size],
                    "neg_items_tgt": LongTensor [batch_size]
                }
            所有 ID 均为重编号后的 ID（从 1 开始，无 padding）
        """
        cur_data = self.dataset[self.pr: self.pr + self.batch_size]
        self.pr += self.batch_size

        users_src, pos_items_src, neg_items_src = [], [], []
        users_tgt, pos_items_tgt, neg_items_tgt = [], [], []

        for _, row in cur_data.iterrows():
            u, i, d = row["user"], row["item"], row["domain"]

            if d == 0:
                users_src.append(u)
                pos_items_src.append(i)
                neg_items_src.append(self._sample_neg_item_from_domain_for_u(u, domain=0))
                u_another_domain, i_another_domain = self._sample_interaction_from_domain(1)
                users_tgt.append(u_another_domain)
                pos_items_tgt.append(i_another_domain)
                neg_items_tgt.append(self._sample_neg_item_from_domain_for_u(u_another_domain, domain=1))
            else:
                users_tgt.append(u)
                pos_items_tgt.append(i)
                neg_items_tgt.append(self._sample_neg_item_from_domain_for_u(u, domain=1))
                u_another_domain, i_another_domain = self._sample_interaction_from_domain(0)
                users_src.append(u_another_domain)
                pos_items_src.append(i_another_domain)
                neg_items_src.append(self._sample_neg_item_from_domain_for_u(u_another_domain, domain=0))

        return{
            "users_src": torch.tensor(users_src, dtype=torch.long, device=self.device),
            "pos_items_src": torch.tensor(pos_items_src, dtype=torch.long, device=self.device),
            "neg_items_src": torch.tensor(neg_items_src, dtype=torch.long, device=self.device),
            "users_tgt": torch.tensor(users_tgt, dtype=torch.long, device=self.device),
            "pos_items_tgt": torch.tensor(pos_items_tgt, dtype=torch.long, device=self.device),
            "neg_items_tgt": torch.tensor(neg_items_tgt, dtype=torch.long, device=self.device)
        }

    def _sample_overlap(self):
        """
        功能：
            在 OVERLAP 训练状态下，为 overlap 用户同时构造
            源域（src）和目标域（tgt）的正负样本。

        数据来源：
            - 使用 dataset 中的活跃训练 DataFrame（train_overlap）
            - 每条记录包含 (user, item, domain)，且 user 均为 overlap 用户

        处理逻辑：
            - 顺序读取 batch_size 条训练交互
            - 对每个 overlap 用户 u：
                * 同时为 src 和 tgt 域各采样一个负样本
                * 若当前交互来自 src 域：
                    - src 正样本使用当前 item
                    - tgt 正样本从该用户在 tgt 域的正样本集合中随机采样
                * 若当前交互来自 tgt 域：
                    - tgt 正样本使用当前 item
                    - src 正样本从该用户在 src 域的正样本集合中随机采样

        输出：
            dict:
                {
                    "users":          LongTensor [batch_size],
                    "pos_items_src":  LongTensor [batch_size],
                    "neg_items_src":  LongTensor [batch_size],
                    "pos_items_tgt":  LongTensor [batch_size],
                    "neg_items_tgt":  LongTensor [batch_size]
                }
            所有 ID 均为重编号后的 ID（从 1 开始，无 padding）
        """
        cur_data = self.dataset[self.pr: self.pr + self.batch_size]
        self.pr += self.batch_size

        users = []
        pos_items_src, neg_items_src = [], []
        pos_items_tgt, neg_items_tgt = [], []

        for _, row in cur_data.iterrows():
            u, i, d = row["user"], row["item"], row["domain"]
            users.append(u)
            neg_items_src.append(self._sample_neg_item_from_domain_for_u(u, domain=0))
            neg_items_tgt.append(self._sample_neg_item_from_domain_for_u(u, domain=1))
            if d == 0:
                pos_items_src.append(i)
                pos_items_tgt.append(self._sample_pos_item_from_domain_for_u(u, domain=1))
            else:
                pos_items_tgt.append(i)
                pos_items_src.append(self._sample_pos_item_from_domain_for_u(u, domain=0))

        return{
            "users": torch.tensor(users, dtype=torch.long, device=self.device),
            "pos_items_src": torch.tensor(pos_items_src, dtype=torch.long, device=self.device),
            "neg_items_src": torch.tensor(neg_items_src, dtype=torch.long, device=self.device),
            "pos_items_tgt": torch.tensor(pos_items_tgt, dtype=torch.long, device=self.device),
            "neg_items_tgt": torch.tensor(neg_items_tgt, dtype=torch.long, device=self.device)
        }

    def _sample_overlap_user(self):
        """
        功能：
            在 OVERLAP_USER 训练状态下，
            从 overlap 用户集合中按 batch 取出用户 ID。

        数据来源：
            - 使用 dataset 中的活跃 DataFrame（train_overlap_user）
            - 该 DataFrame 仅包含一列：
                * user（overlap 用户的 user ID）

        处理逻辑：
            - 顺序读取 batch_size 个 overlap 用户 ID
            - 不涉及物品、不进行正负采样

        输出：
            dict:
                {
                    "users_overlapped": LongTensor [batch_size]
                }
            user ID 为重编号后的 ID（从 1 开始，无 padding）
        """
        cur_data = self.dataset[self.pr: self.pr + self.batch_size]
        self.pr += self.batch_size

        return {
            "users_overlapped": torch.tensor(cur_data["user"].values, dtype=torch.long, device=self.device)
        }

    def _sample_pos_item_from_domain_for_u(self, u, domain):
        """
        功能：
            从指定用户在指定域中的正样本集合里，
            随机采样一个正样本物品。

        数据来源：
            - domain == 0（src）：
                使用 dataset.positive_items_src[u]
            - domain == 1（tgt）：
                使用 dataset.positive_items_tgt[u]
            - 正样本集合仅基于训练集构建

        输入：
            u: int
                重编号后的 user ID（从 1 开始）
            domain: int
                0 表示源域（src），1 表示目标域（tgt）

        输出：
            int
                随机采样得到的正样本 item ID
                （从 1 开始，无 padding）
        """
        if domain == 0:
            return random.sample(self.dataset.positive_items_src[u], 1)[0]
        else:
            return random.sample(self.dataset.positive_items_tgt[u], 1)[0]

    def _sample_neg_item_from_domain_for_u(self, u, domain):
        """
        功能：
            在指定域中为指定用户随机采样一个负样本物品。

        数据来源：
            - domain == 0（src）：
                * 物品空间：[1, num_items_src]
                * 用户历史正样本：dataset.positive_items_src[u]
            - domain == 1（tgt）：
                * 物品空间：[1, num_items_tgt]
                * 用户历史正样本：dataset.positive_items_tgt[u]

        处理逻辑：
            - 在对应域的物品 ID 空间内随机采样
            - 若采样结果出现在该用户的正样本集合中，则重新采样
            - 直到得到一个未被交互过的物品为止

        输入：
            u: int
                重编号后的 user ID（从 1 开始）
            domain: int
                0 表示源域（src），1 表示目标域（tgt）

        输出：
            int
                随机采样得到的负样本 item ID
                （从 1 开始，无 padding）
        """
        if domain == 0:
            num_items = self.dataset.num_items_src
            hist = self.dataset.positive_items_src[u]
        else:
            num_items = self.dataset.num_items_tgt
            hist = self.dataset.positive_items_tgt[u]

        neg = np.random.randint(1, num_items+1)
        while neg in hist:
            neg = np.random.randint(1, num_items+1)
        return neg

    def _sample_interaction_from_domain(self, domain):
        """
        功能：
            从指定域的训练交互中随机采样一条 (user, item) 交互。

        数据来源：
            - domain == 0（src）：
                使用 dataset.df['train_src']
            - domain == 1（tgt）：
                使用 dataset.df['train_tgt']

        处理逻辑：
            - 在对应训练交互 DataFrame 中随机选择一行
            - 直接返回该行中的 user 和 item

        输入：
            domain: int
                0 表示源域（src），1 表示目标域（tgt）

        输出：
            Tuple[int, int]
                (user, item)，均为重编号后的 ID（从 1 开始，无 padding）
        """
        df = self.dataset.df["train_src"] if domain == 0 else self.dataset.df["train_tgt"]
        idx = np.random.randint(len(df))
        row = df.iloc[idx]
        # row = df.sample(n=1).iloc[0]
        return row["user"], row["item"]

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        return self.sample_func()



class EvalDataLoader(AbstractDataLoader):
    """
    功能：
        用于推荐模型评估阶段的数据加载器，
        从 RecDataset 中读取 warm / cold 用户的评估数据，
        并按 batch 组织评估所需的用户与物品信息。

    数据来源：
        - 输入为 split() 后得到的 valid_dataset 或 test_dataset（RecDataset）
        - 使用的数据包括：
            * df['warm_tgt']：warm 用户评估交互
            * df['cold_tgt']：cold 用户评估交互
            * positive_items_tgt：目标域训练阶段的正样本集合

    初始化处理：
        - 根据配置决定是否启用评估模式：
            * warm 评估：config['warm_eval']
            * cold-start 评估：config['cold_start_eval']
        - 对每种启用的评估状态（warm / cold），调用：
            _prepare_one_state(df_key)
          并在 self.cache 中缓存结果：
            * self.cache["warm"] ← df_key = "warm_tgt"
            * self.cache["cold"] ← df_key = "cold_tgt"
        - 每个 cache 条目包含以下变量：
            * eval_u               ：评估用户列表（Tensor）
                - 类型：torch.LongTensor
                - shape：[U]
                - 含义：参与评估的用户 ID 列表（重编号后的 user ID）
            * pos_items_per_u      ：训练阶段正样本索引（用于 masking）
                - 类型：torch.LongTensor
                - shape：[2, P]
                - 含义：
                    * 第 0 行：用户在 eval_u 中的相对索引（0-based）
                    * 第 1 行：对应用户的训练阶段正样本 item ID
                - 说明：
                    * 仅用于 warm 评估
                    * cold 评估时为 shape [2, 0] 的空 Tensor
            * train_pos_len_list   ：每个用户的训练正样本数量
                - 类型：List[int]
                - 长度：U
                - 含义：
                    * 每个评估用户在训练集中的正样本数量
                    * 用于在 batch 中切分 pos_items_per_u
            * eval_items_per_u     ：每个用户的评估物品列表
                - 类型：List[np.ndarray]
                - 长度：U
                - 含义：
                    * 每个评估用户在评估集中的 item 列表
                    * item ID 为重编号后的 ID
            * eval_len_list        ：每个用户的评估物品数量
                - 类型：np.ndarray
                - shape：[U]
                - 含义：
                    * 每个评估用户在评估集中的物品数量
        - 以上预处理结果统一缓存在 self.cache 中，
          用于评估阶段直接按 batch 读取，避免重复构建

    评估使用方式：
        - 通过 set_state_for_eval(state) 选择 warm 或 cold 评估模式
        - 每个 batch 返回：
            * 当前 batch 的评估用户
            * 对应的正样本 mask（基于训练集）
        - 供模型进行全物品排序评估（full-sort evaluation）

    说明：
        - 仅用于评估阶段，不用于训练
        - user / item 均为重编号后的 ID
        - 评估数据不进行负采样
    """
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, shuffle=shuffle)

        self.cache = {}
        self.state = None
        self.current = None
        self.warm = False
        self.cold = False

        if config.get("warm_eval", False):
            self.warm =True
            self.cache["warm"] = self._prepare_one_state("warm_tgt")
        if config.get("cold_start_eval", False):
            self.cold =True
            self.cache["cold"] = self._prepare_one_state("cold_tgt")

    def _prepare_one_state(self, df_key):
        """
        功能：
            为指定评估状态（warm 或 cold）预处理评估所需的数据结构，
            并返回一个可直接用于 batch 评估的缓存字典。

        输入：
            df_key: str
                评估数据在 dataset.df 中的键名，
                取值为 "warm_tgt" 或 "cold_tgt"

        输出：
            dict
            {
                "eval_u":               LongTensor [U],
                "pos_items_per_u":      LongTensor [2, P],
                "train_pos_len_list":   List[int] (len = U),
                "eval_items_per_u":     List[np.ndarray] (len = U),
                "eval_len_list":        np.ndarray [U]
            }
        """
        df = self.dataset.df.get(df_key, None)
        if df is None:
            raise KeyError(f"Dataset missing df[{df_key}] for evaluation.")

        eval_users = df["user"].unique()
        if df_key == "warm_tgt" and self.config.get("overlapped_users_for_warm_eval", False):
            overlap_max = self.dataset.num_users_overlap
            mask = (eval_users >= 1) & (eval_users <= overlap_max)
            eval_users = eval_users[mask]

        pos_items_per_u, train_pos_len_list = \
            self._build_pos_items_per_u(eval_users, df_key)

        eval_items_per_u, eval_len_list = \
            self._build_eval_items_per_u(eval_users, df)

        return {
            "eval_u": torch.tensor(eval_users, dtype=torch.long, device=self.device),
            "pos_items_per_u": pos_items_per_u,
            "train_pos_len_list": train_pos_len_list,
            "eval_items_per_u": eval_items_per_u,
            "eval_len_list": eval_len_list
        }

    def _build_pos_items_per_u(self, eval_users, df_key):
        if df_key == "cold_tgt":
            L = len(eval_users)
            return torch.zeros((2, 0), dtype=torch.long, device=self.device), [0] * L

        tgt_pos = self.dataset.positive_items_tgt

        u_ids, i_ids = [], []
        pos_len = []
        for idx, u in enumerate(eval_users):
            items = list(tgt_pos.get(u, []))
            pos_len.append(len(items))

            u_ids.extend([idx] * len(items))
            i_ids.extend(items)

        if len(u_ids) == 0:
            pos_items_per_u = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            pos_items_per_u = torch.tensor([u_ids, i_ids], dtype=torch.long, device=self.device)
        return pos_items_per_u, pos_len

    def _build_eval_items_per_u(self, eval_users, df):
        uid_freq = df.groupby("user")["item"]

        eval_items = []
        eval_len = []
        for u in eval_users:
            if u in uid_freq.groups:
                items = uid_freq.get_group(u).values
            else:
                items = np.array([], dtype=int)

            eval_items.append(items)
            eval_len.append(len(items))

        return eval_items, np.array(eval_len, dtype=int)

    def set_state_for_eval(self, state):
        """
            设置当前评估阶段的评估状态（warm 或 cold），
            并切换到对应的评估缓存数据。
        """
        assert isinstance(state, EvalDataLoaderState)
        self.state = state

        if state == EvalDataLoaderState.WARM:
            self.current = self.cache["warm"]
        else:
            self.current = self.cache["cold"]

        self.pr = 0
        self.inter_pr = 0

    @property
    def pr_end(self):
        return len(self.current["eval_u"])

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        """
        功能：
            在评估阶段按 batch 返回当前批次的评估用户及其对应的正样本 mask，用于给将用户训练集交互过的物品的sroces设置-∞

        输出：
            List:
            [
                batch_users,   # LongTensor [batch_size]
                mask           # LongTensor [2, cnt]
            ]
        """
        batch_users = self.current["eval_u"][self.pr:self.pr + self.batch_size]

        cnt = sum(self.current["train_pos_len_list"][self.pr:self.pr + self.batch_size])

        mask = self.current["pos_items_per_u"][:, self.inter_pr:self.inter_pr + cnt].clone()
        mask[0] -= self.pr

        self.inter_pr += cnt
        self.pr += self.batch_size

        return [batch_users, mask]

    def get_eval_users(self):
        """
        返回当前评估状态下的全部评估用户列表。
            torch.LongTensor
                shape = [U]
                当前评估状态下的所有评估用户 ID
        """
        return self.current["eval_u"].cpu()

    def get_eval_items(self):
        """
        功能：
            返回当前评估状态下，每个评估用户对应的评估物品列表。

        数据来源：
            - 使用 set_state_for_eval() 选定的评估缓存数据 self.current
            - 对应字段为 self.current["eval_items_per_u"]

        结构说明：
            - 类型：List[np.ndarray]
            - 长度：U（评估用户数量）
            - 第 i 个元素表示第 i 个评估用户的评估物品 ID 列表

        输出：
            List[np.ndarray]
                每个评估用户对应的评估物品集合
        """
        return self.current["eval_items_per_u"]

    def get_eval_len_list(self):
        """
        功能：
            返回当前评估状态下，每个评估用户对应的评估物品数量。

        数据来源：
            - 使用 set_state_for_eval() 选定的评估缓存数据 self.current
            - 对应字段为 self.current["eval_len_list"]

        结构说明：
            - 类型：np.ndarray
            - shape：[U]
            - 第 i 个元素表示第 i 个评估用户的评估物品数量

        输出：
            np.ndarray
                每个评估用户对应的评估物品数量列表
        """
        return self.current["eval_len_list"]