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
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, shuffle=shuffle)
        self.state = None

    def set_state_for_train(self, state):
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
        if domain == 0:
            return self.dataset.positive_items_src[user]
        else:
            return self.dataset.positive_items_tgt[user]

    def inter_matrix(self, domain, form='coo'):
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
        Returns:
        dict: {
            "users": Tensor(batch_size),
            "pos_items": Tensor(batch_size),
            "neg_items": Tensor(batch_size)
        }
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
        Returns:
        dict: {
            "users_src": Tensor(batch_size),
            "pos_items_src": Tensor(batch_size),
            "neg_items_src": Tensor(batch_size)
            "users_tgt": Tensor(batch_size),
            "pos_items_tgt": Tensor(batch_size),
            "neg_items_tgt": Tensor(batch_size)
        }
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
        Returns:
        dict: {
            "users_src": Tensor(batch_size),
            "pos_items_src": Tensor(batch_size),
            "neg_items_src": Tensor(batch_size)
            "pos_items_tgt": Tensor(batch_size),
            "neg_items_tgt": Tensor(batch_size)
        }
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
        Returns:
        dict: {
            "users_overlapped": Tensor(batch_size),
        }
        """
        cur_data = self.dataset[self.pr: self.pr + self.batch_size]
        self.pr += self.batch_size

        return {
            "users_overlapped": torch.tensor(cur_data["user"].values, dtype=torch.long, device=self.device)
        }

    def _sample_pos_item_from_domain_for_u(self, u, domain):
        if domain == 0:
            return random.sample(self.dataset.positive_items_src[u], 1)[0]
        else:
            return random.sample(self.dataset.positive_items_tgt[u], 1)[0]

    def _sample_neg_item_from_domain_for_u(self, u, domain):
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
        batch_users = self.current["eval_u"][self.pr:self.pr + self.batch_size]

        cnt = sum(self.current["train_pos_len_list"][self.pr:self.pr + self.batch_size])

        mask = self.current["pos_items_per_u"][:, self.inter_pr:self.inter_pr + cnt].clone()
        mask[0] -= self.pr

        self.inter_pr += cnt
        self.pr += self.batch_size

        return [batch_users, mask]

    def get_eval_users(self):
        return self.current["eval_u"].cpu()

    def get_eval_items(self):
        return self.current["eval_items_per_u"]

    def get_eval_len_list(self):
        return self.current["eval_len_list"]