import math
import torch
import random
import numpy as np
from logging import getLogger
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.df)
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.device = config['device']

        self.pr = 0
        self.inter_pr_src = 0
        self.inter_pr_tgt = 0

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
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr_src = 0
            self.inter_pr_tgt = 0
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
        super().__init__(config, dataset, additional_dataset=None,
                         batch_size=batch_size, shuffle=shuffle)

        # special for training dataloader
        self.history_source_items_per_u = dict()
        self.history_target_items_per_u = dict()
        # full items in training.
        self.all_source_items = self.dataset.get_source_df()[self.dataset.iid_field].unique().tolist()
        self.all_target_items = self.dataset.get_target_df()[self.dataset.iid_field].unique().tolist()
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()
        self.all_source_items_set = set(self.all_source_items)
        self.all_target_items_set = set(self.all_target_items)
        self.all_users_set = set(self.all_uids)
        self.all_source_item_len = len(self.all_source_items)
        self.all_target_item_len = len(self.all_target_items)

        self.sample_func = self._get_sample

        self._get_history_items_u()

    def pretrain_setup(self):
        """
        Reset dataloader. Outputing the same positive & negative samples with each training.
        :return:
        """
        # sort & random
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
        self.all_source_items.sort()
        self.all_target_items.sort()
        self.all_uids.sort()
        random.shuffle(self.all_source_items)
        random.shuffle(self.all_target_items)

    def inter_matrix(self, form='coo', value_field=None):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        domain_field = self.dataset.domain_field
        if not uid_field or not iid_field:
            raise ValueError("dataset doesn't exist uid/iid, cannot convert to sparse matrix")
        df_source = self.dataset.get_source_df()
        mat_source = self._create_sparse_matrix(
            df_source, uid_field, iid_field,
            form=form, value_field=value_field,
            user_num=self.dataset.user_num,
            item_num=self.dataset.source_item_num
        )
        df_target = self.dataset.get_target_df()
        mat_target = self._create_sparse_matrix(
            df_target, uid_field, iid_field,
            form=form, value_field=value_field,
            user_num=self.dataset.user_num,
            item_num=self.dataset.target_item_num
        )
        return mat_source, mat_target

    def _create_sparse_matrix(self, df_feat, source_field, target_field,
                              form='coo', value_field=None,
                              user_num=None, item_num=None):
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values

        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError(f"value_field [{value_field}] should be one of df_feat's features.")
            data = df_feat[value_field].values

        mat = coo_matrix((data, (src, tgt)), shape=(user_num, item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f"sparse matrix format [{form}] not implemented.")

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        return self.sample_func()

    def _get_sample(self):
        """
        return tensor with shape (5, batch_size)：
        [ user, source_pos, source_neg, target_pos, target_neg ]
        """
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step

        users, src_pos, src_neg, tgt_pos, tgt_neg = [], [], [], [], []
        for _, row in cur_data.iterrows():
            u, i, d = row[self.dataset.uid_field], row[self.dataset.iid_field], row[self.dataset.domain_field]
            if d == 0:
                src_pos_iid = i
                tgt_pos_iid = self._sample_pos_from_domain(u, domain=1)
            else:
                tgt_pos_iid = i
                src_pos_iid = self._sample_pos_from_domain(u, domain=0)

            src_neg_iid = self._sample_neg_from_domain(u, domain=0)
            tgt_neg_iid = self._sample_neg_from_domain(u, domain=1)
            users.append(u)
            src_pos.append(src_pos_iid)
            src_neg.append(src_neg_iid)
            tgt_pos.append(tgt_pos_iid)
            tgt_neg.append(tgt_neg_iid)

        user_tensor = torch.tensor(users, dtype=torch.long, device=self.device)
        src_pos_tensor = torch.tensor(src_pos, dtype=torch.long, device=self.device)
        src_neg_tensor = torch.tensor(src_neg, dtype=torch.long, device=self.device)
        tgt_pos_tensor = torch.tensor(tgt_pos, dtype=torch.long, device=self.device)
        tgt_neg_tensor = torch.tensor(tgt_neg, dtype=torch.long, device=self.device)

        return torch.stack([user_tensor, src_pos_tensor, src_neg_tensor, tgt_pos_tensor, tgt_neg_tensor], dim=0)

    def _sample_pos_from_domain(self, u, domain):
        if domain == 0:
            hist = self.history_source_items_per_u.get(u, None)
        else:
            hist = self.history_target_items_per_u.get(u, None)
        if not hist:
            return None
        return random.sample(hist, 1)[0]

    def _sample_neg_from_domain(self, u, domain):
        if domain == 0:
            pool = self.all_source_items
            hist = self.history_source_items_per_u.get(u, set())
        else:
            pool = self.all_target_items
            hist = self.history_target_items_per_u.get(u, set())

        neg = random.sample(pool, 1)[0]
        while neg in hist:
            neg = random.sample(pool, 1)[0]
        return neg

    def _get_history_items_u(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        domain_field = self.dataset.domain_field

        source_df = self.dataset.get_source_df()
        uid_freq_source = source_df.groupby(uid_field)[iid_field]
        for u, i_ls in uid_freq_source:
            self.history_source_items_per_u[u] = set(i_ls.values)
        target_df = self.dataset.get_target_df()
        uid_freq_target = target_df.groupby(uid_field)[iid_field]
        for u, i_ls in uid_freq_target:
            self.history_target_items_per_u[u] = set(i_ls.values)


class EvalDataLoader(AbstractDataLoader):
    """
        additional_dataset: training dataset in evaluation
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_items_per_u_source = []
        self.eval_items_per_u_target = []
        self.eval_len_list_source = []
        self.eval_len_list_target = []
        self.train_pos_len_list_source = []
        self.train_pos_len_list_target = []

        assert (self.dataset.get_source_df()[self.dataset.uid_field].nunique() ==
                self.dataset.get_target_df()[self.dataset.uid_field].nunique())
        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        # special for eval dataloader
        self.pos_items_per_u_source, self.pos_items_per_u_target = self._get_pos_items_per_u(self.eval_u)
        self.pos_items_per_u_source = self.pos_items_per_u_source.to(self.device)
        self.pos_items_per_u_target = self.pos_items_per_u_target.to(self.device)
        self._get_eval_items_per_u(self.eval_u)
        # to device
        self.eval_u = torch.tensor(self.eval_u).type(torch.LongTensor).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt_src = sum(self.train_pos_len_list_source[self.pr: self.pr + self.step])
        batch_users = self.eval_u[self.pr: self.pr + self.step]
        batch_mask_src = self.pos_items_per_u_source[:, self.inter_pr_src: self.inter_pr_src + inter_cnt_src].clone()
        batch_mask_src[0] -= self.pr
        self.inter_pr_src += inter_cnt_src

        inter_cnt_tgt = sum(self.train_pos_len_list_target[self.pr: self.pr + self.step])
        batch_mask_tgt = self.pos_items_per_u_target[:, self.inter_pr_tgt: self.inter_pr_tgt + inter_cnt_tgt].clone()
        batch_mask_tgt[0] -= self.pr
        self.inter_pr_tgt += inter_cnt_tgt

        self.pr += self.step
        return [batch_users, batch_mask_src, batch_mask_tgt]

    def _get_pos_items_per_u(self, eval_users):
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        src_uid_freq = self.additional_dataset.get_source_df().groupby(uid_field)[iid_field]
        tgt_uid_freq = self.additional_dataset.get_target_df().groupby(uid_field)[iid_field]

        u_ids_src, i_ids_src = [], []
        u_ids_tgt, i_ids_tgt = [], []

        for i, u in enumerate(eval_users):
            if u in src_uid_freq.groups:
                src_items = src_uid_freq.get_group(u).values
                self.train_pos_len_list_source.append(len(src_items))
                u_ids_src.extend([i] * len(src_items))
                i_ids_src.extend(src_items)
            else:
                self.train_pos_len_list_source.append(0)
            if u in tgt_uid_freq.groups:
                tgt_items = tgt_uid_freq.get_group(u).values
                self.train_pos_len_list_target.append(len(tgt_items))
                u_ids_tgt.extend([i] * len(tgt_items))
                i_ids_tgt.extend(tgt_items)
            else:
                self.train_pos_len_list_target.append(0)

        pos_items_per_u_source = torch.tensor([u_ids_src, i_ids_src], dtype=torch.long)
        pos_items_per_u_target = torch.tensor([u_ids_tgt, i_ids_tgt], dtype=torch.long)

        return pos_items_per_u_source, pos_items_per_u_target

    def _get_eval_items_per_u(self, eval_users):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field

        src_df = self.dataset.get_source_df()
        src_uid_freq = src_df.groupby(uid_field)[iid_field]

        tgt_df = self.dataset.get_target_df()
        tgt_uid_freq = tgt_df.groupby(uid_field)[iid_field]

        for u in eval_users:
            if u in src_uid_freq.groups:
                src_items = src_uid_freq.get_group(u).values
                self.eval_items_per_u_source.append(src_items)
                self.eval_len_list_source.append(len(src_items))
            else:
                self.eval_items_per_u_source.append(np.array([], dtype=int))
                self.eval_len_list_source.append(0)
            if u in tgt_uid_freq.groups:
                tgt_items = tgt_uid_freq.get_group(u).values
                self.eval_items_per_u_target.append(tgt_items)
                self.eval_len_list_target.append(len(tgt_items))
            else:
                self.eval_items_per_u_target.append(np.array([], dtype=int))
                self.eval_len_list_target.append(0)

        self.eval_len_list_source = np.asarray(self.eval_len_list_source)
        self.eval_len_list_target = np.asarray(self.eval_len_list_target)

    # return pos_items for each u
    def get_eval_items(self):
        return self.eval_items_per_u_source, self.eval_items_per_u_target

    def get_eval_len_list(self):
        return self.eval_len_list_source, self.eval_len_list_target

    def get_eval_users(self):
        return self.eval_u.cpu()