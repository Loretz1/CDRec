from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.domain_field = self.config['DOMAIN_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None :
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['source_inter_file_name'], self.config['target_inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(self.config['source_inter_file_name'], self.config['target_inter_file_name'])


    def load_inter_graph(self, source_file_name, target_file_name):
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        source_inter_file = os.path.join(self.dataset_path, source_file_name)
        df_source = pd.read_csv(source_inter_file, usecols=cols, sep=self.config['field_separator'])
        if not df_source.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(source_file_name))
        df_source[self.domain_field] = 0

        target_inter_file = os.path.join(self.dataset_path, target_file_name)
        df_target = pd.read_csv(target_inter_file, usecols=cols, sep=self.config['field_separator'])
        if not df_target.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(target_file_name))
        df_target[self.domain_field] = 1

        self.df = pd.concat([df_source, df_target], ignore_index=True)
        assert max(df_source[self.uid_field].values) == max(df_target[self.uid_field].values)
        self.user_num = int(max(df_source[self.uid_field].values)) + 1
        self.source_item_num = int(max(df_source[self.iid_field].values)) + 1
        self.target_item_num = int(max(df_target[self.iid_field].values)) + 1

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        nxt = RecDataset(self.config, new_df)

        nxt.source_item_num = self.source_item_num
        nxt.target_item_num = self.target_item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_source_item_num(self):
        return self.source_item_num

    def get_target_item_num(self):
        return self.target_item_num

    def get_source_df(self):
        return self.df[self.df[self.domain_field] == 0]

    def get_target_df(self):
        return self.df[self.df[self.domain_field] == 1]

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [f"Dataset name: {self.dataset_name}"]

        self.inter_num = len(self.df)
        info.append(f"The number of interactions: {self.inter_num}")
        uni_u = pd.unique(self.df[self.uid_field])
        tmp_user_num = len(uni_u)
        info.append(f"The number of users: {tmp_user_num}")
        source_df = self.get_source_df()
        target_df = self.get_target_df()
        source_inter_num = len(source_df)
        target_inter_num = len(target_df)
        source_users = pd.unique(source_df[self.uid_field])
        avg_actions_of_source_users = source_inter_num / len(source_users) if len(source_users) > 0 else 0
        info.append(f"Average actions of source users: {avg_actions_of_source_users:.2f}")
        target_users = pd.unique(target_df[self.uid_field])
        avg_actions_of_target_users = target_inter_num / len(target_users) if len(target_users) > 0 else 0
        info.append(f"Average actions of target users: {avg_actions_of_target_users:.2f}")
        source_items = pd.unique(source_df[self.iid_field])
        target_items = pd.unique(target_df[self.iid_field])
        info.append(f"The number of source items: {len(source_items)}")
        info.append(f"The number of target items: {len(target_items)}")
        avg_actions_of_source_items = source_inter_num / len(source_items) if len(source_items) > 0 else 0
        avg_actions_of_target_items = target_inter_num / len(target_items) if len(target_items) > 0 else 0
        info.append(f"Average actions of source items: {avg_actions_of_source_items:.2f}")
        info.append(f"Average actions of target items: {avg_actions_of_target_items:.2f}")
        total_item_num = len(source_items) + len(target_items)
        sparsity = 1 - self.inter_num / (tmp_user_num * total_item_num)
        info.append(f"The sparsity of the dataset: {sparsity * 100:.2f}%")
        return '\n'.join(info)

