from logging import getLogger
import os
import json
import pandas as pd
import numpy as np
from utils.enum_type import TrainDataLoaderState, EvalDataLoaderState
import random


class RecDataset(object):
    # Fields for train/valid/test
    META_FIELDS = [
        "config", "logger",
        "all_users",
        "num_users_overlap", "num_users_src", "num_users_tgt",
        "num_items_src", "num_items_tgt",
    ]

    # Optional Basic Field For train/valid/test
    BASIC_DATA_FIELDS = [
        "sent_emb_dim", "sent_embeddings",
        "positive_items_src", "positive_items_tgt",
        "id2user_or_item"
    ]

    # Optional DF Field For train/valid/test, create a dict named "df" for storage of DataFrame
    DATAFRAME_FIELDS = [
        "train_src", "train_tgt", "train_both", "train_overlap",
        "valid_cold_tgt", "test_cold_tgt",
        "valid_warm_tgt", "test_warm_tgt",
    ]
    def __init__(self, config, skip=False):
        self.config = config
        self.logger = getLogger()
        self.df = {}

        if skip:
            return

        dataset_path = os.path.join(self.config['data_path'], config['dataset'], '+'.join(config['domains']),
                                    'only_overlap_users' if config['only_overlap_users'] else 'all_users')

        self.sent_emb_dim = self.config['sent_emb_pca'] if 'sent_emb_pca' in self.config else self.config[
            'sent_emb_dim']
        all_item_seqs, id_mapping, sent_embeddings = self._load_data(dataset_path)
        pad_embedding = np.zeros((1, self.sent_emb_dim), dtype=np.float32)
        self.sent_embeddings = {
            domain: np.concatenate([pad_embedding, sent_embeddings[domain]], axis=0)
            for domain in ['src', 'tgt']
        }

        self.all_users, user2id_src, user2id_tgt, id2user_src, id2user_tgt = self._split_users(all_item_seqs)

        (self.train_src, self.train_tgt, self.valid_cold_tgt, self.test_cold_tgt, self.valid_warm_tgt,
         self.test_warm_tgt), (self.positive_items_src, self.positive_items_tgt) = self._split_interation(
            all_item_seqs, id_mapping, user2id_src, user2id_tgt)

        all_train_stages = [TrainDataLoaderState[i['state']] for i in self.config['training_stages']]
        if TrainDataLoaderState.BOTH in all_train_stages:
            self.train_both = self._build_train_both_df(self.train_src, self.train_tgt)
        if TrainDataLoaderState.OVERLAP in all_train_stages:
            self.train_overlap = self._build_train_overlap_df(self.train_src, self.train_tgt, len(self.all_users['overlap_users']))

        #   - The 5 user groups are mutually exclusive (no overlaps).
        #   - overlap_users: assigned to the same ID range [1 .. len(overlap_users)] in BOTH src and tgt domains.
        #                    → These are warm users shared across domains.
        #   - src_only_users + valid_cold_users + test_cold_users:
        #                    → These are single-domain users of source domain (only src has interactions).
        #                    → Reindexed starting from len(overlap_users) + 1 in source domain.
        #   - tgt_only_users: single-domain users of target domain (only tgt has interactions),
        #                    → Reindexed starting from len(overlap_users) + 1 in target domain.
        self.all_users = {
            'overlap_users': {user2id_src[u] for u in self.all_users['overlap_users']},
            'valid_cold_users': {user2id_src[u] for u in self.all_users['valid_cold_users']},
            'test_cold_users': {user2id_src[u] for u in self.all_users['test_cold_users']},
            'src_only_users': {user2id_src[u] for u in self.all_users['src_only_users']},
            'tgt_only_users': {user2id_tgt[u] for u in self.all_users['tgt_only_users']}
        }
        self.id2user_or_item ={
            "src": {
                "id2user": id2user_src,
                "id2item": id_mapping['src']['id2item']
            },
            "tgt": {
                "id2user": id2user_tgt,
                "id2item": id_mapping['tgt']['id2item']
            }
        }
        self.num_users_overlap = len(self.all_users['overlap_users'])
        self.num_users_src = len(self.all_users['overlap_users']) + len(self.all_users['valid_cold_users']) + len(self.all_users['test_cold_users']) + len(self.all_users['src_only_users'])
        self.num_users_tgt = len(self.all_users['overlap_users']) + len(self.all_users['tgt_only_users'])
        self.num_items_src = len(id_mapping['src']['item2id'])
        self.num_items_tgt = len(id_mapping['tgt']['item2id'])

    def _load_data(self, dataset_path):
        self.logger.info('[TRAINING] Loading dataset from {}'.format(dataset_path))

        # Load Data
        all_item_seqs = {}
        id_mapping = {}
        sent_embeddings ={}
        for domain in ['src', 'tgt']:
            path = os.path.join(dataset_path, domain)
            with open(os.path.join(path, 'all_item_seqs.json'), 'r') as f:
                all_item_seqs[domain] = json.load(f)
                if self.config.get("shuffle_user_sequence", True):
                    for uid, items in all_item_seqs[domain].items():
                        random.shuffle(items)
            with open(os.path.join(path, 'id_mapping.json'), 'r') as f:
                id_mapping[domain] = json.load(f)
            sent_embeddings[domain] = np.load(os.path.join(path, f'final_sent_embeddings_{self.sent_emb_dim}.npy'))

        return all_item_seqs, id_mapping, sent_embeddings

    def _split_users(self, all_item_seqs):
        # Split User Set
        users_src = set(all_item_seqs['src'].keys())
        users_tgt = set(all_item_seqs['tgt'].keys())
        overlap_users = users_src & users_tgt
        src_only_users = users_src - overlap_users
        tgt_only_users = users_tgt - overlap_users

        overlap_list = np.random.permutation(list(overlap_users))
        num_overlap = len(overlap_list)
        num_valid_cold = int(num_overlap * self.config['t_cold_valid'])
        num_test_cold = int(num_overlap * self.config['t_cold_test'])
        assert num_valid_cold + num_test_cold <= num_overlap, "Sum of t_cold_valid and t_cold_test causes user overflow."
        valid_cold_users = set(overlap_list[:num_valid_cold])
        test_cold_users = set(overlap_list[num_valid_cold:num_valid_cold + num_test_cold])
        overlap_users = set(overlap_list[num_valid_cold + num_test_cold:])

        all_users = {
            'overlap_users': overlap_users,
            'valid_cold_users': valid_cold_users,
            'test_cold_users': test_cold_users,
            'src_only_users': src_only_users,
            'tgt_only_users': tgt_only_users
        }

        # Reindex User ID for Each Domain
        # Overlapped User: 1 -> num(overlap_users)
        # Single Domain User: num(overlap_users) + 1 -> num(domain_user)
        user2id_src, user2id_tgt = {}, {}
        id2user_src, id2user_tgt = ["PAD"], ["PAD"]
        for i, u in enumerate(all_users['overlap_users'], start=1):
            user2id_src[u] = i
            id2user_src.append(u)
            user2id_tgt[u] = i
            id2user_tgt.append(u)
        for i, u in enumerate(all_users['valid_cold_users'] | all_users['test_cold_users'] | all_users[
            'src_only_users'], start=len(all_users['overlap_users']) + 1):
            user2id_src[u] = i
            id2user_src.append(u)
        for i, u in enumerate(all_users['tgt_only_users'], start=len(all_users['overlap_users']) + 1):
            user2id_tgt[u] = i
            id2user_tgt.append(u)
        return all_users, user2id_src, user2id_tgt, id2user_src, id2user_tgt

    def _split_interation(self, all_item_seqs, id_mapping, user2id_src, user2id_tgt):
        train_src, train_tgt = [], []
        valid_cold_tgt, test_cold_tgt = [], []
        valid_warm_tgt, test_warm_tgt = [], []
        positive_items_src, positive_items_tgt = {}, {}

        for raw_uid, item_seq in all_item_seqs['src'].items():
            uid = user2id_src[raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping['src']['item2id'][raw_iid]
                train_src.append([uid, iid])
                if uid not in positive_items_src:
                    positive_items_src[uid] = set()
                positive_items_src[uid].add(iid)

        for raw_uid, item_seq in all_item_seqs['tgt'].items():
            if raw_uid in self.all_users["valid_cold_users"]:
                uid = user2id_src[raw_uid]
                for raw_iid in item_seq:
                    iid = id_mapping['tgt']['item2id'][raw_iid]
                    valid_cold_tgt.append([uid, iid])
            elif raw_uid in self.all_users["test_cold_users"]:
                uid = user2id_src[raw_uid]  # cold users only have src IDs
                for raw_iid in item_seq:
                    iid = id_mapping['tgt']['item2id'][raw_iid]
                    test_cold_tgt.append([uid, iid])
            elif raw_uid in user2id_tgt:
                uid = user2id_tgt[raw_uid]
                # if not self.config['warm_eval']:
                #     for raw_iid in item_seq:
                #         iid = id_mapping['tgt']['item2id'][raw_iid]
                #         train_tgt.append([uid, iid])
                #         if uid not in positive_items_tgt:
                #             positive_items_tgt[uid] = set()
                #         positive_items_tgt[uid].add(iid)
                # else:
                seq_len = len(item_seq)
                num_test = max(1, int(seq_len * self.config['warm_test_ratio']))
                num_valid = max(1, int(seq_len * self.config['warm_valid_ratio']))
                num_train = seq_len - num_valid - num_test
                for raw_iid in item_seq[:num_train]:
                    train_tgt.append([uid, id_mapping['tgt']['item2id'][raw_iid]])
                    if uid not in positive_items_tgt:
                        positive_items_tgt[uid] = set()
                    positive_items_tgt[uid].add(id_mapping['tgt']['item2id'][raw_iid])
                for raw_iid in item_seq[num_train:num_train+num_valid]:
                    valid_warm_tgt.append([uid, id_mapping['tgt']['item2id'][raw_iid]])
                for raw_iid in item_seq[num_train+num_valid:]:
                    test_warm_tgt.append([uid, id_mapping['tgt']['item2id'][raw_iid]])

        return (self._to_df(train_src), self._to_df(train_tgt), self._to_df(valid_cold_tgt), self._to_df(
            test_cold_tgt), self._to_df(valid_warm_tgt), self._to_df(test_warm_tgt)), (positive_items_src,positive_items_tgt)

    def _to_df(self, inter):
        return pd.DataFrame(inter, columns=['user', 'item'])

    def _build_train_both_df(self, df_src, df_tgt):
        src_df = df_src.copy()
        src_df["domain"] = 0

        tgt_df = df_tgt.copy()
        tgt_df["domain"] = 1

        return pd.concat([src_df, tgt_df], axis=0, ignore_index=True)

    def _build_train_overlap_df(self, df_src, df_tgt, overlap_u_max_id):
        src_df = df_src.copy()
        src_df["domain"] = 0
        src_df = src_df[src_df["user"] <= overlap_u_max_id]

        tgt_df = df_tgt.copy()
        tgt_df["domain"] = 1
        tgt_df = tgt_df[tgt_df["user"] <= overlap_u_max_id]

        return pd.concat([src_df, tgt_df], axis=0, ignore_index=True)

    def split(self):
        train_dataset = self.copy(
            keep_fields=["train_both", "train_src", "train_tgt", "train_overlap",
                         "sent_emb_dim", "sent_embeddings",
                         "positive_items_src", "positive_items_tgt", "id2user_or_item"])
        valid_dataset = self.copy(keep_fields=["valid_cold_tgt", "valid_warm_tgt",
                                               "positive_items_tgt"])
        test_dataset = self.copy(keep_fields=["test_cold_tgt", "test_warm_tgt",
                                              "positive_items_tgt"])
        valid_dataset.df["cold_tgt"] = valid_dataset.df.pop("valid_cold_tgt")
        valid_dataset.df["warm_tgt"] = valid_dataset.df.pop("valid_warm_tgt")
        test_dataset.df["cold_tgt"] = test_dataset.df.pop("test_cold_tgt")
        test_dataset.df["warm_tgt"] = test_dataset.df.pop("test_warm_tgt")
        return train_dataset, valid_dataset, test_dataset

    def copy(self, keep_fields):
        new_dataset = RecDataset(self.config, skip=True)
        for key in self.META_FIELDS:
            setattr(new_dataset, key, getattr(self, key))
        for key in self.BASIC_DATA_FIELDS:
            if key in keep_fields and hasattr(self, key):
                setattr(new_dataset, key, getattr(self, key))
        for key in self.DATAFRAME_FIELDS:
            if key in keep_fields and hasattr(self, key):
                new_dataset.df[key] = getattr(self, key)
        return new_dataset

    def set_state_for_train(self, state):
        assert isinstance(state, TrainDataLoaderState), "state must be a TrainDataLoaderState"
        self.state = state

        if state == TrainDataLoaderState.BOTH:
            self._active_df = self.df["train_both"]
        elif state == TrainDataLoaderState.SOURCE:
            self._active_df = self.df["train_src"]
        elif state == TrainDataLoaderState.TARGET:
            self._active_df = self.df["train_tgt"]
        elif state == TrainDataLoaderState.OVERLAP:
            self._active_df = self.df["train_overlap"]
        else:
            raise ValueError(f"Unsupported state: {state}")

        self._active_df.reset_index(drop=True, inplace=True)
        return self

    def get_active_df(self):
        return self._active_df

    def shuffle(self):
        if hasattr(self, "_active_df") and isinstance(self._active_df, pd.DataFrame):
            # self._active_df = self._active_df.sample(frac=1, replace=False).reset_index(drop=True)
            shuffled_index = np.random.permutation(len(self._active_df))
            self._active_df.iloc[:] = self._active_df.iloc[shuffled_index].values
        else:
            self.logger.warning("No active DataFrame to shuffle. Did you call set_state() first?")
        return self

    def __len__(self):
        if hasattr(self, '_active_df'):
            return len(self._active_df)
        raise RuntimeError("Call set_state() before using dataset")

    def __getitem__(self, idx):
        if not hasattr(self, '_active_df'):
            raise RuntimeError("Call set_state() before fetching items")
        return self._active_df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        lines = [
            f"Dataset: {self.config.get('dataset', 'Unknown')} "
            f"({' + '.join(self.config.get('domains', []))})",
            f"Users → src: {self.num_users_src}, tgt: {self.num_users_tgt}, overlap: {self.num_users_overlap}",
            f"Items → src: {self.num_items_src}, tgt: {self.num_items_tgt}",
        ]

        # if self.config.get('cold_start_eval', False):
        valid_cold = len(self.all_users.get('valid_cold_users', []))
        test_cold = len(self.all_users.get('test_cold_users', []))
        lines.append(f"Cold-start evaluation enabled → Valid-cold: {valid_cold}, Test-cold: {test_cold}")
        # else:
        #     lines.append("Cold-start evaluation disabled")

        # if self.config.get('warm_eval', False):
        lines.append(
            f"Warm-start split → valid_ratio={self.config.get('warm_valid_ratio', 0)}, "
            f"test_ratio={self.config.get('warm_test_ratio', 0)}"
        )
        # else:
        #     lines.append("Warm-start evaluation disabled")

        if not len(self.df):
            lines.append("Interaction splits:")
            for name in ["train_src", "train_tgt", "valid_cold_tgt", "valid_warm_tgt", "test_cold_tgt", "test_warm_tgt"]:
                if hasattr(self, name):
                    lines.append(f"  └─ {name}: {len(getattr(self, name))}")

        if hasattr(self, 'sent_emb_dim'):
            lines.append(f"Embedding dimension: {self.sent_emb_dim}")

        src_interactions = len(self.train_src) if hasattr(self, "train_src") else len(self.df["train_src"]) if "train_src" in self.df else 0
        tgt_interactions = 0
        if not len(self.df):
            for name in ["train_tgt", "warm_tgt", "warm_tgt"]:
                if hasattr(self, name):
                    tgt_interactions += len(getattr(self, name))
        else:
            for name in ["train_tgt", "warm_tgt", "warm_tgt"]:
                if name in self.df:
                    tgt_interactions += len(self.df[name])

        if self.num_users_src > 0 and self.num_items_src > 0:
            sparsity_src = 1 - src_interactions / (self.num_users_src * self.num_items_src)
            lines.append(f"Sparsity (source): {sparsity_src * 100:.2f}%")
        if self.num_users_tgt > 0 and self.num_items_tgt > 0:
            sparsity_tgt = 1 - tgt_interactions / (self.num_users_tgt * self.num_items_tgt)
            lines.append(f"Sparsity (target): {sparsity_tgt * 100:.2f}%")

        return "\n".join(lines)


