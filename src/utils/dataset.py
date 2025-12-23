from logging import getLogger
import os
import json
import pandas as pd
import numpy as np
from utils.enum_type import TrainDataLoaderState
import random

class RecDataset(object):
    """
    功能：
        从 joint dataset 目录加载跨域推荐所需的数据文件，
        并按训练阶段组织为可直接使用的交互数据视图。

    数据来源：
        从以下路径读取数据：
            {data_path}/{dataset}/{src}+{tgt}/{only_overlap_users|all_users}/{split_dir}/

        读取的文件包括：
            - all_users.json
            - id_mapping.json
            - train_src.pkl
            - train_tgt.pkl
            - valid_cold_tgt.pkl
            - test_cold_tgt.pkl
            - valid_warm_tgt.pkl
            - test_warm_tgt.pkl
            - modality_emb_src/*.npy（可选）
            - modality_emb_tgt/*.npy（可选）

    数据处理：
        - 加载 src / tgt 的交互数据（DataFrame，列为 [user, item]）
        - 加载并对齐多模态 embedding（为 0 号 padding 添加全 0 向量）
        - 从训练集构建正样本集合：
            * positive_items_src
            * positive_items_tgt
        - 根据训练阶段需要构建派生数据：
            * train_both（src + tgt，带 domain 标记）
            * train_overlap（仅 overlap 用户的 src + tgt）
            * train_overlap_user（仅 overlap 用户 ID）
    对外功能：
        - split()：
            返回 train / valid / test 三个 RecDataset 视图
        - set_state_for_train(state)：
            根据 TrainDataLoaderState 选择当前活跃的交互 DataFrame
        - __getitem__ / __len__：
            按当前 state 提供样本访问接口
    """
    # Fields for train/valid/test
    META_FIELDS = [
        "config", "logger",
        "all_users",
        "num_users_overlap", "num_users_src", "num_users_tgt",
        "num_items_src", "num_items_tgt",
    ]

    # Optional Basic Field For train/valid/test
    BASIC_DATA_FIELDS = [
        "modality_embeddings",
        "positive_items_src", "positive_items_tgt",
        "id_mapping"
    ]

    # Optional DF Field For train/valid/test, create a dict named "df" for storage of DataFrame
    DATAFRAME_FIELDS = [
        "train_src", "train_tgt", "train_both", "train_overlap", "train_overlap_user",
        "valid_cold_tgt", "test_cold_tgt",
        "valid_warm_tgt", "test_warm_tgt",
    ]
    def __init__(self, config, skip=False):
        self.config = config
        self.logger = getLogger()
        self.df = {}

        if skip:
            return

        split_dir = (
            f"WarmValid{config['warm_valid_ratio']}_"
            f"WarmTest{config['warm_test_ratio']}_"
            f"ColdValid{config['t_cold_valid']}_"
            f"ColdTest{config['t_cold_test']}"
        )
        if config['only_overlap_users']:
            split_dir += f'_{config["k_cores"]}cores'
        dataset_path = os.path.join(
            self.config['data_path'],
            config['dataset'],
            '+'.join(config['domains']),
            "only_overlap_users" if config['only_overlap_users'] else "all_users",
            split_dir
        )

        self.all_users, self.id_mapping, self.train_src, self.train_tgt, self.valid_cold_tgt, self.test_cold_tgt, self.valid_warm_tgt, self.test_warm_tgt, self.modality_embeddings = self._load_data(
            dataset_path)

        self.modality_embeddings = self._add_padding_for_modality_embeddings(self.modality_embeddings)

        self.positive_items_src, self.positive_items_tgt = self._get_positive_items_set(self.train_src, self.train_tgt)

        all_train_stages = [TrainDataLoaderState[i['state']] for i in self.config['training_stages']]
        if TrainDataLoaderState.BOTH in all_train_stages:
            self.train_both = self._build_train_both_df(self.train_src, self.train_tgt)
        if TrainDataLoaderState.OVERLAP in all_train_stages:
            self.train_overlap = self._build_train_overlap_df(self.train_src, self.train_tgt, len(self.all_users['overlap_users']))
        if TrainDataLoaderState.OVERLAP_USER in all_train_stages:
            self.train_overlap_user = pd.DataFrame(range(1, len(self.all_users['overlap_users']) + 1), columns=['user'])

        self.num_users_overlap = len(self.all_users['overlap_users'])
        self.num_users_src = len(self.all_users['overlap_users']) + len(self.all_users['valid_cold_users']) + len(self.all_users['test_cold_users']) + len(self.all_users['src_only_users'])
        self.num_users_tgt = len(self.all_users['overlap_users']) + len(self.all_users['tgt_only_users'])
        self.num_items_src = len(self.id_mapping['src']['item2id'])
        self.num_items_tgt = len(self.id_mapping['tgt']['item2id'])

    def _load_data(self, dataset_path):
        self.logger.info('[TRAINING] Loading dataset from {}'.format(dataset_path))

        all_users_path = os.path.join(dataset_path, 'all_users.json')
        id_mapping_path = os.path.join(dataset_path, 'id_mapping.json')
        with open(all_users_path, 'r') as f:
            all_users = json.load(f)
        with open(id_mapping_path, 'r') as f:
            id_mapping = json.load(f)

        train_src = pd.read_pickle(os.path.join(dataset_path, 'train_src.pkl'))
        train_tgt = pd.read_pickle(os.path.join(dataset_path, 'train_tgt.pkl'))
        valid_cold_tgt = pd.read_pickle(os.path.join(dataset_path, 'valid_cold_tgt.pkl'))
        test_cold_tgt = pd.read_pickle(os.path.join(dataset_path, 'test_cold_tgt.pkl'))
        valid_warm_tgt = pd.read_pickle(os.path.join(dataset_path, 'valid_warm_tgt.pkl'))
        test_warm_tgt = pd.read_pickle(os.path.join(dataset_path, 'test_warm_tgt.pkl'))

        modality_embeddings = {"src": {},"tgt": {}}
        for modality in self.config['modalities']:
            if not modality.get('enabled', False):
                continue
            emb_name = modality['name'] + '_final_emb_'+ str(modality['emb_pca'])+ '.npy'
            src_emb_path = os.path.join(dataset_path, 'modality_emb_src', emb_name)
            tgt_emb_path = os.path.join(dataset_path, 'modality_emb_tgt', emb_name)

            modality_embeddings["src"][modality['name']] = np.load(src_emb_path)
            modality_embeddings["tgt"][modality['name']] = np.load(tgt_emb_path)

        self.logger.info('[TRAINING] Dataset loading finished.')

        return (all_users, id_mapping, train_src, train_tgt, valid_cold_tgt, test_cold_tgt, valid_warm_tgt,
                test_warm_tgt, modality_embeddings)

    def _add_padding_for_modality_embeddings(self, modality_embeddings):
        for domain in ['src', 'tgt']:
            for modality_name, emb in modality_embeddings[domain].items():
                emb_dim = emb.shape[1]
                pad_embedding = np.zeros((1, emb_dim), dtype=emb.dtype)
                modality_embeddings[domain][modality_name] = np.concatenate(
                    [pad_embedding, emb], axis=0
                )
        return modality_embeddings

    def _get_positive_items_set(self, train_src, train_tgt):
        positive_items_src = {}
        positive_items_tgt = {}
        for user, item in train_src.values:
           if user not in positive_items_src:
               positive_items_src[user] = set()
           positive_items_src[user].add(item)
        for user, item in train_tgt.values:
           if user not in positive_items_tgt:
               positive_items_tgt[user] = set()
           positive_items_tgt[user].add(item)
        return positive_items_src, positive_items_tgt

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
        """
        功能：
            将当前 RecDataset 按用途拆分为训练、验证和测试三个数据集视图。
        处理逻辑：
            - 训练集（train_dataset）：
                * 保留训练阶段使用的数据：
                  train_src / train_tgt / train_both /
                  train_overlap / train_overlap_user (如果构建了的话)
                * 同时保留：
                  modality_embeddings、positive_items、id_mapping
            - 验证集（valid_dataset）：
                * 使用目标域验证数据：
                  valid_cold_tgt、valid_warm_tgt
                * 重命名为：
                  cold_tgt、warm_tgt
            - 测试集（test_dataset）：
                * 使用目标域测试数据：
                  test_cold_tgt、test_warm_tgt
                * 重命名为：
                  cold_tgt、warm_tgt
        输出：
            train_dataset : RecDataset
            valid_dataset : RecDataset
            test_dataset  : RecDataset
        """
        train_dataset = self.copy(
            keep_fields=["train_both", "train_src", "train_tgt", "train_overlap", "train_overlap_user",
                         "modality_embeddings",
                         "positive_items_src", "positive_items_tgt", "id_mapping"])
        valid_dataset = self.copy(keep_fields=["valid_cold_tgt", "valid_warm_tgt",
                                               "positive_items_tgt", "id_mapping"])
        test_dataset = self.copy(keep_fields=["test_cold_tgt", "test_warm_tgt",
                                              "positive_items_tgt", "id_mapping"])
        valid_dataset.df["cold_tgt"] = valid_dataset.df.pop("valid_cold_tgt")
        valid_dataset.df["warm_tgt"] = valid_dataset.df.pop("valid_warm_tgt")
        test_dataset.df["cold_tgt"] = test_dataset.df.pop("test_cold_tgt")
        test_dataset.df["warm_tgt"] = test_dataset.df.pop("test_warm_tgt")
        return train_dataset, valid_dataset, test_dataset

    def copy(self, keep_fields):
        """
        功能：
            基于当前 RecDataset 创建一个浅拷贝的数据集对象，
            仅保留指定字段，用于构造 train / valid / test 数据视图。
            由Split方法调用。
        处理逻辑：
            - 创建一个新的 RecDataset 实例（skip=True，不重新加载数据）
            - 复制所有 META_FIELDS（配置、用户与规模信息）
            - 仅当字段名在 keep_fields 中时，才复制：
                * BASIC_DATA_FIELDS 中的属性
                * DATAFRAME_FIELDS 中对应的 DataFrame 到 new_dataset.df
        输入：
            keep_fields: List[str]
                需要保留的字段名称列表
        输出：
            new_dataset: RecDataset
                仅包含指定字段的数据集视图
        """
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
        """
        功能：
            在 split() 之后生成的训练数据集上，
            根据训练阶段状态选择当前使用的训练数据视图。
        处理逻辑：
            - 根据传入的 TrainDataLoaderState，
              将对应的 DataFrame 设为当前活跃数据（_active_df）：
                * BOTH          → train_both
                * SOURCE        → train_src
                * TARGET        → train_tgt
                * OVERLAP       → train_overlap
                * OVERLAP_USER  → train_overlap_user
            - 重置活跃 DataFrame 的索引
        输入：
            state: TrainDataLoaderState
                指定当前训练阶段的数据使用方式
                （仅适用于 split() 返回的 train_dataset）
        输出：
            self
                返回自身，支持链式调用
        """
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
        elif state == TrainDataLoaderState.OVERLAP_USER:
            self._active_df = self.df["train_overlap_user"]
        else:
            raise ValueError(f"Unsupported state: {state}")

        self._active_df.reset_index(drop=True, inplace=True)
        return self

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


