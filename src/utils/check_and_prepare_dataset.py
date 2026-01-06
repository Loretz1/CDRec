import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from logging import getLogger
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.amazon_data_processor import AmazonDataProcessor
from utils.amazon_modality_processor import AmazonModalityProcessor

logger = getLogger()

def load_all_item_seqs(src_processed_dir, tgt_processed_dir):
    with open(os.path.join(src_processed_dir, 'all_item_seqs.json'), 'r') as f:
        src_all_item_seqs = json.load(f)
    with open(os.path.join(tgt_processed_dir, 'all_item_seqs.json'), 'r') as f:
        tgt_all_item_seqs = json.load(f)
    all_item_seqs = {
        "src": src_all_item_seqs,
        "tgt": tgt_all_item_seqs
    }
    return all_item_seqs

def split_users_and_reindex(config, joint_path, all_item_seqs):
    """
    功能：
        对跨域用户进行划分并重新编号（reindex），
        构建跨域推荐中统一、可对齐的 user / item ID 空间。

    用户划分说明：
        - overlap_users：
            同时出现在源域（src）和目标域（tgt）的用户
        - valid_cold_users / test_cold_users：
            从 overlap_users 中按比例划分得到，
            在目标域中作为冷启动用户用于评估
        - src_only_users：
            仅出现在源域的用户
        - tgt_only_users：
            仅出现在目标域的用户
    ID 对齐规则：
        - overlap_users：
            在 src / tgt 两个域中分配到相同的 user ID 区间
            [1 .. len(overlap_users)]，
            表示跨域共享的 warm 用户
        - src_only_users + valid_cold_users + test_cold_users：
            仅在源域（src）中存在交互的用户
            在 src 域中从 len(overlap_users) + 1 开始重新编号
        - tgt_only_users：
            仅在目标域（tgt）中存在交互的用户
            在 tgt 域中从 len(overlap_users) + 1 开始重新编号
        - item ID 在各自域内独立编号，0 号为 padding
    说明：
        - 所有用户分组互斥
        - 划分结果与 ID 映射会缓存到 joint_path，支持复用
    输入：
        config: dict
            用户划分比例配置
        joint_path: str
            联合数据集存储路径
        all_item_seqs: dict
            原始 src / tgt 用户交互序列
    输出：
        all_users: dict
            使用新 user ID 的用户分组结果
        id_mapping: dict
            src / tgt 域的 user / item ID 映射表
    """
    #   - The 5 user groups are mutually exclusive (no overlaps).
    #   - overlap_users: assigned to the same ID range [1 .. len(overlap_users)] in BOTH src and tgt domains.
    #                    → These are warm users shared across domains.
    #   - src_only_users + valid_cold_users + test_cold_users:
    #                    → These are single-domain users of source domain (only src has interactions).
    #                    → Reindexed starting from len(overlap_users) + 1 in source domain.
    #   - tgt_only_users: single-domain users of target domain (only tgt has interactions),
    #                    → Reindexed starting from len(overlap_users) + 1 in target domain.
    all_users_file_path = os.path.join(joint_path, "all_users.json")
    id_mapping_file_path = os.path.join(joint_path, "id_mapping.json")
    if os.path.exists(all_users_file_path) and os.path.exists(id_mapping_file_path):
        logger.info('[DATASET] User splitting and reindex have been done...')
        with open(all_users_file_path, 'r') as f:
            all_users = json.load(f)
        with open(id_mapping_file_path, 'r') as f:
            id_mapping = json.load(f)
        return all_users, id_mapping

    users_src = sorted(all_item_seqs['src'].keys())
    users_tgt = sorted(all_item_seqs['tgt'].keys())

    users_src_set = set(users_src)
    users_tgt_set = set(users_tgt)
    overlap_users = sorted(users_src_set & users_tgt_set)
    src_only_users = sorted(users_src_set - set(overlap_users))
    tgt_only_users = sorted(users_tgt_set - set(overlap_users))

    rng = np.random.RandomState(999)
    overlap_list = rng.permutation(overlap_users)
    num_overlap = len(overlap_list)
    num_valid_cold = int(num_overlap * config['t_cold_valid'])
    num_test_cold = int(num_overlap * config['t_cold_test'])
    assert num_valid_cold + num_test_cold <= num_overlap, "Sum of t_cold_valid and t_cold_test causes user overflow."
    valid_cold_users = sorted(overlap_list[:num_valid_cold])
    test_cold_users = sorted(overlap_list[num_valid_cold:num_valid_cold + num_test_cold])
    overlap_users = sorted(overlap_list[num_valid_cold + num_test_cold:])

    all_users = {
        'overlap_users': overlap_users,
        'valid_cold_users': valid_cold_users,
        'test_cold_users': test_cold_users,
        'src_only_users': src_only_users,
        'tgt_only_users': tgt_only_users
    }

    id_mapping = {'src': {'user2id': {}, 'item2id': {}, 'id2user': ['[PAD]'], 'id2item': ['[PAD]']},
                  'tgt': {'user2id': {}, 'item2id': {}, 'id2user': ['[PAD]'], 'id2item': ['[PAD]']}}
    for i, u in enumerate(all_users['overlap_users'], start=1):
        id_mapping['src']['user2id'][u] = i
        id_mapping['src']['id2user'].append(u)
        id_mapping['tgt']['user2id'][u] = i
        id_mapping['tgt']['id2user'].append(u)
    src_extra_users = (
            all_users['valid_cold_users']
            + all_users['test_cold_users']
            + all_users['src_only_users']
    )
    for i, u in enumerate(src_extra_users, start=len(all_users['overlap_users']) + 1):
        id_mapping['src']['user2id'][u] = i
        id_mapping['src']['id2user'].append(u)
    for i, u in enumerate(all_users['tgt_only_users'], start=len(all_users['overlap_users']) + 1):
        id_mapping['tgt']['user2id'][u] = i
        id_mapping['tgt']['id2user'].append(u)

    for domain in ['src', 'tgt']:
        all_items = set()
        for items in all_item_seqs[domain].values():
            all_items.update(items)
        for item in sorted(all_items):
            id_mapping[domain]['item2id'][item] = len(id_mapping[domain]['id2item'])
            id_mapping[domain]['id2item'].append(item)

    all_users = {
        'overlap_users': [id_mapping['src']['user2id'][u] for u in all_users['overlap_users']],
        'valid_cold_users': [id_mapping['src']['user2id'][u] for u in all_users['valid_cold_users']],
        'test_cold_users': [id_mapping['src']['user2id'][u] for u in all_users['test_cold_users']],
        'src_only_users': [id_mapping['src']['user2id'][u] for u in all_users['src_only_users']],
        'tgt_only_users': [id_mapping['tgt']['user2id'][u] for u in all_users['tgt_only_users']]
    }

    logger.info('[DATASET] Saving user splitting and id mapping...')
    with open(all_users_file_path, 'w') as f:
        json.dump(all_users, f)
    with open(id_mapping_file_path, 'w') as f:
        json.dump(id_mapping, f)
    return all_users, id_mapping

def filter_overlap_users(all_item_seqs, k, joint_dataset_name):
    """
    功能：
        对源域（src）和目标域（tgt）的重叠用户执行双域 K-Core 过滤，
        用于构建交互密度较高的 overlap 用户子集。
    说明：
        - 仅保留同时出现在 src 和 tgt 中的用户
        - 要求用户在两个域中的交互数均不少于 k
        - 同时对两个域的 item 施加最小出现次数约束（≥ k）
        - 过滤过程在 src / tgt 两个域上交替执行，直至收敛
    使用场景：
        - 当 config['only_overlap_users'] == True 时启用
        - 用于复现部分工作中常见的 dual-domain k-core 设定
    输入：
        all_item_seqs: dict
            原始 src / tgt 用户交互序列
        k: int
            K-Core 阈值
        joint_dataset_name: str
            联合数据集名称（用于日志输出）
    输出：
        all_item_seqs: dict
            经过 K-Core 过滤后的 src / tgt 用户交互序列
    """
    def dual_domain_kcore_loop(src, tgt, k):
        prev_src, prev_tgt = None, None
        iteration = 0
        while prev_src != src or prev_tgt != tgt:
            iteration += 1
            prev_src, prev_tgt = src, tgt
            overlap_users = [u for u in src.keys() if u in tgt]
            src = {u: seq for u, seq in src.items() if u in overlap_users}
            tgt = {u: seq for u, seq in tgt.items() if u in overlap_users}
            src = {u: seq for u, seq in src.items() if len(seq) >= k}
            tgt = {u: seq for u, seq in tgt.items() if len(seq) >= k}

            def filter_items(data):
                item_count = defaultdict(int)
                for seq in data.values():
                    for i in seq:
                        item_count[i] += 1
                valid_items = set(i for i, c in item_count.items() if c >= k)
                return {u: [i for i in seq if i in valid_items] for u, seq in data.items()}

            src = filter_items(src)
            tgt = filter_items(tgt)
            src = {u: seq for u, seq in src.items() if len(seq) > 0}
            tgt = {u: seq for u, seq in tgt.items() if len(seq) > 0}
            logger.info(f"[JOINT] Iter {iteration}: {len(src)} overlapping users remaining")
        return src, tgt

    all_item_seqs['src'], all_item_seqs['tgt'] = dual_domain_kcore_loop(all_item_seqs['src'], all_item_seqs['tgt'], k)
    if not len(all_item_seqs['src']):
        logger.error(
            f"[JOINT] K-Core filtering resulted in **zero users** for dataset {joint_dataset_name} "
            f"with k={k}. Try lowering k_core or use all_users mode."
        )
        raise ValueError(f"K-Core filtering failed: no overlapping users remain for {joint_dataset_name}.")
    return all_item_seqs

def to_df(inter):
    df = pd.DataFrame(inter, columns=['user', 'item'])
    df = df.sort_values(['user', 'item']).reset_index(drop=True)
    return df

def split_interation(config, joint_path, all_item_seqs, all_users, id_mapping):
    """
    功能：
        将跨域用户交互数据切分为训练、验证和测试集合，
    切分结果：
        - 源域（src）：
            * train_src（全部用于训练）
        - 目标域（tgt）：
            * train_tgt        ：warm 用户训练集
            * valid_warm_tgt   ：warm 用户验证集
            * test_warm_tgt    ：warm 用户测试集
            * valid_cold_tgt   ：cold 用户验证集
            * test_cold_tgt    ：cold 用户测试集
        以上均为Dataframe结构并存储为文件，包含字段：['user', 'item']
    说明：
        - cold 用户（valid_cold / test_cold）：
            * 从 overlap 用户中按比例划分得到
            * 在目标域中视为冷启动用户，仅用于目标域评估，不参与目标域训练，保留源域交互
        - warm 用户：
            * 指目标域中的其余用户（包含 overlap 中未被选为 cold 的用户以及 tgt-only 用户）
            * 将每个warm用户的交互按比例划分为 train / valid / test
        - 所有交互均使用重编号后的 user / item ID
    """
    train_src_path = os.path.join(joint_path, "train_src.pkl")
    train_tgt_path = os.path.join(joint_path, "train_tgt.pkl")
    valid_cold_tgt_path = os.path.join(joint_path, "valid_cold_tgt.pkl")
    test_cold_tgt_path = os.path.join(joint_path, "test_cold_tgt.pkl")
    valid_warm_tgt_path = os.path.join(joint_path, "valid_warm_tgt.pkl")
    test_warm_tgt_path = os.path.join(joint_path, "test_warm_tgt.pkl")
    if os.path.exists(train_src_path) and os.path.exists(train_tgt_path) and os.path.exists(
            valid_cold_tgt_path) and os.path.exists(test_cold_tgt_path) and os.path.exists(
        valid_warm_tgt_path) and os.path.exists(
        test_warm_tgt_path):
        logger.info('[DATASET] Interaction Splitting has been done...')
        train_src = pd.read_pickle(train_src_path)
        train_tgt = pd.read_pickle(train_tgt_path)
        valid_cold_tgt = pd.read_pickle(valid_cold_tgt_path)
        test_cold_tgt = pd.read_pickle(test_cold_tgt_path)
        valid_warm_tgt = pd.read_pickle(valid_warm_tgt_path)
        test_warm_tgt = pd.read_pickle(test_warm_tgt_path)
        return train_src, train_tgt, valid_cold_tgt, test_cold_tgt, valid_warm_tgt, test_warm_tgt

    train_src, train_tgt = [], []
    valid_cold_tgt, test_cold_tgt = [], []
    valid_warm_tgt, test_warm_tgt = [], []

    for raw_uid in sorted(all_item_seqs['src']):
        item_seq = all_item_seqs['src'][raw_uid]
        uid = id_mapping['src']['user2id'][raw_uid]
        for raw_iid in item_seq:
            iid = id_mapping['src']['item2id'][raw_iid]
            train_src.append([uid, iid])

    valid_cold_raw_users = {id_mapping['src']['id2user'][uid] for uid in all_users['valid_cold_users']}
    test_cold_raw_users = {id_mapping['src']['id2user'][uid] for uid in all_users['test_cold_users']}
    for raw_uid in sorted(all_item_seqs['tgt']):
        item_seq = all_item_seqs['tgt'][raw_uid]
        if raw_uid in valid_cold_raw_users:
            uid = id_mapping['src']['user2id'][raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping['tgt']['item2id'][raw_iid]
                valid_cold_tgt.append([uid, iid])
        elif raw_uid in test_cold_raw_users:
            uid = id_mapping['src']['user2id'][raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping['tgt']['item2id'][raw_iid]
                test_cold_tgt.append([uid, iid])
        elif raw_uid in id_mapping['tgt']['user2id']:
            uid = id_mapping['tgt']['user2id'][raw_uid]
            seq_len = len(item_seq)
            num_test = max(1, int(seq_len * config['warm_test_ratio']))
            num_valid = max(1, int(seq_len * config['warm_valid_ratio']))
            num_train = seq_len - num_valid - num_test
            assert num_train > 0
            for raw_iid in item_seq[:num_train]:
                train_tgt.append([uid, id_mapping['tgt']['item2id'][raw_iid]])
            for raw_iid in item_seq[num_train:num_train + num_valid]:
                valid_warm_tgt.append([uid, id_mapping['tgt']['item2id'][raw_iid]])
            for raw_iid in item_seq[num_train + num_valid:]:
                test_warm_tgt.append([uid, id_mapping['tgt']['item2id'][raw_iid]])

    train_src = to_df(train_src)
    train_tgt = to_df(train_tgt)
    valid_cold_tgt = to_df(valid_cold_tgt)
    test_cold_tgt = to_df(test_cold_tgt)
    valid_warm_tgt = to_df(valid_warm_tgt)
    test_warm_tgt = to_df(test_warm_tgt)

    train_src.to_pickle(train_src_path)
    train_tgt.to_pickle(train_tgt_path)
    valid_cold_tgt.to_pickle(valid_cold_tgt_path)
    test_cold_tgt.to_pickle(test_cold_tgt_path)
    valid_warm_tgt.to_pickle(valid_warm_tgt_path)
    test_warm_tgt.to_pickle(test_warm_tgt_path)
    logger.info('[DATASET] Interaction Splitting finished and saved.')

    return (
        train_src,
        train_tgt,
        valid_cold_tgt,
        test_cold_tgt,
        valid_warm_tgt,
        test_warm_tgt,
    )

def prepare_modality_emb(config, domains, id_mapping, train_src, train_tgt, joint_path):
    """
    功能：
        为跨域联合数据集构建并保存多模态 embedding。
        该方法在 joint 数据集构建阶段被调用，
        根据配置并行处理源域（src）和目标域（tgt）的模态特征，并行调用AmazonModalityProcessor进行处理
        并将最终 embedding 保存到 joint_path。
    说明：
        - 仅在配置中启用模态（modality['enabled'] == True）时生效
        - embedding涉及的交互信息只来自训练集
        - 支持 src / tgt 域独立的模态处理流程
    输入：
        config: dict
            模态相关配置（是否启用、PCA 维度等）
        domains: List[str]
            域名称列表，顺序为 [src_domain, tgt_domain]
        id_mapping: dict
            src / tgt 域的 item ID 映射
        train_src / train_tgt: pd.DataFrame
            用于确定训练阶段可见的 item
        joint_path: str
            联合数据集存储路径
    """
    domain_role_map = {
        domains[0]: {"role": "src"},
        domains[1]: {"role": "tgt"},
    }
    all_id_mapping = {
        "src": id_mapping["src"],
        "tgt": id_mapping["tgt"],
    }
    all_train_df = {
        "src": train_src,
        "tgt": train_tgt,
    }

    with ThreadPoolExecutor(max_workers=len(domains)) as executor:
        future_to_domain = {}

        for domain in domains:
            info = domain_role_map[domain]

            processor = AmazonModalityProcessor(
                config=config,
                domain=domain,
                role=info["role"],
                id_mapping=all_id_mapping,
                train_df=all_train_df,
                joint_path=joint_path,
            )

            future = executor.submit(processor.run_full_pipeline)
            future_to_domain[future] = domain

        results = {}
        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                result = future.result()
                results[domain] = result
                logger.info(
                    f"[TRAINING] [{domain}] Modality Preparation "
                    f"{'completed' if result else 'failed'}"
                )
            except Exception as e:
                logger.error(
                    f"[ERROR] [{domain}] Exception during modality preparation: {e}",
                    exc_info=True
                )
                results[domain] = False

def check_and_prepare_Amazon2014_single(config, domain):
    """
    功能：
        检查并准备 Amazon2014 的单域（single-domain）数据。
        该方法负责：
        - 检查指定 domain 的单域 processed 数据是否已存在
        - 若缺失，则自动调用 AmazonDataProcessor 重新构建
    必要检查文件：
        - processed/all_item_seqs.json
          （单域内所有用户的完整交互序列）
    构建逻辑：
        - 若必要文件存在：直接返回，跳过处理
        - 若必要文件缺失：调用 AmazonDataProcessor.run_full_pipeline()
    """
    domain_path = os.path.join(config['data_path'], config['dataset'], domain)
    processed_dir = os.path.join(domain_path, 'processed')

    required_files = [
        os.path.join(processed_dir, 'all_item_seqs.json'),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logger.info(f"[TRAINING] [{domain}] Missing files detected, starting data processing pipeline...")
        logger.info(f"[TRAINING] [{domain}] Missing files: {missing_files}")

        try:
            processor = AmazonDataProcessor(config, domain, config['data_path'])
            processor.run_full_pipeline()
            logger.info(f"[TRAINING] [{domain}] Data processing pipeline completed")

            still_missing = [f for f in required_files if not os.path.exists(f)]
            if still_missing:
                logger.error(f"[ERROR] [{domain}] Files still missing after processing: {still_missing}")
                return False

        except Exception as e:
            logger.error(f"[ERROR] [{domain}] Error during data processing: {e}", exc_info=True)
            return False
    else:
        logger.info(f"[TRAINING] [{domain}] All required data files exist, skipping data processing")

    return True

def create_joint_dataset(domains: List[str], config: dict):
    """
    功能：
        构建 Amazon2014 的跨域联合数据集（joint dataset）。
        该方法是 joint 数据集的统一构建入口，负责从两个单域数据中
        生成完整的跨域推荐训练与评估所需文件。
    主要流程：
        1. 加载源域 / 目标域的 all_item_seqs.json
        2. （可选）仅保留 overlap 用户并执行双域 k-core 过滤
        3. 用户划分（overlap / warm / cold）并重新编号，生成all_users.json和id_mapping.json
        4. 构建 src / tgt 的 train / valid / test 交互数据，生成六个pkl
        5. （可选）构建并保存多模态 embedding
    """
    data_path = config['data_path'] if 'data_path' in config else '../data/'
    joint_dataset_name = "+".join(domains)
    dataset_type = "only_overlap_users" if config['only_overlap_users'] else "all_users"
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}"
    )
    if config['only_overlap_users']:
        split_dir += f'_{config["k_cores"]}cores'
    joint_path = os.path.join(data_path, 'Amazon2014', joint_dataset_name, dataset_type, split_dir)

    os.makedirs(os.path.join(joint_path, 'modality_emb_src'), exist_ok=True)
    os.makedirs(os.path.join(joint_path, 'modality_emb_tgt'), exist_ok=True)
    logger.info(f"[JOINT] Creating joint dataset: {joint_dataset_name}")

    logger.info("\n=== STEP 1: Load all_item_seqs from src & tgt ===")
    src_processed_dir = os.path.join(data_path, 'Amazon2014', domains[0], 'processed')
    tgt_processed_dir = os.path.join(data_path, 'Amazon2014', domains[1], 'processed')
    all_item_seqs = load_all_item_seqs(src_processed_dir, tgt_processed_dir)
    if config['only_overlap_users']:
        all_item_seqs = filter_overlap_users(all_item_seqs, config['k_cores'], joint_dataset_name)

    logger.info("\n=== STEP 2: Split users and reindex ===")
    all_users, id_mapping = split_users_and_reindex(config, joint_path, all_item_seqs)

    logger.info("\n=== STEP 3: Split train/valid/test ===")
    train_src, train_tgt, valid_cold_tgt, test_cold_tgt, valid_warm_tgt, test_warm_tgt =\
        split_interation(config, joint_path, all_item_seqs, all_users, id_mapping)

    logger.info("\n=== STEP 4: Prepare modality embeddings ===")
    prepare_modality_emb(config, domains, id_mapping, train_src, train_tgt, joint_path)

    logger.info(f"\n[JOINT] All joint dataset files created successfully!")
    return joint_path

def check_and_prepare_Amazon2014(config):
    """
    功能：
        Amazon2014 跨域推荐 benchmark 的「统一数据检查与自动构建入口」。
        负责：
        1）单域（single-domain）数据集的完整性检查与构建
        2）联合（joint-domain）数据集的完整性检查与构建
    --------------------------------------------------
    一、Single-domain（单域数据集）
    --------------------------------------------------
    1. 针对 config['domains'] 中的每一个 domain：
        并行执行 check_and_prepare_Amazon2014_single(config, domain)
        同时检查多个domain文件完整性
    --------------------------------------------------
    二、Joint-domain（联合数据集）
    --------------------------------------------------
    1.检查 joint_path 下是否存在所有必要文件
        Joint-domain 必要检查文件（必须存在）：
           - all_users.json
           - id_mapping.json
           - train_src.pkl
           - train_tgt.pkl
           - valid_cold_tgt.pkl
           - test_cold_tgt.pkl
           - valid_warm_tgt.pkl
           - test_warm_tgt.pkl
        Joint-domain 可选检查文件（模态相关，按配置启用）：
           - modality_emb_src/{modality_name}_final_emb_{emb_pca}.npy
           - modality_emb_tgt/{modality_name}_final_emb_{emb_pca}.npy
           其中：
           - 仅当 modality['enabled'] == True 时才要求存在
           - 模态文件缺失将视为 joint 数据集不完整
    2. 若任意 joint-domain 必要 / 启用的可选文件缺失：
       → 直接调用：create_joint_dataset(domains, config)
         重新构建整个 joint 数据集

    --------------------------------------------------
    输出：
        bool
            - True ：所有 single / joint 数据集文件均已准备完成
            - False：任一阶段构建失败
    """
    domains = config['domains']

    # STEP 1: Checking and preparing individual Amazon datasets
    logger.info(f"[TRAINING] Starting parallel data preparation for dataset: Amazon2014, processing domains individually: {domains}")
    with ThreadPoolExecutor(max_workers=len(domains)) as executor:
        future_to_dataset = {
            executor.submit(check_and_prepare_Amazon2014_single, config, domain): domain
            for domain in domains
        }

        results = {}
        for future in as_completed(future_to_dataset):
            domain_name = future_to_dataset[future]
            try:
                result = future.result()
                results[domain_name] = result
                logger.info(f"[TRAINING] [{domain_name}] Data preparation {'completed' if result else 'failed'}")
            except Exception as e:
                logger.error(f"[ERROR] [{domain_name}] Exception during data preparation: {e}", exc_info=True)
                results[domain_name] = False

    failed_datasets = [name for name, success in results.items() if not success]
    if failed_datasets:
        logger.error(f"[ERROR] Failed to prepare data for datasets: {failed_datasets}")
        return False
    logger.info(f"[TRAINING] All single-domain datasets are prepared successfully: {list(results.keys())}")

    # STEP 2: Checking and creating joint Amazon dataset
    if len(domains) != 2:
        logger.info("[TRAINING] Single dataset mode, skipping joint dataset creation")
        return True

    data_path = config['data_path'] if 'data_path' in config else '../data/'
    joint_dataset_name = "+".join(domains)
    dataset_type = "only_overlap_users" if config['only_overlap_users'] else "all_users"
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}"
    )
    if config['only_overlap_users']:
        split_dir += f'_{config["k_cores"]}cores'
    joint_path = os.path.join(data_path, 'Amazon2014', joint_dataset_name, dataset_type, split_dir)

    required_joint_files = [
        os.path.join(joint_path, 'all_users.json'),
        os.path.join(joint_path, 'id_mapping.json'),
        os.path.join(joint_path, 'train_src.pkl'),
        os.path.join(joint_path, 'train_tgt.pkl'),
        os.path.join(joint_path, 'valid_cold_tgt.pkl'),
        os.path.join(joint_path, 'test_cold_tgt.pkl'),
        os.path.join(joint_path, 'valid_warm_tgt.pkl'),
        os.path.join(joint_path, 'test_warm_tgt.pkl'),
    ]

    for modality in config['modalities']:
        if not modality['enabled']:
            continue
        final_embs_file_name = modality['name'] + '_final_emb_' + str(modality['emb_pca']) + '.npy'
        required_joint_files.append(os.path.join(joint_path, 'modality_emb_src', final_embs_file_name))
        required_joint_files.append(os.path.join(joint_path, 'modality_emb_tgt', final_embs_file_name))

    missing_joint_files = [f for f in required_joint_files if not os.path.exists(f)]

    if missing_joint_files:
        logger.info(f"[TRAINING] One or more joint dataset files are missing. Recreating ALL joint files.")
        logger.info(f"Missing files: {missing_joint_files}")
        try:
            create_joint_dataset(domains=domains, config=config)

            still_missing = [f for f in required_joint_files if not os.path.exists(f)]
            if still_missing:
                logger.error(f"[ERROR] Joint dataset creation failed, still missing: {still_missing}")
                return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to create joint dataset: {e}", exc_info=True)
            return False
    else:
        logger.info(
            f"[TRAINING] All required joint dataset files (including embeddings) already exist. Skipping creation.")

    return True

def check_and_prepare_dataset(config):
    """
    数据集检查与构建入口。
    """
    dataset = config['dataset']

    if dataset == "Amazon2014":
        check_and_prepare_Amazon2014(config)