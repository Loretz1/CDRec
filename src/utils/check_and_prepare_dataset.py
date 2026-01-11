import os
import json
import numpy as np
import pandas as pd
from typing import List
from logging import getLogger
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.data_preprocessing.amazon_data_processor import AmazonDataProcessor
from utils.data_preprocessing.amazon_modality_processor import AmazonModalityProcessor
from utils.data_preprocessing.douban_data_processor import DoubanDataProcessor

logger = getLogger()

def load_all_item_seqs(src_processed_dir, tgt_processed_dir, shuffle):
    with open(os.path.join(src_processed_dir, f'all_item_seqs_{"shuffle" if shuffle else "noshuffle"}.json'), 'r') as f:
        src_all_item_seqs = json.load(f)
    with open(os.path.join(tgt_processed_dir, f'all_item_seqs_{"shuffle" if shuffle else "noshuffle"}.json'), 'r') as f:
        tgt_all_item_seqs = json.load(f)
    all_item_seqs = {
        "src": src_all_item_seqs,
        "tgt": tgt_all_item_seqs
    }
    return all_item_seqs

def split_users_and_reindex(config, joint_path, all_item_seqs):
    # 如果存在这两个文件，就不重新生成了，之后的文件也都是这样的，不会重复生成。
    all_users_file_path = os.path.join(joint_path, "all_users.json")
    id_mapping_file_path = os.path.join(joint_path, "id_mapping.json")
    if os.path.exists(all_users_file_path) and os.path.exists(id_mapping_file_path):
        logger.info('[DATASET] User splitting and reindex have been done...')
        with open(all_users_file_path, 'r') as f:
            all_users = json.load(f)
        with open(id_mapping_file_path, 'r') as f:
            id_mapping = json.load(f)
        return all_users, id_mapping

    # 源域、目标域所有用户，这里是raw id存的
    users_src = sorted(all_item_seqs['src'].keys())
    users_tgt = sorted(all_item_seqs['tgt'].keys())

    # 按照raw id把用户分成三类：原始重叠用户、src单域用户、tgt单域用户
    users_src_set = set(users_src)
    users_tgt_set = set(users_tgt)
    overlap_users = sorted(users_src_set & users_tgt_set)
    src_only_users = sorted(users_src_set - set(overlap_users))
    tgt_only_users = sorted(users_tgt_set - set(overlap_users))

    rng = np.random.RandomState(999)
    overlap_list = rng.permutation(overlap_users)
    # 按照Yaml超参数：t_cold_valid、t_cold_test，按比例把原始重叠用户分成：重叠用户、valid冷用户、test冷用户
    # 因此就形成了五类用户：重叠用户、valid冷用户、test冷用户 （前面这三类是由原始重叠用户拆开来的）、 src单域用户、tgt单域用户
    num_overlap = len(overlap_list)
    num_valid_cold = int(num_overlap * config['t_cold_valid'])
    num_test_cold = int(num_overlap * config['t_cold_test'])
    assert num_valid_cold + num_test_cold <= num_overlap, "Sum of t_cold_valid and t_cold_test causes user overflow."
    valid_cold_users = sorted(overlap_list[:num_valid_cold])
    test_cold_users = sorted(overlap_list[num_valid_cold:num_valid_cold + num_test_cold])
    overlap_users = sorted(overlap_list[num_valid_cold + num_test_cold:])

    # 形成了all_users，这里还是raw id存的，后面映射为id后，再存成json
    all_users = {
        'overlap_users': overlap_users,
        'valid_cold_users': valid_cold_users,
        'test_cold_users': test_cold_users,
        'src_only_users': src_only_users,
        'tgt_only_users': tgt_only_users
    }

    # 开始给用户、物品编号
    id_mapping = {'src': {'user2id': {}, 'item2id': {}, 'id2user': ['[PAD]'], 'id2item': ['[PAD]']},
                  'tgt': {'user2id': {}, 'item2id': {}, 'id2user': ['[PAD]'], 'id2item': ['[PAD]']}}
    # 先给重叠用户overlap_users编号，他们在两个域编号相同，且都靠在前面，比如有3128个重叠用户，那么 1-3128。
    for i, u in enumerate(all_users['overlap_users'], start=1):
        id_mapping['src']['user2id'][u] = i
        id_mapping['src']['id2user'].append(u)
        id_mapping['tgt']['user2id'][u] = i
        id_mapping['tgt']['id2user'].append(u)
    src_extra_users = (  #这些是源域的单域用户，给他们接着编号，比如从3129开始编号
            all_users['valid_cold_users']
            + all_users['test_cold_users']
            + all_users['src_only_users']
    )
    # 源域单域用户（valid_cold_users + test_cold_users + src_only_users）编号，比如从3129开始编号
    for i, u in enumerate(src_extra_users, start=len(all_users['overlap_users']) + 1):
        id_mapping['src']['user2id'][u] = i
        id_mapping['src']['id2user'].append(u)
    # 目标域单域用户tgt_only_users 编号，比如从3129开始
    for i, u in enumerate(all_users['tgt_only_users'], start=len(all_users['overlap_users']) + 1):
        id_mapping['tgt']['user2id'][u] = i
        id_mapping['tgt']['id2user'].append(u)

    # 物品不重叠，直接分域编一下
    for domain in ['src', 'tgt']:
        all_items = set()
        for items in all_item_seqs[domain].values():
            all_items.update(items)
        for item in sorted(all_items):
            id_mapping[domain]['item2id'][item] = len(id_mapping[domain]['id2item'])
            id_mapping[domain]['id2item'].append(item)

    # 把all_user从raw id 映射到 id
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

    # 读一下src的所有交互，这些交互不会被分到任何valid/test，所以直接放到train_src里面。
    for raw_uid in sorted(all_item_seqs['src']):
        item_seq = all_item_seqs['src'][raw_uid]
        uid = id_mapping['src']['user2id'][raw_uid]
        for raw_iid in item_seq:
            iid = id_mapping['src']['item2id'][raw_iid]
            train_src.append([uid, iid])

    # 读取tgt所有交互，把每一个交互归入train_tgt/valid_cold_tgt/test_cold_tgt/valid_warm_tgt/test_warm_tgt
    valid_cold_raw_users = {id_mapping['src']['id2user'][uid] for uid in all_users['valid_cold_users']}
    test_cold_raw_users = {id_mapping['src']['id2user'][uid] for uid in all_users['test_cold_users']}
    for raw_uid in sorted(all_item_seqs['tgt']):
        item_seq = all_item_seqs['tgt'][raw_uid]
        # 用户分成四类：overlap_users、valid_cold_users、test_cold_users、tgt_only_users
        if raw_uid in valid_cold_raw_users:
            # 如果是valid_cold_users，他的交互全部放到valid_cold_tgt
            uid = id_mapping['src']['user2id'][raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping['tgt']['item2id'][raw_iid]
                valid_cold_tgt.append([uid, iid])
        elif raw_uid in test_cold_raw_users:
            # 如果是test_cold_users，他的交互全部放到test_cold_tgt
            uid = id_mapping['src']['user2id'][raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping['tgt']['item2id'][raw_iid]
                test_cold_tgt.append([uid, iid])
        elif raw_uid in id_mapping['tgt']['user2id']:
            # overlap_users/tgt_only_users
            # 他的交互按照warm_test_ratio、warm_valid_ratio比例，分别放入train_tgt/valid_warm_tgt/test_warm_tgt
            # 这里如果shuffle_user_sequence=False，用户交互List没有重排序的话，这里最新的交互会被放入valid/test，引入时间信息
            uid = id_mapping['tgt']['user2id'][raw_uid]
            seq_len = len(item_seq)
            num_test = max(1, int(seq_len * config['warm_test_ratio'])) if config['warm_test_ratio'] else 0
            num_valid = max(1, int(seq_len * config['warm_valid_ratio'])) if config['warm_valid_ratio'] else 0
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
    all_id_mapping = {
        "src": id_mapping["src"],
        "tgt": id_mapping["tgt"],
    }
    train_df = {
        "src": train_src,
        "tgt": train_tgt,
    }

    logger.info(f"[TRAINING] Start joint modality preparation for domains: {domains}")

    try:
        processor = AmazonModalityProcessor(
            config=config,
            domains=domains,              # ["Clothing...", "Sports..."]
            id_mapping=all_id_mapping,  # {"src": ..., "tgt": ...}
            train_df=train_df,            # {"src": df, "tgt": df}
            joint_path=joint_path,
        )

        result = processor.run_full_pipeline()

        logger.info(
            "[TRAINING] Joint modality preparation "
            f"{'completed' if result else 'failed'}"
        )

        return result

    except Exception as e:
        logger.error(
            "[ERROR] Exception during joint modality preparation",
            exc_info=True
        )
        return False

def check_and_prepare_Amazon2014_single(config, domain):
    # 检查是否该域是否缺失all_item_seqs_{shuffle|noshuffle}.json
    # 如果没有该文件，跑一个AmazonDataProcessor类的run_full_pipeline()生成这个json
    domain_path = os.path.join(config['data_path'], config['dataset'], domain)
    processed_dir = os.path.join(domain_path, 'processed')
    seq_file = (
        f"all_item_seqs_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}.json"
    )

    required_files = [
        os.path.join(processed_dir, seq_file),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logger.info(f"[TRAINING] [{domain}] Missing files detected, starting data processing pipeline...")
        logger.info(f"[TRAINING] [{domain}] Missing files: {missing_files}")

        try:
            processor = AmazonDataProcessor(config, domain, config['data_path'])
            processor.run_full_pipeline()   # 生成json文件入口
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
    # 联合跨域数据集生成逻辑，主要包含四个步骤
    data_path = config['data_path'] if 'data_path' in config else '../data/'
    joint_dataset_name = "+".join(domains)
    dataset_type = "only_overlap_users" if config['only_overlap_users'] else "all_users"
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config['only_overlap_users']:
        split_dir += f'_{config["k_cores"]}cores'
    joint_path = os.path.join(data_path, 'Amazon2014', joint_dataset_name, dataset_type, split_dir)

    os.makedirs(os.path.join(joint_path, 'modality_emb'), exist_ok=True)
    logger.info(f"[JOINT] Creating joint dataset: {joint_dataset_name}")

    # 第一步：读入两个域 单域处理好的 all_item_seqs_{shuffle|noshuffle}.json
    # 如果 Yaml中配置only_overlap_users=True，额外进行一步操作：双域k-cores操作
    # 这个操作会保证： 1.每个用户在每一个域都至少有k个交互物品（因此最终留下的都是重叠用户） 2.每个物品在该域内至少有k个用户交互过
    # 如果only_overlap_users=True，那么数据集只剩下重叠用户了
    # 这里也决定了联合跨域数据集目录是all_users（only_overlap_users=False），还是only_overlap_users（only_overlap_users=True）
    logger.info("\n=== STEP 1: Load all_item_seqs from src & tgt ===")
    src_processed_dir = os.path.join(data_path, 'Amazon2014', domains[0], 'processed')
    tgt_processed_dir = os.path.join(data_path, 'Amazon2014', domains[1], 'processed')
    all_item_seqs = load_all_item_seqs(src_processed_dir, tgt_processed_dir, config['shuffle_user_sequence'])
    if config['only_overlap_users']:
        all_item_seqs = filter_overlap_users(all_item_seqs, config['k_cores'], joint_dataset_name)

    # 第二步：把用户划分成为五类，给用户、物品编号
    # 具体逻辑在里面
    # 最后保存all_users.json和id_mapping.json
    # all_users.json例子：
    # {
    #     "overlap_users": [1, 2, 3, ...],
    #     "valid_cold_users": [3131, 3136, 3222, ...],
    #     "test_cold_users": [3132, 3138, 3199, ...],
    #     "src_only_users": [3129, 3130, 3133, ...],
    #     "tgt_only_users":[3129, 3130, 3131, ...],
    # }
    # id_mapping.json例子：
    # {
    #     "src": {
    #         "user2id": {"raw_user_id": 1, "...": "..."},
    #         "id2user": ["[PAD]", "raw_user_id", "..."],
    #         "item2id": {"raw_item_id": 1, "...": "..."},
    #         "id2item": ["[PAD]", "raw_item_id", "..."]
    #     },
    #     "tgt": {
    #         "user2id": {"raw_user_id": 1, "...": "..."},
    #         "id2user": ["[PAD]", "..."],
    #         "item2id": {"raw_item_id": 1, "...": "..."},
    #         "id2item": ["[PAD]", "..."]
    #     }
    # }
    # 每一个域对于物品、用户的编号都是从1开始的。
    # 对于用户来说，重叠用户的编号靠前，且同一个用户两个域的编号相同，比如1-3128是重叠用户，源域的1号用户和目标域1号用户是同一个用户。
    # 3129开始就是每个域的单域用户了，源域的3129号用户和目标域3129号用户不是同一个用户。
    logger.info("\n=== STEP 2: Split users and reindex ===")
    all_users, id_mapping = split_users_and_reindex(config, joint_path, all_item_seqs)

    # 第三步，划分train/valid/test，生成六个pkl文件，每个文件都是DataFrame，列名：['user', 'item']，记录一组交互。
    # 具体逻辑在里面
    # 每个pkl文件都是以id形式保存物品、用户的，而不是raw id。
    logger.info("\n=== STEP 3: Split train/valid/test ===")
    train_src, train_tgt, valid_cold_tgt, test_cold_tgt, valid_warm_tgt, test_warm_tgt =\
        split_interation(config, joint_path, all_item_seqs, all_users, id_mapping)

    # 第四步，处理Yaml中每一个Enable=true的模态，每一个模态分别在modality_emb文件夹中生成三个文件：
    # modality_emb /
    # ├── < name > _metadata.json
    # ├── < name > _ < emb_model > _ < emb_dim >.npy
    # └── < name > _final_emb_ < emb_pca >.npy
    # metadata文件是一个json，一般按照用户/物品的id为key，该用户、物品的profile作为value。
    # < name > _ < emb_model > _ < emb_dim >.npy是嵌入模型直接生成的embedding。
    # 最后一个final_emb是最终处理后的emb。
    # 这三个文件由 prepare_modality_emb 里面的三个方法分别生成，这三个方法都是可以在<model>.py中由每个模型去实现的。
    # 由于处理逻辑都是<model>.py实现的，也是模型自已拿来用，实际上最后只要能给出两个final_emb就可以了，额外提供json、原始npy文件的保存方法，是为了避免重复调用大模型API。
    logger.info("\n=== STEP 4: Prepare modality embeddings ===")
    prepare_modality_emb(config, domains, id_mapping, train_src, train_tgt, joint_path)

    logger.info(f"\n[JOINT] All joint dataset files created successfully!")
    return joint_path

def check_and_prepare_Amazon2014(config):
    # 分成两步
    domains = config['domains']

    # 第一步分别单独处理两个域的数据集，下载raw文件，初步处理用户交互数据并保存为all_item_seqs_{shuffle|noshuffle}.json，
    # 最终形成如下文件：
    # Amazon2014 /
    # ├── Clothing_Shoes_and_Jewelry /  单域数据集1
    # │   ├── raw /
    # │   │   ├── meta_Clothing_Shoes_and_Jewelry.json.gz
    # │   │   └── reviews_Clothing_Shoes_and_Jewelry_5.json.gz
    # │   └── processed /
    # │       └── all_item_seqs_{shuffle|noshuffle}.json
    # ├── Sports_and_Outdoors /    单域数据集2
    # │   ├── raw /
    # │   │   ├── meta_Sports_and_Outdoors.json.gz
    # │   │   └── reviews_Sports_and_Outdoors_5.json.gz
    # │   └── processed /
    # │       └── all_item_seqs_{shuffle|noshuffle}.json
    # STEP 1: Checking and preparing individual Amazon datasets
    logger.info(f"[TRAINING] Starting parallel data preparation for dataset: Amazon2014, processing domains individually: {domains}")
    with ThreadPoolExecutor(max_workers=len(domains)) as executor:
        # 每个域跑一个 check_and_prepare_Amazon2014_single，进行单域处理
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

    failed_domains = [name for name, success in results.items() if not success]
    if failed_domains:
        logger.error(f"[ERROR] Failed to prepare data for domains: {failed_domains}")
        return False
    logger.info(f"[TRAINING] All single-domain datasets are prepared successfully: {list(results.keys())}")

    # 第二步生成联合跨域数据集，这个数据集最终用于模型训练与评估，它是单向的，只评估模型在target域上的热/冷推荐表现，
    # 最终形成如下文件：
    # Amazon2014 /
    # └── Clothing_Shoes_and_Jewelry + Sports_and_Outdoors /   联合数据集文件夹
    # └── all_users / (或者 only_overlap_users /,  `only_overlap_users`控制该目录)
    #       └── WarmValid{w_v}_WarmTest{w_t}_ColdValid{c_v}_ColdTest{c_t}_{shuffle|noshuffle}_{kcores} /
    #             ├── train_src.pkl
    #             ├── train_tgt.pkl
    #             ├── valid_warm_tgt.pkl
    #             ├── test_warm_tgt.pkl
    #             ├── valid_cold_tgt.pkl
    #             ├── test_cold_tgt.pkl
    #             ├── all_users.json
    #             ├── id_mapping.json
    #             └── modality_emb /
    # STEP 2: Checking and creating joint Amazon dataset
    if len(domains) != 2:
        logger.info("[TRAINING] Single dataset mode, skipping joint dataset creation")
        return True
    # 先检查文件是否齐全，包括：
    # all_users.json、id_mapping.json 这两个json
    # 六个pkl文件
    # 每个Yaml配置中Enable = True的模态在modality_emb文件夹中的文件：{modality['name']}_final_emb_{str(modality['emb_pca'])}.npy
    # 如果少了任何一个，就用create_joint_dataset开始生成
    data_path = config['data_path'] if 'data_path' in config else '../data/'
    joint_dataset_name = "+".join(domains)
    dataset_type = "only_overlap_users" if config['only_overlap_users'] else "all_users"
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
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
        required_joint_files.append(os.path.join(joint_path, 'modality_emb', final_embs_file_name))

    missing_joint_files = [f for f in required_joint_files if not os.path.exists(f)]

    if missing_joint_files:
        logger.info(f"[TRAINING] One or more joint dataset files are missing. Recreating ALL joint files.")
        logger.info(f"Missing files: {missing_joint_files}")
        try:
            # 这里是联合跨域数据集处理入口
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

def check_and_prepare_Douban_single(config, domain):
    # 检查是否该域是否缺失all_item_seqs_{shuffle|noshuffle}.json
    # 如果没有该文件，跑一个DoubanDataProcessor类的run_full_pipeline()生成这个json
    domain_path = os.path.join(config['data_path'], config['dataset'], domain)
    processed_dir = os.path.join(domain_path, 'processed')
    seq_file = (
        f"all_item_seqs_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}.json"
    )

    required_files = [
        os.path.join(processed_dir, seq_file),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logger.info(f"[TRAINING] [{domain}] Missing files detected, starting data processing pipeline...")
        logger.info(f"[TRAINING] [{domain}] Missing files: {missing_files}")

        try:
            processor = DoubanDataProcessor(config, domain, config['data_path'])
            processor.run_full_pipeline()   # 生成json文件入口
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

def check_and_prepare_Douban(config):
    # 分成两步
    domains = config['domains']

    # STEP 1: Checking and preparing individual Amazon datasets
    logger.info(f"[TRAINING] Starting parallel data preparation for dataset: Douban, processing domains individually: {domains}")
    with ThreadPoolExecutor(max_workers=len(domains)) as executor:
        # 每个域跑一个 check_and_prepare_Douban_single，进行单域处理
        future_to_dataset = {
            executor.submit(check_and_prepare_Douban_single, config, domain): domain
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

    failed_domains = [name for name, success in results.items() if not success]
    if failed_domains:
        logger.error(f"[ERROR] Failed to prepare data for domains: {failed_domains}")
        return False
    logger.info(f"[TRAINING] All single-domain datasets are prepared successfully: {list(results.keys())}")

    # STEP 2: Checking and creating joint Amazon dataset


    return True


def check_and_prepare_dataset(config):
    """
    数据集检查与构建入口。
    """
    dataset = config['dataset']

    if dataset == "Amazon2014":
        check_and_prepare_Amazon2014(config)
    elif dataset == "Douban":
        check_and_prepare_Douban(config)