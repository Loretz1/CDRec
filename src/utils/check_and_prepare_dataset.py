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

    users_src = set(all_item_seqs['src'].keys())
    users_tgt = set(all_item_seqs['tgt'].keys())
    overlap_users = users_src & users_tgt
    src_only_users = users_src - overlap_users
    tgt_only_users = users_tgt - overlap_users

    overlap_list = np.random.permutation(list(overlap_users))
    num_overlap = len(overlap_list)
    num_valid_cold = int(num_overlap * config['t_cold_valid'])
    num_test_cold = int(num_overlap * config['t_cold_test'])
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

    id_mapping = {'src': {'user2id': {}, 'item2id': {}, 'id2user': ['[PAD]'], 'id2item': ['[PAD]']},
                  'tgt': {'user2id': {}, 'item2id': {}, 'id2user': ['[PAD]'], 'id2item': ['[PAD]']}}
    for i, u in enumerate(all_users['overlap_users'], start=1):
        id_mapping['src']['user2id'][u] = i
        id_mapping['src']['id2user'].append(u)
        id_mapping['tgt']['user2id'][u] = i
        id_mapping['tgt']['id2user'].append(u)
    for i, u in enumerate(all_users['valid_cold_users'] | all_users['test_cold_users'] | all_users[
        'src_only_users'], start=len(all_users['overlap_users']) + 1):
        id_mapping['src']['user2id'][u] = i
        id_mapping['src']['id2user'].append(u)
    for i, u in enumerate(all_users['tgt_only_users'], start=len(all_users['overlap_users']) + 1):
        id_mapping['tgt']['user2id'][u] = i
        id_mapping['tgt']['id2user'].append(u)
    for domain in ['src', 'tgt']:
        for user, items in all_item_seqs[domain].items():
            for item in items:
                if item not in id_mapping[domain]['item2id']:
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
    return pd.DataFrame(inter, columns=['user', 'item'])

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

    for raw_uid, item_seq in all_item_seqs['src'].items():
        uid = id_mapping['src']['user2id'][raw_uid]
        for raw_iid in item_seq:
            iid = id_mapping['src']['item2id'][raw_iid]
            train_src.append([uid, iid])

    for raw_uid, item_seq in all_item_seqs['tgt'].items():
        if raw_uid in [id_mapping['src']['id2user'][i] for i in all_users["valid_cold_users"]]:
            uid = id_mapping['src']['user2id'][raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping['tgt']['item2id'][raw_iid]
                valid_cold_tgt.append([uid, iid])
        elif raw_uid in [id_mapping['src']['id2user'][i] for i in all_users["test_cold_users"]]:
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
    domain_role_map = {
        domains[0]: {
            "role": "src",
            "id_mapping": id_mapping["src"],
            "train_df": train_src,
        },
        domains[1]: {
            "role": "tgt",
            "id_mapping": id_mapping["tgt"],
            "train_df": train_tgt,
        }
    }

    with ThreadPoolExecutor(max_workers=len(domains)) as executor:
        future_to_domain = {}

        for domain in domains:
            info = domain_role_map[domain]

            processor = AmazonModalityProcessor(
                config=config,
                domain=domain,
                role=info["role"],
                id_mapping=info["id_mapping"],
                train_df=info["train_df"],
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
    dataset = config['dataset']

    if dataset == "Amazon2014":
        check_and_prepare_Amazon2014(config)