import os
import json
import numpy as np
from typing import Dict, List
from logging import getLogger
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.amazon_data_processor import AmazonDataProcessor

logger = getLogger()

def check_and_prepare_Amazon2014_single(config, domain):
    domain_path = os.path.join(config['data_path'], config['dataset'], domain)
    processed_dir = os.path.join(domain_path, 'processed')

    required_files = [
        os.path.join(processed_dir, 'all_item_seqs.json'),
        os.path.join(processed_dir, 'id_mapping.json'),
        os.path.join(processed_dir, 'metadata.sentence.json')
    ]

    sent_emb_model = config['sent_emb_model'] if 'sent_emb_model' in config else 'text-embedding-3-large'
    sent_emb_dim = config['sent_emb_dim'] if 'sent_emb_dim' in config else 3072
    sent_emb_path = os.path.join(
        processed_dir,
        f"{os.path.basename(sent_emb_model)}_{sent_emb_dim}.npy"
    )
    required_files.append(sent_emb_path)

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
    joint_path = os.path.join(data_path, 'Amazon2014', joint_dataset_name, dataset_type)
    src_path = os.path.join(data_path, 'Amazon2014', domains[0], 'processed')
    tgt_path = os.path.join(data_path, 'Amazon2014', domains[1], 'processed')

    model_name = os.path.basename(config['sent_emb_model'] if 'sent_emb_model' in config else 'text-embedding-3-large')
    sent_emb_dim = config['sent_emb_dim'] if 'sent_emb_dim' in config else 3072
    embedding_filename = f'{model_name}_{sent_emb_dim}.npy'

    os.makedirs(os.path.join(joint_path, 'src'), exist_ok=True)
    os.makedirs(os.path.join(joint_path, 'tgt'), exist_ok=True)
    logger.info(f"[JOINT] Creating joint dataset: {joint_dataset_name}")

    # Load
    with open(os.path.join(src_path, 'all_item_seqs.json'), 'r') as f:
        all_item_seqs_src = json.load(f)
    with open(os.path.join(src_path, 'id_mapping.json'), 'r') as f:
        id_mapping_src = json.load(f)
    embeddings_src = np.load(os.path.join(src_path, embedding_filename))
    with open(os.path.join(tgt_path, 'all_item_seqs.json'), 'r') as f:
        all_item_seqs_tgt = json.load(f)
    with open(os.path.join(tgt_path, 'id_mapping.json'), 'r') as f:
        id_mapping_tgt = json.load(f)
    embeddings_tgt = np.load(os.path.join(tgt_path, embedding_filename))

    if config['only_overlap_users']:
        k = config['k_cores'] if 'k_cores' in config else 3

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

        all_item_seqs_src, all_item_seqs_tgt = dual_domain_kcore_loop(all_item_seqs_src, all_item_seqs_tgt, k)
        if not len(all_item_seqs_src):
            logger.error(
                f"[JOINT] K-Core filtering resulted in **zero users** for dataset {joint_dataset_name} "
                f"with k={k}. Try lowering k_core or use all_users mode."
            )
            raise ValueError(f"K-Core filtering failed: no overlapping users remain for {joint_dataset_name}.")
        final_users = sorted(all_item_seqs_src.keys())
        final_items_src = sorted({i for seq in all_item_seqs_src.values() for i in seq})
        final_items_tgt = sorted({i for seq in all_item_seqs_tgt.values() for i in seq})

        def rebuild_id_mapping(final_users, final_items):
            new_user2id = {}
            new_id2user = ["[PAD]"]
            new_item2id = {}
            new_id2item = ["[PAD]"]
            new_id2user.extend(final_users)
            new_id2item.extend(final_items)
            for idx, u in enumerate(final_users, start=1):
                new_user2id[u] = idx
            for idx, i in enumerate(final_items, start=1):
                new_item2id[i] = idx
            return {
                'user2id': new_user2id,
                'item2id': new_item2id,
                'id2user': new_id2user,
                'id2item': new_id2item
            }

        id_mapping_src_filtered = rebuild_id_mapping(final_users, final_items_src)
        id_mapping_tgt_filtered = rebuild_id_mapping(final_users, final_items_tgt)
        embeddings_src = embeddings_src[[id_mapping_src['item2id'][i] - 1 for i in final_items_src]]
        embeddings_tgt = embeddings_tgt[[id_mapping_tgt['item2id'][i] - 1 for i in final_items_tgt]]
        id_mapping_src = id_mapping_src_filtered
        id_mapping_tgt = id_mapping_tgt_filtered

    # PCA
    sent_emb_dim = (
        config['sent_emb_pca']
        if 'sent_emb_pca' in config
        else (config['sent_emb_dim'] if 'sent_emb_dim' in config else 3072)
    )
    if 'sent_emb_pca' in config and config['sent_emb_pca'] != 0:
        print(f'[TOKENIZER] Applying PCA to sentence embeddings...')
        try:
            from sklearn.decomposition import PCA
            pca_src = PCA(n_components=config['sent_emb_pca'], whiten=True)
            embeddings_src = pca_src.fit_transform(embeddings_src)
            pca_tgt = PCA(n_components=config['sent_emb_pca'], whiten=True)
            embeddings_tgt = pca_tgt.fit_transform(embeddings_tgt)
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")

    # Save
    with open(os.path.join(joint_path, 'src', 'all_item_seqs.json'), 'w') as f:
        json.dump(all_item_seqs_src, f)
    with open(os.path.join(joint_path, 'src', 'id_mapping.json'), 'w') as f:
        json.dump(id_mapping_src, f)
    np.save(os.path.join(joint_path, 'src', f'final_sent_embeddings_{sent_emb_dim}.npy'), embeddings_src)
    with open(os.path.join(joint_path, 'tgt', 'all_item_seqs.json'), 'w') as f:
        json.dump(all_item_seqs_tgt, f)
    with open(os.path.join(joint_path, 'tgt', 'id_mapping.json'), 'w') as f:
        json.dump(id_mapping_tgt, f)
    np.save(os.path.join(joint_path, 'tgt', f'final_sent_embeddings_{sent_emb_dim}.npy'), embeddings_tgt)

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
    joint_path = os.path.join(data_path, 'Amazon2014', joint_dataset_name, dataset_type)

    sent_emb_dim = config['sent_emb_pca'] if 'sent_emb_pca' in config else config['sent_emb_dim']

    required_joint_files = [
        os.path.join(joint_path, 'src', 'all_item_seqs.json'),
        os.path.join(joint_path, 'src', 'id_mapping.json'),
        os.path.join(joint_path, 'src', f'final_sent_embeddings_{sent_emb_dim}.npy'),
        os.path.join(joint_path, 'tgt', 'all_item_seqs.json'),
        os.path.join(joint_path, 'tgt', 'id_mapping.json'),
        os.path.join(joint_path, 'tgt', f'final_sent_embeddings_{sent_emb_dim}.npy')
    ]

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