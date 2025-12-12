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
    ]

    for modality in config['modalities']:
        if not modality['enabled']:
            continue
        metadata_file_name = modality['name'] + '_metadata.json'
        emb_file_name = modality['name'] + '_' + modality['emb_model'] + '_' + str(modality['emb_dim']) + '.npy'
        pca_file_name = modality['name'] + '_final_emb_' + str(modality['emb_pca']) + '.npy'
        required_files.append(os.path.join(processed_dir, metadata_file_name))
        required_files.append(os.path.join(processed_dir, emb_file_name))
        required_files.append(os.path.join(processed_dir, pca_file_name))

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
    joint_path = os.path.join(data_path, 'Amazon2014', joint_dataset_name)
    src_path = os.path.join(data_path, 'Amazon2014', domains[0], 'processed')
    tgt_path = os.path.join(data_path, 'Amazon2014', domains[1], 'processed')

    os.makedirs(os.path.join(joint_path, 'src'), exist_ok=True)
    os.makedirs(os.path.join(joint_path, 'tgt'), exist_ok=True)
    logger.info(f"[JOINT] Creating joint dataset: {joint_dataset_name}")

    # Load
    with open(os.path.join(src_path, 'all_item_seqs.json'), 'r') as f:
        all_item_seqs_src = json.load(f)
    with open(os.path.join(src_path, 'id_mapping.json'), 'r') as f:
        id_mapping_src = json.load(f)
    with open(os.path.join(tgt_path, 'all_item_seqs.json'), 'r') as f:
        all_item_seqs_tgt = json.load(f)
    with open(os.path.join(tgt_path, 'id_mapping.json'), 'r') as f:
        id_mapping_tgt = json.load(f)

    for modality in config['modalities']:
        if not modality['enabled']:
            continue
        final_embs_file_name = modality['name'] + '_final_emb_' + str(modality['emb_pca']) + '.npy'
        embeddings_src = np.load(os.path.join(src_path, final_embs_file_name))
        embeddings_tgt = np.load(os.path.join(tgt_path, final_embs_file_name))
        np.save(os.path.join(joint_path, 'src', final_embs_file_name), embeddings_src)
        np.save(os.path.join(joint_path, 'tgt', final_embs_file_name), embeddings_tgt)

    with open(os.path.join(joint_path, 'src', 'all_item_seqs.json'), 'w') as f:
        json.dump(all_item_seqs_src, f)
    with open(os.path.join(joint_path, 'src', 'id_mapping.json'), 'w') as f:
        json.dump(id_mapping_src, f)
    with open(os.path.join(joint_path, 'tgt', 'all_item_seqs.json'), 'w') as f:
        json.dump(all_item_seqs_tgt, f)
    with open(os.path.join(joint_path, 'tgt', 'id_mapping.json'), 'w') as f:
        json.dump(id_mapping_tgt, f)

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
    joint_path = os.path.join(data_path, 'Amazon2014', joint_dataset_name)

    required_joint_files = [
        os.path.join(joint_path, 'src', 'all_item_seqs.json'),
        os.path.join(joint_path, 'src', 'id_mapping.json'),
        os.path.join(joint_path, 'tgt', 'all_item_seqs.json'),
        os.path.join(joint_path, 'tgt', 'id_mapping.json'),
    ]

    for modality in config['modalities']:
        if not modality['enabled']:
            continue
        final_embs_file_name = modality['name'] + '_final_emb_' + str(modality['emb_pca']) + '.npy'
        required_joint_files.append(os.path.join(joint_path, 'src', final_embs_file_name))
        required_joint_files.append(os.path.join(joint_path, 'tgt', final_embs_file_name))

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