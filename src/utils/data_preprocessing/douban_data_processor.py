import os
import gzip
import json
import random
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import threading
from logging import getLogger
import zipfile

logger = getLogger()

class DoubanDataProcessor:
    def __init__(self, config, domain: str, data_path: str = "../data/"):
        self.config = config
        self.domain = domain
        self.domain_path = os.path.join(data_path, 'Douban', domain)
        self.raw_dir = os.path.join(self.domain_path, 'raw')
        self.processed_dir = os.path.join(self.domain_path, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.kaggle_user_name = config['kaggle_user_name'] if config['kaggle_user_name'] else '1'
        self.kaggle_api_token = config['kaggle_api_token'] if config['kaggle_api_token'] else '1'

        self.all_item_seqs = {}

    def _check_available_domain(self):
        available_domains = [
            'Book', 'Movie', 'Music'
        ]
        assert self.domain in available_domains, f'domain "{self.domain}" not available. Available categories: {available_domains}'

    def _authenticate(self):
        try:
            os.environ['KAGGLE_USERNAME'] = self.kaggle_user_name
            os.environ['KAGGLE_KEY'] = self.kaggle_api_token

            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            logger.info(f"Authentication successful for domain: {self.domain}")
            return api
        except Exception as e:
            logger.error(f"Authentication failed for domain: {self.domain}.")
            logger.error("Please check your Kaggle credentials in the Douban.yaml file.")
            raise ValueError(
                "Authentication failed. Please configure the correct kaggle_user_name and kaggle_api_token in Douban.yaml.")

    def _download_raw(self) -> str:
        dataset = 'fengzhujoey/douban-datasetratingreviewside-information'
        zip_filepath = os.path.join(self.raw_dir, 'douban-datasetratingreviewside-information.zip')

        if not os.path.exists(zip_filepath):
            logger.info(f"Downloading Douban dataset from Kaggle...")
            api = self._authenticate()
            api.dataset_download_files(dataset, path=self.raw_dir, unzip=False)
        return zip_filepath

    def _process_raw_files(self, zip_path: str):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            if self.domain == 'Book':
                logger.info("Processing Book Domain...")
                reviews_path, user_meta_path, item_meta_path = self._process_book_domain(zip_ref, file_names)
            elif self.domain == 'Movie':
                logger.info("Processing Movie Domain...")
                reviews_path, user_meta_path, item_meta_path = self._process_movie_domain(zip_ref, file_names)
            elif self.domain == 'Music':
                logger.info("Processing Music Domain...")
                reviews_path, user_meta_path, item_meta_path = self._process_music_domain(zip_ref, file_names)
            else:
                logger.error(f"Unsupported domain: {self.domain}")
                return None, None
            return reviews_path, user_meta_path, item_meta_path

    def _process_book_domain(self, zip_ref, file_names):
        reviews_out = os.path.join(self.raw_dir, "reviews.json.gz")
        user_meta_out = os.path.join(self.raw_dir, "user_meta.json.gz")
        item_meta_out = os.path.join(self.raw_dir, "item_meta.json.gz")

        if os.path.exists(reviews_out) and os.path.exists(user_meta_out) and os.path.exists(item_meta_out):
            logger.info("Processed Book files already exist. Skip rebuilding.")
            return reviews_out, user_meta_out, item_meta_out

        # Find required raw files in zip
        users_txt = None
        reviews_txt = None
        for name in file_names:
            if name.endswith("users_cleaned.txt"):
                users_txt = name
            elif name.endswith("bookreviews_cleaned.txt"):
                reviews_txt = name

        if users_txt is None or reviews_txt is None:
            raise RuntimeError(
                f"Book domain: required raw txt files not "
                f"found in zip: [\"users_cleaned.txt\", \"bookreviews_cleaned.txt\"].\nCurrent:{file_names}")

        logger.info("Building Book domain raw json.gz files...")

        # Read raw txt files from zip
        with zip_ref.open(users_txt) as f:
            users_lines = f.read().decode("utf-8").splitlines()

        with zip_ref.open(reviews_txt) as f:
            reviews_lines = f.read().decode("utf-8").splitlines()

        self._build_book_jsons(users_lines, reviews_lines,
                               reviews_out, user_meta_out, item_meta_out)

        return reviews_out, user_meta_out, item_meta_out

    def _build_book_jsons(self, users_lines, reviews_lines,
                          reviews_out, user_meta_out, item_meta_out):

        # -------- Step 1: parse users_cleaned.txt --------
        raw_header = users_lines[0].split("\t")
        header = [h.strip().strip('"') for h in raw_header]
        uid_idx = header.index("UID")
        living_idx = header.index("living_place")
        join_idx = header.index("join_time")
        stmt_idx = header.index("self_statement")

        user_meta_map = {}
        for line in users_lines[1:]:
            parts = [p.strip().strip('"') for p in line.split("\t")]
            uid = parts[uid_idx]
            user_meta_map[uid] = {
                "uid": uid,
                "living_place": parts[living_idx],
                "join_time": parts[join_idx],
                "self_statement": parts[stmt_idx]
            }

        # -------- Step 2: parse bookreviews_cleaned.txt --------
        raw_header = reviews_lines[0].split("\t")
        header = [h.strip().strip('"') for h in raw_header]
        u_idx = header.index("user_id")
        i_idx = header.index("book_id")
        r_idx = header.index("rating")
        l_idx = header.index("labels")
        c_idx = header.index("comment")
        t_idx = header.index("time")

        interactions = []
        used_users = set()
        item_labels = defaultdict(set)
        for line in reviews_lines[1:]:
            parts = [p.strip().strip('"') for p in line.split("\t")]
            uid = parts[u_idx]
            iid = parts[i_idx]

            used_users.add(uid)

            # build interaction
            interactions.append({
                "uid": uid,
                "iid": iid,
                "rating": float(parts[r_idx]),
                "comments": parts[c_idx],
                "time": parts[t_idx]
            })

            # merge labels
            labels = parts[l_idx].split("|")
            for l in labels:
                if l.strip():
                    item_labels[iid].add(l.strip())

        # -------- Step 3: write reviews.json.gz --------
        with gzip.open(reviews_out, "wt", encoding="utf-8") as f:
            for x in interactions:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        # -------- Step 4: write user_meta.json.gz (only users that appear in interactions) --------
        with gzip.open(user_meta_out, "wt", encoding="utf-8") as f:
            for uid in used_users:
                if uid in user_meta_map:
                    f.write(json.dumps(user_meta_map[uid], ensure_ascii=False) + "\n")

        # -------- Step 5: write item_meta.json.gz (merged labels) --------
        with gzip.open(item_meta_out, "wt", encoding="utf-8") as f:
            for iid, label_set in item_labels.items():
                labels = "|".join(sorted(label_set))
                f.write(json.dumps({"iid": iid, "labels": labels}, ensure_ascii=False) + "\n")

        logger.info("Book domain json.gz files built successfully.")

    def _process_movie_domain(self, zip_ref, file_names):
        pass

    def _process_music_domain(self, zip_ref, file_names):
        pass

    def run_full_pipeline(self):
        # 分两步
        logger.info(f"Starting Douban dataset processing - Domain: {self.domain}")

        self._check_available_domain()

        logger.info("\n=== Step 1: Download raw data ===")
        zip_path = self._download_raw()
        reviews_path, user_meta_path, item_meta_path = self._process_raw_files(zip_path)

        logger.info("\n=== Step 2: Process reviews ===")

