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
import csv
import io

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

    def _kcore_filter(self, interactions, k=3):
        while True:
            user_cnt = defaultdict(int)
            item_cnt = defaultdict(int)

            for x in interactions:
                user_cnt[x["uid"]] += 1
                item_cnt[x["iid"]] += 1

            new_interactions = [
                x for x in interactions
                if user_cnt[x["uid"]] >= k and item_cnt[x["iid"]] >= k
            ]

            if len(new_interactions) == len(interactions):
                break
            interactions = new_interactions

        return interactions

    def _iter_tsv_rows_from_zip(self, zip_ref, inner_path: str):
        # csv module needs newline='' to correctly handle embedded newlines in quoted fields
        with zip_ref.open(inner_path, "r") as bf:
            tf = io.TextIOWrapper(bf, encoding="utf-8", newline="")
            reader = csv.reader(tf, delimiter="\t", quotechar='"')
            for row in reader:
                yield row

    def _parse_movie_reviews_from_zip(self, zip_ref, reviews_txt):
        it = self._iter_tsv_rows_from_zip(zip_ref, reviews_txt)
        header = next(it)
        header = [h.strip().strip('"') for h in header]

        u_idx = header.index("user_id")
        i_idx = header.index("movie_id")
        r_idx = header.index("rating")
        c_idx = header.index("comment")
        t_idx = header.index("time")

        seen = set()
        raw_interactions = []

        for row in it:
            if len(row) < len(header):
                continue  # skip broken row

            rating_str = (row[r_idx] or "").strip()
            if rating_str == "":
                continue  # drop no-rating

            uid = (row[u_idx] or "").strip().strip('"')
            iid = (row[i_idx] or "").strip().strip('"')
            if uid == "" or iid == "":
                continue

            key = (uid, iid)
            if key in seen:
                continue
            seen.add(key)

            raw_interactions.append({
                "uid": uid,
                "iid": iid,
                "rating": float(rating_str),
                "comments": row[c_idx],
                "time": row[t_idx]
            })

        return raw_interactions

    def _parse_movies_meta_from_zip(self, zip_ref, movies_txt, used_items):
        it = self._iter_tsv_rows_from_zip(zip_ref, movies_txt)
        header = next(it)
        header = [h.strip().strip('"') for h in header]

        mid_idx = header.index("UID")

        movie_meta_map = {}

        for row in it:
            if len(row) < len(header):
                continue

            mid = (row[mid_idx] or "").strip().strip('"')
            if mid == "" or mid not in used_items:
                continue

            meta = {"iid": mid}
            for j, col in enumerate(header):
                if col == "movie_id" or col == "UID":
                    continue
                meta[col] = row[j]
            movie_meta_map[mid] = meta

        return movie_meta_map

    def _build_user_item_jsons(self, users_lines, reviews_lines,
                               item_id_field,
                               reviews_out, user_meta_out, item_meta_out):

        # =========================
        # Step 1: parse users_cleaned.txt
        # =========================
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

        # =========================
        # Step 2: parse raw reviews + deduplicate
        # =========================
        raw_header = reviews_lines[0].split("\t")
        header = [h.strip().strip('"') for h in raw_header]

        u_idx = header.index("user_id")
        i_idx = header.index(item_id_field)
        r_idx = header.index("rating")
        l_idx = header.index("labels")
        c_idx = header.index("comment")
        t_idx = header.index("time")

        seen = set()
        raw_interactions = []
        raw_labels = defaultdict(set)

        for line in reviews_lines[1:]:
            parts = [p.strip().strip('"') for p in line.split("\t")]

            uid = parts[u_idx]
            iid = parts[i_idx]

            key = (uid, iid)
            if key in seen:
                continue
            seen.add(key)

            raw_interactions.append({
                "uid": uid,
                "iid": iid,
                "rating": float(parts[r_idx]),
                "comments": parts[c_idx],
                "time": parts[t_idx]
            })

            for l in parts[l_idx].split("|"):
                if l.strip():
                    raw_labels[iid].add(l.strip())

        # =========================
        # Step 3: 5-core filtering
        # =========================
        interactions = self._kcore_filter(raw_interactions, k=3)

        # =========================
        # Step 4: recompute used users & items
        # =========================
        used_users = {x["uid"] for x in interactions}
        used_items = {x["iid"] for x in interactions}

        # rebuild labels only for kept items
        item_labels = {iid: raw_labels.get(iid, set()) for iid in used_items}

        # =========================
        # Step 5: write reviews.json.gz
        # =========================
        with gzip.open(reviews_out, "wt", encoding="utf-8") as f:
            for x in interactions:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        # =========================
        # Step 6: write user_meta.json.gz (exact match)
        # =========================
        with gzip.open(user_meta_out, "wt", encoding="utf-8") as f:
            for uid in used_users:
                if uid in user_meta_map:
                    f.write(json.dumps(user_meta_map[uid], ensure_ascii=False) + "\n")
                else:
                    # user missing in users_cleaned.txt → must still exist
                    f.write(json.dumps({
                        "uid": uid,
                        "living_place": "",
                        "join_time": "",
                        "self_statement": ""
                    }, ensure_ascii=False) + "\n")

        # =========================
        # Step 7: write item_meta.json.gz
        # =========================
        with gzip.open(item_meta_out, "wt", encoding="utf-8") as f:
            for iid in used_items:
                labels = "|".join(sorted(item_labels.get(iid, [])))
                f.write(json.dumps({"iid": iid, "labels": labels}, ensure_ascii=False) + "\n")

    def _build_movie_jsons(self, zip_ref, users_lines, reviews_txt, movies_txt,
                           reviews_out, user_meta_out, item_meta_out, k=3):

        # =========================
        # Step 1: parse users_cleaned.txt
        # =========================
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

        # =========================
        # Step 2: parse moviereviews_cleaned.txt (CSV-safe)
        # =========================
        raw_interactions = self._parse_movie_reviews_from_zip(zip_ref, reviews_txt)

        # =========================
        # Step 3: k-core filtering
        # =========================
        interactions = self._kcore_filter(raw_interactions, k=k)

        used_users = {x["uid"] for x in interactions}
        used_items = {x["iid"] for x in interactions}

        # =========================
        # Step 4: parse movies_cleaned.txt (CSV-safe)
        # =========================
        movie_meta_map = self._parse_movies_meta_from_zip(zip_ref, movies_txt, used_items)

        # =========================
        # Step 5: write reviews.json.gz
        # =========================
        with gzip.open(reviews_out, "wt", encoding="utf-8") as f:
            for x in interactions:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        # =========================
        # Step 6: write user_meta.json.gz
        # =========================
        with gzip.open(user_meta_out, "wt", encoding="utf-8") as f:
            for uid in used_users:
                if uid in user_meta_map:
                    f.write(json.dumps(user_meta_map[uid], ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({
                        "uid": uid,
                        "living_place": "",
                        "join_time": "",
                        "self_statement": ""
                    }, ensure_ascii=False) + "\n")

        # =========================
        # Step 7: write item_meta.json.gz
        # =========================
        with gzip.open(item_meta_out, "wt", encoding="utf-8") as f:
            for iid in used_items:
                if iid in movie_meta_map:
                    f.write(json.dumps(movie_meta_map[iid], ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({"iid": iid}, ensure_ascii=False) + "\n")

        logger.info("Movie domain json.gz built successfully.")

    def _parse_gz(self, path: str):
        with gzip.open(path, 'r') as g:
            for line in g:
                line = line.replace(b'true', b'True').replace(b'false', b'False')
                yield eval(line)

    def _load_reviews(self, path: str) -> List[Tuple]:
        logger.info('[DATASET] Loading reviews...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['uid']
            item = inter['iid']
            time = inter['time']
            reviews.append((user, item, time))
        return reviews

    def _get_item_seqs(self, reviews: List[Tuple]) -> Dict:
        item_seqs = defaultdict(list)
        for user, item, time in reviews:
            item_seqs[user].append((item, time))

        # fix the seed for shuffle
        rng = random.Random(999)

        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            seq = [item for item, _ in item_time]
            if self.config.get("shuffle_user_sequence", True):
                rng.shuffle(seq)
            item_seqs[user] = seq
        return item_seqs




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

        with zip_ref.open(users_txt) as f:
            users_lines = f.read().decode("utf-8").splitlines()

        with zip_ref.open(reviews_txt) as f:
            reviews_lines = f.read().decode("utf-8").splitlines()

        self._build_user_item_jsons(
            users_lines,
            reviews_lines,
            "book_id",
            reviews_out,
            user_meta_out,
            item_meta_out
        )

        logger.info("Book domain json.gz files built successfully.")
        return reviews_out, user_meta_out, item_meta_out

    def _process_music_domain(self, zip_ref, file_names):
        reviews_out = os.path.join(self.raw_dir, "reviews.json.gz")
        user_meta_out = os.path.join(self.raw_dir, "user_meta.json.gz")
        item_meta_out = os.path.join(self.raw_dir, "item_meta.json.gz")

        if os.path.exists(reviews_out) and os.path.exists(user_meta_out) and os.path.exists(item_meta_out):
            logger.info("Processed Music files already exist. Skip rebuilding.")
            return reviews_out, user_meta_out, item_meta_out

        users_txt = None
        reviews_txt = None
        for name in file_names:
            if name.endswith("users_cleaned.txt"):
                users_txt = name
            elif name.endswith("musicreviews_cleaned.txt"):
                reviews_txt = name

        if users_txt is None or reviews_txt is None:
            raise RuntimeError(
                f"Music domain: required raw txt files not "
                f"found in zip: [\"users_cleaned.txt\", \"musicreviews_cleaned.txt\"].\nCurrent:{file_names}")

        with zip_ref.open(users_txt) as f:
            users_lines = f.read().decode("utf-8").splitlines()

        with zip_ref.open(reviews_txt) as f:
            reviews_lines = f.read().decode("utf-8").splitlines()

        self._build_user_item_jsons(
            users_lines,
            reviews_lines,
            "music_id",
            reviews_out, user_meta_out, item_meta_out
        )

        logger.info("Music domain json.gz files built successfully.")
        return reviews_out, user_meta_out, item_meta_out

    def _process_movie_domain(self, zip_ref, file_names):
        reviews_out = os.path.join(self.raw_dir, "reviews.json.gz")
        user_meta_out = os.path.join(self.raw_dir, "user_meta.json.gz")
        item_meta_out = os.path.join(self.raw_dir, "item_meta.json.gz")

        if os.path.exists(reviews_out) and os.path.exists(user_meta_out) and os.path.exists(item_meta_out):
            logger.info("Processed Movie files already exist. Skip rebuilding.")
            return reviews_out, user_meta_out, item_meta_out

        users_txt = None
        reviews_txt = None
        movies_txt = None

        for name in file_names:
            if name.endswith("users_cleaned.txt"):
                users_txt = name
            elif name.endswith("moviereviews_cleaned.txt"):
                reviews_txt = name
            elif name.endswith("movies_cleaned.txt"):
                movies_txt = name

        if not users_txt or not reviews_txt or not movies_txt:
            raise RuntimeError(
                f"Music domain: required raw txt files not "
                f"found in zip: [\"users_cleaned.txt\", \"moviereviews_cleaned.txt\", \"movies_cleaned.txt\"].\nCurrent:{file_names}")

        with zip_ref.open(users_txt) as f:
            users_lines = f.read().decode("utf-8").splitlines()

        self._build_movie_jsons(
            zip_ref=zip_ref,
            users_lines=users_lines,
            reviews_txt=reviews_txt,
            movies_txt=movies_txt,
            reviews_out=reviews_out,
            user_meta_out=user_meta_out,
            item_meta_out=item_meta_out,
            k=3
        )

        logger.info("Movie domain json.gz files built successfully.")
        return reviews_out, user_meta_out, item_meta_out

    def _process_reviews(self, input_path: str) -> Dict:
        seq_file = (
            f"all_item_seqs_"
            f"{'shuffle' if self.config['shuffle_user_sequence'] else 'noshuffle'}.json"
        )
        seq_file = os.path.join(self.processed_dir, seq_file)

        if os.path.exists(seq_file):
            logger.info('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            return all_item_seqs

        logger.info('[DATASET] Processing reviews...')
        reviews = self._load_reviews(input_path)
        all_item_seqs = self._get_item_seqs(reviews)

        logger.info('[DATASET] Saving seq data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)

        return all_item_seqs

    def run_full_pipeline(self):
        # 分两步
        logger.info(f"Starting Douban dataset processing - Domain: {self.domain}")

        self._check_available_domain()

        logger.info("\n=== Step 1: Download raw data ===")
        zip_path = self._download_raw()
        # 从原始zip中提取出本域需要的三个json.gz文件,放在raw文件夹中
        # 对于交互文件，进行了去重，3-cores操作
        # 对于物品侧文件、用户侧文件，只留下了存在交互的物品和用户
        reviews_path, user_meta_path, item_meta_path = self._process_raw_files(zip_path)

        logger.info("\n=== Step 2: Process reviews ===")
        self.all_item_seqs = self._process_reviews(reviews_path)

        logger.info(f"\n=== Processing completed ===")
        logger.info(f"Data saved in: {self.domain_path}")
        logger.info(f"Raw data: {self.raw_dir}")
        logger.info(f"Processed data: {self.processed_dir}")

        logger.info("\nGenerated files:")
        for root, dirs, files in os.walk(self.domain_path):
            level = root.replace(self.domain_path, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                logger.info(f"{subindent}{file}")

