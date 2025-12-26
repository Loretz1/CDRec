import os
import gzip
import json
import random
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import requests
from logging import getLogger

logger = getLogger()

class AmazonDataProcessor:
    def __init__(self, config, domain: str, data_path: str = "../data/"):
        self.config = config
        self.domain = domain
        self.domain_path = os.path.join(data_path, 'Amazon2014', domain)
        self.raw_dir = os.path.join(self.domain_path, 'raw')
        self.processed_dir = os.path.join(self.domain_path, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.all_item_seqs = {}

    def _check_available_domain(self):
        available_domains = [
            'Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl',
            'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Kindle_Store',
            'Sports_and_Outdoors', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games', 'Video_Games',
            'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android',
            'Office_Products', 'Pet_Supplies', 'Automotive',
            'Grocery_and_Gourmet_Food', 'Patio_Lawn_and_Garden', 'Baby',
            'Digital_Music', 'Musical_Instruments', 'Amazon_Instant_Video'
        ]
        assert self.domain in available_domains, f'domain "{self.domain}" not available. Available categories: {available_domains}'

    def download_file(self, url: str, local_path: str):
        if os.path.exists(local_path):
            logger.info(f"File already exists: {local_path}")
            return

        logger.info(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(
                desc=os.path.basename(local_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def _download_raw(self, data_type: str = 'reviews') -> str:
        url = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{data_type}_{self.domain}{"_5" if data_type == "reviews" else ""}.json.gz'
        base_name = os.path.basename(url)
        local_filepath = os.path.join(self.raw_dir, base_name)

        if not os.path.exists(local_filepath):
            self.download_file(url, local_filepath)
        return local_filepath

    def _parse_gz(self, path: str):
        with gzip.open(path, 'r') as g:
            for line in g:
                line = line.replace(b'true', b'True').replace(b'false', b'False')
                yield eval(line)

    def _load_reviews(self, path: str) -> List[Tuple]:
        logger.info('[DATASET] Loading reviews...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
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

    def _process_reviews(self, input_path: str) -> Dict:
        seq_file = os.path.join(self.processed_dir, 'all_item_seqs.json')

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
        """
        功能：
            执行 Amazon2014 单域数据集的完整处理流程，生成该 domain 的基础 processed 数据。
        主要流程：
            1. 下载原始 reviews / meta 数据（若本地不存在）
            2. 解析 reviews，构建用户的交互序列，如果config['shuffle_user_sequence']则打乱每个用户的交互，否则交互按时间顺序排序
            3. 生成并保存 processed/all_item_seqs.json
        产出文件：
            - processed/all_item_seqs.json
              （单域内所有用户的完整交互序列）
                {
                  "<raw_user_id_1>": [
                    "<raw_item_id_1>",
                    "<raw_item_id_2>",
                    "<raw_item_id_3>",
                    ...
                  ],
                  "<raw_user_id_2>": [
                    "<raw_item_id_4>",
                    "<raw_item_id_5>",
                    ...
                  ],
                  ...
                }
        说明：
            - 若 processed 数据已存在，将自动跳过重复处理
            - 该方法仅处理单域数据，不涉及跨域划分或重编号
        """
        logger.info(f"Starting Amazon Reviews 2014 dataset processing - Domain: {self.domain}")

        self._check_available_domain()

        logger.info("\n=== Step 1: Download raw data ===")
        reviews_path = self._download_raw('reviews')
        meta_path = self._download_raw('meta')

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

