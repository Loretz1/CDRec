import os
import gzip
import json
import numpy as np
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
        self.id_mapping = {
            'user2id': {},
            'item2id': {},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = []

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
            print(f"File already exists: {local_path}")
            return

        print(f"Downloading: {url}")
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
        print('[DATASET] Loading reviews...')
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

        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [item for item, _ in item_time]
        return item_seqs

    def _remap_ids(self, item_seqs: Dict) -> Tuple[Dict, Dict]:
        print('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)

            iids = []
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)
            self.all_item_seqs[user] = iids

        return self.all_item_seqs, self.id_mapping

    def _process_reviews(self, input_path: str) -> Tuple[Dict, Dict]:
        seq_file = os.path.join(self.processed_dir, 'all_item_seqs.json')
        id_mapping_file = os.path.join(self.processed_dir, 'id_mapping.json')

        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            print('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping

        print('[DATASET] Processing reviews...')
        reviews = self._load_reviews(input_path)
        item_seqs = self._get_item_seqs(reviews)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)

        print('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)

        return all_item_seqs, id_mapping

    def _load_metadata(self, path: str, item2id: Dict) -> Dict:
        print('[DATASET] Loading metadata...')
        data = {}
        item_asins = set(item2id.keys())
        for info in tqdm(self._parse_gz(path)):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
        return data

    def clean_text(self, raw_text: str) -> str:
        import re
        import html

        if isinstance(raw_text, list):
            raw_text = ' '.join(str(item) for item in raw_text)

        text = str(raw_text)
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def _sent_process(self, raw) -> str:
        sentence = ""
        if isinstance(raw, float):
            sentence += str(raw) + '.'
        elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            for v1 in raw:
                for v in v1:
                    sentence += self.clean_text(str(v))[:-1] + ', '
            sentence = sentence[:-2] + '.'
        elif isinstance(raw, list):
            for v1 in raw:
                sentence += self.clean_text(str(v1))
        else:
            sentence = self.clean_text(str(raw))
        return sentence + ' '

    def _extract_meta_sentences(self, metadata: Dict) -> Dict:
        print('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            meta_sentence = ''
            keys = set(meta.keys())
            features_needed = ['title', 'price', 'brand', 'feature', 'categories', 'description']
            for feature in features_needed:
                if feature in keys:
                    meta_sentence += self._sent_process(meta[feature])
            item2meta[item] = meta_sentence
        return item2meta

    def _process_meta(self, input_path: str) -> Optional[Dict]:
        result = []

        # judge whether to load metadata or not
        for id, modality in enumerate(self.config['modalities']):
            if not modality['enabled']:
                continue
            meta_file = os.path.join(self.processed_dir, modality['name'] + '_metadata.json')
            if not os.path.exists(meta_file):
                metadata = self._load_metadata(path=input_path, item2id=self.id_mapping['item2id'])
                break

        for modality in self.config['modalities']:
            if not modality['enabled']:
                continue

            meta_file = os.path.join(self.processed_dir, modality['name'] + '_metadata.json')
            if os.path.exists(meta_file):
                print(f'[DATASET] Metadata [{modality["name"]}] has been processed...')
                with open(meta_file, 'r') as f:
                    result.append(json.load(f))
                continue

            print(f'[DATASET] Processing metadata, mode: {modality["name"]}')

            if modality["name"] == 'sentence':
                proceeded_metadata = self._extract_meta_sentences(metadata=metadata)

            with open(meta_file, 'w') as f:
                json.dump(proceeded_metadata, f)
            result.append(proceeded_metadata)

        return result

    def _encode_sent_emb(self, output_path: str, modality, id) -> np.ndarray:
        meta_sentences = []
        for i in range(1, len(self.id_mapping['id2item'])):
            item = self.id_mapping['id2item'][i]
            meta_sentences.append(self.item2meta[id][item])

        if 'sentence-transformers' in modality['emb_model']:
            try:
                from sentence_transformers import SentenceTransformer
                device = self.config.get('device', 'cpu')
                sent_emb_model = SentenceTransformer(modality['emb_model']).to(device)

                sent_embs = sent_emb_model.encode(
                    meta_sentences,
                    convert_to_numpy=True,
                    batch_size=modality['emb_batch_size'],
                    show_progress_bar=True,
                    device=device
                )
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        elif 'text-embedding-3' in modality['emb_model']:
            if not self.config['openai_api_key']:
                raise ValueError("OpenAI API key required for OpenAI embeddings")

            try:
                from openai import OpenAI

                client_kwargs = {'api_key': self.config['openai_api_key']}
                if 'openai_base_url' in self.config and self.config['openai_base_url']:
                    client_kwargs['base_url'] = self.config['openai_base_url']

                client = OpenAI(**client_kwargs)

                sent_embs = []
                for i in tqdm(range(0, len(meta_sentences), modality['emb_batch_size']), desc='Encoding'):
                    batch = meta_sentences[i:i + modality['emb_batch_size']]
                    try:
                        responses = client.embeddings.create(
                            input=batch,
                            model=modality['emb_model']
                        )
                        for response in responses.data:
                            sent_embs.append(response.embedding)
                    except Exception as e:
                        print(f'Encoding failed {i} - {i + modality["emb_batch_size"]}: {e}')

                        try:
                            new_batch = []
                            for sent in batch:
                                if len(sent) > 8000:
                                    new_batch.append(sent[:8000])
                                else:
                                    new_batch.append(sent)

                            print(f'[TOKENIZER] Retrying batch {i} - {i + modality["emb_batch_size"]}')
                            import time
                            time.sleep(2)

                            responses = client.embeddings.create(
                                input=new_batch,
                                model=modality['emb_model']
                            )
                            for response in responses.data:
                                sent_embs.append(response.embedding)
                        except Exception as retry_e:
                            print(f'Retry also failed: {retry_e}')
                            raise retry_e

                sent_embs = np.array(sent_embs, dtype=np.float32)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError(f"Unsupported embedding model: {modality['emb_model']}")

        np.save(output_path, sent_embs)
        print(f'[TOKENIZER] Sentence embeddings saved to: {output_path}')
        return sent_embs

    def _pca_sent_emb(self, output_path: str, modality, embs) -> np.ndarray:
        if modality['emb_pca'] == modality['emb_dim']:
            pca_embs = embs
        elif modality['emb_pca'] < modality['emb_dim']:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=modality['emb_pca'], whiten=True)
                pca_embs = pca.fit_transform(embs)
            except ImportError:
                raise ImportError("Please install scikit-learn: pip install scikit-learn")
        else:
            raise ValueError(f"The dimension of emb_pca must be less than or equal to emb_dim for {modality['name']}")

        np.save(output_path, pca_embs)
        print(f'[TOKENIZER] PCA sentence embeddings saved to: {output_path}')
        return pca_embs

    def generate_embeddings(self):
        for id, modality in enumerate(self.config['modalities']):
            if not modality['enabled']:
                continue

            # emb
            emb_file_path = os.path.join(self.processed_dir, modality['name'] + '_' + modality['emb_model'] + '_' + str(modality['emb_dim']) + '.npy')
            if os.path.exists(emb_file_path):
                print(f'[TOKENIZER] Loading {modality["name"]} embeddings: {emb_file_path}...')
                embs = np.load(emb_file_path)
            else:
                print(f'[TOKENIZER] Encoding {modality["name"]} embeddings...')
                if modality["name"] == 'sentence':
                    embs = self._encode_sent_emb(emb_file_path, modality, id)
            print(f'[TOKENIZER] {modality["name"]} embeddings shape: {embs.shape}')

            # pca
            pca_file_path = os.path.join(self.processed_dir, modality['name'] + '_final_emb_' + str(modality['emb_pca']) + '.npy')
            if os.path.exists(pca_file_path):
                print(f'[TOKENIZER] Loading {modality["name"]} pca embeddings: {pca_file_path}...')
                pca_embs = np.load(pca_file_path)
            else:
                print(f'[TOKENIZER] Applying PCA to {modality["name"]} embeddings...')
                if modality["name"] == 'sentence':
                    pca_embs = self._pca_sent_emb(pca_file_path, modality, embs)
            print(f'[TOKENIZER] {modality["name"]} pca embeddings shape: {pca_embs.shape}')

    def run_full_pipeline(self):
        print(f"Starting Amazon Reviews 2014 dataset processing - Domain: {self.domain}")

        self._check_available_domain()

        print("\n=== Step 1: Download raw data ===")
        reviews_path = self._download_raw('reviews')
        meta_path = self._download_raw('meta')

        print("\n=== Step 2: Process reviews ===")
        self.all_item_seqs, self.id_mapping = self._process_reviews(reviews_path)

        print("\n=== Step 3: Process metadata ===")
        self.item2meta = self._process_meta(meta_path)

        if self.item2meta:
            print("\n=== Step 4: Generate embeddings ===")
            self.generate_embeddings()

        print(f"\n=== Processing completed ===")
        print(f"Data saved in: {self.domain_path}")
        print(f"Raw data: {self.raw_dir}")
        print(f"Processed data: {self.processed_dir}")

        print("\nGenerated files:")
        for root, dirs, files in os.walk(self.domain_path):
            level = root.replace(self.domain_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

