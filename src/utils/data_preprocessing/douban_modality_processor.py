import logging
import os
import gzip
import json
import importlib
from tqdm import tqdm
import numpy as np

logger = logging.getLogger()

class DoubanModalityProcessor:
    def __init__(self, config, domains, id_mapping, train_df, joint_path):
        self.config = config
        self.domains = domains
        self.id_mapping = id_mapping
        self.interaction = train_df
        self.joint_path = os.path.join(joint_path, 'modality_emb')
        self.reviews = {}
        self.user_metadata = {}
        self.item_metadata = {}

    def _parse_gz(self, path: str):
        with gzip.open(path, 'r') as g:
            for line in g:
                line = line.replace(b'true', b'True').replace(b'false', b'False')
                yield eval(line)

    def _load_raw(self, reviews_path, user_metadata_path, item_metadata_path, id_mapping, interaction):
        reviews = {}
        user_metadata = {}
        item_metadata = {}

        # 读取raw的时候需要进行一些筛选，
        # 去除掉无效user、item
        # 去掉不在train中的u-i交互，防止处理模态数据时发生数据泄露
        valid_users = set(id_mapping['user2id'].keys())
        valid_items = set(id_mapping['item2id'].keys())

        interaction_pairs = set(
            (id_mapping['id2user'][u], id_mapping['id2item'][i])
            for u, i in interaction[['user', 'item']].to_numpy()
        )

        for inter in tqdm(self._parse_gz(reviews_path)):
            user = inter['uid']
            item = inter['iid']
            text = inter['comments']
            if user not in valid_users or item not in valid_items:
                continue
            if (user, item) not in interaction_pairs:
                continue
            reviews[(user, item)] = text
        for info in tqdm(self._parse_gz(user_metadata_path)):
            user = info.get('uid')
            if user not in valid_users:
                continue
            user_metadata[user] = info
        for info in tqdm(self._parse_gz(item_metadata_path)):
            item = info.get('iid')
            if item not in valid_items:
                continue
            item_metadata[item] = info
        return reviews, user_metadata, item_metadata

    def _get_modality_handler(self, func_name):
        # Get the processing method from model
        model_name = self.config["model"].lower()
        try:
            model_module = importlib.import_module(f"models.{model_name}")
            if hasattr(model_module, func_name):
                return getattr(model_module, func_name)
        except ModuleNotFoundError:
            logger.warning(f"[WARN] model.{model_name} not found")

        # Get the processing method from self
        if hasattr(self, func_name):
            return getattr(self, func_name)

        raise NotImplementedError(
            f"Handler '{func_name}' was not found.\n"
            f"Searched in:\n"
            f"  - model/{model_name}.py\n"
            f"  - {self.__class__.__name__}\n"
            f"Please implement '{func_name}' in one of the above locations."
        )

    def _create_modality_data(self, modality):
        modality_file_name = modality['name'] + '_metadata.json'
        modality_data_path = os.path.join(self.joint_path, modality_file_name)
        if os.path.exists(modality_data_path):
            logger.info(f'[DATASET] Modality data: {modality["name"]} is already created.')
            with open(modality_data_path, 'r') as f:
                modality_data = json.load(f)
            return modality_data

        if self.reviews == {} and self.user_metadata == {} and self.item_metadata == {}:
            for role, domain in zip(['src', 'tgt'], self.domains):
                reviews_path = os.path.join(self.config['data_path'], self.config['dataset'], domain, 'raw',
                                            'reviews.json.gz')
                user_metadata_path = os.path.join(self.config['data_path'], self.config['dataset'], domain, 'raw',
                                            'user_meta.json.gz')
                item_metadata_path = os.path.join(self.config['data_path'], self.config['dataset'], domain, 'raw',
                                            'item_meta.json.gz')
                self.reviews[role], self.user_metadata[role], self.item_metadata[role] \
                    = self._load_raw(reviews_path, user_metadata_path, item_metadata_path, self.id_mapping[role], self.interaction[role])

        func_name = f"extract_{modality['name']}_modality_data"
        handler = self._get_modality_handler(func_name)
        logger.info(f'[DATASET] Extracting meta: {modality["name"]}...')
        modality_data = handler(self.config, modality, self.interaction, self.id_mapping, [self.reviews, self.user_metadata, self.item_metadata])
        logger.info(f'[DATASET] Saving modality data: {modality["name"]}..')
        with open(modality_data_path, 'w') as f:
            json.dump(modality_data, f)
        return modality_data

    def _create_embs(self, modality, modality_data):
        embs_file_name = modality['name'] + '_' + modality['emb_model'] + '_' + str(modality['emb_dim']) + '.npy'
        embs_path = os.path.join(self.joint_path, embs_file_name)
        if os.path.exists(embs_path):
            logger.info(f'[DATASET] Embs: {modality["name"]} is already created.')
            embs = np.load(embs_path)
            return embs

        func_name = f"generate_{modality['name']}_embs"
        handler = self._get_modality_handler(func_name)
        logger.info(f'[DATASET] Generatinng embs: {modality["name"]}...')
        embs = handler(self.config, modality, self.interaction, self.id_mapping, modality_data)
        logger.info(f'[DATASET] Saving embs: {modality["name"]}..')
        np.save(embs_path, embs)
        return embs

    def _create_final_embs(self, modality, embs):
        final_embs_file_name = modality['name'] + '_final_emb_' + str(modality['emb_pca']) + '.npy'
        final_embs_path = os.path.join(self.joint_path, final_embs_file_name)
        if os.path.exists(final_embs_path):
            logger.info(f'[DATASET] Final embs: {modality["name"]} is already created.')
            final_embs = np.load(final_embs_path)
            return final_embs

        func_name = f"generate_{modality['name']}_final_embs"
        handler = self._get_modality_handler(func_name)
        logger.info(f'[DATASET] Generatinng final embs: {modality["name"]}...')
        final_embs = handler(self.config, modality, self.interaction, self.id_mapping, embs)
        logger.info(f'[DATASET] Saving final embs: {modality["name"]}..')
        np.save(final_embs_path, final_embs)
        return final_embs

    def run_full_pipeline(self):
        try:
            for id, modality in enumerate(self.config['modalities']):
                if not modality['enabled']:
                    continue

                final_embs_file_name = modality['name'] + '_final_emb_' + str(modality['emb_pca']) + '.npy'
                if os.path.exists(os.path.join(self.joint_path, final_embs_file_name)):
                    continue

                # 三步，每一步生成一个文件
                # _create_modality_data会在本文件和<model>.py中查找：extract_<name>_modality_data方法，来处理此模态，最后生成一个json文件
                # 比如这里给出了一个extract_sentence_modality_data例子，<model>.py的优先级高于本文件的同名方法。
                modality_data = self._create_modality_data(modality)
                # _create_embs生成一个npy文件，查找：generate_<name>_embs，可以参考：generate_sentence_embs方法
                embs = self._create_embs(modality, modality_data)
                # _create_final_embs生成一个npy文件，查找：generate_<name>_final_embs，可以参考：generate_sentence_final_embs方法
                final_embs = self._create_final_embs(modality, embs)
            return True
        except Exception as e:
            logger.error(
                f"[ERROR] [{self.domain}] Modality pipeline failed: {e}",
                exc_info=True
            )
            return False

    def extract_sentence_modality_data(self, config, modality, interaction, id_mapping, raw_data_list):
        def clean_text(raw_text: str) -> str:
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

        def _sent_process(raw) -> str:
            sentence = ""
            if isinstance(raw, float):
                sentence += str(raw) + '.'
            elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
                for v1 in raw:
                    for v in v1:
                        sentence += clean_text(str(v))[:-1] + ', '
                sentence = sentence[:-2] + '.'
            elif isinstance(raw, list):
                for v1 in raw:
                    sentence += clean_text(str(v1))
            else:
                sentence = clean_text(str(raw))
            return sentence + ' '

        reviews, user_metadata, item_metadata = raw_data_list

        item2meta = {"src": {}, "tgt": {}}
        for role in ['src', 'tgt']:
            for item, meta in item_metadata[role].items():
                if item not in id_mapping[role]['item2id'].keys():
                    continue
                meta_sentence = ''
                keys = set(meta.keys())
                features_needed = ['labels', 'name', 'director', 'summary', 'writer', 'country', 'pubdate', 'language',
                                   'rating', 'tag', 'fullname', 'CategoryID']
                for feature in features_needed:
                    if feature in keys:
                        meta_sentence += _sent_process(meta[feature])
                item2meta[role][item] = meta_sentence
        return item2meta

    def generate_sentence_embs(self, config, modality, interaction, id_mapping, modality_data):
        meta_sentences = {"src": [], "tgt": []}
        for role in ['src', 'tgt']:
            for i in range(1, len(id_mapping[role]['id2item'])):
                item = id_mapping[role]['id2item'][i]
                meta_sentences[role].append(modality_data[role][item])

        if 'sentence-transformers' in modality['emb_model']:
            try:
                from sentence_transformers import SentenceTransformer
                device = config.get('device', 'cpu')
                sent_emb_model = SentenceTransformer(modality['emb_model']).to(device)

                sent_embs_src = sent_emb_model.encode(
                    meta_sentences["src"],
                    convert_to_numpy=True,
                    batch_size=modality['emb_batch_size'],
                    show_progress_bar=True,
                    device=device
                )
                sent_embs_tgt = sent_emb_model.encode(
                    meta_sentences["tgt"],
                    convert_to_numpy=True,
                    batch_size=modality['emb_batch_size'],
                    show_progress_bar=True,
                    device=device
                )
                sent_embs = np.concatenate((sent_embs_src, sent_embs_tgt), axis=0)
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        elif 'text-embedding-3' in modality['emb_model']:
            if not config['openai_api_key']:
                raise ValueError("OpenAI API key required for OpenAI embeddings")

            try:
                from openai import OpenAI
                import tiktoken
                _enc = tiktoken.get_encoding("cl100k_base")

                def truncate_to_max_tokens(text, max_tokens=7000):
                    # 防止Text过长
                    tokens = _enc.encode(text)
                    if len(tokens) <= max_tokens:
                        return text
                    return _enc.decode(tokens[:max_tokens])

                client_kwargs = {'api_key': config['openai_api_key']}
                if 'openai_base_url' in config and config['openai_base_url']:
                    client_kwargs['base_url'] = config['openai_base_url']

                client = OpenAI(**client_kwargs)

                sent_embs = []
                for role in ['src', 'tgt']:
                    for i in tqdm(range(0, len(meta_sentences[role]), modality['emb_batch_size']), desc='Encoding'):
                        batch = [
                            truncate_to_max_tokens(x, 7000)
                            for x in meta_sentences[role][i:i + modality['emb_batch_size']]
                        ]
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

        return sent_embs

    def generate_sentence_final_embs(self, config, modality, interaction, id_mapping, embs):
        src_item_num = len(id_mapping['src']['item2id'])
        tgt_item_num = len(id_mapping['tgt']['item2id'])
        assert (src_item_num + tgt_item_num == embs.shape[0])

        src_embs = embs[:src_item_num]
        tgt_embs = embs[src_item_num:src_item_num + tgt_item_num]

        if modality['emb_pca'] == modality['emb_dim']:
            pca_embs = embs
        elif modality['emb_pca'] < modality['emb_dim']:
            try:
                from sklearn.decomposition import PCA
                pca_src = PCA(n_components=modality['emb_pca'], whiten=True)
                pca_src_embs = pca_src.fit_transform(src_embs)
                pca_tgt = PCA(n_components=modality['emb_pca'], whiten=True)
                pca_tgt_embs = pca_tgt.fit_transform(tgt_embs)
                pca_embs = np.concatenate((pca_src_embs, pca_tgt_embs), axis=0)
            except ImportError:
                raise ImportError("Please install scikit-learn: pip install scikit-learn")
        else:
            raise ValueError(f"The dimension of emb_pca must be less than or equal to emb_dim for {modality['name']}")
        return pca_embs

