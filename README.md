# CDRec

## Overview

CDRec is a unified benchmark for cross-domain recommendation (CDR), designed to support
systematic evaluation under different settings.

CDRec focuses on two representative evaluation protocols, both of which
evaluate recommendation performance **exclusively on the target domain**:
- **Warm-start CDR**: all users are shared across domains; training uses both source-
  and target-domain interactions, while evaluation is conducted only on held-out
  target-domain interactions.
- **Cold-start CDR**: only a subset of users overlap across domains; cold-start users
  have no target-domain interactions during training, and evaluation measures their
  recommendation performance on the target domain.

Beyond these two standard protocols, CDRec is designed with high extensibility and
flexible data preprocessing. The benchmark does not restrict users to fixed
evaluation scenarios.
By adjusting dataset configuration options, users can easily construct customized
CDR datasets.
This flexibility enables CDRec to support a wide range of experimental designs
and emerging CDR scenarios.

Dataset preprocessing in CDRec is fully configurable and controls user overlap,
interaction filtering, and train–validation–test splitting, which together define
the resulting evaluation scenario. All preprocessing options are specified via
YAML files to ensure reproducibility and flexibility.

CDRec provides two configurable YAML files for **Dataset Setting**:
- **Model configuration (High Priority)**:  
  `configs/model/<model_name>.yaml`
- **Dataset configuration (Low Priority)**:  
  `configs/dataset/Amazon2014.yaml`

**Detailed explanations of the dataset preprocessing options** in these YAML files
are provided in the **Dataset Preprocessing Details** section below.

In addition to flexible dataset settings, CDRec also supports **multi-stage
training** for cross-domain recommendation models. By specifying a list of
`training_stages` in the model configuration file
`configs/model/<model_name>.yaml`, users can easily define and execute
multi-stage training pipelines.

**Detailed explanations of the multi-stage training configuration** and the
corresponding YAML options are provided in the
**Multi-stage Training Details** section.


CDRec also supports **grid search** for hyper-parameter tuning. By specifying
the `hyper_parameters` field in the model configuration file
`configs/model/<model_name>.yaml`, users can define searchable hyper-parameter
spaces and automatically evaluate different configurations.

**Detailed explanations of the grid search configuration and supported options**
are provided in the **Grid Search Details** section.

---

## Quick Start

### Command Line Arguments
All experiments in CDRec are launched via the following command format:
```bash
python -u main.py --model <MODEL> --dataset <DATASET> --domains <SRC_DOMAIN> <TGT_DOMAIN>
```
These three arguments control **which model is evaluated, on which dataset, and on which pair of domains**.

- `--model` specifies the CDR model to evaluate.
The model must be implemented in `src/model/<MODEL>.py` and configured in `configs/model/<MODEL>.yaml`.
- `--dataset` specifies the dataset. Currently supported: `Amazon2014` and `Douban`.
- `--domains` specifies the source and target domains in the dataset (<SRC_DOMAIN> <TGT_DOMAIN>).
The first domain is the source domain, the second is the target domain, and **evaluation is always performed on the target domain**

For `Amazon2014`, valid domain names include:
```text
Books, Electronics, Movies_and_TV, CDs_and_Vinyl,
Clothing_Shoes_and_Jewelry, Home_and_Kitchen, Kindle_Store,
Sports_and_Outdoors, Cell_Phones_and_Accessories,
Health_and_Personal_Care, Toys_and_Games, Video_Games,
Tools_and_Home_Improvement, Beauty, Apps_for_Android,
Office_Products, Pet_Supplies, Automotive,
Grocery_and_Gourmet_Food, Patio_Lawn_and_Garden, Baby,
Digital_Music, Musical_Instruments, Amazon_Instant_Video
```
For `Douban`, valid domain names are:
```text
Book, Movie, Music
```

### Typical Warm-Start Evaluation Scenario

All users are **fully overlapped across domains**. During training, the model observes all source-domain interactions
and a subset of target-domain interactions. Evaluation is conducted only on **target domain**.

To run warm-start evaluation, make sure the following settings are correctly configured
in the YAML file:
```yaml
only_overlap_users: True
k_cores: 3
shuffle_user_sequence: True
warm_valid_ratio: 0.1
warm_test_ratio: 0.1
t_cold_valid: 0
t_cold_test: 0
```
#### Running Examples
The following command runs a typical warm-start cross-domain recommendation experiment
on the Amazon2014 dataset, using *Clothing_Shoes_and_Jewelry* as the source domain
and *Sports_and_Outdoors* as the target domain:
```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py --model DisenCDR --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors
```
The corresponding dataset setting for warm-start evaluation is specified in
`configs/model/DisenCDR.yaml`.
To reverse the source and target domains, exchange their order in the
`--domains` argument. For example:
```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py --model DisenCDR --dataset Amazon2014 --domains Sports_and_Outdoors Clothing_Shoes_and_Jewelry
```

---

### Typical Cold-Start Evaluation Scenario

Users are **partially overlapped across domains**. Cold-start users are
observed only in the source domain during training, with no target-domain
interactions. Evaluation is performed on interactions between these cold-start users and target-domain items.

To run cold-start evaluation, make sure the following settings are correctly configured
in the YAML file:
```yaml
only_overlap_users: False
k_cores: 3
shuffle_user_sequence: True
warm_valid_ratio: 0
warm_test_ratio: 0
t_cold_valid: 0.1
t_cold_test: 0.1
```

#### Running Examples
The following command runs a typical cold-start cross-domain recommendation experiment
on the Amazon2014 dataset, using *Clothing_Shoes_and_Jewelry* as the source domain
and *Sports_and_Outdoors* as the target domain:
```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py --model EMCDR --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors
```
The corresponding dataset setting for cold-start evaluation is specified in
`configs/model/EMCDR.yaml`.
To reverse the source and target domains, exchange their order in the
`--domains` argument. For example:
```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py --model EMCDR --dataset Amazon2014 --domains Sports_and_Outdoors Clothing_Shoes_and_Jewelry
```
---

## Dataset Preprocessing Details

This section describes the end-to-end dataset preprocessing pipeline used in CDRec.
We take Amazon2014 as an example, with Clothing_Shoes_and_Jewelry as the source
domain and Sports_and_Outdoors as the target domain, to illustrate how raw data
are transformed into ready-to-use benchmark datasets.

Overall, dataset preprocessing in CDRec consists of two stages:
1. Single-domain preprocessing, where each domain is processed independently
2. Joint-domain preprocessing, where cross-domain user alignment and data splitting
are performed according to the specified evaluation setting

### 1. Single-Domain Preprocessing
Each domain is first processed independently from raw Amazon review data.
If the required raw files are not found locally, CDRec will automatically
download the corresponding Amazon review and metadata files and place them
under the `raw/` directory.
For each domain, CDRec maintains a separate directory containing raw files and
processed interaction sequences.
```text
Amazon2014/
├── Clothing_Shoes_and_Jewelry/
│   ├── raw/
│   │   ├── meta_Clothing_Shoes_and_Jewelry.json.gz
│   │   └── reviews_Clothing_Shoes_and_Jewelry_5.json.gz
│   └── processed/
│       └── all_item_seqs_{shuffle|noshuffle}.json
├── Sports_and_Outdoors/
│   ├── raw/
│   │   ├── meta_Sports_and_Outdoors.json.gz
│   │   └── reviews_Sports_and_Outdoors_5.json.gz
│   └── processed/
│       └── all_item_seqs_{shuffle|noshuffle}.json
```
- raw/ contains the original Amazon metadata and review files.
- `processed/all_item_seqs_{shuffle|noshuffle}.json` stores cleaned user–item interaction sequences
  for the corresponding domain, organized as a JSON dictionary:
```json
{
  "A1KLRMWW2FWPL4": ["B003U3GOFO", "B009H6NPBE", "B00400N6XE", "..."],
  "A2G5TCU2WDFZ65": ["B0019K9WDQ", "B0036FSXI2", "B00B7TBDTU", "..."],
  "A1RLQXYNCMWRWN": ["B0007YVP1W", "B004LXVIDK", "B000LSWXWO", "..."]
}
```
Each key is a user **raw ID**, and each value is the corresponding list of
interacted **raw item IDs**. The item sequence is ordered by time by default,
and is randomly shuffled when `shuffle_user_sequence: True` is specified in
the YAML configuration. **This flag also determines the directory suffix
(`all_item_seqs_shuffle.json` or `all_item_seqs_noshuffle.json`) used to cache
the processed sequences.**

### 2. Joint-Domain Preprocessing
After single-domain preprocessing, CDRec performs joint-domain preprocessing
for a specific source–target domain pair.
Using Amazon2014 with Clothing_Shoes_and_Jewelry (source) and
Sports_and_Outdoors (target) as an example, the resulting directory structure is:
```text
Amazon2014/
└── Clothing_Shoes_and_Jewelry+Sports_and_Outdoors/
    └── all_users/ (or only_overlap_users/, controlled by `only_overlap_users`)
        └── WarmValid{w_v}_WarmTest{w_t}_ColdValid{c_v}_ColdTest{c_t}_{shuffle|noshuffle}_{kcores}/
            ├── train_src.pkl
            ├── train_tgt.pkl
            ├── valid_warm_tgt.pkl
            ├── test_warm_tgt.pkl
            ├── valid_cold_tgt.pkl
            ├── test_cold_tgt.pkl
            ├── all_users.json
            ├── id_mapping.json
            └── modality_emb/
```
Here, {w_v}, {w_t}, {c_v}, and {c_t} are determined by the YAML parameters
warm_valid_ratio, warm_test_ratio, t_cold_valid, and t_cold_test,
respectively.
The `{shuffle|noshuffle}` part is controlled by the `shuffle_user_sequence` parameter:
- `_shuffle` indicates that user sequences are shuffled before splitting into source and target domains.
- `_noshuffle` indicates that user sequences are preserved in temporal order, where the source domain contains past interactions and the target domain contains future interactions.

The `{kcores}` suffix is determined by the combination of the `only_overlap_users` and `k_cores` parameters:
- If `only_overlap_users=true`, the directory name will include the number of cores used for parallel processing (e.g., `_3cores`).
- If `only_overlap_users=false`, the directory will not include the `kcores` suffix.

**Typical cold-start experiments use all_users/, while typical warm-start experiments use only_overlap_users/.**

Although CDRec always generates six split files
(`train_src.pkl`, `train_tgt.pkl`, `valid_warm_tgt.pkl`, `test_warm_tgt.pkl`, `valid_cold_tgt.pkl`, `test_cold_tgt.pkl`),
which files are actually used depends on the evaluation scenario.

**Cold-start setting.**
CDRec first identifies all overlapped users and then selects a subset of them as cold users.
Their target-domain interactions are removed from training and split into validation and test sets.
The model is evaluated on its ability to predict target-domain interactions for these cold users.
Therefore, only the following files are non-empty and used:
- `train_src.pkl` (source-domain training data)
- `train_tgt.pkl` (target-domain training data)
- `valid_cold_tgt.pkl` (target interactions of validation cold users)
- `test_cold_tgt.pkl` (target interactions of test cold users)

while `valid_warm_tgt.pkl` and `test_warm_tgt.pkl` are empty.

**Warm-start setting.**
All non-overlapped users are removed, so all remaining users are warm users with interactions in both domains.
Their target-domain interactions are split into training, validation, and test sets (typically 8:1:1).
Therefore, only the following files are non-empty and used:
- `train_src.pkl` (source-domain training data)
- `train_tgt.pkl` (target-domain training data)
- `valid_warm_tgt.pkl` (target interactions for validation)
- `test_warm_tgt.pkl` (target interactions for testing)

while `valid_cold_tgt.pkl` and `test_cold_tgt.pkl` are empty.


---

Joint dataset construction proceeds through four sequential steps:
1. Load and filter users and items
2. Split users and reindex user/item IDs
3. Split interactions into train/validation/test sets
4. Prepare modality embeddings

####  Step 1: Load Interaction Sequences and Apply Overlap Filtering 
After loading `all_item_seqs_{shuffle|noshuffle}.json` from both the source and target domains,
CDRec determines whether user–item filtering is required according to the YAML
configuration. When `only_overlap_users: True` is specified, CDRec applies
dual-domain *k*-core filtering controlled by `k_cores`: each user must have at
least `k_cores` interacted items in **both** domains, and each item must be
interacted with by at least `k_cores` users within its domain. This guarantees
that the remaining users are fully overlapped across domains and that all
interactions satisfy a minimum frequency constraint. By default, `k_cores = 3`,
which allows each user sequence to be effectively split into
train/validation/test sets while preserving as many users and items as possible.
According to the value of `only_overlap_users`, the resulting files are stored
under `all_users/` or `only_overlap_users/`.

**`all_users/` is used for cold-start evaluation, while `only_overlap_users/` is used for warm-start evaluation.**

#### Step 2: Split Users and Reindex IDs
In Step 2, CDRec **splits users** into disjoint categories **using raw user IDs**
according to the YAML parameters `t_cold_valid` and `t_cold_test`, and then
**reindexes** users and items to generate integer ID mappings for model training.

Based on the original user distribution, all users can be categorized into three
types according to their domain presence: source-only users, target-only users,
and overlapped users. **Under the cold-start evaluation scenario**, CDRec then**further 
splits the original overlapped users** into multiple subsets according to 
the YAML parameters `t_cold_valid` and `t_cold_test`.

Specifically, a proportion of overlapped users is sampled and reassigned as
`valid_cold_users` and `test_cold_users`, while the remaining overlapped users
are kept as `overlap_users`. The target-domain interactions of `valid_cold_users` 
and `test_cold_users` are reserved exclusively for evaluation and are never used 
during training.
Together with the original source-only and target-only users, this process
produces five **mutually exclusive** user groups:


**An example of User Splitting in the Warm-Start Scenario:**
```json
{
  "overlap_users": ["A036147939NFPC389VLK", "A100L918633LUO", "A100WFKYVRPVX7", "..."],
  "valid_cold_users": [],
  "test_cold_users": [],
  "src_only_users": [],
  "tgt_only_users": []
}
```
**An example of User Splitting in the Cold-Start Scenario:**
```json
{
  "overlap_users": ["A021943320Y3C5B58IY79", "A036147939NFPC389VLK", "..."],
  "valid_cold_users": ["A00046902LP5YSDV0VVNF", "A0029274J35Q1MYNKUWO", "..."],
  "test_cold_users": ["A11GCF3KECY6HO", "A11GF0R6HVHDJG", "..."],
  "src_only_users": ["A001114613O3F18Q5NVR6", "A00146182PNM90WNNAZ5Q", "..."],
  "tgt_only_users": ["A2PAFNZ5D4J4WN", "A2S26YGSVXBCFL", "..."]
}
```
These five user sets are disjoint, and their union forms the complete user set.
They are saved in `all_users.json` after raw string user IDs 
are reindexed into internal integer IDs according to `id_mapping.json`.

CDRec also generates `id_mapping.json` to map raw user/item IDs to consecutive
integer IDs for model training:

For user indexing, CDRec builds domain-specific user vocabularies,
with users indexed separately in the source and target domains:
- **Source-domain users** include: `overlap_users`, `valid_cold_users`,
  `test_cold_users`, and `src_only_users` (4 groups).
- **Target-domain users** include: `overlap_users` and `tgt_only_users`
  (2 groups).

User and item IDs are indexed **separately for each domain**, starting from 1, following the ordered user groups defined above.
In the source domain, users are indexed in the order `overlap_users → valid_cold_users → test_cold_users → src_only_users`, 
while in the target domain they are indexed as `overlap_users → tgt_only_users`.
**Overlapped users are assigned identical ID ranges in both domains (`1 ... num_overlap_users`)
to ensure that the same user has consistent indices across domains.**

**An Example of ID Reindexing under Warm- and Cold-Start Scenarios:**
```json
{
  "src": {
    "user2id": { "A021943320Y3C5B58IY79": 1, "A036147939NFPC389VLK": 2, "...": "..." },
    "id2user": ["[PAD]", "A021943320Y3C5B58IY79", "A036147939NFPC389VLK", "..."],
    "item2id": {"0000031887": 1, "0123456479": 2, "...": "..." },
    "id2item": ["[PAD]", "0000031887", "0123456479", "..."]
  },
  "tgt": {
    "user2id": { "A021943320Y3C5B58IY79": 1, "A036147939NFPC389VLK": 2, "...": "..." },
    "id2user": ["[PAD]", "A021943320Y3C5B58IY79", "A036147939NFPC389VLK", "..."],
    "item2id": { "1881509818": 1, "2094869245": 2, "...": "..." },
    "id2item": ["[PAD]", "1881509818", "2094869245", "..."]
  }
}
```

In this example, `id2user` is implemented as a list rather than a dictionary 
so that direct indexing can be used: id2user[1] = "A021943320Y3C5B58IY79" means that the raw user ID "A021943320Y3C5B58IY79" is mapped to internal ID 1.
By convention, id2user[0] = "PAD", so that all valid user IDs start from 1 and index 0 is reserved for padding.

#### Step 3: Split Interactions into Train / Validation / Test Sets

In Step 3, CDRec splits cross-domain user–item interactions into training,
validation, and test sets according to the user categories defined in Step 2.
This step is controlled by the YAML parameters `warm_valid_ratio`,
`warm_test_ratio`, `t_cold_valid`, and `t_cold_test`, and generates six
interaction files: `train_src.pkl`, `train_tgt.pkl`, `valid_warm_tgt.pkl`,
`test_warm_tgt.pkl`, `valid_cold_tgt.pkl`, and `test_cold_tgt.pkl`.

All interaction files generated in this step are stored in Pickle (.pkl)
format. Each file contains a pandas DataFrame with the following columns:
```text
['user', 'item']
```
where user and item are reindexed integer IDs defined in `id_mapping.json`.

Note that in the warm-start evaluation scenario, only `train_src.pkl`, `train_tgt.pkl`,
`valid_warm_tgt.pkl`, and `test_warm_tgt.pkl` are non-empty and used,
while `valid_cold_tgt.pkl` and `test_cold_tgt.pkl` are empty.

In the cold-start evaluation scenario, only `train_src.pkl`, `train_tgt.pkl`, 
`valid_cold_tgt.pkl`, and `test_cold_tgt.pkl` are non-empty and used,
while `valid_warm_tgt.pkl` and `test_warm_tgt.pkl` are empty.

It is important to note that all validation and test sets are constructed exclusively for the target domain.

#### Step 4: Prepare Modality Embeddings
In Step 4, CDRec prepares modality embeddings according to the
`modalities` configuration in the YAML file.
Each modality entry must contain the following fields: `name`, `emb_model`, `emb_dim`, `emb_pca` , `enabled`,
Only modalities with enabled: true are processed. For each enabled modality,
CDRec generates three files per domain, stored under `modality_emb/`:
```text
modality_emb/
├──<name>_metadata.json
├── <name>_<emb_model>_<emb_dim>.npy
└── <name>_final_emb_<emb_pca>.npy
```
These files store the corresponding metadata, the raw modality embeddings and the post-processed (e.g., PCA)
embeddings, respectively. Modalities with
`enabled: false` are skipped and no files are generated.

For each `enabled` modality, CDRec processes the modality data
through three sequential functions, each responsible for generating one of
the modality-related files:
```text
_create_modality_data  →  _create_embs  →  _create_final_embs
```

Specifically:

- `_create_modality_data(modality)`
This function looks for a method named
`extract_<name>_modality_data` in the corresponding model file, where <name>
is the modality name.
The method extracts modality-specific raw data and returns a JSON object,
which is saved as: `<name>_metadata.json`
- `_create_embs(modality, modality_data)`
This function looks for a method named `generate_<name>_embs`.
The method generates raw modality embeddings as a NumPy array and saves
them as: `<name>_<emb_model>_<emb_dim>.npy`
- _create_final_embs(modality, embs)
This function looks for a method named `generate_<name>_final_embs`.
The method performs post-processing (e.g., dimensionality reduction) on the
raw embeddings and saves the resulting NumPy array as: `<name>_final_emb_<emb_pca>.npy`

The concrete logic of these three functions is model-dependent and is
implemented in the corresponding `<model>.py` file.
Users can refer to the `sentence` modality as an example, which defines the
following methods in `amazon_modality_processor.py`:
- `extract_sentence_modality_data`
- `generate_sentence_embs`
- `generate_sentence_final_embs`

This design enables new modalities to be added by implementing the corresponding
`extract_*`, `generate_*`, and `generate_*_final_embs` methods in `<model>.py`,
without modifying the core preprocessing pipeline.

---

## Multi-stage Training Details

CDRec provides multiple data loading modes for cross-domain recommendation,
which determine which users and interactions are returned by the dataloader in each training iteration.
The following data loading modes are supported:
- **BOTH**
Returns interactions from both source and target domains, including:
  - `users_src`: user IDs for source-domain interactions
  - `pos_items_src`: positive items in the source domain
  - `neg_items_src`: negative items in the source domain
  - `users_tgt`: user IDs for target-domain interactions
  - `pos_items_tgt`: positive items in the target domain
  - `neg_items_tgt`: negative items in the target domain

- **SOURCE**
Returns only source-domain interactions, including:
  - `users`: user IDs
  - `pos_items`: positive items in the source domain
  - `neg_items`: negative items in the source domain

- **TARGET**
Returns only target-domain interactions, including:
  - `users`: user IDs
  - `pos_items`: positive items in the target domain
  - `neg_items`: negative items in the target domain

- **OVERLAP**
Returns interactions of overlapped users in both domains, including:
  - `users`: overlapped user IDs
  - `pos_items_src`: positive items in the source domain
  - `neg_items_src`: negative items in the source domain
  - `pos_items_tgt`: positive items in the target domain
  - `neg_items_tgt`: negative items in the target domain

- **OVERLAP_USER**
Returns only overlapped user IDs, including:
  - `users_overlapped`: IDs of overlapped users


CDRec supports **multi-stage training** for cross-domain recommendation models,
allowing different training phases to adopt different data loading strategies,
optimization settings, and training schedules.
Multi-stage training is fully configured through the `training_stages` field in
the model configuration file `configs/model/<model_name>.yaml`.

**Only the final training stage performs evaluation and early stopping**;
all preceding stages are executed as pure training phases without validation
or test performance monitoring.

Accordingly, each model implementation (`<model>.py`) must implement
`set_train_stage(stage)` to control which model parameters are trainable
in different stages by enabling or freezing their gradients.

A multi-stage training pipeline is defined as a list of stages, for example:
```yaml
training_stages:
  - name: source_training
    state: SOURCE
    epochs: 10
    train_batch_size: 2048
    learner: adam
    learning_rate: 0.01
    learning_rate_scheduler: [1.0, 50]
    weight_decay: 0.0

  - name: target_training
    state: TARGET
    epochs: 10
    train_batch_size: 2048
    learner: adam
    learning_rate: 0.01
    learning_rate_scheduler: [1.0, 50]
    weight_decay: 0.0

  - name: mapping_training
    state: OVERLAP_USER
    epochs: 1000
    train_batch_size: 2048
    learner: adam
    learning_rate: [0.001, 0.01]
    learning_rate_scheduler: [1.0, 50]
    weight_decay: 0.0
```

The above example illustrates a typical three-stage training pipeline for cold-start mapping methods:
source-domain pretraining, target-domain pretraining, and final cross-domain user mapping on overlapped users.

Each training stage must specify the following **required parameters**:
- `name`: unique identifier of the training stage
- `state`: data loading mode for this stage
- `epochs`: number of training epochs
- `learner`: optimization algorithm 
- `learning_rate`: learning rate
- `learning_rate_scheduler`: scheduler configuration
- `weight_decay`: weight decay coefficient
The following parameter is **optional**:
- `clip_grad_norm`: gradient clipping configuration for stabilizing training


---

## Grid Search Details

CDRec supports grid search–based hyper-parameter tuning through YAML
configuration. Users specify which hyper-parameters should be searched and
provide their candidate values directly in the model configuration file
`configs/model/<model_name>.yaml`.

### Defining Searchable Hyper-parameters
Hyper-parameters to be searched are listed under the `hyper_parameters` field. For example:
```yaml
hyper_parameters:
  - "training_stages.0.learning_rate"
  - "history_len"
  - "aggregator"
  - "mask_rate"
  - "lambda_loss"
```

### Specifying Search Ranges
For each parameter listed in `hyper_parameters`, its value in the YAML file must
be provided as **a list of candidate values**, defining the grid search space.
For example:
```yaml
history_len: [10, 20, 40]
aggregator: ["mean", "attention"]
mask_rate: [0.1, 0.3]
lambda_loss: [0.1, 1.0]
```
Nested parameters follow the same rule. For instance, to search over learning
rates in the first training stage:
```yaml
training_stages:
  - name: source_training
    state: SOURCE
    epochs: 10
    learner: adam
    learning_rate: [0.001, 0.01]
    learning_rate_scheduler: [1.0, 50]
    weight_decay: 0.0
```
During grid search, CDRec enumerates all combinations of the specified candidate
values and runs training accordingly.