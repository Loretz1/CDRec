# CDRec

## Overview

CDRec is a unified benchmark for cross-domain recommendation (CDR), designed to support
systematic evaluation under different settings.

CDRec focuses on two representative evaluation protocols, both of which
evaluate recommendation performance **exclusively on the target domain**:
- **Warm-start CDR (Intra domain)**: all users are shared across domains; training uses both source-
  and target-domain interactions, while evaluation is conducted only on held-out
  target-domain interactions.
- **Cold-start CDR (Inter domain)**: only a subset of users overlap across domains; cold-start users
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
- **Model configuration (High Priority, model-specific dataset configuration)**:  
  `configs/model/<model_name>.yaml`
- **Dataset configuration (Low Priority, default dataset configuration)**:  
  `configs/dataset/<dataset>.yaml`

**We recommend specifying dataset settings in the model configuration
(`configs/model/<model_name>.yaml`), since it overrides `<dataset>.yaml` and enables
model-specific dataset configurations.**

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


### Available Domains in a Dataset

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


### Quick Start for Models under Different Evaluation Scenarios

The following models in CDRec are designed for the **warm-start CDR setting**,
where all users are shared across domains and the model leverages
**both source- and target-domain interactions** to improve
target-domain recommendation.

```text
Base, LightGCN, DisenCDR, GDCCDR, CUT, CUT_MF, PicCDR
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

The following models in CDRec are designed for the **cold-start CDR setting**,
where only a subset of users overlap across domains and the model predicts
target-domain preferences for **cold users** who have no target-domain
interactions during training.
```text
EMCDR, CDRIB, PTUPCDR, Disco, DMCDR, DiffCDR
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

---

The following models in CDRec support **both warm-start and cold-start CDR settings**
and can be evaluated under different **dataset settings** using the same model.


```text
UniCDR, CD_CDR
```

#### Running Examples

The following command runs an evaluation using UniCDR on Amazon2014.
By changing the **dataset settings** in `configs/model/UniCDR.yaml`,
the same model can be evaluated under different CDR scenarios.

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py --model UniCDR --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors
```

**To run under the warm-start setting**, configure `configs/model/UniCDR.yaml` as:
```yaml
only_overlap_users: True # Whether to construct a fully-overlapped (warm-start) or partially-overlapped (cold-start) user set
k_cores: 3 # only work with only_overlap_users, for dual domain k-cores, at least 3 otherwise can not split train/valid/test
shuffle_user_sequence: True  # weather shuffle users' item sequence before split train/valid/test dataset
warm_valid_ratio: 0.1 # Ratio of interactions from warm users in the target domain to be assigned to the validation set
warm_test_ratio: 0.1 # Ratio of interactions from warm users in the target domain to be assigned to the test set
t_cold_valid: 0 # cold user in the target domain valid set
t_cold_test: 0 # cold user in the target domain test set

warm_eval: True  # Whether to enable warm user evaluation
cold_start_eval: False # Whether to enable cold start user evaluation
```

**To run under the cold-start setting**, configure `configs/model/UniCDR.yaml` as:
```yaml
only_overlap_users: False # Whether to construct a fully-overlapped (warm-start) or partially-overlapped (cold-start) user set
k_cores: 3 # only work with only_overlap_users, for dual domain k-cores, at least 3 otherwise can not split train/valid/test
shuffle_user_sequence: True  # weather shuffle users' item sequence before split train/valid/test dataset
warm_valid_ratio: 0 # Ratio of interactions from warm users in the target domain to be assigned to the validation set
warm_test_ratio: 0 # Ratio of interactions from warm users in the target domain to be assigned to the test set
t_cold_valid: 0.1 # cold user in the target domain valid set
t_cold_test: 0.1 # cold user in the target domain test set

warm_eval: False  # Whether to enable warm user evaluation
cold_start_eval: True # Whether to enable cold start user evaluation
```

---


## Dataset Preprocessing Details

### Dataset Construction under Different Evaluation Scenarios
This section describes the end-to-end dataset preprocessing pipeline used in CDRec.
We take Amazon2014 as an example, with Clothing_Shoes_and_Jewelry as the source
domain and Sports_and_Outdoors as the target domain, to illustrate how raw data
are transformed into ready-to-use benchmark datasets.

Overall, dataset preprocessing in CDRec is **scenario-dependent**.
Different evaluation settings (warm-start vs. cold-start) require
different user filtering, user splitting, and interaction partitioning.
By configuring the **dataset settings in YAML**, CDRec constructs
different benchmark datasets from the same raw data to support
different evaluation scenarios.


The following dataset-related configuration options control both
**dataset construction** and **evaluation activation**:
```yaml
# Dataset Setting
only_overlap_users
k_cores
shuffle_user_sequence
warm_valid_ratio
warm_test_ratio
t_cold_valid
t_cold_test

# Eval Setting
warm_eval
cold_start_eval
```

These options jointly determine which users are kept,
how interactions are split,
and which evaluation protocol is enabled.

In the following, we separately describe the dataset configurations
and preprocessing pipelines for the **warm-start** **and cold-start**
evaluation scenarios.

---

### Warm-Start Dataset Construction


A typical warm-start evaluation scenario keeps **only overlapped users**
and splits their target-domain interactions into training, validation,
and test sets.

A typical warm-start dataset configuration is:
```yaml
only_overlap_users: True # For warm-start scenarios, this must be True.
k_cores: 3 # only work with only_overlap_users, for dual domain k-cores, at least 3 otherwise can not split train/valid/test
shuffle_user_sequence: True  # weather shuffle users' item sequence before split train/valid/test dataset
warm_valid_ratio: 0.1 # Ratio of interactions from warm users in the target domain to be assigned to the validation set
warm_test_ratio: 0.1 # Ratio of interactions from warm users in the target domain to be assigned to the test set
t_cold_valid: 0 # For warm-start scenarios, this must be 0.
t_cold_test: 0 # For warm-start scenarios, this must be 0.

warm_eval: True  # For warm-start scenarios, this must be True.
cold_start_eval: False # For warm-start scenarios, this must be False.
```

In this configuration, the first seven parameters control **dataset construction**
(i.e., user filtering, interaction splitting),
while the last two parameters (`warm_eval` and `cold_start_eval`) control
which **evaluation protocol** is enabled.
For warm-start experiments, `warm_eval` is set to `True` and
`cold_start_eval` is set to `False`, meaning that only warm-user evaluation
is performed.

Overall, dataset preprocessing consists of two stages:

1. Single-domain preprocessing, where each domain is processed independently.
2. Joint-domain preprocessing, where cross-domain user alignment and data splitting
   are performed according to the warm evaluation setting.

### 1. Single-Domain Preprocessing for Warm-Start Scenarios

Single-domain preprocessing is affected by `shuffle_user_sequence`,
which controls whether user interaction sequences are randomly shuffled 
and determines the corresponding cached file
(`all_item_seqs_shuffle.json` or `all_item_seqs_noshuffle.json`).

Each domain is processed independently from raw Amazon review data.
If the required raw files are not found locally, CDRec will automatically
download the corresponding Amazon review and metadata files and place them
under the `raw/` directory.

For each domain, CDRec maintains a separate directory containing raw files and
processed interaction sequences:



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

- raw/ contains the original Amazon item metadata and user-item review files.
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
and is randomly shuffled when <span style="color:#7AAEF7;">shuffle_user_sequence: True</span> is specified in
the YAML configuration. **This flag also determines the directory suffix
(`all_item_seqs_shuffle.json` or `all_item_seqs_noshuffle.json`) used to cache
the processed sequences.**

### 2. Joint-Domain Preprocessing for Warm-Start Scenarios

After single-domain preprocessing, CDRec performs joint-domain preprocessing
for a specific source–target domain pair.
Using Amazon2014 with Clothing_Shoes_and_Jewelry (source) and
Sports_and_Outdoors (target) as an example, the resulting **warm-start**
dataset directory structure is:

```text
Amazon2014/
└── Clothing_Shoes_and_Jewelry+Sports_and_Outdoors/
    └── only_overlap_users/
        └── WarmValid{w_v}_WarmTest{w_t}_ColdValid{c_v}_ColdTest{c_t}_{shuffle|noshuffle}_{kcores}/
            ├── train_src.pkl
            ├── train_tgt.pkl
            ├── valid_warm_tgt.pkl
            ├── test_warm_tgt.pkl
            ├── valid_cold_tgt.pkl (empty in warm-start and not used for evaluation)
            ├── test_cold_tgt.pkl (empty in warm-start and not used for evaluation)
            ├── all_users.json
            ├── id_mapping.json
            └── modality_emb/
```

The dataset directory structure is determined by the dataset configuration parameters.
Here, `{w_v}`, `{w_t}`, `{c_v}`, and `{c_t}` correspond to the YAML parameters
`warm_valid_ratio`, `warm_test_ratio`, `t_cold_valid`, and `t_cold_test`, respectively.
The `{shuffle|noshuffle}` suffix is controlled by `shuffle_user_sequence`,
and `{kcores}` corresponds to the value of `k_cores`.

Although CDRec always generates six split files
(`train_src.pkl`, `train_tgt.pkl`, `valid_warm_tgt.pkl`, `test_warm_tgt.pkl`,
`valid_cold_tgt.pkl`, and `test_cold_tgt.pkl`), two of them 
(`valid_cold_tgt.pkl` and `test_cold_tgt.pkl`) are empty and are not used for evaluation.
They are still created to keep a **unified file format** across warm-start
and cold-start dataset constructions.

In the warm-start setting, joint-domain preprocessing keeps only overlapped users
and constructs a fully shared user set across domains.
After dual-domain *k*-core filtering, all remaining users have interactions in both domains,
and their target-domain interactions are split into training, validation,
and test sets (typically 8:1:1).

Joint dataset construction proceeds through four sequential steps:

1. Load and filter users and items
2. Split users and reindex user/item IDs
3. Split interactions into train/validation/test sets
4. Prepare modality embeddings

#### Step 1: Load Interaction Sequences and Apply Overlap Filtering (Warm-Start Scenarios)

The dataset configuration parameters relevant to this step include
`only_overlap_users`(whether to apply overlap filtering, and must be set to `True`
in the warm-start setting) and `k_cores`(the minimum number of interactions required per user and item
during overlap filtering).

In this step, after loading `all_item_seqs_{shuffle|noshuffle}.json` from both
the source and target domains, CDRec decides whether to apply additional
user–item filtering based on the YAML configuration.

Overlap filtering is applied only when `only_overlap_users = True`;
since warm-start datasets are constructed with `only_overlap_users` enabled,
this filtering is always performed in the warm-start setting.

Under overlap filtering, CDRec performs **dual-domain *k*-core filtering**
controlled by `k_cores`. Specifically, each user must have interacted
with at least `k_cores` items in **both** domains, and each item must
be interacted with by at least `k_cores` users within its own domain.
This filtering step ensures that the remaining users are **fully
overlapped across domains** and that all interactions satisfy a minimum
frequency constraint. By default, `k_cores = 3`, which enables each user
sequence to be effectively split into train/validation/test sets while
preserving as many users and items as possible.

#### Step 2: Split Users and Reindex IDs (Warm-Start Scenarios)

The dataset configuration parameters relevant to this step include
`t_cold_valid` and `t_cold_test` (both are set to 0 in the warm-start setting),
so this step has no tunable parameters under warm-start scenarios.

In Step 2, CDRec **splits users** into disjoint categories **using raw user IDs**,
and then **reindexes** users and items to generate integer ID mappings for model training.

Under the unified warm–cold preprocessing framework, **user splitting**
divides users into five conceptual categories: `overlap_users`,
`valid_cold_users`, `test_cold_users`, `src_only_users`, and `tgt_only_users`.
However, in the warm-start setting, **all users belong to `overlap_users`**,
and the other four groups are empty.
These four groups are still retained to maintain a **unified user-splitting
and file format** across warm-start and cold-start dataset construction.

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

This dictionary is then reindexed and stored as `all_users.json`,
with raw user IDs replaced by internal integer IDs.

After user splitting, CDRec **reindexes** both users and items for model training.
User and item IDs are indexed **separately for each domain**, starting from 1.
In the warm-start setting, all users are overlapped users and therefore occupy
the same ID range (`1 ... num_users`) in both domains, so the same user has
identical IDs in the source and target domains.
Items are indexed independently in the two domains, each starting from 1.

**An Example of ID Reindexing under Warm-Start Scenarios:**

```json
{
  "src": {
    "user2id": { "A036147939NFPC389VLK": 1, "A100L918633LUO": 2, "...": "..." },
    "id2user": ["[PAD]", "A036147939NFPC389VLK", "A100L918633LUO", "..."],
    "item2id": {"1608299953": 1, "1617160377": 2, "...": "..." },
    "id2item": ["[PAD]", "1608299953", "1617160377", "..."]
  },
  "tgt": {
    "user2id": { "A036147939NFPC389VLK": 1, "A100L918633LUO": 2, "...": "..." },
    "id2user": ["[PAD]", "A036147939NFPC389VLK", "A100L918633LUO", "..."],
    "item2id": { "7245456313": 1, "B00000IURU": 2, "...": "..." },
    "id2item": ["[PAD]", "7245456313", "B00000IURU", "..."]
  }
}
```

In this example, `id2user` is implemented as a list rather than a dictionary 
so that direct indexing can be used: id2user[1] = "A100L918633LUO" means that the raw user ID "A100L918633LUO" is mapped to internal ID 1.
By convention, id2user[0] = "PAD", so that all valid user IDs start from 1 and index 0 is reserved for padding.

In the warm-start setting, `user2id` and `id2user` are identical in the source
and target domains because all users are fully overlapped,
while `item2id` and `id2item` are different since items are domain-specific.

The dictionary above is then stored as `id_mapping.json`.

#### Step 3: Split Interactions into Train / Validation / Test Sets (Warm-Start Scenarios)

The dataset configuration parameters relevant to this step include
`warm_valid_ratio` and `warm_test_ratio`, which control how target-domain
interactions of users are split into training, validation, and test sets.

In Step 3 , all source-domain interactions of users are assigned
to the training set (`train_src.pkl`).
All target-domain interactions of users are split into
training/validation/test according to `warm_valid_ratio` and `warm_test_ratio`,
producing `train_tgt.pkl`, `valid_warm_tgt.pkl`, and `test_warm_tgt.pkl`.

For format consistency, CDRec still outputs `valid_cold_tgt.pkl` (empty in warm-start)
and `test_cold_tgt.pkl` (empty in warm-start).

All interaction files generated in this step are stored in Pickle (.pkl)
format. Each file contains a pandas DataFrame with the following columns:


```text
['user', 'item']
```

where user and item are reindexed integer IDs defined in `id_mapping.json`.

It is important to note that all validation and test sets are constructed exclusively for the target domain.


#### Step 4: Prepare Modality Embeddings (Warm-Start Scenarios)

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
embeddings, respectively. Modalities with `enabled: false` are skipped and no files are generated.

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


### Cold-Start Dataset Construction

A typical cold-start evaluation scenario selects a subset of overlapped users
as validation and test cold users.
Their source-domain interactions are used for training,
while **all** of their target-domain interactions are reserved exclusively
for validation or test.

A typical cold-start dataset configuration is:
```yaml
only_overlap_users: False # For cold-start scenarios, this must be False.
k_cores: 3 # Not work at cold-start scenarios, cause it only works when only_overlap_users=True.
shuffle_user_sequence: True  # weather shuffle users' item sequence before split train/valid/test dataset
warm_valid_ratio: 0 # For cold-start scenarios, this must be 0.
warm_test_ratio: 0 # For cold-start scenarios, this must be 0.
t_cold_valid: 0 # cold user in the target domain valid set
t_cold_test: 0 # cold user in the target domain test set

warm_eval: False  # For cold-start scenarios, this must be False.
cold_start_eval: true # For cold-start scenarios, this must be true.
```

For cold-start experiments, `warm_eval` is set to `False` and
`cold_start_eval` is set to `True`, meaning that only cold-user evaluation
is performed.


Overall, dataset preprocessing under the cold-start setting follows
the same two-stage pipeline as the warm-start setting:

1. Single-domain preprocessing, where each domain is processed independently.
2. Joint-domain preprocessing, where cross-domain user alignment and data splitting
   are performed according to the cold evaluation setting.


### 1. Single-Domain Preprocessing for Cold-Start Scenarios

The single-domain preprocessing pipeline for cold-start scenarios is
identical to that for warm-start scenarios.


### 2. Joint-Domain Preprocessing for Cold-Start Scenarios

The joint-domain preprocessing pipeline for cold-start scenarios
follows the same overall structure as the warm-start pipeline,
but differs in user splitting, interaction partitioning,
and the resulting dataset files.
Using Amazon2014 with Clothing_Shoes_and_Jewelry (source) and
Sports_and_Outdoors (target) as an example, the resulting **cold-start**
dataset directory structure is:

```text
Amazon2014/
└── Clothing_Shoes_and_Jewelry+Sports_and_Outdoors/
    └── all_users/
        └── WarmValid{w_v}_WarmTest{w_t}_ColdValid{c_v}_ColdTest{c_t}_{shuffle|noshuffle}/
            ├── train_src.pkl
            ├── train_tgt.pkl
            ├── valid_warm_tgt.pkl (empty in cold-start and not used for evaluation)
            ├── test_warm_tgt.pkl (empty in cold-start and not used for evaluation)
            ├── valid_cold_tgt.pkl
            ├── test_cold_tgt.pkl
            ├── all_users.json
            ├── id_mapping.json
            └── modality_emb/
```

The dataset directory structure is determined by the dataset configuration parameters.
Here, `{w_v}`, `{w_t}`, `{c_v}`, and `{c_t}` correspond to the YAML parameters
`warm_valid_ratio`, `warm_test_ratio`, `t_cold_valid`, and `t_cold_test`, respectively.
The `{shuffle|noshuffle}` suffix is controlled by `shuffle_user_sequence`.

Although CDRec always generates six split files
(`train_src.pkl`, `train_tgt.pkl`, `valid_warm_tgt.pkl`, `test_warm_tgt.pkl`,
`valid_cold_tgt.pkl`, and `test_cold_tgt.pkl`), two of them 
(`valid_warm_tgt.pkl` and `test_warm_tgt.pkl`) are empty and are not used for evaluation.
They are still created to keep a **unified file format** across warm-start
and cold-start dataset constructions.

In the cold-start setting, joint-domain preprocessing keeps both overlapped
and non-overlapped users, and selects a subset of overlapped users as
validation and test cold users.
Their source-domain interactions are used for training,
while all of their target-domain interactions are reserved exclusively
for validation or test.

Joint dataset construction proceeds through four sequential steps:

1. Load and filter users and items
2. Split users and reindex user/item IDs
3. Split interactions into train/validation/test sets
4. Prepare modality embeddings



#### Step 1: Load Interaction Sequences and Apply Overlap Filtering (Cold-Start Scenarios)

For cold-start scenarios, `only_overlap_users` is set to `False`,
so no overlap filtering is applied.
In this step, CDRec simply loads
`all_item_seqs_{shuffle|noshuffle}.json` from the source and target domains.

#### Step 2: Split Users and Reindex IDs (Cold-Start Scenarios)

The dataset configuration parameters relevant to this step include
`t_cold_valid` and `t_cold_test`, which specify the proportions of overlapped
users to be selected as validation and test cold users, respectively.

In Step 2, CDRec **splits users** into disjoint categories **using raw user IDs**,
and then **reindexes** users and items to generate integer ID mappings for model training.

Users are first categorized into three groups based on their domain presence:
`src_only_users`, `tgt_only_users`, and `overlapped_users`.
From the original `overlapped_users`, a subset is then sampled according to
`t_cold_valid` and `t_cold_test` and reassigned as `valid_cold_users`
and `test_cold_users`, while the remaining users stay in `overlap_users`.
As a result, CDRec produces five disjoint user groups:
`overlap_users`, `valid_cold_users`, `test_cold_users`,
`src_only_users`, and `tgt_only_users`.

The target-domain interactions of `valid_cold_users` and `test_cold_users`
are reserved exclusively for evaluation and are never used during training.

**An example of User Splitting in the Cold-Start Scenario
(The length of `overlap_users` is 3128.):**
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

Then, CDRec generates `id_mapping.json` to map raw user/item IDs to consecutive
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

**An Example of ID Reindexing under Cold-Start Scenarios:**

```json
{
  "src": {
    "user2id": { "A021943320Y3C5B58IY79": 1, "A036147939NFPC389VLK": 2,
      "...": "...", "AZY513DDDRN0Q": 3128, "A10AVFDDU87KJ4": 3129, "...": "..."},
    "id2user": ["[PAD]", "A021943320Y3C5B58IY79", "A036147939NFPC389VLK", "..."],
    "item2id": {"0000031887": 1, "0123456479": 2, "...": "..." },
    "id2item": ["[PAD]", "0000031887", "0123456479", "..."]
  },
  "tgt": {
    "user2id": { "A021943320Y3C5B58IY79": 1, "A036147939NFPC389VLK": 2,
      "...": "...", "AZY513DDDRN0Q": 3128, "A00046902LP5YSDV0VVNF": 3129, "...": "..."},
    "id2user": ["[PAD]", "A021943320Y3C5B58IY79", "A036147939NFPC389VLK", "..."],
    "item2id": { "1881509818": 1, "2094869245": 2, "...": "..." },
    "id2item": ["[PAD]", "1881509818", "2094869245", "..."]
  }
}
```

In this example, the users in the `overlap_users` group (one of the five user categories)
are indexed as `1 ... 3128` in **both** the source and target domains,
so user IDs in this range have a one-to-one correspondence across domains.
Starting from ID 3129, users belong to the other four categories
(`valid_cold_users`, `test_cold_users`, `src_only_users`, or `tgt_only_users`)
and are therefore domain-specific and do not correspond across domains.
In other words, user ID 3128 refers to the **same user** in both domains,
while user ID 3129 in the source and target domains refers to **different users**.


#### Step 3: Split Interactions into Train / Validation / Test Sets (Cold-Start Scenarios)

In Step 3, all source-domain interactions of all users are assigned
to the training set (`train_src.pkl`).

For users in `overlap_users`, their target-domain interactions are also used
for training and stored in `train_tgt.pkl`.

For users in `valid_cold_users` and `test_cold_users`, **all** of their
target-domain interactions are reserved exclusively for evaluation and stored in
`valid_cold_tgt.pkl` and `test_cold_tgt.pkl`, respectively.

For users in `tgt_only_users`, their target-domain interactions are assigned
to the training set (`train_tgt.pkl`).

For format consistency, `valid_warm_tgt.pkl` and `test_warm_tgt.pkl` are still
generated but are empty in the cold-start setting.

All interaction files generated in this step are stored in Pickle (.pkl)
format. Each file contains a pandas DataFrame with the following columns:


```text
['user', 'item']
```

where user and item are reindexed integer IDs defined in `id_mapping.json`.

It is important to note that all validation and test sets are constructed exclusively for the target domain.


#### Step 4: Prepare Modality Embeddings (Cold-Start Scenarios)

The modality embedding preparation pipeline for cold-start scenarios
is identical to that for warm-start scenarios.
The final generated files are:
```text
modality_emb/
├──<name>_metadata.json
├── <name>_<emb_model>_<emb_dim>.npy
└── <name>_final_emb_<emb_pca>.npy
```


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