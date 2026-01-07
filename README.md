# CDRec

## Evaluation Scenarios

CDRec supports **warm-start** and **cold-start** evaluation scenarios, which are controlled by the
**Dataset Setting** in YAML configuration files.

CDRec provides two configurable YAML files for **Dataset Setting**:
- **Model configuration**:  
  `configs/model/<model_name>.yaml`
- **Dataset configuration**:  
  `configs/dataset/Amazon2014.yaml`

If the same dataset setting is specified in both files, the priority is: model YAML > dataset YAML

---

## Warm-Start Evaluation
All users are **fully overlapped across domains**. During training, the model observes all source-domain interactions
and a subset of target-domain interactions from warm users. Evaluation is conducted only on **warm users**.

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

### Properties
- Full user overlap across domains
- Dual-domain k-core filtering
- Validation and test sets contain only warm target-domain interactions

## Running Examples
CUDA_VISIBLE_DEVICES=0 python -u main.py --model DisenCDR --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors

---

## Cold-Start Evaluation
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

### Properties
- Partial user overlap across domains
- Validation and test sets include cold-start users from source domains

## Running Examples
CUDA_VISIBLE_DEVICES=0 python -u main.py --model EMCDR --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors