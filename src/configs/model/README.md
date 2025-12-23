# {model_name}.yaml
- 功能：配置模型训练和评估时的参数
- 需要包含：
  1. warm_eval：是否启用热评估
  2. overlapped_users_for_warm_eval：warm_eval=True时，是否仅对包含源域交互的热用户进行评估
  3. cold_start_eval：是否启用冷评估
  4. training_stages：List，模型多阶段训练配置，每一项需包含：
     - name: 阶段名称
     - state：TrainDataloader数据加载模式，符合枚举类TrainDataLoaderState
     - epochs
     - train_batch_size
     - learner
     - learning_rate
     - learning_rate_scheduler
     - weight_decay
     - （可选）clip_grad_norm
  5. 其它模型所需参数: 如feature_dim等
  6. hyper_parameters：grid search超参数列表，其中的超参数需给出一个search的List，
        比如hyper_parameters:\["training_stages.0.learning_rate", "beta"\]，则
        training_stages:
            - name: stage1
            - learning_rate = \[0.1, 0.001\]
        beta: \[0.3, 0.7\]
  7. （可选）log_model_suffix: Log的模型后缀，用于区分不同dataset_setting/model_structure下的训练