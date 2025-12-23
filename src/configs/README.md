# overall.yaml
- 功能：配置全局参数
- 需要包含：
  1. use_gpu：gpu/cpu
  2. seed：种子
  3. hyper_parameters：grid search超参数List，与{model_name}.yaml中配置的hyper_parameters共同组成grid search超参数列表
- 以及模型训练的默认配置，如果{model_name}.yaml配置了同样的参数，将会覆盖overall.yaml中的配置 
  1. eval_step：eval模型的epoch间隔 
  2. stopping_step：早停步数
  3. metrics
  4. topk
  5. valid_metric：早停依据的metric
  6. eval_batch_size
  8. skip_hyper_tuple_num：在模型grid search时，跳过 1-skip_hyper_tuple_num的组合，当模型训练中断时配置以继续grid search 
  9. log_model_suffix: Log的模型后缀，用于区分不同dataset_setting/model_structure下的训练