from logging import getLogger
from itertools import product
from utils.check_and_prepare_dataset import check_and_prepare_dataset
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str, metrics_dict2str, get_config_by_path, set_config_by_path
from utils.enum_type import TrainDataLoaderState
import platform
import os
import torch.utils.tensorboard as tb

def quick_start(model, dataset, domains, save_model=True):
    config = Config(model, dataset, domains)
    logfilename = init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # prepare dataset
    check_and_prepare_dataset(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value_warm = 0.0
    best_test_value_cold = 0.0
    best_test_idx_warm = None
    best_test_idx_cold = None
    idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(get_config_by_path(config, i) or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
             set_config_by_path(config, j, k)
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        if 'skip_hyper_tuple_num' in config and idx+1 <= config['skip_hyper_tuple_num']:
            hyper_ret.append((hyper_tuple, None, None, None, None))
            idx += 1
            continue

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        tbdir = f"runs/{logfilename[0]}-{logfilename[1]}-{logfilename[2]}-{logfilename[3]}-{config['hyper_parameters']}={hyper_tuple}" # tb ablation study要改名字
        tbdir_short = f"runs/{logfilename[0]}-{logfilename[1]}-{logfilename[2]}-{logfilename[3]}-{hyper_tuple}"
        writer = tb.SummaryWriter(log_dir=(tbdir if len(tbdir) <= 240 else tbdir_short)) # tb

        # trainer loading and initialization
        trainer = get_trainer()(config, model)

        # multi-stage training 多阶段训练，根据模型Yaml中配置的training_stages。
        # 只有最后一个stage有eval和早停
        for i, stage_config in enumerate(config['training_stages']):
            logger.info("Training stage {}: {}".format(i + 1, stage_config['name']))
            # 不同阶段设置不同的数据加载方式，由Yaml中的state控制
            # state需要满足TrainDataLoaderState这个枚举类其中的一个，代表一种数据加载方式。
            # 详细的interaction格式可以看项目README.md的Multi-stage Training Details部分，有详细说明。
            train_data.set_state_for_train(TrainDataLoaderState[stage_config['state']])
            train_data.set_batch_size(stage_config['train_batch_size'])

            eval = True if i == len(config['training_stages']) - 1 else False
            trainer.set_train_stage(i, stage_config, eval)  # 设置一下不同阶段训练参数，比如optimizer等

            # model training
            (best_valid_score_warm, best_valid_result_warm, best_test_upon_valid_warm,
             best_valid_score_cold, best_valid_result_cold, best_test_upon_valid_cold), (stage_train_time_total, stop_epoch) \
                = trainer.fit(i, train_data, valid_data=valid_data, test_data=test_data, saved=save_model, writer=writer)
            logger.info(f"=Train time of stage {i} total %.2fs, train time of stage {i} average: %.2fs"
                        % (stage_train_time_total, stage_train_time_total / (stop_epoch + 1)))

            if not eval:
                continue

            hyper_ret.append((hyper_tuple, best_valid_result_warm, best_test_upon_valid_warm, best_valid_result_cold,
                              best_test_upon_valid_cold))

            # save best test
            if best_test_upon_valid_warm[val_metric] > best_test_value_warm:
                best_test_value_warm = best_test_upon_valid_warm[val_metric]
                best_test_idx_warm = idx
            if best_test_upon_valid_cold[val_metric] > best_test_value_cold:
                best_test_value_cold = best_test_upon_valid_cold[val_metric]
                best_test_idx_cold = idx
            idx += 1

            logger.info("\n" + "=" * 100)
            if config.get("warm_eval", False):
                logger.info(f"🌐 [Warm Evaluation] Best Validation Result:\n{metrics_dict2str(best_valid_result_warm)}")
                logger.info(f"🧪 [Warm Evaluation] Test Result (upon best valid):\n"
                            f"{metrics_dict2str(best_test_upon_valid_warm)}")
            if config.get("cold_start_eval", False):
                logger.info(f"🎯 [Cold Evaluation] Best Validation Result:\n{metrics_dict2str(best_valid_result_cold)}")
                logger.info(f"🧪 [Cold Evaluation] Test Result (upon best valid):\n"
                            f"{metrics_dict2str(best_test_upon_valid_cold)}")

            logger.info(f"\n{'█' * 10} Current BEST (per enabled evaluation mode) {'█' * 10}")
            if config.get("warm_eval", False) and best_test_idx_warm is not None:
                logger.info(f"\n🏆 Warm Evaluation:")
                logger.info(f"📊 Best Hyper-parameters: {config['hyper_parameters']} = "
                            f"{hyper_ret[best_test_idx_warm][0]}")
                logger.info(f"   Valid:\n{metrics_dict2str(hyper_ret[best_test_idx_warm][1])}")
                logger.info(f"   Test:\n{metrics_dict2str(hyper_ret[best_test_idx_warm][2])}")
            if config.get("cold_start_eval", False) and best_test_idx_cold is not None:
                logger.info(f"\n🏆 Cold Evaluation:")
                logger.info(f"📊 Best Hyper-parameters: {config['hyper_parameters']} = "
                            f"{hyper_ret[best_test_idx_cold][0]}")
                logger.info(f"   Valid:\n{metrics_dict2str(hyper_ret[best_test_idx_cold][3])}")
                logger.info(f"   Test:\n{metrics_dict2str(hyper_ret[best_test_idx_cold][4])}")

    # log info
    logger.info('\n============ All Over ============\n')
    for (p, valid_warm, test_warm, valid_cold, test_cold) in hyper_ret:
        if valid_warm == None:
            continue
        logger.info(f"Parameters: {config['hyper_parameters']} = {p}")
        if config.get("warm_eval", False):
            logger.info("🌐 Warm Evaluation:")
            logger.info(f"   Valid:\n{metrics_dict2str(valid_warm, indent=8)}")
            logger.info(f"   Test:\n{metrics_dict2str(test_warm, indent=8)}")
        if config.get("cold_start_eval", False):
            logger.info("🎯 Cold Evaluation:")
            logger.info(f"   Valid:\n{metrics_dict2str(valid_cold, indent=8)}")
            logger.info(f"   Test:\n{metrics_dict2str(test_cold, indent=8)}")
        logger.info('-' * 100)

    logger.info('\n\n█████████████ BEST RESULTS (per enabled evaluation mode) ████████████████')
    if config.get("warm_eval", False) and best_test_idx_warm is not None:
        logger.info(f"\n🏆 Warm Evaluation:")
        logger.info(f"📊 Best Parameters: {config['hyper_parameters']} = "
                    f"{hyper_ret[best_test_idx_warm][0]}")
        logger.info("   Valid:\n"
                    f"{metrics_dict2str(hyper_ret[best_test_idx_warm][1], indent=8)}")
        logger.info("   Test:\n"
                    f"{metrics_dict2str(hyper_ret[best_test_idx_warm][2], indent=8)}")
    else:
        logger.info("\n🏆 Warm Evaluation: Disabled or No Record")
    if config.get("cold_start_eval", False) and best_test_idx_cold is not None:
        logger.info(f"\n🏆 Cold Evaluation:")
        logger.info(f"📊 Best Parameters: {config['hyper_parameters']} = "
                    f"{hyper_ret[best_test_idx_cold][0]}")
        logger.info("   Valid:\n"
                    f"{metrics_dict2str(hyper_ret[best_test_idx_cold][3], indent=8)}")
        logger.info("   Test:\n"
                    f"{metrics_dict2str(hyper_ret[best_test_idx_cold][4], indent=8)}")
    else:
        logger.info("\n🏆 Cold Evaluation: Disabled or No Record")

    logger.info("\n" + "=" * 100 + "\n")
    writer.close()  # tb
