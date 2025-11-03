from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str, metrics_dict2str
import platform
import os

def quick_start(model, dataset, config_dict, save_model=True):
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

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
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value_src = 0.0
    best_test_value_tgt = 0.0
    best_test_idx_src = 0
    best_test_idx_tgt = 0
    idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        (best_valid_score_src, best_valid_result_src, best_test_upon_valid_src,
         best_valid_score_tgt, best_valid_result_tgt, best_test_upon_valid_tgt) \
            = trainer.fit(train_data, valid_data=valid_data,test_data=test_data, saved=save_model)

        hyper_ret.append((hyper_tuple, best_valid_result_src, best_test_upon_valid_src, best_valid_result_tgt, best_test_upon_valid_tgt))

        # save best test
        if best_test_upon_valid_src[val_metric] > best_test_value_src:
            best_test_value_src = best_test_upon_valid_src[val_metric]
            best_test_idx_src = idx
        if best_test_upon_valid_tgt[val_metric] > best_test_value_tgt:
            best_test_value_tgt = best_test_upon_valid_tgt[val_metric]
            best_test_idx_tgt = idx
        idx += 1

        logger.info("\n" + "=" * 100)
        logger.info(f"🌐 [Source Domain] Best Validation Result:\n{metrics_dict2str(best_valid_result_src)}")
        logger.info(f"🧪 [Source Domain] Test Result (upon best valid):\n{metrics_dict2str(best_test_upon_valid_src)}")
        logger.info(f"🎯 [Target Domain] Best Validation Result:\n{metrics_dict2str(best_valid_result_tgt)}")
        logger.info(f"🧪 [Target Domain] Test Result (upon best valid):\n{metrics_dict2str(best_test_upon_valid_tgt)}")
        logger.info("\n████ Current BEST (per domain) ████")
        logger.info(f"\n🏆 Source Domain:")
        logger.info(f"📊 Best Hyper-parameters: {config['hyper_parameters']} = {hyper_ret[best_test_idx_src][0]}")
        logger.info(f"   Valid:\n{metrics_dict2str(hyper_ret[best_test_idx_src][1])}")
        logger.info(f"   Test:\n{metrics_dict2str(hyper_ret[best_test_idx_src][2])}")
        logger.info(f"\n🎯 Target Domain:")
        logger.info(f"📊 Best Hyper-parameters: {config['hyper_parameters']} = {hyper_ret[best_test_idx_tgt][0]}")
        logger.info(f"   Valid:\n{metrics_dict2str(hyper_ret[best_test_idx_tgt][3])}")
        logger.info(f"   Test:\n{metrics_dict2str(hyper_ret[best_test_idx_tgt][4])}")

    # log info
    logger.info('\n============ All Over ============\n')
    for (p, valid_src, test_src, valid_tgt, test_tgt) in hyper_ret:
        logger.info(f"Parameters: {config['hyper_parameters']} = {p}")
        logger.info(f"🌐 Source Domain:")
        logger.info(f"   Valid:\n{metrics_dict2str(valid_src, indent=8)}")
        logger.info(f"   Test:\n{metrics_dict2str(test_src, indent=8)}")
        logger.info(f"🎯 Target Domain:")
        logger.info(f"   Valid:\n{metrics_dict2str(valid_tgt, indent=8)}")
        logger.info(f"   Test:\n{metrics_dict2str(test_tgt, indent=8)}")
        logger.info('-' * 100)

    logger.info('\n\n█████████████ BEST RESULTS (per domain) ████████████████')
    logger.info(f"\n🏆 Source Domain:")
    logger.info(f"📊 Best Parameters: {config['hyper_parameters']} = {hyper_ret[best_test_idx_src][0]}")
    logger.info(f"   Valid:\n{metrics_dict2str(hyper_ret[best_test_idx_src][1], indent=8)}")
    logger.info(f"   Test:\n{metrics_dict2str(hyper_ret[best_test_idx_src][2], indent=8)}")
    logger.info(f"\n🎯 Target Domain:")
    logger.info(f"📊 Best Parameters: {config['hyper_parameters']} = {hyper_ret[best_test_idx_tgt][0]}")
    logger.info(f"   Valid:\n{metrics_dict2str(hyper_ret[best_test_idx_tgt][3], indent=8)}")
    logger.info(f"   Test:\n{metrics_dict2str(hyper_ret[best_test_idx_tgt][4], indent=8)}")
    logger.info("\n" + "=" * 100 + "\n")