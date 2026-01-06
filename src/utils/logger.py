import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    model_name = config['model']
    if 'log_model_suffix' in config and config['log_model_suffix']:
        model_name = f"{model_name}_{config['log_model_suffix']}"
    dataset_str = config['dataset']
    domains_str = '+'.join(config['domains'])
    time = get_local_time()
    logfilename = '{}-{}-{}-{}.log'.format(model_name, dataset_str, domains_str, time)

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = u"%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    # comment following 3 lines and handlers = [sh, fh] to cancel file dump.
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        #handlers=[sh]
        handlers = [sh, fh]
    )

    return (model_name, dataset_str, domains_str, time)

