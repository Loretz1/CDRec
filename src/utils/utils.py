# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import numpy as np
import torch
import importlib
import datetime
import random
from collections import defaultdict


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag

def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str

def metrics_dict2str(result_dict, indent=6):
    """
    Convert evaluation metric dictionary to grouped string.
    Sort metrics by numeric @K ascending (e.g. @5, @10, @20, @50).
    """
    grouped = defaultdict(list)

    for metric, value in result_dict.items():
        if "@" in metric:
            prefix, k = metric.split("@", 1)
        else:
            prefix, k = metric, ""
        grouped[prefix].append((k, value))

    lines = []
    for prefix, values in sorted(grouped.items()):
        # 对每个指标组按 K 的数值大小排序
        values.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        values_str = ", ".join(f"@{k}={v:.4f}" if k else f"{v:.4f}" for k, v in values)
        lines.append(" " * indent + f"{prefix.lower():<12}: {values_str}")
    return "\n".join(lines)

def get_model(model_name):
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class

def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')
