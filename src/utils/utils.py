# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""
import os

import numpy as np
import torch
import importlib
import datetime
import random
from collections import defaultdict
import gzip
import pandas as pd


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
    if value == None:
        return best, cur_step, True, False
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

def get_config_by_path(config, path: str):
    """
    path: e.g., 'training_stages.1.learning_rate'
    """
    parts = path.split('.')
    cur = config
    for p in parts:
        if isinstance(cur, list) and p.isdigit():
            cur = cur[int(p)]
        else:
            cur = cur[p]
    return cur

def set_config_by_path(config, path: str, value):
    parts = path.split('.')
    cur = config
    for p in parts[:-1]:
        if isinstance(cur, list) and p.isdigit():
            cur = cur[int(p)]
        else:
            cur = cur[p]
    last = parts[-1]
    if isinstance(cur, list) and last.isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value

def _parse_gz(path: str):
    with gzip.open(path, 'r') as g:
        for line in g:
            line = line.replace(b'true', b'True').replace(b'false', b'False')
            yield eval(line)

def get_dict_from_raw_data_for_Amazon2014(dataset_path, domain, is_review, keys, values):
    if is_review:
        file_name = "reviews_" + domain + "_5.json.gz"
        ALLOWED_FIELDS = ["reviewerID", "asin", "reviewerName", "helpful", "reviewText", "overall", "summary",
                          "unixReviewTime", "reviewTime"]
    else:
        file_name = "meta_" + domain + ".json.gz"
        ALLOWED_FIELDS = ["asin", "title", "price", "imUrl", "related", "salesRank", "brand",
                          "description"]
    file_path = os.path.join(dataset_path, domain, 'raw', file_name)

    invalid = [f for f in keys if f not in ALLOWED_FIELDS] + [f for f in values if f not in ALLOWED_FIELDS]
    if invalid:
        raise ValueError(f"Invalid field(s): {invalid}. "
                         f"Allowed fields: {ALLOWED_FIELDS}")

    dict = {}
    for inter in _parse_gz(file_path):
        key = tuple(inter.get(f, None) for f in keys)
        dict[key] = {value: inter[value] for value in values if value in inter}

    return dict