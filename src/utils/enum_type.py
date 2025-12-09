# -*- coding: utf-8 -*-

"""
DiffICDR.utils.enum_type
#######################
"""

from enum import Enum


class TrainDataLoaderState(Enum):
    """States for TrainDataLoader.

    - ``BOTH``: Return both interactions in source domain and target domain.
    - ``SOURCE``: Only return the interactions in source domain.
    - ``TARGET``: Only return the interactions in target domain.
    - ``OVERLAP``: Return the interactions of overlap users
    - ``OVERLAP_USER``: Return the id of overlap users
    """

    BOTH = 1
    SOURCE = 2
    TARGET = 3
    OVERLAP = 4
    OVERLAP_USER = 5


# train_mode2state = {'BOTH': TrainDataLoaderState.BOTH,
#                     'SOURCE': TrainDataLoaderState.SOURCE,
#                     'TARGET': TrainDataLoaderState.TARGET,
#                     'OVERLAP': TrainDataLoaderState.OVERLAP}

class EvalDataLoaderState(Enum):
    """States for EvalDataLoader."""

    WARM = 1
    COLD = 2