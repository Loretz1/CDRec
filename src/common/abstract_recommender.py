import os
import numpy as np
import torch
import torch.nn as nn

class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def set_train_stage(self, stage_id):
        r"""
         Configure the model for the current training stage.
         Control which parameters participate in gradient updates, freeze or unfreeze model components by adjusting `requires_grad`.
        """
        pass

    def calculate_loss(self, interaction, epoch_idx):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction, is_warm):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all target items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all target items, which includes item 0 which is 'PAD'
            shape: [n_batch_users * (n_target_items + 1)]
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.num_users_overlap = dataloader.dataset.num_users_overlap
        self.num_users_src = dataloader.dataset.num_users_src
        self.num_users_tgt = dataloader.dataset.num_users_tgt
        self.num_items_src = dataloader.dataset.num_items_src
        self.num_items_tgt = dataloader.dataset.num_items_tgt

        self.device = config['device']
        self.stage_id = None

    def set_train_stage(self, stage_id):
        self.stage_id = stage_id
