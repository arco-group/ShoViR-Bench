# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import json
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, classification_report, jaccard_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss

from transformers import AdamW

from findings_classifier.chexpert_model import ChexpertClassifier

class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)

class LitIGClassifier(pl.LightningModule):
    def __init__(self, num_classes, class_names, class_weights=None, learning_rate=1e-5):
        super().__init__()

        # Model
        self.model = ChexpertClassifier(num_classes)

        # Loss with class weights
        if class_weights is None:
            self.criterion = BCEWithLogitsLoss()
        else:
            self.criterion = BCEWithLogitsLoss(pos_weight=class_weights)

        # Learning rate
        self.learning_rate = learning_rate
        self.class_names = class_names

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
