import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task.

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass

    @property
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass

    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends)."""
        self.record = []
        self.bs = []


# accuracy
class AccMetric(AbsMetric):
    r"""Calculate the accuracy."""

    def __init__(self):
        super(AccMetric, self).__init__()

    def update_fun(self, pred, gt):
        r""" """
        pred = pred.argmax(dim=1) - 1
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r""" """
        return [(sum(self.record) / sum(self.bs))]


# L1 Error
class L1Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE)."""

    def __init__(self):
        super(L1Metric, self).__init__()

    def update_fun(self, pred, gt):
        r""" """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r""" """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records * batch_size).sum() / (sum(batch_size))]
