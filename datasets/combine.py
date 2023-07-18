import torch
import torchvision

import numpy as np

from torch.utils.data import Dataset
import torchvision.io

import os
from multiprocessing import Lock

import random

import utils.mem


class Fusion(Dataset):
    """
    """

    def __init__(
            self,
            datasets
            ):
        
        self.datasets = datasets

        self._precalculate_fused_idxs()


    def _precalculate_fused_idxs(self):

        self.fused_idxs = []
        lengths = np.asarray([len(dataset) for dataset in self.datasets])

        counts = np.asarray([0 for length in lengths])
        percs = np.asarray([0 for length in lengths], dtype=float)

        for _ in range(sum(lengths)):
            
            sidx = percs.argmin()
            
            self.fused_idxs.append((sidx, counts[sidx]))
            counts[sidx] += 1
            percs[sidx] = counts[sidx] / lengths[sidx]


    def __len__(self):

        return sum([len(dataset) for dataset in self.datasets])


    def __getitem__(self, idx):

        dataset_idx, item_idx = self.fused_idxs[idx]
        return dataset_idx, self.datasets[dataset_idx][item_idx]
