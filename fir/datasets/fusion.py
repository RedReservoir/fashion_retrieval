import torch
import torchvision

import numpy as np

from torch.utils.data import Dataset
import torchvision.io

import os
from multiprocessing import Lock

import random



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


    def collate_fn(original_batch):

        new_batch = []

        dataset_idxs = np.asarray([item[0] for item in original_batch])
        unique_dataset_idxs = np.unique(dataset_idxs)

        for dataset_idx in unique_dataset_idxs:

            dataset_batch = []

            dataset_items = [item[1] for item in original_batch if item[0] == dataset_idx]

            for subitem_idx in range(len(dataset_items[0])):

                dataset_subitems = [item[subitem_idx] for item in dataset_items]
                if type(dataset_subitems[0]) == int:
                    dataset_subitems = list(map(torch.tensor, dataset_subitems))

                dataset_batch.append(torch.stack(dataset_subitems))

            new_batch.append((dataset_idx, dataset_batch))

        return new_batch
