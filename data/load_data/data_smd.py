import torch
import pickle
#import git
import sys
import numpy as np
import os
from torch_geometric.data import InMemoryDataset, Data, Dataset
from data.data_utils.general_data_utils import (
    StandardScaler,
    MultilabelBalancedRandomSampler,
    DistributedSamplerWrapper,
)
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch_geometric
import pytorch_lightning as pl
from data.data_utils.general_data_utils import ImbalancedDatasetSampler
import pandas as pd



class SMDDataset(InMemoryDataset):
    def __init__(
        self,
        raw_data_dir,
        split,
        num_nodes,
        adj_mat_dir=None,
        sampling_freq=1/60,
        transform=None,
        pre_transform=None,
    ):
        self.raw_data_dir = raw_data_dir
        self.split = split
        self.num_nodes = num_nodes
        self.sampling_freq = sampling_freq
        self.adj_mat_dir = adj_mat_dir

        self.X = np.load(
            os.path.join(raw_data_dir, "X_{}.npy".format(split)), allow_pickle=True
        )

        self.y = np.load(
            os.path.join(raw_data_dir, "y_{}.npy".format(split)), allow_pickle=True
        )


        # all ecg ids
        self.ecg_ids = np.arange(self.X.shape[0])

        # process
        super().__init__(None, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.raw_data_dir]

    def len(self):
        return len(self.ecg_ids)

    def _get_combined_graph(self, adj_mat_dir):
        with open(adj_mat_dir, "rb") as pf:
            adj_mat = pickle.load(pf)
            adj_mat = adj_mat[-1]
        return adj_mat

    def get_labels(self):

        return torch.FloatTensor(self.y)

    def get(self, idx):

        x = self.X[idx]  # (seq_len*freq, num_nodes)
        y = self.y[idx]
        seq_len = x.shape[0]

        x = np.expand_dims(x, axis=-1)  # (max_seq_len*freq, num_nodes, 1)
        x = np.transpose(x, axes=(1, 0, 2))  # (num_nodes, max_seq_len*freq, 1)

        # get edge index
        if self.adj_mat_dir is not None:
            adj_mat = self._get_combined_graph()
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(
                torch.FloatTensor(adj_mat)
            )
        else:
            edge_index = None
            edge_weight = None

        # pyg graph
        x = torch.FloatTensor(x)
        #y = torch.FloatTensor(y).unsqueeze(0)
        y = torch.FloatTensor([y])
        seq_len = torch.LongTensor([seq_len])
        data = Data(x=x, y=y, seq_len=seq_len)
        if edge_index is not None:
            data.edge_index = edge_index.contiguous()
            data.edge_attr = edge_weight
            data.adj_mat = torch.FloatTensor(adj_mat).unsqueeze(0)

        data.writeout_fn = str(self.ecg_ids[idx])

        return data


class SMD_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_data_dir,
        num_nodes,
        train_batch_size,
        test_batch_size,
        num_workers,
        adj_mat_dir=None,
        sampling_freq=100,
        balanced_sampling=False,
        pin_memory=False,
        ddp=False,
    ):
        super().__init__()

        self.raw_data_dir = raw_data_dir
        self.num_nodes = num_nodes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.adj_mat_dir = adj_mat_dir
        self.sampling_freq = sampling_freq
        self.balanced_sampling = balanced_sampling
        self.pin_memory = pin_memory
        self.ddp = ddp

        self.train_dataset = SMDDataset(
            # root=self.raw_data_dir,
            raw_data_dir=self.raw_data_dir,
            split="train",
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            sampling_freq=self.sampling_freq,
        )

        self.val_dataset = SMDDataset(
            # root=self.raw_data_dir,
            raw_data_dir=self.raw_data_dir,
            split="valid",
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            sampling_freq=self.sampling_freq,
        )

        self.test_dataset = SMDDataset(
            # root=self.raw_data_dir,
            raw_data_dir=self.raw_data_dir,
            split="test",
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            sampling_freq=self.sampling_freq,
        )

    def train_dataloader(self):
        #print(self.train_dataset._get_labels())
        if self.balanced_sampling:
            num_nor = torch.sum(self.train_dataset.get_labels() == 0)
            sampler = ImbalancedDatasetSampler(
                dataset=self.train_dataset,
                num_samples=num_nor * 3,
                replacement=False,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            shuffle=shuffle,
            sampler=None,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return train_dataloader



    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return val_dataloader

    def test_dataloader(self):

        num_normal = torch.sum(self.test_dataset.get_labels() == 0)
        num_abnormal = torch.sum(self.test_dataset.get_labels() == 1)


        shuffle = False

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return test_dataloader

    def teardown(self, stage=None):
        # clean up after fit or test
        # called on every process in DDP
        pass
