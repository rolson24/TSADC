"""
iEEG dataloader following the same format as data_tusz.py
"""

import sys
import os
import pytorch_lightning as pl
import pickle
import numpy as np
import pandas as pd
import mne
import torch
import torch_geometric
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Dataset
from typing import Optional
from tqdm import tqdm
from scipy import signal
from data.data_utils.data_utils import StandardScaler

FILEMARKER_DIR = "data/data_ieeg/data_and_labels"


class IEEGDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            raw_data_path,
            file_marker,
            split,
            seq_len, # Window size I think?
            num_nodes,
            adj_mat_dir,
            freq=1000,  # Default frequency for iEEG
            scaler=None,
            transform=None,
            pre_transform=None,
            repreproc=False,
    ):
        self.root = root
        self.raw_data_path = raw_data_path
        self.file_marker = file_marker
        self.split = split
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.adj_mat_dir = adj_mat_dir
        self.freq = freq
        self.scaler = scaler

        # Build file list from directory structure
        self._build_file_list()

        # process
        super().__init__(root, transform, pre_transform)

    def _build_file_list(self):
        """Build file list from patient subdirectories"""
        self.file_paths = []
        self.labels = []
        self.clip_idxs = []
        self.channel_idxs = []  # Add channel index tracking
        
        # If file_marker is provided, use it; otherwise scan directory
        if self.file_marker is not None and not self.file_marker.empty:
            self.file_paths = self.file_marker["file_path"].tolist()
            self.labels = self.file_marker["label"].tolist()
            self.clip_idxs = self.file_marker["clip_index"].tolist()
            # Handle channel indices if they exist in file marker
            if "channel_index" in self.file_marker.columns:
                self.channel_idxs = self.file_marker["channel_index"].tolist()
            else:
                self.channel_idxs = [0] * len(self.file_paths)  # Default to channel 0
        else:
            print(f"No file marker provided, scanning directory {self.raw_data_path} for files...")
            # Scan directory structure
            for patient_dir in os.listdir(self.raw_data_path):
                patient_path = os.path.join(self.raw_data_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue
                    
                for file_name in os.listdir(patient_path):
                    if not (file_name.endswith('.pkl') or file_name.endswith('.pickle')):
                        continue
                        
                    file_path = os.path.join(patient_dir, file_name)
                    
                    # Extract label from filename
                    if '_preictal_' in file_name:
                        label = 0
                    elif '_ictal_' in file_name:
                        label = 1
                    elif '_postictal_' in file_name:
                        label = 2
                    else:
                        continue  # Skip files that don't match expected pattern
                    
                    # Load data trace from original file
                    full_path = os.path.join(self.raw_data_path, file_path)
                    data_trace, curr_freq = self._load_ieeg_data(full_path)

                    total_time_points = data_trace.shape[1]
                    clip_duration = int(self.freq * self.seq_len)
                    num_clips = total_time_points // clip_duration
                    num_channels = data_trace.shape[0]
                    
                    # Add entries for each possible clip and channel combination
                    for clip_idx in range(num_clips):
                        for channel_idx in range(num_channels):
                            self.file_paths.append(file_path)
                            self.labels.append(label)
                            self.clip_idxs.append(clip_idx)
                            self.channel_idxs.append(channel_idx)

    @property
    def raw_file_names(self):
        """Return all pickle files in patient subdirectories"""
        raw_files = []
        for patient_dir in os.listdir(self.raw_data_path):
            patient_path = os.path.join(self.raw_data_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for file_name in os.listdir(patient_path):
                if file_name.endswith('.pkl') or file_name.endswith('.pickle'):
                    raw_files.append(os.path.join(patient_dir, file_name))
        return raw_files

    @property
    def processed_file_names(self):
        # Now we can accurately list all channel files since we track channels in _build_file_list
        processed_files = []
        for idx in range(len(self.file_paths)):
            base_fn = "{}_clip{}_ch{:02d}.pt".format(
                self.file_paths[idx].replace('.pkl', '').replace('.pickle', '').replace('/', '_'),
                self.clip_idxs[idx],
                self.channel_idxs[idx]
            )
            processed_files.append(base_fn)
        return processed_files

    def len(self):
        return len(self.file_paths)

    def _get_combined_graph(self):
        with open(self.adj_mat_dir, "rb") as pf:
            adj_mat = pickle.load(pf)
            if isinstance(adj_mat, list):
                adj_mat = adj_mat[-1]
        return adj_mat

    def get_labels(self):
        return torch.LongTensor(self.labels)  # Use LongTensor for multi-class

    def _load_ieeg_data(self, file_path):
        """Load iEEG data from pickle file and extract data trace"""
        with open(file_path, 'rb') as file:
            data_obj = pickle.load(file)

        # print(f"Type of loaded data: {type(data_obj)}")
        # resample the data to self.freq if it has type mne.io.Raw
        if type(data_obj) == mne.io.array._array.RawArray:
            data_obj = data_obj.resample(self.freq, npad="auto")
            # print(f"Resampled data to {self.freq} Hz")

        # Extract data using get_data() method as shown in checkout_data.py
        data_trace = data_obj.get_data()
        sample_rate = data_obj.info['sfreq'] if hasattr(data_obj, 'info') else 1000  # Default to 1000 Hz if not available

        # print(f"Loaded data from {file_path} with shape {data_trace.shape} at {sample_rate} Hz")

        return data_trace, sample_rate


    def process(self):
        # Group processing by unique file+clip combinations to avoid reloading the same data
        processed_combinations = set()
        
        for idx in tqdm(range(len(self.file_paths))):
            file_path = self.file_paths[idx]
            y = self.labels[idx]
            clip_idx = self.clip_idxs[idx]
            
            # Create a unique key for this file+clip combination
            file_clip_key = (file_path, clip_idx)
            
            if file_clip_key in processed_combinations:
                continue  # Already processed all channels for this file+clip
            
            processed_combinations.add(file_clip_key)
            
            writeout_fn = "{}_clip{}".format(
                file_path.replace('.pkl', '').replace('.pickle', '').replace('/', '_'),
                clip_idx
            )

            # Load resampled data
            full_file_path = os.path.join(self.raw_data_path, file_path)
            x, curr_sfreq = self._load_ieeg_data(full_file_path)
            
            # Now curr_sfreq should equal self.freq
            assert curr_sfreq == self.freq, f"Expected frequency {self.freq}, got {curr_sfreq}"
            
            # Extract the specified time segment
            time_start_idx = clip_idx * int(self.freq * self.seq_len)
            time_end_idx = time_start_idx + int(self.freq * self.seq_len)
            x = x[:, time_start_idx:time_end_idx]  # (num_nodes, seq_len*freq)
            assert x.shape[1] == self.freq * self.seq_len
            
            # Process each channel separately
            for channel_idx in range(x.shape[0]):
                # Check if this specific channel file already exists
                channel_writeout_fn = f"{writeout_fn}_ch{channel_idx:02d}"
                if os.path.exists(os.path.join(self.processed_dir, "{}.pt".format(channel_writeout_fn))):
                    continue
                
                # Extract single channel data
                single_channel_x = x[channel_idx, :]  # (seq_len*freq,)
                single_channel_x = np.expand_dims(single_channel_x, axis=0)  # (1, seq_len*freq)
                single_channel_x = np.expand_dims(single_channel_x, axis=-1)  # (1, seq_len*freq, 1)
                
                # Create identity matrix for single channel (1x1 identity matrix)
                adj_mat = np.eye(1)
                
                edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(
                    torch.FloatTensor(adj_mat)
                )

                # pyg graph for single channel
                single_channel_x = torch.FloatTensor(single_channel_x)  # (1, seq_len*freq, 1)
                y_tensor = torch.LongTensor([y])  # Use LongTensor for multi-class
                data = Data(
                    x=single_channel_x,
                    edge_index=edge_index.contiguous(),
                    edge_attr=edge_weight,
                    y=y_tensor,
                    adj_mat=torch.FloatTensor(adj_mat).unsqueeze(0),
                )

                data.writeout_fn = channel_writeout_fn

                torch.save(
                    data,
                    os.path.join(self.processed_dir, "{}.pt".format(channel_writeout_fn)),
                )

    def get(self, idx):
        # Now we can directly use the tracked indices
        file_path = self.file_paths[idx]
        y = self.labels[idx]
        clip_idx = self.clip_idxs[idx]
        channel_idx = self.channel_idxs[idx]

        writeout_fn = "{}_clip{}_ch{:02d}".format(
            file_path.replace('.pkl', '').replace('.pickle', '').replace('/', '_'),
            clip_idx,
            channel_idx
        )

        data = torch.load(os.path.join(self.processed_dir, "{}.pt".format(writeout_fn)), weights_only=False)

        if self.scaler is not None:
            # standardize
            data.x = self.scaler.transform(data.x)

        data.x = data.x.float()

        return data


class IEEG_DataModule(pl.LightningDataModule):
    def __init__(
            self,
            raw_data_path,
            preproc_save_dir,
            seq_len,
            num_nodes,
            train_batch_size,
            test_batch_size,
            num_workers,
            freq=1000,
            adj_mat_dir=None,
            standardize=True,
            balanced_sampling=False,
            pin_memory=False,
    ):
        super().__init__()

        self.raw_data_path = raw_data_path
        self.preproc_save_dir = preproc_save_dir
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.freq = freq
        self.adj_mat_dir = adj_mat_dir
        self.standardize = standardize
        self.balanced_sampling = balanced_sampling
        self.pin_memory = pin_memory

        # Load file markers if they exist, otherwise datasets will scan directory
        self.file_markers = {}
        for split in ["train", "val", "test"]:
            marker_file = os.path.join(
                FILEMARKER_DIR, "{}_file_markers_{}s.csv".format(split, seq_len)
            )
            if os.path.exists(marker_file):
                self.file_markers[split] = pd.read_csv(marker_file)
            else:
                self.file_markers[split] = pd.DataFrame()  # Empty dataframe

        if standardize:
            # Get unique files from train split
            if not self.file_markers["train"].empty:
                train_files = list(set(self.file_markers["train"]["file_path"].tolist()))
            else:
                # Scan directory for training files
                train_files = []
                for patient_dir in os.listdir(raw_data_path):
                    patient_path = os.path.join(raw_data_path, patient_dir)
                    if not os.path.isdir(patient_path):
                        continue
                    for file_name in os.listdir(patient_path):
                        if file_name.endswith('.pkl') or file_name.endswith('.pickle'):
                            train_files.append(os.path.join(patient_dir, file_name))
            
            train_files = [os.path.join(raw_data_path, fn) for fn in train_files]
            self.mean, self.std = self._compute_mean_std(train_files)
            print("mean:", self.mean.shape)

            self.scaler = StandardScaler(mean=self.mean, std=self.std)
        else:
            self.scaler = None

        self.train_dataset = IEEGDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["train"],
            split="train",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            freq=self.freq,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        self.val_dataset = IEEGDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["val"],
            split="val",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            freq=self.freq,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        self.test_dataset = IEEGDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["test"],
            split="test",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            freq=self.freq,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=False,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=True,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            shuffle=True,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return test_dataloader

    def _load_ieeg_data(self, file_path):
        """Load iEEG data from pickle file and extract data trace"""
        with open(file_path, 'rb') as file:
            data_obj = pickle.load(file)
        
        # Extract data using get_data() method as shown in checkout_data.py
        data_trace = data_obj.get_data()
        return data_trace

    def _compute_mean_std(self, train_files):
        """Compute mean and std across all training files, handling variable number of nodes"""
        if ".pkl" in train_files[0] or ".pickle" in train_files[0]:
            count = 0
            all_signals = []
            
            # First pass: collect all signals to determine max number of nodes
            max_nodes = 0
            for idx in tqdm(range(len(train_files)), desc="First pass - finding max nodes"):
                signal = self._load_ieeg_data(train_files[idx])  # (num_nodes, time_points)
                max_nodes = max(max_nodes, signal.shape[0])
                all_signals.append(signal)
            
            # Second pass: compute statistics using max_nodes
            signal_sum = np.zeros((max_nodes))
            signal_sum_sqrt = np.zeros((max_nodes))
            
            for signal in tqdm(all_signals, desc="Second pass - computing statistics"):
                # Pad signal to max_nodes if necessary
                if signal.shape[0] < max_nodes:
                    padded_signal = np.zeros((max_nodes, signal.shape[1]))
                    padded_signal[:signal.shape[0], :signal.shape[1]] = signal
                    signal = padded_signal
                
                signal_sum += signal.sum(axis=-1)
                signal_sum_sqrt += (signal ** 2).sum(axis=-1)
                count += signal.shape[-1]
                
            total_mean = signal_sum / count
            total_var = (signal_sum_sqrt / count) - (total_mean ** 2)
            total_std = np.sqrt(total_var)
        else:
            raise NotImplementedError

        return np.expand_dims(np.expand_dims(total_mean, -1), -1), np.expand_dims(
            np.expand_dims(total_std, -1), -1
        )

    def teardown(self, stage=None):
        # clean up after fit or test
        # called on every process in DDP
        pass
