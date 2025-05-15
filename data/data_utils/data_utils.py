"""
Some preprocessing data functions are adapted https://github.com/tsy935/graphs4mer/tree/main

"""


import numpy as np
import os
import sys
import torch
import pandas as pd
import h5py

import os
import pyedflib
from scipy.signal import resample
from sensors import TUH_LABEL_DICT


def read_dreem_data(data_dir, file_name):
    with h5py.File(os.path.join(data_dir, file_name), "r") as f:
        labels = f["hypnogram"][()]

        signals = []
        channels = []
        for key in f["signals"].keys():
            for ch in f["signals"][key].keys():
                signals.append(f["signals"][key][ch][()])
                channels.append(ch)
    signals = np.stack(signals, axis=0)

    return signals, channels, labels


class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, batched=True):
        """
        Masked inverse transform
        Args:
            data: data for inverse scaling
            is_tensor: whether data is a tensor
            device: device
            mask: shape (batch_size,) nodes where some signals are masked
        """
        device = data.device

        mean = self.mean.copy()
        std = self.std.copy()

        if batched:
            mean = np.expand_dims(mean, 0)
            std = np.expand_dims(std, 0)

        if torch.is_tensor(data):
            mean = torch.FloatTensor(mean).to(device).squeeze(-1)
            std = torch.FloatTensor(std).to(device).squeeze(-1)

        return data * std + mean

def getOrderedChannels(file_name, verbose, labels_object, channel_names):
    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def getEDFsignals(edf):
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals


def resampleData(signals, to_freq=200, window_size=4):
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)
    return resampled

def getSeizureTimes(file_name):
    tse_file = file_name.split(".edf")[0] + ".tse_bi"

    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times

