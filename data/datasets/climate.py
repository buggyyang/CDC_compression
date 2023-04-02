import numpy as np

import torch
import os
from torch.utils.data import Dataset


class ClimateData(Dataset):
    def __init__(self, path, mode):
        data = np.load(os.path.join(path, "climate_timestep/W_fields.npy"), mmap_mode="r")
        data = np.reshape(data, (-1, 192, 30, 128), order="F")
        data = np.reshape(data, (-1, 24, 8, 30, 128))
        self.mean = data.mean()
        self.std = np.std(data)
        data = (data - self.mean) / self.std

        if mode == "train":

            self.t = 20
            train = data[:, :20, :, :, :]
            del data
            train = np.reshape(train, (-1, 8, 30, 128))
            train = np.reshape(train, (-1, 30, 128))
            train = np.pad(train, ((0, 0), (1, 1), (0, 0)), "symmetric")
            self.data = torch.from_numpy(train).float()

        else:

            self.t = 4
            test = data[:, 20:, :, :, :]
            del data
            test = np.reshape(test, (-1, 8, 30, 128))
            test = np.reshape(test, (-1, 30, 128))
            test = np.pad(test, ((0, 0), (1, 1), (0, 0)), "symmetric")
            self.data = torch.from_numpy(test).float()

    def __len__(self):

        return self.data.size()[0]

    def __getitem__(self, idx):

        width = self.t * 8
        start = int(idx / (width))
        p = idx % width
        if p > width - 8:
            p = width - 8
        begin = start * width + p
        return self.data[begin : begin + 8, :, :].unsqueeze(1)

