import torch
import numpy as np
import torchvision.transforms.functional as G
from torch.utils.data import Dataset


class Simulation(Dataset):
    
    def __init__(self, path, number_of_frame, train, size, transform=None):
        
        data = np.load(path).astype(np.single)
        mmin = data.min()
        mmax = data.max()
        self.number_of_frame = number_of_frame
        self.transform = transform
        
        if train:
            self.t = 1000
            train = data[:8000, :, :]
            train = (train - mmin) / (mmax - mmin)
            self.data = torch.from_numpy(train)
            self.data = self.data.unsqueeze(1)
            self.data = G.resize(self.data, size)
            
        else:
            
            self.t = 250
            test = data[8000:, :, :]
            test = (test - mmin) / (mmax - mmin)
            self.data = torch.from_numpy(test)
            self.data = self.data.unsqueeze(1)
            self.data = G.resize(self.data, size)
            
    def __len__(self):
        
        return self.data.size()[0]

    def __getitem__(self, idx):
        
        width = self.t
        start = int(idx/(width))
        p = idx % width
        if p > width - self.number_of_frame:
            p = width - self.number_of_frame
        begin = start * width + p
        frames = self.data[begin:begin + self.number_of_frame, :, :, :]
        return frames