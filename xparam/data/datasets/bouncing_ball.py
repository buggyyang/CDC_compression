import os
import torch
import numpy as np
from torch.utils.data import Dataset


class BouncingBall(Dataset):
    """
    Dataset class for moving MNIST dataset.

    Args:
        path (str): path to the .mat dataset
        transform (torchvision.transforms): image/video transforms
    """
    def __init__(self, path, sequence_lengh):
        assert os.path.exists(path), 'Invalid path to Bouncing Ball data set: ' + path
        self.sequence_length = sequence_lengh
        self.data = np.load(path)

    def __getitem__(self, ind):
        imgs = self.data[ind,:,:,:].astype('float32')
        s, h, w = imgs.shape
        imgs = imgs.reshape(s, 1, h, w)

        imgs = imgs[:self.sequence_length, :, :, :]

        return torch.FloatTensor(imgs).contiguous()

    def __len__(self):
        return self.data.shape[0]
