import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torch


class CITY(Dataset):
    def __init__(self, path, num_of_frame, train=True, transform=None, add_noise=False):
        assert os.path.exists(path), "Invalid path to CITY data set: " + path
        self.path = path
        self.transform = transform
        self.train = train
        if train:
            self.frame_list = Path(os.path.join(path, "leftImg8bit_sequence/train")).glob("*/*.png")
        else:
            self.frame_list = Path(os.path.join(path, "leftImg8bit_sequence/val")).glob("*/*.png")
        self.add_noise = add_noise
        self.num_of_frame = num_of_frame
        self.frame_list = sorted(self.frame_list)

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        first_frame_ind = ind * 30
        last_frame_ind = (ind+1) * 30
        if self.train:
            start_ind = torch.randint(first_frame_ind, last_frame_ind - self.num_of_frame, (1,)).item()
        else:
            start_ind = first_frame_ind
        imgs = [Image.open(self.frame_list[start_ind + i]) for i in range(self.num_of_frame)]
        if self.transform is not None:
            imgs = self.transform(imgs)

        if self.add_noise:
            imgs = imgs + (torch.rand_like(imgs) - 0.5) / 256.0

        return imgs

    def __len__(self):
        # total number of videos
        return len(self.frame_list) // 30
