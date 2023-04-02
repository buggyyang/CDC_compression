import os
import numpy as np
from PIL import Image, ImageChops
from torch.utils.data import Dataset
import torch
import random


class Youtube(Dataset):
    def __init__(self, path, nframe=3, train=True, transform=None, seed=1212):
        assert os.path.exists(path), 'Invalid path to youtube data set: ' + path
        random.seed(seed)
        ldir = os.listdir(path)
        random.shuffle(ldir)
        self.transform = transform
        video_list = np.core.defchararray.add(f'{path}/', ldir)
        if train:
            self.video_list = video_list[:-32]
        else:
            self.video_list = video_list[-32:]
        self.nframe = nframe

    def __getitem__(self, ind):
        tot_nframe = len(os.listdir(self.video_list[ind]))
        assert tot_nframe >= self.nframe
        start_ind = torch.randint(1, 1 + tot_nframe - self.nframe, (1, )).item()
        imgs = [
            Image.open(os.path.join(self.video_list[ind],
                                    str(img_name) + '.png')) for img_name in range(start_ind, start_ind + self.nframe)
        ]
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs

    def __len__(self):
        # total number of videos
        return len(self.video_list)
