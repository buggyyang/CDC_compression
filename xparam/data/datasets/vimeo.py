import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class VIMEO(Dataset):

    def __init__(self, path, train=True, transform=None, add_noise=False):
        assert os.path.exists(
            path), 'Invalid path to VIMEO data set: ' + path
        self.path = path
        self.transform = transform
        if train:
            self.video_list = os.path.join(path, 'sep_trainlist.txt')
        else:
            self.video_list = os.path.join(path, 'sep_testlist.txt')
        self.video_list = np.loadtxt(self.video_list, dtype=str)
        self.video_list = np.core.defchararray.add(f'{os.path.join(path, "sequences")}/', self.video_list)

        self.add_noise = add_noise

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img_names = os.listdir(str(self.video_list[ind]))
        imgs = [Image.open(os.path.join(self.video_list[ind], str(img_name)))
                for img_name in img_names]
        if self.transform is not None:
            imgs = self.transform(imgs)

        if self.add_noise:
            imgs = imgs + (torch.rand_like(imgs)-0.5) / 256.

        return imgs

    def __len__(self):
        # total number of videos
        return len(self.video_list)
