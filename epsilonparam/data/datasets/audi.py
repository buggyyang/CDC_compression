import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torch


class AUDI(Dataset):
    def __init__(self, path, num_of_frame, train=True, transform=None, add_noise=False):
        assert os.path.exists(path), "Invalid path to AUDI data set: " + path
        self.path = path
        self.transform = transform
        self.train = train
        if train:
            self.video_list = list(
                Path(os.path.join(path, "camera_lidar_semantic")).glob("*/camera/cam_front_center")
            )[:-1]
        else:
            self.video_list = list(
                Path(os.path.join(path, "camera_lidar_semantic")).glob("*/camera/cam_front_center")
            )[-1:]
        self.add_noise = add_noise
        self.num_of_frame = num_of_frame
        self.img_paths = []
        for each in self.video_list:
            self.img_paths.append(sorted(list(each.glob("**/*small.png"))))

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        if self.train:
            start_index = torch.randint(0, len(self.img_paths[ind]) - self.num_of_frame, (1,)).item()
        else:
            start_index = 525
        imgs = [Image.open(self.img_paths[ind][start_index + i]) for i in range(self.num_of_frame)]
        if self.transform is not None:
            imgs = self.transform(imgs)

        if self.add_noise:
            imgs = imgs + (torch.rand_like(imgs) - 0.5) / 256.0

        return imgs

    def __len__(self):
        # total number of videos
        return len(self.video_list)
