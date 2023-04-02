import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class BAIRRobotPushing(Dataset):
    """
    Dataset object for BAIR robot pushing dataset. The dataset must be stored
    with each video in a separate directory:
        /path
            /0
                /0.png
                /1.png
                /...
            /1
                /...
    """

    def __init__(self, path, transform=None, add_noise=False):
        assert os.path.exists(path), 'Invalid path to BAIR data set: ' + path
        self.path = path
        self.transform = transform
        self.video_list = os.listdir(self.path)

        self.add_noise = add_noise

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img_names = os.listdir(os.path.join(self.path, self.video_list[ind]))
        img_names = [img_name.split('.')[0] for img_name in img_names]
        img_names.sort(key=float)
        imgs = [Image.open(os.path.join(self.path, self.video_list[ind], i + '.png')) for i in img_names]
        if self.transform is not None:
            # apply the image/video transforms
            imgs = self.transform(imgs)

        # imgs = imgs.unsqueeze(1)

        if self.add_noise:
            imgs = imgs + (torch.rand_like(imgs)-0.5) / 256.

        return imgs

    def __len__(self):
        # total number of videos
        return len(self.video_list)
