import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch


class BIG(Dataset):
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

    def __init__(self, path, transform=None, add_noise=False, img_mode=False):
        assert os.path.exists(
            path), 'Invalid path to UCF+HMDB data set: ' + path
        self.path = path
        self.transform = transform
        self.video_list = os.listdir(self.path)
        self.img_mode = img_mode
        self.add_noise = add_noise

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img_names = os.listdir(os.path.join(
            self.path, self.video_list[ind]))
        img_names = [img_name.split('.')[0] for img_name in img_names]
        img_names.sort(key=float)
        if not self.img_mode:
            imgs = [Image.open(os.path.join(
                self.path, self.video_list[ind], i + '.png')) for i in img_names]
        else:
            select = torch.randint(0, len(img_names), (1,))
            imgs = [Image.open(os.path.join(
                self.path, self.video_list[ind], img_names[select] + '.png'))]
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
