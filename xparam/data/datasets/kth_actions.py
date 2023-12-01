import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class KTHActions(Dataset):
    """
    Dataset object for KTH actions dataset. The dataset must be stored
    with each video (action sequence) in a separate directory:
        /path
            /person01_walking_d1_0
                /0.png
                /1.png
                /...
            /person01_walking_d1_1
                /...
    """
    def __init__(self, path, transform=None, add_noise=False):
        assert os.path.exists(path), 'Invalid path to KTH actions data set: ' + path
        self.path = path
        self.transform = transform
        self.video_list = os.listdir(self.path)
        self.add_noise = add_noise

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img_names = os.listdir(os.path.join(self.path, self.video_list[ind]))
        img_names = [img_name.split('.')[0] for img_name in img_names]
        img_names.sort(key=float)
        imgs = [Image.open(os.path.join(self.path, self.video_list[ind], i + '.png')).convert('L') for i in img_names]
        if self.transform is not None:
            # apply the image/video transforms
            imgs = self.transform(imgs)

        if self.add_noise:
            imgs += torch.randn_like(imgs)/256

        return imgs

    def __len__(self):
        # returns the total number of videos
        return len(os.listdir(self.path))
