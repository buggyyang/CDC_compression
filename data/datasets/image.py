import os
from PIL import Image
from torch.utils.data import Dataset


class IMG(Dataset):
    def __init__(self, path, transform=None):
        assert os.path.exists(path), 'Invalid path to IMAGE data set: ' + path
        self.path = path
        self.transform = transform
        self.img_list = os.listdir(self.path)

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img = [Image.open(os.path.join(self.path, self.img_list[ind]))]
        if self.transform is not None:
            img = self.transform(img)
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        return img

    def __len__(self):
        # total number of videos
        return len(self.img_list)
