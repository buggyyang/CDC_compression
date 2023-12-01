import socket
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch


class StochasticMovingMNIST(Dataset):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self,
                 train,
                 data_root,
                 seq_len=20,
                 num_digits=2,
                 image_size=64,
                 deterministic=True,
                 add_noise=False,
                 epoch_size=0):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 1
        self.add_noise = add_noise
        self.epoch_size = epoch_size

        self.data = datasets.MNIST(path,
                                   train=train,
                                   download=True,
                                   transform=transforms.Compose(
                                       [transforms.Scale(self.digit_size),
                                        transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        if self.epoch_size > 0:
            return self.epoch_size
        else:
            return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len, image_size, image_size, self.channels), dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size - digit_size)
            sy = np.random.randint(image_size - digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size - 32:
                    sy = image_size - 32 - 1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size - 32:
                    sx = image_size - 32 - 1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[t, sy:sy + 32, sx:sx + 32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x = torch.FloatTensor(x).permute(0, 3, 1, 2).contiguous()
        if self.add_noise:
            x += torch.randn_like(x) / 256

        x[x < 0] = 0.
        x[x > 1] = 1.
        return x
