import torch
import numpy as np
import torchvision.transforms as torch_transforms
import torchvision.transforms.functional as VF
from PIL import Image, ImageChops
import torch.nn.functional as F

Compose = torch_transforms.Compose


def get_bbox(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return bbox


class RandomRotation(object):
    """
    Rotates a PIL image or sequence of PIL images by a random amount.
    """

    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, input):
        angle = np.random.randint(-self.max_angle, self.max_angle, (1,)).item()
        if type(input) == list:
            return [im.rotate(angle) for im in input]
        return input.rotate(angle)


class RandomCrop(object):
    """
    Randomly crops a PIL image or sequence of PIL images.
    """

    def __init__(self, output_size, black_trim=False):
        if type(output_size) != tuple and type(output_size) != list:
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.black_trim = black_trim

    def __call__(self, input):
        img = input
        if type(input) == list:
            img = img[0]
        width, height = img.size
        black_trim = self.black_trim
        if black_trim:
            bbox = get_bbox(img)
            timg = img.crop(bbox)
            tr_width, tr_height = timg.size
            if (tr_width < self.output_size[0]) or (tr_height < self.output_size[1]):
                black_trim = False
            else:
                width, height = tr_width, tr_height
        left = torch.randint(0, width - self.output_size[0] + 1, (1,)).item()
        top = torch.randint(0, height - self.output_size[1] + 1, (1,)).item()
        if type(input) == list:
            if black_trim:
                return [
                    im.crop(bbox).crop(
                        (left, top, left + self.output_size[0], top + self.output_size[1])
                    )
                    for im in input
                ]
            return [
                im.crop((left, top, left + self.output_size[0], top + self.output_size[1]))
                for im in input
            ]
        if black_trim:
            return input.crop(bbox).crop(
                (left, top, left + self.output_size[0], top + self.output_size[1])
            )
        return input.crop((left, top, left + self.output_size[0], top + self.output_size[1]))


class RandomHorizontalFlip(object):
    """
    Randomly flips a PIL image or sequence of PIL images horizontally.
    """

    def __init__(self):
        pass

    def __call__(self, input):
        flip = torch.rand(1) > 0.5
        if flip:
            if type(input) == list:
                return [im.transpose(Image.FLIP_LEFT_RIGHT) for im in input]
            return input.transpose(Image.FLIP_LEFT_RIGHT)
        return input


class Resize(object):
    """
    Resizes a PIL image or sequence of PIL images.
    img_size can be an int, list or tuple (width, height)
    """

    def __init__(self, img_size):
        if type(img_size) != tuple and type(img_size) != list:
            img_size = (img_size, img_size)
        self.img_size = img_size

    def __call__(self, input):
        if type(input) == list:
            return [im.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR) for im in input]
        return input.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)

class VFResize(object):
    """
    Resizes a PIL image or sequence of PIL images. Use torchvision built-in resize
    img_size can be an int, list or tuple (width, height)
    """

    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, input):
        if type(input) == list:
            return [VF.resize(im, self.img_size) for im in input]
        return VF.resize(input, self.img_size)

class RandomSequenceCrop(object):
    """
    Randomly crops a sequence (list or tensor) to a specified length.
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, input):
        if type(input) == list:
            input_seq_len = len(input)
        elif "shape" in dir(input):
            input_seq_len = input.shape[0]
        max_start_ind = input_seq_len - self.seq_len + 1
        assert max_start_ind > 0, (
            "Sequence length longer than input sequence length: " + str(input_seq_len) + "."
        )
        # start_ind = np.random.choice(range(max_start_ind))
        start_ind = torch.randint(0, max_start_ind, (1,)).item()
        return input[start_ind : start_ind + self.seq_len]


class FixedSequenceCrop(object):
    """
    Randomly crops a sequence (list or tensor) to a specified length.
    """

    def __init__(self, seq_len, start_index=0):
        self.seq_len = seq_len
        self.start_index = start_index

    def __call__(self, input):
        return input[self.start_index : self.start_index + self.seq_len]


class ConcatSequence(object):
    """
    Concatenates a sequence (list of tensors) along a new axis.
    """

    def __init__(self):
        pass

    def __call__(self, input):
        return torch.stack(input)


class ImageToTensor(object):
    """
    Converts a PIL image or sequence of PIL images into (a) PyTorch tensor(s).
    """

    def __init__(self):
        self.to_tensor = torch_transforms.ToTensor()

    def __call__(self, input):
        if type(input) == list:
            return [self.to_tensor(i) for i in input]
        return self.to_tensor(input)


class ToTensor(object):
    """
    Converts a numpy array into (a) PyTorch tensor(s).
    """

    def __init__(self):
        pass

    def __call__(self, input):
        return torch.from_numpy(input)


class NormalizeImage(object):
    """
    Normalizes a PyTorch image tensor or a list of PyTorch image tensors.

    Args:
        mean (int, tensor): mean to subtract
        std (int, tensor): standard deviation by which to divide
    """

    def __init__(self, mean, std):
        self.normalize = torch_transforms.Normalize(mean, std)

    def __call__(self, input):
        if type(input) == list:
            return [self.normalize(i) for i in input]
        return self.normalize(input)


class Normalize(object):
    """
    Normalizes a PyTorch tensor or a list of PyTorch tensors.

    Args:
        mean (int, tensor): mean to subtract
        std (int, tensor): standard deviation by which to divide
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, input):
        return (input - self.mean) / self.std

    def __call__(self, input):
        if type(input) == list:
            return [self.normalize(i) for i in input]
        return self.normalize(input)


class ChannelFirst(object):
    def __init__(self):
        pass

    def __call__(self, input):
        return input.permute(0, 3, 1, 2)


class ResizeFrameSeq(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, input):
        return F.interpolate(input.float(), size=(self.h, self.w), mode="bilinear")


class ResizeSeqMin(object):
    def __init__(self, min_len):
        self.min_len = min_len

    def __call__(self, input):
        assert len(input.shape) == 4
        T, C, H, W = input.shape
        if H < W:
            return F.interpolate(
                input.float(), size=(self.min_len, int(float(W) / float(H) * self.min_len))
            )
        else:
            return F.interpolate(
                input.float(), size=(int(float(H) / float(W) * self.min_len), self.min_len)
            )


class SegmentCrop(object):
    """
        T, C, H, W input
    """

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, input):
        assert self.h <= input.shape[2]
        assert self.w <= input.shape[-1]
        c_h = torch.randint(0, input.shape[2] - self.h + 1, (1,)).item()
        c_w = torch.randint(0, input.shape[-1] - self.w + 1, (1,)).item()
        return input[:, :, c_h : c_h + self.h, c_w : c_w + self.w]


class FixedSegmentCrop(object):
    """
        T, C, H, W input
    """

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, input):
        assert self.h <= input.shape[2]
        assert self.w <= input.shape[-1]
        return input[:, :, : self.h, : self.w]


class BinSequence(object):
    """
    Reshapes a sequence into a series of bins of the same width. Used in modeling
    audio data.

    Args:
        window (int): the window over which consecutive samples are aggregated
    """

    def __init__(self, window):
        self.window = window

    def __call__(self, input):
        if type(input) == list:
            input = np.array(input)
        n_bins = int(input.shape[0] / self.window)
        input = input[: n_bins * self.window]
        if type(input) == np.ndarray:
            return input.reshape(-1, self.window)
        else:
            return input.view(-1, self.window)


class CentercropList(object):
    def __init__(self, img_size):
        self.crop = torch_transforms.CenterCrop(img_size)

    def __call__(self, inputs):
        return [self.crop(input) for input in inputs]
