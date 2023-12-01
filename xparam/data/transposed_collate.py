from torch.utils.data.dataloader import default_collate
import torch


def train_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    # assert transposed_batch.shape[0] >= 4
    # idx = torch.randint(4, transposed_batch.shape[0] + 1, size=(1,)).item()
    # return transposed_batch[:idx]
    return transposed_batch


def test_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    return transposed_batch
