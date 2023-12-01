from .load_dataset import load_dataset
from .transposed_collate import train_transposed_collate, test_transposed_collate
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def load_data(data_config, batch_size, num_workers=4, pin_memory=True, distributed=False):
    """
    Wrapper around load_dataset. Gets the dataset, then places it in a DataLoader.

    Args:
        data_config (dict): data configuration dictionary
        batch_size (dict): run configuration dictionary
        num_workers (int): number of threads of multi-processed data Loading
        pin_memory (bool): whether or not to pin memory in cpu
        sequence (bool): whether data examples are sequences, in which case the
                         data loader returns transposed batches with the sequence
                         step as the first dimension and batch index as the
                         second dimension
    """
    train, val = load_dataset(data_config)
    train_spl = DistributedSampler(train) if distributed else None
    val_spl = DistributedSampler(val, shuffle=False) if distributed else None

    if train is not None:
        train = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=False if distributed else True,
            collate_fn=train_transposed_collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_spl
        )

    if val is not None:
        val = DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_transposed_collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=val_spl
        )
    return train, val
