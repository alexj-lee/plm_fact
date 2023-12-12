from torch.utils import data
import numpy as np
from sklearn import model_selection
from s3fs.core import S3FileSystem
from dask import array as dda
from typing import Optional, List


def try_load_dataset(dataset_path: str):
    try:
        return dda.from_zarr(dataset_path)
    except:
        raise ValueError(
            f'Tried to load dataset from S3 using `dda.from_zarr`, but failed. Check that the path is correct. Path is: {dataset_path}'
        )


def try_load_offsets(offset_path: str):
    s3_fs = S3FileSystem()
    try:
        return np.load(s3_fs.open(offset_path))
    except:
        raise ValueError(
            f'Tried to load offset from S3 but failed. Check that the path is correct. Path is: {offset_path}'
        )


def get_long_indices(index_list: List[int], cumsum_offsets: List[int]):
    """

    This function takes a list of indices and a cumulative offset length list. We want to convert something like:

    index_list = [0, 2]
    cumsum_offsets = [5, 11, 14] # lengths of sequences for lengths (5, 6, 3)

    desired output: [0, 1, 2, 3, 4, 11, 12, 13]

    Then we can just give a dataloader this list of indices to randomly shuffle to index using dask from S3.

    Args:
        index_list (List[int]): List of indices to convert to long--these are the indices of the sequences.
        cumsum_offsets (List[int]): The lengths of the sequences as a cumulative sum.

    Returns:
        List[int]: List of long indices.

    """

    def _long_indices_from_indx_pair(a: int):
        if a == 0:
            _start = 0
        else:
            _start = cumsum_offsets[a - 1]

        _end = cumsum_offsets[a]
        return range(_start, _end)

    long_indices = []

    for idx in index_list:
        _long_indices = _long_indices_from_indx_pair(idx)
        long_indices.extend(_long_indices)

    return long_indices


def get_data_loaders_rosetta(
    test_size: Optional[float] = 0.2,
    random_state: Optional[int] = 9999,
    includes_register_toks: Optional[bool] = True,
):
    try:
        0 < float(test_size) <= 1
    except:
        raise ValueError(
            f"test_size must be a float between 0 and 1. You passed {test_size}"
        )

    dataset_path = "brainform-data/rosetta"
    offset_path = "s3://brainform-data/rosetta_offsets.npy"

    dataset = try_load_dataset(dataset_path)
    indices_offsets = try_load_offsets(offset_path)

    all_indices = range(
        len(indices_offsets)
    )  # this gives the length of the sequences, is len == num_seqs

    _train_indices, _valid_indices = model_selection.train_test_split(
        all_indices, test_size=test_size, random_state=random_state
    )

    _multiplier = 33 if includes_register_toks else 31
    cumsum_offsets = (
        np.cumsum(indices_offsets) * _multiplier
    )  # still len == num_seqs, but now the element is the index in the actual dataset

    train_indices = get_long_indices(_train_indices, cumsum_offsets)
    valid_indices = get_long_indices(_valid_indices, cumsum_offsets)

    # realized too late I should have just used a subset sampler, but that's OK
    train_dataset = S3WithOffsetsLoader(dataset, train_indices)
    test_dataset = S3WithOffsetsLoader(dataset, valid_indices)

    return train_dataset, test_dataset


class S3WithOffsetsLoader(data.Dataset):
    def __init__(
        self,
        dataset: dda.core.Array,
        indices: List[int],
    ):
        # self.dataset_path = dataset_path
        # self.offset_path = offset_path

        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        _index = self.indices[index]
        return self.dataset[_index]