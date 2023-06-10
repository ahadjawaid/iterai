from torch.utils.data import Dataset, RandomSampler, SequentialSampler
import pytest
from data import *


class MockDataset(Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return idx


@pytest.mark.parametrize("train_len, valid_len, bs", [(100, 200, 10), (1, 1, 1)])
def test_get_dls(train_len, valid_len, bs):
    train_ds = MockDataset(train_len)
    valid_ds = MockDataset(valid_len)

    train_dl, valid_dl = get_dls(train_ds, valid_ds, bs)

    assert isinstance(train_dl, DataLoader)
    assert isinstance(valid_dl, DataLoader)

    assert len(train_dl.dataset) == train_len
    assert len(valid_dl.dataset) == valid_len

    assert train_dl.batch_size == bs
    assert valid_dl.batch_size == bs * 2

    assert isinstance(train_dl.sampler, RandomSampler)
    assert isinstance(valid_dl.sampler, SequentialSampler)
