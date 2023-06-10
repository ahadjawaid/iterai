from torch.utils.data import Dataset, RandomSampler, SequentialSampler
import pytest
from data import *
import torch


class MockDataset(Dataset):
    def __init__(self, len: int = 1, features: list = []):
        self.len = len
        self.features = features

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {feature: idx for feature in self.features}


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


@pytest.mark.parametrize("features, len", [(['a', 'b'], 100)])
def test_collate_dict(features, len):
    ds = MockDataset(len, features)
    batch = [ds[i] for i in range(len)]
    collate_fn = collate_dict(ds)
    result = collate_fn(batch)

    result_dict = {key: value for key, value in zip(ds.features, result)}

    assert set(result_dict.keys()) == set(features)

    for key in features:
        expected_value = default_collate([b[key] for b in batch])
        assert torch.equal(result_dict[key], expected_value)
