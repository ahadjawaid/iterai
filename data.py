from torch.utils.data import DataLoader, Dataset, default_collate


def get_dls(train_ds: Dataset, valid_ds: Dataset, bs: int, **kwargs) -> tuple[DataLoader, DataLoader]:
    return (DataLoader(dataset=train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(dataset=valid_ds, batch_size=bs * 2, **kwargs))