from torch.utils.data import DataLoader, Dataset, default_collate
from operator import itemgetter
from collections.abc import  Mapping
import torch

def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
class DataLoaders:
    def __init__(self, *dls):
        self.train,self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        f = collate_dict(dd['train'])
        return cls(*get_dls(*dd.values(), bs=batch_size, collate_fn=f, **kwargs))


def get_dls(train_ds: Dataset, valid_ds: Dataset, bs: int, **kwargs) -> tuple[DataLoader, DataLoader]:
    return (DataLoader(dataset=train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(dataset=valid_ds, batch_size=bs * 2, **kwargs))


def collate_dict(ds):
    get = itemgetter(*ds.features)

    def _f(b):
        return get(default_collate(b))

    return _f

def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)

def to_cpu(x):
    if isinstance(x, Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res

def collate_device(b):
    return to_device(default_collate(b))