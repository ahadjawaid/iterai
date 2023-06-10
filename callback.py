from learner import CancelBatchException, CancelEpochException, CancelFitException
from torcheval.metrics import Mean
from copy import copy
import fastcore as fc
from data import def_device, to_device, to_cpu
import math
import matplotlib.pyplot as plt
from collections.abc import Mapping
import torch
from torch.optim.lr_scheduler import ExponentialLR
from fastprogress import progress_bar,master_bar



class Callback():
    order = 0


class SingleBatchCB(Callback):
    order = 1
    def after_batch(self, learn): raise CancelFitException()


class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d): print(d)
    def before_fit(self, learn): learn.metrics = self
    def before_epoch(self, learn): [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k:f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.model.training else 'eval'
        self._log(log)

    def after_batch(self, learn):
        x,y,*_ = to_cpu(learn.batch)
        for m in self.metrics.values(): m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))


class DeviceCB(Callback):
    def __init__(self, device=def_device): 
        fc.store_attr()

    def before_fit(self, learn):
        if hasattr(learn.model, 'to'): learn.model.to(self.device)

    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)


class TrainCB(Callback):
    def __init__(self, n_inp=1): self.n_inp = n_inp

    def predict(self, learn): learn.preds = learn.model(*learn.batch[:self.n_inp])

    def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp:])

    def backward(self, learn): learn.loss.backward()

    def step(self, learn): learn.opt.step()

    def zero_grad(self, learn): learn.opt.zero_grad()


class ProgressCB(Callback):
    order = MetricsCB.order + 1

    def __init__(self, plot=False):
        self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses: self.mbar.update_graph([[fc.L.range(self.losses), self.losses], [
                fc.L.range(learn.epoch).map(lambda x: (x + 1) * len(learn.dls.train)), self.val_losses]])

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
                self.mbar.update_graph([[fc.L.range(self.losses), self.losses],
                                        [fc.L.range(learn.epoch + 1).map(lambda x: (x + 1) * len(learn.dls.train)),
                                         self.val_losses]])


class LRFinderCB(Callback):
    def __init__(self, gamma=1.3, max_mult=3):
        fc.store_attr()

    def before_fit(self, learn):
        self.sched = ExponentialLR(learn.opt, self.gamma)
        self.lrs, self.losses = [], []
        self.min = math.inf

    def after_batch(self, learn):
        if not learn.training: raise CancelEpochException()
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        loss = to_cpu(learn.loss)
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if math.isnan(loss) or (loss > self.min * self.max_mult):
            raise CancelFitException()
        self.sched.step()

    def cleanup_fit(self, learn):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')


class BaseSchedCB(Callback):
    def __init__(self, sched): self.sched = sched
    def before_fit(self, learn): self.schedo = self.sched(learn.opt)
    def _step(self, learn):
        if learn.training: self.schedo.step()


class BatchSchedCB(BaseSchedCB):
    def after_batch(self, learn): self._step(learn)


class HasLearnCB(Callback):
    def before_fit(self, learn): self.learn = learn 
    def after_fit(self, learn): self.learn = None


class RecorderCB(Callback):
    def __init__(self, **d): self.d = d
    def before_fit(self, learn):
        self.recs = {k:[] for k in self.d}
        self.pg = learn.opt.param_groups[0]
    
    def after_batch(self, learn):
        if not learn.training: return
        for k,v in self.d.items():
            self.recs[k].append(v(self))

    def plot(self):
        for k,v in self.recs.items():
            plt.plot(v, label=k)
            plt.legend()
            plt.show()


class EpochSchedCB(BaseSchedCB):
    def after_epoch(self, learn): self._step(learn)