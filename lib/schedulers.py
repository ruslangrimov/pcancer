from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler


class CosineAnnealingLRWithWarmup(_LRScheduler):
    def __init__(self, optimizer, multiplier, warmup_epoch, total_epoch):
        for pg in optimizer.param_groups:
            pg['lr'] /= multiplier
        self.cosine_scheduler = CosineAnnealingLR(optimizer,
                                                  total_epoch-warmup_epoch,
                                                  eta_min=0, last_epoch=-1)
        self.warmup_scheduler = \
            GradualWarmupScheduler(optimizer,
                                   multiplier=multiplier,
                                   total_epoch=warmup_epoch,
                                   after_scheduler=self.cosine_scheduler)
        super().__init__(optimizer)

    def get_lr(self):
        return self.warmup_scheduler.get_lr()

    def step(self, epoch=None):
        return self.warmup_scheduler.step(epoch)
