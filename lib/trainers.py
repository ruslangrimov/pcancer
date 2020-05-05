import torch
from torch import optim
from torch.optim import lr_scheduler
from torch import nn
# from torch.nn import functional as F

from pytorch_lightning.core import LightningModule

from .schedulers import DelayedScheduler
from .losses import SmoothLoss
from .datasets import actual_lbl_nums


class GeneralModule(LightningModule):
    def __init__(self, model, hparams, log_train_every_batch=False):
        super().__init__()
        self.model = model
        self.hparams = hparams
        self.log_train_every_batch = log_train_every_batch
        self.optimizer = None
        self.scheduler = None
        self.warmup_steps = 0

    def forward(self, x):
        return self.model(x)

    @classmethod
    def _accuracy(cls, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        eq = pred.eq(target.view_as(pred))
        return eq.float().mean()
        # return eq.all(dim=-1).float().mean()

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_idx, optimizer,
                       optimizer_idx, second_order_closure=None):
        # warm up lr
        if 'warmup_steps' in self.hparams:
            if self.trainer.global_step < self.hparams['warmup_steps']:
                lr_scale = min(1., float(self.trainer.global_step + 1) /
                               self.hparams['warmup_steps'])
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.hparams['learning_rate']

        # update params
        return super().optimizer_step(current_epoch, batch_idx, optimizer,
                                      optimizer_idx, second_order_closure)

    def configure_optimizers(self):
        omtimizer_class = getattr(optim, self.hparams['optimizer']['name'])
        self.optimizer = omtimizer_class(self.parameters(),
                                         lr=self.hparams['learning_rate'],
                                         **self.hparams['optimizer']['params'])

        # optimizers = [{k: v for k, v in self.hparams['optimizer'].items()
        #                if k not in {'name', 'params'}}]
        #
        # optimizers[0]['optimizer'] = self.optimizer

        optimizers = [self.optimizer, ]

        scheduler_class = getattr(lr_scheduler,
                                  self.hparams['scheduler']['name'])

        self.scheduler = scheduler_class(self.optimizer,
                                         **self.hparams['scheduler']['params'])

        if 'warmup_steps' in self.hparams:
            self.scheduler = DelayedScheduler(self.optimizer,
                                              self.hparams['warmup_steps'],
                                              self.scheduler)
        elif 'warmup_epochs' in self.hparams:
            self.scheduler = DelayedScheduler(self.optimizer,
                                              self.hparams['warmup_epochs'],
                                              self.scheduler)
            self.hparams['warmup_steps'] = (self.hparams['warmup_epochs'] *
                                            self.hparams['steps_in_epoh'])

        schedulers = [{k: v for k, v in self.hparams['scheduler'].items()
                       if k not in {'name', 'params'}}]

        schedulers[0]['scheduler'] = self.scheduler

        print(optimizers, schedulers)

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        if self.log_train_every_batch:
            return {}
        else:
            keys = outputs[0].keys()
            o_dict = {}
            for k in keys:
                # if k != 'loss':
                o_dict[k] = sum([o[k] for o in outputs]) / len(outputs)

        return {'log': o_dict}

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        o_dict = {}
        for k in keys:
            o_dict[k] = sum([o[k] for o in outputs]) / len(outputs)

        return {'log': o_dict}


class PatchesModuleV1(GeneralModule):
    def __init__(self, model, hparams, log_train_every_batch=False):
        super().__init__(model, hparams, log_train_every_batch)
        self.hparams = hparams
        self.model = model
        self.log_train_every_batch = log_train_every_batch

        self.dec_loss = nn.MSELoss()
        self.mask_loss = SmoothLoss(nn.KLDivLoss(), smoothing=0.2,
                                    one_hot_target=True)
        self.lbl_loss = SmoothLoss(nn.KLDivLoss(), smoothing=0.2,
                                   one_hot_target=False)

        self.loss_weights = self.hparams['loss_weights']

        self.rgb_mean = torch.tensor(0.0, dtype=torch.float32)
        self.rgb_std = torch.tensor(1.0, dtype=torch.float32)

    def _loss(self, o_masks, o_labels, o_imgs, masks, labels, imgs, n_imgs,
              provider):
        pr0_idx = provider == 0
        pr1_idx = provider == 1

        dec_loss = self.dec_loss(o_imgs, imgs)

        mask_loss = (self.mask_loss(o_masks[pr0_idx, :6], masks[pr0_idx]) +
                     self.mask_loss(o_masks[pr1_idx, -3:], masks[pr1_idx]-6))

        label_loss = (self.lbl_loss(o_labels[pr0_idx, :6],
                                    labels[pr0_idx, :6]) +
                      self.lbl_loss(o_labels[pr1_idx, -3:],
                                    labels[pr1_idx, -3:]))

        loss = (self.loss_weights['dec'] * dec_loss +
                self.loss_weights['mask'] * mask_loss +
                self.loss_weights['label'] * label_loss)

        return loss, dec_loss, mask_loss, label_loss

    def step(self, batch, batch_idx, is_train):
        imgs, masks, labels, provider, isup_grade, g_score = batch
        b = imgs.shape[0]
        n_imgs = imgs - self.rgb_mean / self.rgb_std

        o_masks, o_labels, o_imgs = self(n_imgs)

        loss, dec_loss, mask_loss, label_loss =\
            self._loss(o_masks, o_labels, o_imgs, masks, labels, imgs, n_imgs,
                       provider)

        lbl_acc = self._accuracy(o_labels, labels.argmax(dim=1))
        mask_acc = self._accuracy(o_masks.view(b, actual_lbl_nums, -1),
                                  masks.view(b, -1))

        lr = self.optimizer.param_groups[0]['lr']

        pr = '' if is_train else 'val_'

        log_dict = {
            pr+'loss': loss.item(),
            pr+'dec_loss': dec_loss.item(),
            pr+'mask_loss': mask_loss.item(),
            pr+'label_loss': label_loss.item(),
            pr+'lbl_acc': lbl_acc.item(),
            pr+'mask_acc': mask_acc.item(),
            pr+'lr': lr
        }

        if is_train and self.log_train_every_batch:
            return {'loss': loss, 'log': log_dict}
        else:
            return log_dict

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, False)

    def _apply(self, fn):
        super()._apply(fn)
        self.rgb_mean = fn(self.rgb_mean)
        self.rgb_std = fn(self.rgb_std)

        return self
