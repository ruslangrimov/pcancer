import os
import psutil

import torch
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import cohen_kappa_score

from pytorch_lightning.core import LightningModule

from .schedulers import DelayedScheduler
from .losses import SmoothLoss
from .datasets import actual_lbl_nums
from .utils import get_module_attr


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
        omtimizer_class = get_module_attr(self.hparams['optimizer']['name'])
        self.optimizer = omtimizer_class(self.parameters(),
                                         lr=self.hparams['learning_rate'],
                                         **self.hparams['optimizer']['params'])

        # optimizers = [{k: v for k, v in self.hparams['optimizer'].items()
        #                if k not in {'name', 'params'}}]
        #
        # optimizers[0]['optimizer'] = self.optimizer

        optimizers = [self.optimizer, ]

        scheduler_class = get_module_attr(self.hparams['scheduler']['name'])

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
        if len(outputs) > 0:
            keys = outputs[0].keys()
            o_dict = {}
            for k in keys:
                o_dict[k] = sum([o[k] for o in outputs]) / len(outputs)
        else:
            o_dict = {}

        return {'log': o_dict}


class PatchesModuleV1(GeneralModule):
    def __init__(self, model, hparams, log_train_every_batch=False):
        super().__init__(model, hparams, log_train_every_batch)
        self.hparams = hparams
        self.model = model
        self.log_train_every_batch = log_train_every_batch

        self.dec_loss = nn.MSELoss()
        mask_smoothing = self.hparams['loss']['mask_smoothing']
        self.mask_loss = SmoothLoss(nn.KLDivLoss(), smoothing=mask_smoothing,
                                    one_hot_target=True)
        label_smoothing = self.hparams['loss']['label_smoothing']
        self.lbl_loss = SmoothLoss(nn.KLDivLoss(), smoothing=label_smoothing,
                                   one_hot_target=False)

        self.loss_weights = self.hparams['loss']['weights']

        self.rgb_mean = torch.tensor(hparams['dataset']['rgb_mean'],
                                     dtype=torch.float32)
        self.rgb_std = torch.tensor(hparams['dataset']['rgb_std'],
                                    dtype=torch.float32)

    def _loss(self, o_masks, o_labels, o_imgs, masks, labels, imgs, n_imgs,
              provider):
        def ls(x):
            return torch.log_softmax(x, dim=1)

        pr0_idx = provider == 0
        pr1_idx = provider == 1

        if self.loss_weights['dec'] > 0:
            dec_loss = self.dec_loss(o_imgs, imgs)
        else:
            dec_loss = 0

        mask_loss = 0
        if self.loss_weights['mask'] > 0:
            if pr0_idx.any():
                mask_loss = mask_loss +\
                    self.mask_loss(ls(o_masks[pr0_idx, :6]),
                                   masks[pr0_idx])
            if pr1_idx.any():
                mask_loss = mask_loss +\
                    self.mask_loss(ls(o_masks[pr1_idx, -3:]),
                                   masks[pr1_idx]-6)

        label_loss = 0
        if self.loss_weights['label'] > 0:
            if pr0_idx.any():
                label_loss = label_loss +\
                    self.lbl_loss(ls(o_labels[pr0_idx, :6]),
                                  labels[pr0_idx, :6])
            if pr1_idx.any():
                label_loss = label_loss +\
                    self.lbl_loss(ls(o_labels[pr1_idx, -3:]),
                                  labels[pr1_idx, -3:])

        loss = (self.loss_weights['dec'] * dec_loss +
                self.loss_weights['mask'] * mask_loss +
                self.loss_weights['label'] * label_loss)

        return loss, dec_loss, mask_loss, label_loss

    def step(self, batch, batch_idx, is_train):
        imgs, masks, labels, provider, isup_grade, g_score = batch
        b = imgs.shape[0]
        n_imgs = (imgs - self.rgb_mean) / self.rgb_std

        if self.loss_weights['dec'] > 0 and self.loss_weights['label'] > 0:
            o_masks, o_labels, o_imgs = self(n_imgs)
        else:
            o_labels, o_imgs = None, None
            o_masks = self(n_imgs)[0]

        loss, dec_loss, mask_loss, label_loss =\
            self._loss(o_masks, o_labels, o_imgs, masks, labels, imgs, n_imgs,
                       provider)

        if self.loss_weights['label'] > 0:
            lbl_acc = self._accuracy(o_labels, labels.argmax(dim=1))
        else:
            lbl_acc = 0

        if self.loss_weights['mask'] > 0:
            mask_acc = self._accuracy(o_masks.view(b, actual_lbl_nums, -1),
                                      masks.view(b, -1))
        else:
            mask_acc = 0

        lr = self.optimizer.param_groups[0]['lr']

        pr = '' if is_train else 'val_'

        log_dict = {
            pr+'loss': float(loss),
            pr+'dec_loss': float(dec_loss),
            pr+'mask_loss': float(mask_loss),
            pr+'label_loss': float(label_loss),
            pr+'lbl_acc': float(lbl_acc),
            pr+'mask_acc': float(mask_acc),
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


class WSIModuleV1(GeneralModule):
    def __init__(self, model, hparams, log_train_every_batch=False):
        super().__init__(model, hparams, log_train_every_batch)
        self.hparams = hparams
        self.model = model
        self.log_train_every_batch = log_train_every_batch

        self.reg_loss = nn.MSELoss()

        label_smoothing = self.hparams['loss']['label_smoothing']
        self.class_loss = SmoothLoss(nn.KLDivLoss(), smoothing=label_smoothing,
                                     one_hot_target=True)

        self.loss_weights = self.hparams['loss']['weights']

        self.rgb_mean = torch.tensor(hparams['dataset']['rgb_mean'],
                                     dtype=torch.float32)
        self.rgb_std = torch.tensor(hparams['dataset']['rgb_std'],
                                    dtype=torch.float32)

        self.process = psutil.Process(os.getpid())

        self.max_lbl_nums = hparams['dataset']['classes']

    @classmethod
    def _accuracy(cls, output, target):
        pred = output
        eq = pred.eq(target.view_as(pred))
        return eq.float().mean()

    def step(self, batch, batch_idx, is_train):
        # features, ys, xs, provider, isup_grade, gleason_score = batch
        # features = features.mean(-1).mean(-1).transpose(1, -1)
        features, ys, xs, provider, isup_grade, gleason_score = batch

        # b = features.shape[0]

        labels = isup_grade

        labels_reg = labels[:, None].float()
        labels_class = labels

        o_labels_reg, o_labels_class = self.model(features, ys, xs)

        o_labels_reg = torch.sigmoid(o_labels_reg) * self.max_lbl_nums - 0.5
        o_labels_class = F.log_softmax(o_labels_class, dim=-1)

        reg_loss = self.reg_loss(o_labels_reg, labels_reg)
        class_loss = self.class_loss(o_labels_class, labels_class)

        loss = (self.loss_weights['reg'] * reg_loss +
                self.loss_weights['class'] * class_loss)

        o_labels_reg = o_labels_reg.round().long().\
            clamp(0, self.max_lbl_nums-1)
        o_labels_class = o_labels_class.argmax(dim=-1)

        acc_reg = self._accuracy(o_labels_reg, labels)
        acc_class = self._accuracy(o_labels_class, labels)

        qwk_reg = cohen_kappa_score(o_labels_reg.cpu().numpy(),
                                    labels.cpu().numpy(),
                                    weights="quadratic")
        qwk_class = cohen_kappa_score(o_labels_class.cpu().numpy(),
                                      labels.cpu().numpy(),
                                      weights="quadratic")

        lr = self.optimizer.param_groups[0]['lr']

        pr = '' if is_train else 'val_'

        log_dict = {
            pr+'loss': loss,
            pr+'reg_loss': float(reg_loss),
            pr+'class_loss': float(class_loss),
            pr+'acc_reg': float(acc_reg),
            pr+'acc_class': float(acc_class),
            pr+'qwk_reg': float(qwk_reg),
            pr+'qwk_class': float(qwk_class),
            pr+'lr': lr,
            pr+'memory': self.process.memory_info().rss
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
