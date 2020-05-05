import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SmoothLoss(_Loss):
    "A simple loss compute and train function."
    def __init__(self, criterion, smoothing=0.1, one_hot_target=True):
        super().__init__()
        self.criterion = criterion
        self.smoothing = smoothing
        self.confidence = 1 - smoothing
        self.one_hot_target = one_hot_target

    def forward(self, output, target):
        if self.one_hot_target:
            smooth_target = torch.full_like(output, self.smoothing /
                                            (output.size(1) - 1))
            smooth_target.scatter_(1, target[:, None], self.confidence)
        else:
            smooth_target = target * self.confidence
            smooth_target += self.smoothing * output.size(1)
        loss = self.criterion(output, smooth_target)
        return loss


class _Labels2PSmoothLoss:
    def __init__(self, confidence0=0.8, confidence1=0.8):
        self.confidence0 = confidence0
        self.confidence1 = confidence1

    def __call__(self, output, target, provider):
        pr0_idx = provider == 0
        pr1_idx = provider == 1

        s_labels = target.clone()

        s_labels[pr0_idx] *= self.confidence0
        s_labels[pr1_idx] *= self.confidence1

        s_labels[pr0_idx, :5] += (1-self.confidence0) / 5
        s_labels[pr1_idx, -3:] += (1-self.confidence1) / 3

        loss = F.kl_div(output[pr0_idx, :5], s_labels[pr0_idx, :5]) +\
            F.kl_div(output[pr1_idx, -3:], s_labels[pr1_idx, -3:])
        return loss
