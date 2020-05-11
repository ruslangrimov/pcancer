import torch
from torch import nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def get_dense_512_64(classes, use_dummy_feature=False, f_d_rate=0.5,
                     d_rate=0.4, max_len=300):
    class MainModel(nn.Module):
        def __init__(self, use_dummy_feature, f_d_rate, d_rate, max_len):
            super().__init__()

            self.use_dummy_feature = use_dummy_feature
            self.max_len = max_len

            if self.use_dummy_feature:
                self.dummy_feature = nn.Parameter(
                    torch.randn((1, 512, 1), dtype=torch.float32))

            self.backbone = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Dropout(f_d_rate),

                nn.Conv1d(512, 64, 1),
                nn.ReLU(inplace=True),
                nn.Dropout(d_rate),
                nn.BatchNorm1d(64),

                nn.Conv1d(64, 64, 1),
                nn.ReLU(inplace=True),
                nn.Dropout(d_rate),
                nn.BatchNorm1d(64),

                nn.AdaptiveMaxPool1d(1),
                LambdaLayer(lambda x: x.view(-1, 64)),

                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(d_rate),
                nn.BatchNorm1d(64),
            )

            self.reg_linear = nn.Linear(64, 1)
            self.class_linear = nn.Linear(64, classes)

        def forward(self, x):
            # b, 300, 512
            b = x.shape[0]

            if self.use_dummy_feature:
                empty_mask = (x == 0).all(dim=1)[:, None, :]
                x = x + empty_mask * self.dummy_feature.expand(b, 512,
                                                               self.max_len)

            x = self.backbone(x)
            return self.reg_linear(x), self.class_linear(x)

    return MainModel(use_dummy_feature, f_d_rate, d_rate, max_len)
