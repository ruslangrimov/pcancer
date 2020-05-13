from torch import nn
from torchvision import models

from .features_map import FeaturesMap


class Resnet_512x1x1(nn.Module):
    def __init__(self, backbone, backbone_features, classes, features_do,
                 max_height=70, max_width=40):
        super().__init__()

        self.f_map = FeaturesMap(True, 512, max_height, max_width)

        self.backbone = getattr(models, backbone)(pretrained=False)
        # self.backbone.conv1 = nn.Conv2d(512, 64, 1)
        self.backbone.conv1 = nn.Sequential(
            nn.Dropout2d(features_do) if features_do > 0 else nn.Identity(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
        )

        self.backbone.fc = nn.Linear(backbone_features, 512)
        self.backbone.maxpool = nn.Identity()

        self.reg_linear = nn.Linear(512, 1)
        self.class_linear = nn.Linear(512, classes)

    def forward(self, features, ys, xs):
        f_map = self.f_map(features, ys, xs)
        x = self.backbone(f_map)
        return self.reg_linear(x), self.class_linear(x)
