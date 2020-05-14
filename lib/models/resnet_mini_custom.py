'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, momentum=0.1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0,
                                                   planes//4, planes//4),
                                                  "constant", 0)
                                            if in_planes != planes
                                            else x[:, :, ::2, ::2])
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes, momentum=momentum)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, p_m=1, num_classes=100, num_targets=1,
                 momentum=0.1, c_do=0, d_do=0, type='simple_conv', first_k_sz=3,
                 first_stride=1, first_l_stride=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = int(16*p_m)
        self.num_classes, self.num_targets = num_classes, num_targets
        self.type = type
        self.conv1 = nn.Conv2d(1, int(16*p_m), kernel_size=first_k_sz,
                               stride=first_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16*p_m), momentum=momentum)
        self.c_do1 = nn.Dropout2d(p=c_do) if c_do > 0 else lambda x: x

        if self.type == 'two_conv':
            self.conv2 = nn.Conv2d(int(16*p_m), 16 * p_m, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(int(16*p_m), momentum=momentum)
            self.c_do2 = nn.Dropout2d(p=c_do) if c_do > 0 else lambda x: x

        self.layer1 = self._make_layer(block, int(16*p_m), num_blocks[0],
                                       stride=first_l_stride,
                                       momentum=momentum)
        self.layer2 = self._make_layer(block, int(32*p_m), num_blocks[1],
                                       stride=2, momentum=momentum)
        self.layer3 = self._make_layer(block, int(64*p_m), num_blocks[2],
                                       stride=2, momentum=momentum)
        self.linear = nn.Linear(int(64*p_m), num_classes*num_targets)
        self.d_do = nn.Dropout(p=d_do) if d_do > 0 else lambda x: x

        # self.apply(_weights_init)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if False:  # isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride, momentum):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, momentum))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        b = x.size(0)
        out = F.max_pool2d(F.relu(self.bn1(self.c_do1(self.conv1(x)))), 2)
        if self.type == 'two_conv':
            out = F.relu(self.bn2(self.c_do2(self.conv2(out))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(b, -1)
        out = self.d_do(self.linear(out))
        if self.num_targets > 1:
            out = out.view(b, self.num_classes, self.num_targets)
        else:
            out = out.view(b, self.num_classes)
        # out = F.log_softmax(out, dim=1)
        return out


def resnet_custom(num_blocks=[3, 3, 3], p_m=1, num_classes=100, num_targets=1,
                  momentum=0.1, c_do=0, d_do=0, type='simple_conv',
                  first_k_sz=3, first_stride=1, first_l_stride=1,
                  zero_init_residual=False):
    return ResNet(BasicBlock, num_blocks, p_m=p_m, num_classes=num_classes,
                  num_targets=num_targets, momentum=momentum, c_do=c_do,
                  d_do=d_do, type=type, first_k_sz=first_k_sz,
                  first_stride=first_stride, first_l_stride=first_l_stride,
                  zero_init_residual=zero_init_residual)


def resnet20(num_classes=100, num_targets=1, p_m=1, momentum=0.1, c_do=0,
             d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do)


def resnet20_c5(num_classes=100, num_targets=1, p_m=1, momentum=0.1, c_do=0,
                d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do, first_k_sz=5)


def resnet20_c7(num_classes=100, num_targets=1, p_m=1, momentum=0.1, c_do=0,
                d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do, first_k_sz=7)


def resnet20_c5_fls2(num_classes=100, num_targets=1, p_m=1, momentum=0.1,
                     c_do=0, d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do, first_k_sz=5,
                         first_l_stride=2)


def resnet20_c5_fls2_small(num_classes=100, num_targets=1, p_m=1, momentum=0.1,
                           c_do=0, d_do=0):
    return resnet_custom(num_blocks=[1, 1, 1], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do, first_k_sz=5,
                         first_l_stride=2)


def resnet20_c7_fls2(num_classes=100, num_targets=1, p_m=1, momentum=0.1,
                     c_do=0, d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do, first_k_sz=7,
                         first_l_stride=2)


def resnet20_c5_fs2(num_classes=100, num_targets=1, p_m=1, momentum=0.1,
                    c_do=0, d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do, first_k_sz=5,
                         first_stride=2)


def resnet20_c7_fs2(num_classes=100, num_targets=1, p_m=1, momentum=0.1,
                    c_do=0, d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do, first_k_sz=7,
                         first_stride=2)


def resnet20_c3x3_fls2(num_classes=100, num_targets=1, p_m=1, momentum=0.1,
                       c_do=0, d_do=0):
    return resnet_custom(num_blocks=[3, 3, 3], p_m=p_m,
                         num_classes=num_classes, num_targets=num_targets,
                         momentum=momentum, c_do=c_do, d_do=d_do,
                         type='two_conv', first_k_sz=3, first_l_stride=2)
# def resnet32():
#    return ResNet(BasicBlock, [5, 5, 5])
