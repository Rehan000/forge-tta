"""CIFAR ResNet-20 (He et al. 2015) — a small BatchNorm-based CNN.

ResNet-20 is the standard tiny CIFAR baseline (~0.27M params). BatchNorm layers
are the hook point for both adaptation levers in this project:
  1. BN running-stat recalibration (Phase 0 baseline)
  2. quantization scale/clip adaptation (Phase 1 novel lever, added later)
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Option A (parameter-free) downsample keeps the model MCU-small.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks, 1)
        self.layer2 = self._make_layer(32, num_blocks, 2)
        self.layer3 = self._make_layer(64, num_blocks, 2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def resnet20(num_classes=10):
    """ResNet-20: 3 stages x 3 blocks. ~0.27M params."""
    return ResNetCIFAR(num_blocks=3, num_classes=num_classes)


# --- MobileNetV2 for CIFAR: a different architecture family (depthwise-separable
#     convs, inverted residuals) but still BatchNorm-based, so fold-then-recalib applies.
#     Every Conv2d is immediately followed by a BatchNorm2d inside an nn.Sequential, so
#     the generic Sequential folder (fold.fold_bn_recalib_seq) handles it.

def _divisible(v, divisor=8):
    nv = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return nv + divisor if nv < 0.9 * v else nv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand):
        super().__init__()
        self.use_res = stride == 1 and inp == oup
        hidden = int(round(inp * expand))
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(inp, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, bias=False), nn.BatchNorm2d(oup),   # project (no activation)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    def __init__(self, num_classes=10, width=0.5, stem_stride=1):
        super().__init__()
        # (expand, channels, num_blocks, stride) — CIFAR strides (first stages keep 32x32)
        cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2),
               (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
        in_c = _divisible(32 * width)
        # stem_stride=2 downsamples 64x64 inputs (Tiny-ImageNet) early -> much less compute.
        self.stem = nn.Sequential(nn.Conv2d(3, in_c, 3, stem_stride, 1, bias=False),
                                  nn.BatchNorm2d(in_c), nn.ReLU6(inplace=True))
        blocks = []
        for t, c, n, s in cfg:
            out_c = _divisible(c * width)
            for i in range(n):
                blocks.append(InvertedResidual(in_c, out_c, s if i == 0 else 1, t))
                in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        last = _divisible(1280 * width)
        self.head = nn.Sequential(nn.Conv2d(in_c, last, 1, bias=False),
                                  nn.BatchNorm2d(last), nn.ReLU6(inplace=True))
        self.linear = nn.Linear(last, num_classes)

    def forward(self, x):
        x = self.head(self.blocks(self.stem(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.linear(x)


def mobilenetv2(num_classes=10, width=0.5, stem_stride=1):
    """Compact MobileNetV2 for CIFAR (depthwise-separable, inverted residuals)."""
    return MobileNetV2CIFAR(num_classes=num_classes, width=width, stem_stride=stem_stride)


def build_model(arch, num_classes=10, width=0.5, stem_stride=1):
    if arch == "mobilenetv2":
        return mobilenetv2(num_classes=num_classes, width=width, stem_stride=stem_stride)
    return resnet20(num_classes=num_classes)
