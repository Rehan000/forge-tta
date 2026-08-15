"""Fold BatchNorm into preceding convolutions — the true integer-only deployment form.

Real full-integer deployment (TFLite-Micro, ESP-DL, CMSIS-NN) merges each
Conv->BN pair into a single Conv with bias:
    W' = W * gamma / sqrt(var + eps)
    b' = beta + (b - mean) * gamma / sqrt(var + eps)
and deletes the BN layer. This is mathematically exact at eval time, so clean
accuracy is unchanged — but it removes the running statistics that the forward-only
BN-stat baseline recalibrates. After folding there is literally nothing for
BN-adapt to update: that absence is the gap the quantization-scale lever fills.

ResNet-20 structure has every BN directly after a Conv, so folding is unambiguous:
  stem:  conv1 + bn1
  block: conv1 + bn1, conv2 + bn2, and (when present) shortcut[0] + shortcut[1]
We replace each folded BN with nn.Identity so the existing forward() is unchanged.
"""
import copy
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from models import BasicBlock


class ChannelRecalib(nn.Module):
    """Forward-only per-channel recalibration — the folded-model analog of BN-adapt.

    Sits where a BatchNorm used to be (after a now-folded conv). The original BN
    guarantees the clean per-channel output distribution has mean beta and std |gamma|,
    so we keep those as the target. In `adapt` mode we EMA-track the live per-channel
    mean/var and renormalize the activations back to (target_mean, target_std). With
    adapt off it is a pass-through identity (clean accuracy preserved exactly).
    No gradients, no learnable parameters.
    """

    def __init__(self, target_mean, target_std, momentum=0.1, eps=1e-5):
        super().__init__()
        tstd = target_std.detach().abs().clamp_min(eps)
        self.register_buffer("target_mean", target_mean.detach().clone())
        self.register_buffer("target_std", tstd.clone())
        self.register_buffer("run_mean", target_mean.detach().clone())
        self.register_buffer("run_var", (tstd ** 2).clone())
        self.momentum = momentum
        self.eps = eps
        self.adapt = False

    def reset(self):
        """Reset the online EMA state to the clean target (start of a fresh stream)."""
        self.run_mean = self.target_mean.clone()
        self.run_var = self.target_std.clone() ** 2

    def forward(self, x):
        if not self.adapt:
            return x
        m = x.mean((0, 2, 3))
        v = x.var((0, 2, 3), unbiased=False)
        self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * m
        self.run_var = (1 - self.momentum) * self.run_var + self.momentum * v
        xn = (x - self.run_mean[None, :, None, None]) / torch.sqrt(self.run_var[None, :, None, None] + self.eps)
        return xn * self.target_std[None, :, None, None] + self.target_mean[None, :, None, None]


def _recalib_from_bn(bn, momentum):
    return ChannelRecalib(bn.bias.data, bn.weight.data, momentum)


def fold_bn(model):
    """Return an eval-mode copy of `model` with all Conv->BN pairs fused."""
    m = copy.deepcopy(model).eval()

    # stem
    m.conv1 = fuse_conv_bn_eval(m.conv1, m.bn1)
    m.bn1 = nn.Identity()

    # residual stages
    for stage in (m.layer1, m.layer2, m.layer3):
        for block in stage:
            assert isinstance(block, BasicBlock)
            block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
            if len(block.shortcut) == 2:                       # conv + bn downsample
                fused = fuse_conv_bn_eval(block.shortcut[0], block.shortcut[1])
                block.shortcut = nn.Sequential(fused)
    return m


def fold_bn_recalib(model, momentum=0.1):
    """Like fold_bn, but replaces each folded BN with a ChannelRecalib (adapt off).

    The folded convs are numerically identical to fold_bn (clean accuracy preserved),
    but each former-BN site now holds a per-channel recalibration module that can be
    switched on at test time. Build this, quantize it, calibrate quant scales, then
    enable adapt for forward-only recovery on the deployed model.
    """
    m = copy.deepcopy(model).eval()

    rc = _recalib_from_bn(m.bn1, momentum)
    m.conv1 = fuse_conv_bn_eval(m.conv1, m.bn1)
    m.bn1 = rc

    for stage in (m.layer1, m.layer2, m.layer3):
        for block in stage:
            assert isinstance(block, BasicBlock)
            rc1 = _recalib_from_bn(block.bn1, momentum)
            block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
            block.bn1 = rc1
            rc2 = _recalib_from_bn(block.bn2, momentum)
            block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
            block.bn2 = rc2
            if len(block.shortcut) == 2:
                rcs = _recalib_from_bn(block.shortcut[1], momentum)
                fused = fuse_conv_bn_eval(block.shortcut[0], block.shortcut[1])
                block.shortcut = nn.Sequential(fused, rcs)
    return m


def _process_children(module, momentum):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            module._modules[name] = _fold_sequential(child, momentum)
        else:
            _process_children(child, momentum)


def _fold_sequential(seq, momentum):
    """Fuse adjacent (Conv2d, BatchNorm2d) inside a Sequential; BN -> ChannelRecalib."""
    items = list(seq)
    out, i = [], 0
    while i < len(items):
        a = items[i]
        b = items[i + 1] if i + 1 < len(items) else None
        if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
            out.append(fuse_conv_bn_eval(a, b))
            out.append(_recalib_from_bn(b, momentum))
            i += 2
        else:
            _process_children(a, momentum)            # recurse (e.g. InvertedResidual)
            out.append(a)
            i += 1
    return nn.Sequential(*out)


def fold_bn_recalib_seq(model, momentum=0.1):
    """Generic fold for any architecture whose conv->BN pairs sit adjacent in nn.Sequential
    (e.g. MobileNetV2). Fuses each pair and replaces the BN with a ChannelRecalib."""
    m = copy.deepcopy(model).eval()
    _process_children(m, momentum)
    return m


def fold_for_arch(model, arch, momentum=0.1):
    """Dispatch to the right folder: ResNet-20 (attribute-based) vs Sequential-based."""
    return fold_bn_recalib(model, momentum) if arch == "resnet20" else fold_bn_recalib_seq(model, momentum)


def count_bn(model):
    return sum(isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)) for mod in model.modules())
