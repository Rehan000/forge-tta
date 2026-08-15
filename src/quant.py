"""Simulated int8 quantization (fake-quant) that PRESERVES BatchNorm modules.

Design note — central to this project:
Standard deployment quantization folds BatchNorm into the preceding conv and
discards the separate BN layers. That would remove exactly the statistics the
Phase 0 forward-only baseline recalibrates. So here we use *fake quantization*
that keeps the module graph (BN included) intact:
  - weights: per-output-channel symmetric int8, scale = max|W_c| / 127  (static)
  - activations (each Conv/Linear input): per-tensor symmetric int8, scale
    calibrated from a few clean batches (max-abs, with optional percentile)

This mirrors how FOA / ZOA / PACE simulate quantization on GPU, so it gives an
apples-to-apples "does adaptation survive int8?" measurement on the host. It is
NOT yet true integer-only execution with folded BN — that is the Phase 2
on-device step, where BN is gone and the novel quantization-scale adaptation
lever becomes the only thing left to adapt.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetric_fake_quant(x, scale, qmax):
    """Round-to-int, clamp to [-qmax, qmax], scale back. Straight float sim."""
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale


def quantize_weight_per_channel(w, num_bits=8):
    """Per-output-channel symmetric fake-quant of a conv/linear weight."""
    qmax = 2 ** (num_bits - 1) - 1
    out_ch = w.shape[0]
    flat = w.reshape(out_ch, -1)
    scale = flat.abs().amax(dim=1).clamp_min(1e-8) / qmax     # [out_ch]
    scale = scale.reshape(out_ch, *([1] * (w.dim() - 1)))     # broadcast shape
    return _symmetric_fake_quant(w, scale, qmax)


class ActFakeQuant(nn.Module):
    """Per-tensor symmetric activation fake-quant with calibratable scale.

    Modes:
      calibrate=True  -> observe running max-abs, pass input through unchanged
      calibrate=False -> fake-quantize input using the frozen scale
    """

    def __init__(self, num_bits=8, percentile=1.0):
        super().__init__()
        self.num_bits = num_bits
        self.qmax = 2 ** (num_bits - 1) - 1
        self.percentile = percentile          # 1.0 = plain max-abs
        self.calibrate = False
        self.adapt = False                    # test-time online scale adaptation
        self.adapt_momentum = 0.1
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("max_abs", torch.tensor(0.0))

    def _observe(self, x):
        if self.percentile >= 1.0:
            return x.detach().abs().amax()
        return torch.quantile(x.detach().abs().flatten(), self.percentile)

    def forward(self, x):
        if self.calibrate:                    # offline calibration: observe, pass through
            self.max_abs = torch.maximum(self.max_abs, self._observe(x))
            return x
        if self.adapt:                        # test-time: EMA-update scale from the live stream
            target = (self._observe(x) / self.qmax).clamp_min(1e-8)
            self.scale = (1 - self.adapt_momentum) * self.scale + self.adapt_momentum * target
        return _symmetric_fake_quant(x, self.scale, self.qmax)

    def freeze(self):
        self.scale = (self.max_abs / self.qmax).clamp_min(1e-8)


class QuantConv2d(nn.Module):
    """Conv2d with fake-quantized input activations and per-channel int8 weights."""

    def __init__(self, conv, w_bits=8, a_bits=8, percentile=1.0):
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=False)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        self.stride, self.padding, self.dilation, self.groups = (
            conv.stride, conv.padding, conv.dilation, conv.groups)
        self.w_bits = w_bits
        self.act_q = ActFakeQuant(a_bits, percentile)

    def forward(self, x):
        x = self.act_q(x)
        w = quantize_weight_per_channel(self.weight, self.w_bits)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Module):
    def __init__(self, lin, w_bits=8, a_bits=8, percentile=1.0):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.detach().clone(), requires_grad=False)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.w_bits = w_bits
        self.act_q = ActFakeQuant(a_bits, percentile)

    def forward(self, x):
        x = self.act_q(x)
        w = quantize_weight_per_channel(self.weight, self.w_bits)
        return F.linear(x, w, self.bias)


def quantize_model(model, w_bits=8, a_bits=8, percentile=1.0):
    """Return a fake-quant copy of `model` with Conv2d/Linear replaced. BN preserved."""
    m = copy.deepcopy(model)
    _replace(m, w_bits, a_bits, percentile)
    return m


def _replace(module, w_bits, a_bits, percentile):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d(child, w_bits, a_bits, percentile))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child, w_bits, a_bits, percentile))
        else:
            _replace(child, w_bits, a_bits, percentile)


def _set_calibrate(model, flag):
    for mod in model.modules():
        if isinstance(mod, ActFakeQuant):
            mod.calibrate = flag


@torch.no_grad()
def calibrate_model(model, loader, device, num_batches=8):
    """Calibrate activation scales on clean in-distribution data, then freeze."""
    model.eval().to(device)
    _set_calibrate(model, True)
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(x.to(device))
    for mod in model.modules():
        if isinstance(mod, ActFakeQuant):
            mod.freeze()
    _set_calibrate(model, False)
    return model
