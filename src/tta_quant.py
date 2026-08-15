"""Forward-only test-time adaptation levers for folded integer-only models.

These adapt a deployed (BN-folded) int8 model using only forward passes — the
regime where the standard BN-stat baseline is unavailable. Two levers:

  scale_adapt   : re-estimate the activation quantization scales online (EMA of
                  observed magnitudes). Near-zero overhead, integer-friendly.
                  Fixes quantizer range mismatch under shift, but NOT feature
                  mean/variance drift, so expected recovery is partial.

  channel_recalib (Phase 1 step 3b, added if scale_adapt under-delivers):
                  the folded-model analog of BN-stat adaptation — re-normalize
                  per-channel conv outputs to their clean-calibration statistics
                  using test-stream running stats. Recovers what BN-adapt did,
                  but on a model with no BN layers.

This file implements scale_adapt. channel_recalib lives alongside once we have
the scale_adapt numbers to compare against.
"""
import copy
import torch

from quant import ActFakeQuant
from fold import ChannelRecalib


def configure_scale_adapt(model, momentum=0.1, percentile=None):
    """Return a copy of a quantized model set to online activation-scale adaptation.

    All parameters frozen; every ActFakeQuant switches to `adapt` mode and updates
    its scale by EMA from the incoming stream. Forward-only, no gradients.
    """
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, ActFakeQuant):
            mod.adapt = True
            mod.adapt_momentum = momentum
            if percentile is not None:
                mod.percentile = percentile
    return m


@torch.no_grad()
def scale_adapt_eval(model, loader, device, momentum=0.1, percentile=None):
    """Online forward-only activation-scale adaptation, evaluated over the stream."""
    m = configure_scale_adapt(model, momentum, percentile).to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = m(x).argmax(1)          # forward pass also EMA-updates the quant scales
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def configure_channel_recalib(model, momentum=None, sites=None):
    """Return a copy of a folded+recalib quantized model with ChannelRecalib adapt ON.

    The folded-model analog of BN-stat adaptation: each former-BN site re-normalizes
    its channels to the clean (beta, |gamma|) target using test-stream running stats.
    Forward-only, no gradients. Requires the model was built with fold_bn_recalib.

    sites: if None, adapt ALL recalib sites. Otherwise a set/list of site indices
    (in forward order, 0..N-1) to adapt — the rest stay frozen (pass-through). Used
    for the selective-recalibration study (adapt only the layers that matter).
    """
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    idx = 0
    for mod in m.modules():
        if isinstance(mod, ChannelRecalib):
            on = (sites is None) or (idx in sites)
            mod.adapt = on
            if on and momentum is not None:
                mod.momentum = momentum
            idx += 1
    if idx == 0:
        raise ValueError("no ChannelRecalib modules — build the model with fold_bn_recalib")
    return m


def num_recalib_sites(model):
    return sum(isinstance(mod, ChannelRecalib) for mod in model.modules())


def set_active_recalib(model, sites=None, momentum=None):
    """In-place: enable adapt on the given site indices (None=all), reset their EMA
    state. No deepcopy — for sweeps over many site subsets on one fixed model."""
    idx = 0
    for mod in model.modules():
        if isinstance(mod, ChannelRecalib):
            mod.adapt = (sites is None) or (idx in sites)
            if momentum is not None:
                mod.momentum = momentum
            mod.reset()
            idx += 1


@torch.no_grad()
def channel_recalib_eval_fast(model, loader, device, sites=None, momentum=None):
    """Like channel_recalib_eval but mutates `model` in place (no deepcopy). The model
    must already be on `device`, in eval mode, with params frozen."""
    set_active_recalib(model, sites, momentum)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


@torch.no_grad()
def channel_recalib_eval(model, loader, device, momentum=None, sites=None):
    """Online forward-only per-channel recalibration, evaluated over the stream.

    sites: subset of recalib site indices to adapt (None = all)."""
    m = configure_channel_recalib(model, momentum, sites).to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = m(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total
