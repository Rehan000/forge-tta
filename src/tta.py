"""Test-time adaptation methods — Phase 0 baselines.

Phase 0 establishes the two reference points every later result is measured against:
  - source        : no adaptation (model.eval()) — the lower bound
  - bn_adapt       : forward-only BatchNorm-statistic recalibration — the floor that
                     the Phase 1 novel quantization-scale lever must beat

bn_adapt is the classic forward-only baseline (Nado et al. 2020 / TENT's BN baseline):
freeze all weights, put BN layers in train mode so they normalize with statistics
estimated from the incoming (shifted) stream instead of the stale source stats.
No gradients, no optimizer — genuinely backprop-free.
"""
import copy
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 accuracy with no adaptation (source baseline). model is set to eval()."""
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def configure_bn_adapt(model, momentum=0.1):
    """Return a copy of `model` set up for forward-only BN-stat adaptation.

    All parameters are frozen; BN layers are switched to train mode so each forward
    pass recomputes normalization statistics from the current batch and updates the
    running stats (momentum EMA). Nothing else changes.
    """
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train()
            mod.momentum = momentum
        else:
            mod.eval()
    return m


@torch.no_grad()
def bn_adapt_eval(model, loader, device, momentum=0.1):
    """Forward-only BN-stat adaptation, evaluated online over the stream.

    Returns top-1 accuracy. Uses the test batch statistics (BN in train mode) — a
    standard forward-only baseline. Batch-size-1 streaming is a Phase 1 concern;
    here we use the dataloader's batch size (set it modestly, e.g. 64).
    """
    m = configure_bn_adapt(model, momentum).to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = m(x).argmax(1)            # forward pass also updates BN running stats
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total
