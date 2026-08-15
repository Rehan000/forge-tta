"""Backprop test-time adaptation baselines for head-to-head comparison.

TENT (Wang et al., ICLR 2021): adapt by minimizing prediction entropy, updating ONLY
the BatchNorm affine parameters (gamma, beta) by gradient descent while BN uses the
current batch's statistics. It needs backpropagation, so it cannot run on a folded
integer-only MCU model — that contrast is the point of including it.

CoTTA's full machinery (teacher-student + augmentation averaging + stochastic restore)
is heavier; TENT is the canonical, widely-cited backprop baseline, so it is the primary
point of comparison. We report it on the FP model (gradients required).
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_entropy(logits):
    """Entropy of the softmax distribution, per sample."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def configure_tent(model):
    """Set up a copy of `model` for TENT: train only BN affine params, BN on batch stats."""
    m = copy.deepcopy(model)
    m.train()                                   # BN uses batch statistics
    m.requires_grad_(False)
    params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True)            # gamma, beta are learnable
            mod.track_running_stats = False     # use the batch, don't accumulate
            mod.running_mean = None
            mod.running_var = None
            params += [mod.weight, mod.bias]
    return m, params


def _cotta_augment(x):
    """Cheap stochastic augmentation for pseudo-label averaging (flip + noise + jitter).
    Operates on already-normalized tensors -- approximate but captures the ensemble."""
    out = x
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=[3])
    out = out + torch.randn_like(out) * 0.05
    scale = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.2
    return out * scale


def cotta_eval(model, loader, device, lr=1e-4, alpha=0.999, rst=0.01, n_aug=8, p_th=0.5):
    # lr=1e-3 (TENT's rate) diverges for CoTTA on a long non-iid continual stream
    # (confirmation-bias collapse); 1e-4 is stable. Use the native continual protocol.
    """CoTTA (Wang et al., 2022): teacher--student with weight EMA, augmentation-averaged
    pseudo-labels (when the teacher is unconfident), and stochastic weight restore.
    Backprop-based -> not runnable on a folded integer-only model. Final prediction = teacher.
    """
    student = copy.deepcopy(model).to(device).train()
    student.requires_grad_(True)
    teacher = copy.deepcopy(model).to(device).eval()
    teacher.requires_grad_(False)
    source = {n: p.detach().clone() for n, p in copy.deepcopy(model).named_parameters()}
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():                                   # teacher pseudo-labels
            prob = teacher(x).softmax(1)
            need = prob.max(1).values < p_th                    # augment-average the unconfident
            if need.any():
                avg = torch.zeros_like(prob)
                for _ in range(n_aug):
                    avg += teacher(_cotta_augment(x)).softmax(1)
                avg /= n_aug
                prob = torch.where(need.unsqueeze(1), avg, prob)
            pseudo = prob
        out = student(x)                                        # student update (soft CE)
        loss = -(pseudo * out.log_softmax(1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), student.parameters()):
                tp.mul_(alpha).add_(sp.detach(), alpha=1 - alpha)   # EMA teacher
            for tb, sb in zip(teacher.buffers(), student.buffers()):
                tb.copy_(sb)
            for n, p in student.named_parameters():             # stochastic restore
                m = torch.rand_like(p) < rst
                p[m] = source[n].to(p.device)[m]
        correct += (pseudo.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def tent_eval(model, loader, device, lr=1e-3, steps=1):
    """Online (continual) TENT over the stream. Returns top-1 accuracy.

    Each batch: forward, minimize mean softmax-entropy w.r.t. BN affine params (`steps`
    gradient steps), then score. Updates persist across the stream (continual).
    """
    m, params = configure_tent(model)
    m.to(device)
    opt = torch.optim.Adam(params, lr=lr)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for _ in range(steps):
            opt.zero_grad()
            out = m(x)
            loss = softmax_entropy(out).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = m(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total
