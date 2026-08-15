"""Export the folded + recalib int8 ResNet-20 to deployable artifacts.

Builds the same model the host experiments use (fold_bn_recalib -> quantize ->
calibrate on clean data), then dumps every layer's integer parameters:
  - per-conv int8 weights (per-output-channel symmetric) + weight scales
  - per-conv input-activation scale (calibrated) + folded bias
  - per-recalib-site target_mean (beta) and target_std (|gamma|)
  - the final linear layer

Outputs:
  deploy/artifacts/resnet20_int8.npz   numeric arrays for the host integer engine
  deploy/artifacts/model_data.h        C arrays + scale tables for the firmware

The .npz is the single source of truth: int8_reference.py runs it as a true
integer engine on the host, and model_data.h is the same data for the ESP32-S3.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import resnet20                      # noqa: E402
from fold import fold_bn_recalib, ChannelRecalib  # noqa: E402
from quant import quantize_model, calibrate_model, QuantConv2d, QuantLinear  # noqa: E402
from data import clean_loaders                   # noqa: E402
from train import pick_device                    # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def _quantize_weight_int8(w):
    """Per-output-channel symmetric int8 weight + per-channel scale."""
    out_ch = w.shape[0]
    flat = w.reshape(out_ch, -1)
    scale = (flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0)
    q = torch.clamp(torch.round(w / scale.reshape(out_ch, *([1] * (w.dim() - 1)))), -127, 127)
    return q.to(torch.int8).cpu().numpy(), scale.cpu().numpy().astype(np.float32)


def export(ckpt="checkpoints/resnet20_cifar10.pt", momentum=0.01, calib_batches=8):
    device = pick_device()
    fp32 = resnet20()
    fp32.load_state_dict(torch.load(ckpt, map_location="cpu"))
    qmodel = quantize_model(fold_bn_recalib(fp32, momentum), 8, 8)
    _, clean_test = clean_loaders("data", 128)

    # Per-conv OUTPUT range over clean calibration data -> out_scale = maxabs/127.
    # esp_nn_conv_s8 requantizes the conv output to int8 at this scale; the firmware
    # needs it (and it is what the host esp_nn-scheme validation used).
    out_absmax = {}
    hooks = []
    for name, mod in qmodel.named_modules():
        if isinstance(mod, QuantConv2d):
            def mk(key):
                def hook(m, i, o):
                    out_absmax[key] = max(out_absmax.get(key, 0.0), o.detach().abs().amax().item())
                return hook
            hooks.append(mod.register_forward_hook(mk(name or "root")))
    calibrate_model(qmodel, clean_test, device, calib_batches)
    for h in hooks:
        h.remove()
    out_scales = {k: (v / 127.0 if v > 0 else 1.0) for k, v in out_absmax.items()}
    qmodel.to("cpu").eval()

    arrays, meta = {}, {"momentum": momentum, "convs": [], "recalibs": []}
    for name, mod in qmodel.named_modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            wq, wscale = _quantize_weight_int8(mod.weight.data)
            key = name or "root"
            arrays[f"{key}.wq"] = wq
            arrays[f"{key}.wscale"] = wscale
            arrays[f"{key}.in_scale"] = np.float32(mod.act_q.scale.item())
            arrays[f"{key}.bias"] = (mod.bias.data.cpu().numpy().astype(np.float32)
                                     if mod.bias is not None else np.zeros(wq.shape[0], np.float32))
            if isinstance(mod, QuantConv2d):                       # esp_nn needs OHWI + out_scale
                arrays[f"{key}.wq_ohwi"] = np.ascontiguousarray(np.transpose(wq, (0, 2, 3, 1)))
                arrays[f"{key}.out_scale"] = np.float32(out_scales[key])
            meta["convs"].append({"name": key, "kind": type(mod).__name__,
                                  "shape": list(wq.shape),
                                  "stride": list(getattr(mod, "stride", (1, 1))),
                                  "padding": list(getattr(mod, "padding", (0, 0)))})
        elif isinstance(mod, ChannelRecalib):
            key = name or "root"
            arrays[f"{key}.target_mean"] = mod.target_mean.cpu().numpy().astype(np.float32)
            arrays[f"{key}.target_std"] = mod.target_std.cpu().numpy().astype(np.float32)
            meta["recalibs"].append({"name": key, "channels": int(mod.target_mean.numel())})

    os.makedirs(ART, exist_ok=True)
    npz_path = os.path.join(ART, "resnet20_int8.npz")
    np.savez(npz_path, meta=np.array(repr(meta)), **arrays)
    _write_c_header(arrays, meta, os.path.join(ART, "model_data.h"))

    n_w = sum(v.size for k, v in arrays.items() if k.endswith(".wq"))
    print(f"exported {len(meta['convs'])} weight layers, {len(meta['recalibs'])} recalib sites")
    print(f"  int8 weights total: {n_w:,} bytes ({n_w/1024:.1f} KB)")
    print(f"  -> {npz_path}")
    print(f"  -> {os.path.join(ART, 'model_data.h')}")


def _write_c_header(arrays, meta, path):
    """Minimal C header: flat int8 weight blobs + float scale/bias/recalib tables."""
    lines = ["// Auto-generated by export_model.py — ResNet-20 int8 + recalib for ESP32-S3",
             "#pragma once", "#include <stdint.h>", ""]
    for k, v in arrays.items():
        cname = "m_" + k.replace(".", "_")
        flat = np.asarray(v).reshape(-1)
        if v.dtype == np.int8:
            body = ",".join(str(int(x)) for x in flat)
            lines.append(f"static const int8_t {cname}[{flat.size}] = {{{body}}};")
        else:
            body = ",".join(f"{float(x):.8g}f" for x in flat)
            lines.append(f"static const float {cname}[{flat.size}] = {{{body}}};")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    export()
