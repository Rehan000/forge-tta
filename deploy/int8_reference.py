"""Int8-convolution ResNet-20 inference in NumPy — the firmware's golden reference.

Every convolution runs as int8 x int8 -> int32 accumulation (the expensive ops are
genuine integer MACs, exactly as the ESP32-S3 will do them). Requantization, bias,
per-channel recalibration, ReLU, pooling and the final linear run in float on what
the ESP32-S3 does with its FPU — a faithful, honest model of the on-device path.

This closes a real validity gap: the host PyTorch results use float "fake" quant.
If this integer engine reproduces that accuracy, the method is confirmed to survive
actual integer convolution execution.

Run via validate_reference.py.
"""
import ast
import numpy as np

EPS = 1e-5


def _quantize(x, scale):
    """float -> int8 (per-tensor symmetric), matching ActFakeQuant."""
    return np.clip(np.round(x / scale), -127, 127).astype(np.int8)


def _conv2d_int(x_i8, w_i8, stride, pad):
    """int8 input/weights -> int32(64) accumulation. x:[N,Cin,H,W] w:[Co,Cin,kh,kw]."""
    N, Cin, H, W = x_i8.shape
    Co, _, kh, kw = w_i8.shape
    sh, sw = stride
    ph, pw = pad
    xp = np.pad(x_i8.astype(np.int64), ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    Ho = (H + 2 * ph - kh) // sh + 1
    Wo = (W + 2 * pw - kw) // sw + 1
    # im2col
    cols = np.empty((N, Cin * kh * kw, Ho * Wo), dtype=np.int64)
    idx = 0
    for c in range(Cin):
        for i in range(kh):
            for j in range(kw):
                patch = xp[:, c, i:i + sh * Ho:sh, j:j + sw * Wo:sw]
                cols[:, idx, :] = patch.reshape(N, -1)
                idx += 1
    wmat = w_i8.astype(np.int64).reshape(Co, -1)          # [Co, Cin*kh*kw]
    acc = np.einsum("oc,ncp->nop", wmat, cols)            # int64 exact
    return acc.reshape(N, Co, Ho, Wo)


class IntResNet20:
    """Integer ResNet-20 (folded conv + float per-channel recalib) over the export npz."""

    def __init__(self, npz_path, momentum=0.01):
        d = np.load(npz_path, allow_pickle=True)
        self.p = {k: d[k] for k in d.files if k != "meta"}
        self.meta = ast.literal_eval(str(d["meta"]))
        self.momentum = self.meta.get("momentum", momentum)
        self.stride = {c["name"]: tuple(c["stride"]) for c in self.meta["convs"]}
        self.pad = {c["name"]: tuple(c["padding"]) for c in self.meta["convs"]}
        # ESP-NN simulation: quantize each conv OUTPUT to int8 @ out_scale (the integer
        # requant esp_nn does is equivalent to this per-tensor fake-quant). out_scales is
        # filled by calibrate_out_scales(); quant_out toggles the behaviour.
        self.quant_out = False
        # out_scales exported by export_model.py (the exact per-conv scales the firmware
        # uses); fall back to empty so calibrate_out_scales() can still recompute them.
        self.out_scales = {c["name"]: float(self.p[f"{c['name']}.out_scale"])
                           for c in self.meta["convs"]
                           if f"{c['name']}.out_scale" in self.p}
        self._calib = None
        self.reset()

    def reset(self):
        """Clear the online recalib EMA state (start of a fresh stream)."""
        self.state = {}
        for r in self.meta["recalibs"]:
            n = r["name"]
            tm = self.p[f"{n}.target_mean"].astype(np.float64)
            ts = self.p[f"{n}.target_std"].astype(np.float64)
            self.state[n] = [tm.copy(), (ts ** 2).copy()]   # run_mean, run_var

    def _conv(self, name, x):
        in_scale = float(self.p[f"{name}.in_scale"])
        wq = self.p[f"{name}.wq"]
        wscale = self.p[f"{name}.wscale"].astype(np.float64)
        bias = self.p[f"{name}.bias"].astype(np.float64)
        acc = _conv2d_int(_quantize(x, in_scale), wq, self.stride[name], self.pad[name])
        out = acc * (in_scale * wscale)[None, :, None, None] + bias[None, :, None, None]
        if self._calib is not None:                       # calibration pass: record range
            cur = np.abs(out).max()
            self._calib[name] = max(self._calib.get(name, 0.0), float(cur))
        if self.quant_out and name in self.out_scales:    # ESP-NN: quantize conv output to int8
            s = self.out_scales[name]
            out = np.clip(np.round(out / s), -127, 127) * s
        return out

    def _recalib(self, name, x, adapt):
        if not adapt:
            return x
        tm = self.p[f"{name}.target_mean"].astype(np.float64)
        ts = self.p[f"{name}.target_std"].astype(np.float64)
        rm, rv = self.state[name]
        m = x.mean((0, 2, 3))
        v = x.var((0, 2, 3))
        rm[:] = (1 - self.momentum) * rm + self.momentum * m
        rv[:] = (1 - self.momentum) * rv + self.momentum * v
        xn = (x - rm[None, :, None, None]) / np.sqrt(rv[None, :, None, None] + EPS)
        return xn * ts[None, :, None, None] + tm[None, :, None, None]

    def calibrate_out_scales(self, images):
        """Record per-conv output ranges over clean images, set out_scales = maxabs/127."""
        self._calib = {}
        for x in images:
            self.forward(x, adapt=False)
        self.out_scales = {k: (v / 127.0 if v > 0 else 1.0) for k, v in self._calib.items()}
        self._calib = None
        return self.out_scales

    def _block(self, prefix, x, downsample, adapt):
        identity = x
        out = np.maximum(self._recalib(f"{prefix}.bn1", self._conv(f"{prefix}.conv1", x), adapt), 0)
        out = self._recalib(f"{prefix}.bn2", self._conv(f"{prefix}.conv2", out), adapt)
        if downsample:
            identity = self._recalib(f"{prefix}.shortcut.1",
                                     self._conv(f"{prefix}.shortcut.0", x), adapt)
        return np.maximum(out + identity, 0)

    def forward(self, x, adapt=False):
        x = np.maximum(self._recalib("bn1", self._conv("conv1", x), adapt), 0)
        for i in range(3):
            x = self._block(f"layer1.{i}", x, False, adapt)
        x = self._block("layer2.0", x, True, adapt)
        for i in (1, 2):
            x = self._block(f"layer2.{i}", x, False, adapt)
        x = self._block("layer3.0", x, True, adapt)
        for i in (1, 2):
            x = self._block(f"layer3.{i}", x, False, adapt)
        x = x.mean((2, 3))                                   # global avg pool -> [N,64]
        # final linear (also quantized input)
        in_scale = float(self.p["linear.in_scale"])
        wq = self.p["linear.wq"]
        wscale = self.p["linear.wscale"].astype(np.float64)
        bias = self.p["linear.bias"].astype(np.float64)
        acc = _quantize(x, in_scale).astype(np.int64) @ wq.astype(np.int64).T
        return acc * (in_scale * wscale)[None, :] + bias[None, :]
