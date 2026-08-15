"""Energy measurement WITHOUT logic triggers (PPK2 logic port unusable on this unit).

The firmware alternates inference (forward, adapt=0) and inference+adapt (forward,
adapt=1). These produce two cleanly-separated window DURATIONS (the adapt step adds
~21 ms of recalibration), so we label each active window by its duration instead of a
GPIO trigger. Current is integrated at the full 100 kHz rate; source voltage is constant
(3.3 V), so energy = V * sum(I * dt).

Setup: PPK2 Source mode powering the 3V3 rail (VOUT->3V3, GND->GND); ESP32 USB unplugged;
PPK2 USB -> Mac. (No logic-port wiring needed.)

Usage:
    python deploy/ppk2_capture_notrig.py --seconds 90
"""
import argparse
import sys
import time

from statistics import median
from ppk2_api.ppk2_api import PPK2_API

DT = 1e-5            # 100 kHz
PORT_DEFAULT = "/dev/cu.usbmodemCEFC62156B8E2"


def detect_windows(cur, v, th_hi=80000, th_lo=60000, min_s=1.5, max_s=4.0):
    """Contiguous current>th_hi runs (hysteresis to th_lo) -> (energy_mJ, dur_ms)."""
    wins, idle, n, i = [], [], len(cur), 0
    while i < n:
        if cur[i] > th_hi:
            j, e = i, 0.0
            while j < n and cur[j] > th_lo:
                e += v * (cur[j] * 1e-6) * DT
                j += 1
            dur = (j - i) * DT
            if min_s < dur < max_s:
                wins.append((e * 1000.0, dur * 1000.0))
            i = j
        else:
            idle.append(cur[i])
            i += 1
    return wins, idle


def split_by_duration(wins):
    """Split into inference / inference+adapt at the largest gap in the CENTRAL region of
    sorted durations (robust to a stray partial window at the edges)."""
    sw = sorted(wins, key=lambda w: w[1])
    durs = [w[1] for w in sw]
    n = len(durs)
    if n < 4:
        return None
    lo_i, hi_i = max(1, n // 5), min(n - 1, (4 * n) // 5)
    _, idx = max((durs[i] - durs[i - 1], i) for i in range(lo_i, hi_i + 1))
    split = (durs[idx] + durs[idx - 1]) / 2.0
    inf = [w for w in sw if w[1] < split]
    adp = [w for w in sw if w[1] >= split]
    print("  durations(ms):", " ".join(f"{d:.0f}" for d in durs))
    print(f"  split at {split:.0f} ms -> {len(inf)} inference / {len(adp)} adapt windows")
    return inf, adp, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--voltage", type=int, default=3300)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--reps", type=int, default=10, help="forward() calls per window (firmware EREPS)")
    args = ap.parse_args()
    v = args.voltage / 1000.0

    ppk2 = PPK2_API(args.port, timeout=1)
    ppk2.get_modifiers()
    ppk2.use_source_meter()
    ppk2.set_source_voltage(args.voltage)
    ppk2.toggle_DUT_power("OFF")
    time.sleep(1.0)
    ppk2.toggle_DUT_power("ON")             # clean boot
    ppk2.start_measuring()
    print(f"source {args.voltage} mV | capturing {args.seconds:.0f}s ...")
    cur, t0 = [], time.time()
    while time.time() - t0 < args.seconds:
        d = ppk2.get_data()
        if d:
            s, _ = ppk2.get_samples(d)
            cur.extend(s)
    ppk2.stop_measuring()
    ppk2.toggle_DUT_power("OFF")

    print(f"captured {len(cur)} samples ({len(cur)*DT:.1f}s), mean {sum(cur)/len(cur)/1000:.1f} mA\n")
    wins, idle = detect_windows(cur, v)
    res = split_by_duration(wins)
    if not res or not res[0] or not res[1]:
        print(f"could not separate the two window types ({len(wins)} windows). "
              "Capture longer or check the board is running.")
        sys.exit(1)
    inf, adp, split = res

    def mean(xs, k):
        return sum(x[k] for x in xs) / len(xs)
    e_inf, d_inf = median(w[0] for w in inf)/args.reps, median(w[1] for w in inf)/args.reps
    e_adp, d_adp = median(w[0] for w in adp)/args.reps, median(w[1] for w in adp)/args.reps
    idle_mW = (sum(idle) / len(idle)) * 1e-6 * v * 1000 if idle else 0.0

    print(f"split at {split:.0f} ms | inference windows={len(inf)} adapt windows={len(adp)}")
    print(f"idle baseline power: {idle_mW:.1f} mW\n")
    print(f"{'operation':<22}{'energy(mJ)':>12}{'time(ms)':>10}{'power(mW)':>11}")
    print("-" * 55)
    print(f"{'inference':<22}{e_inf:>12.1f}{d_inf:>10.1f}{e_inf/(d_inf/1000):>11.0f}")
    print(f"{'inference + adapt':<22}{e_adp:>12.1f}{d_adp:>10.1f}{e_adp/(d_adp/1000):>11.0f}")
    print("-" * 55)
    print(f"{'adaptation overhead':<22}{e_adp-e_inf:>12.2f}{d_adp-d_inf:>10.1f}")
    print(f"\nadaptation adds {e_adp-e_inf:.2f} mJ ({(e_adp-e_inf)/e_inf*100:.2f}% of inference energy), "
          f"{d_adp-d_inf:.1f} ms")


if __name__ == "__main__":
    main()
