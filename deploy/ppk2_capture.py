"""Measure ESP32-S3 inference vs adaptation energy with a Nordic PPK2.

Setup (PPK2 in Source Meter mode, powering the board's 3V3 rail):
  PPK2 VOUT  -> ESP32 3V3 pin
  PPK2 GND   -> ESP32 GND
  ESP32 GPIO5 (INFER) -> PPK2 logic D0
  ESP32 GPIO4 (ADAPT) -> PPK2 logic D1
  ESP32 GND  -> PPK2 logic GND
  ESP32 USB: DISCONNECTED during measurement (board is powered by the PPK2).
  PPK2 USB -> this Mac.

The firmware energy loop alternates inference-only (D0=1,D1=0) and inference+adapt
(D0=1,D1=1) with idle gaps (D0=0). This script segments the current trace by those
GPIOs and reports mJ/inference, mJ/adaptation, and idle baseline power.

Usage:
    python deploy/ppk2_capture.py --seconds 40           # auto-detect PPK2 port
    python deploy/ppk2_capture.py --port /dev/cu.usbmodemXXXX --voltage 3300
"""
import argparse
import glob
import sys
import time

from ppk2_api.ppk2_api import PPK2_API

SAMPLE_HZ = 100000          # PPK2 fixed sample rate
DT = 1.0 / SAMPLE_HZ


def find_ppk2(exclude):
    """Return the PPK2 *control* port — the one that returns calibration metadata.
    list_devices() returns both CDC interfaces in arbitrary order; only one responds."""
    try:
        cands = [d for d in (PPK2_API.list_devices() or []) if d not in exclude]
    except Exception:
        cands = []
    cands += [p for p in glob.glob("/dev/cu.usbmodem*") if p not in exclude and p not in cands]
    for p in cands:
        try:
            ppk2 = PPK2_API(p, timeout=1)
            if ppk2.get_modifiers():        # control port returns metadata
                ppk2.ser.close()
                return p
            ppk2.ser.close()
        except Exception:
            pass
    return None


def segment(currents_uA, d0, d1, voltage):
    """Split into D0-high windows, label by D1, return per-class energy/power stats."""
    windows = []          # (energy_mJ, duration_ms, was_adapt)
    idle_uA = []
    i, n = 0, len(currents_uA)
    while i < n:
        if d0[i]:
            j = i
            e_J = 0.0
            adapt_hi = 0
            while j < n and d0[j]:
                e_J += voltage * (currents_uA[j] * 1e-6) * DT
                adapt_hi += d1[j]
                j += 1
            dur_ms = (j - i) * DT * 1000.0
            windows.append((e_J * 1000.0, dur_ms, adapt_hi > (j - i) / 2))
            i = j
        else:
            idle_uA.append(currents_uA[i])
            i += 1
    return windows, idle_uA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--voltage", type=int, default=3300, help="source voltage (mV)")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--esp-port", default="/dev/cu.usbmodem5B5E1034441")
    args = ap.parse_args()

    port = args.port or find_ppk2(exclude={args.esp_port})
    if not port:
        print("PPK2 not found. Plug the PPK2 into the Mac (and pass --port if needed).")
        sys.exit(1)
    print(f"PPK2 on {port} | source {args.voltage} mV | capturing {args.seconds:.0f}s")

    ppk2 = PPK2_API(port)
    ppk2.get_modifiers()
    ppk2.use_source_meter()
    ppk2.set_source_voltage(args.voltage)
    ppk2.toggle_DUT_power("ON")
    ppk2.start_measuring()

    currents, dig = [], []
    t0 = time.time()
    while time.time() - t0 < args.seconds:        # TIGHT loop (no sleep) to keep up with 100kHz
        data = ppk2.get_data()
        if data:
            s, raw = ppk2.get_samples(data)
            currents.extend(s)
            dig.extend(raw)
    ppk2.stop_measuring()
    ppk2.toggle_DUT_power("OFF")

    ch = ppk2.digital_channels(dig)
    n = min(len(currents), len(ch[0]))
    currents, d0, d1 = currents[:n], ch[0][:n], ch[1][:n]
    print(f"captured {n} samples ({n*DT:.1f}s of data)\n")

    windows, idle_uA = segment(currents, d0, d1, args.voltage / 1000.0)
    inf = [w for w in windows if not w[2]]
    adp = [w for w in windows if w[2]]
    if not inf or not adp:
        print(f"Not enough labeled windows (inference={len(inf)}, adapt={len(adp)}). "
              "Check GPIO->D0/D1 wiring and that the firmware energy loop is running.")
        sys.exit(1)

    def avg(xs, k):
        return sum(x[k] for x in xs) / len(xs)
    idle_mW = (sum(idle_uA) / len(idle_uA)) * 1e-6 * (args.voltage / 1000.0) * 1000 if idle_uA else 0.0
    e_inf, d_inf = avg(inf, 0), avg(inf, 1)
    e_adp, d_adp = avg(adp, 0), avg(adp, 1)

    print(f"idle baseline power: {idle_mW:.2f} mW\n")
    print(f"{'operation':<22}{'energy(mJ)':>12}{'time(ms)':>10}{'avg power(mW)':>15}{'n':>5}")
    print("-" * 64)
    print(f"{'inference only':<22}{e_inf:>12.2f}{d_inf:>10.1f}{e_inf/(d_inf/1000):>15.1f}{len(inf):>5}")
    print(f"{'inference + adapt':<22}{e_adp:>12.2f}{d_adp:>10.1f}{e_adp/(d_adp/1000):>15.1f}{len(adp):>5}")
    print("-" * 64)
    print(f"{'adaptation overhead':<22}{e_adp - e_inf:>12.2f}{d_adp - d_inf:>10.1f}")
    print(f"\nadaptation costs {(e_adp-e_inf):.2f} mJ ({(e_adp-e_inf)/e_inf*100:.1f}% on top of inference)")


if __name__ == "__main__":
    main()
