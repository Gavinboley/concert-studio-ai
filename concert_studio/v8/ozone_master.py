"""Ozone 12 master of a V8 pre-limiter bus. Runs on WINDOWS Python (Ozone is a Windows VST3):

    %CS_OZONE_PYTHON% concert_studio\\v8\\ozone_master.py <premaster.wav> <out.wav>

Chain (module plugins, explicitly configured - the monolithic Ozone 12.vst3 loads with every module at
amount 0, i.e. a pass-through, which is what V6/V7's "Ozone master" actually was):
  Stabilizer (dynamic tonal balance) -> Clarity (moderate) -> Low End Focus (contrast +20, -1.5 dB)
  -> Maximizer (-1.0 dBTP ceiling, transient preservation, input gain solved for -14 LUFS).
Self-contained BS.1770-4 meter so the Windows venv needs no extra packages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pedalboard
import soundfile as sf
from scipy import signal

BASE = r"C:\Program Files\Common Files\VST3\iZotope" + "\\"
TARGET_LUFS = -14.0
CEILING = -1.0


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    # BS.1770 pre-filter (shelf) + RLB high-pass, coefficients from the analytic design (as pyloudnorm)
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    K = np.tan(np.pi * fc / sr)
    Vh, Vb = 10 ** (G / 20), (10 ** (G / 20)) ** 0.499666774155
    a0 = 1 + K / Q + K * K
    b = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    y = signal.lfilter(b, a, x, axis=0)
    Q, fc = 0.5003270373253953, 38.13547087613982
    K = np.tan(np.pi * fc / sr)
    a0 = 1 + K / Q + K * K
    b = [1, -2, 1]
    a = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return signal.lfilter(b, a, y, axis=0)


def lufs(x: np.ndarray, sr: int) -> float:
    y = k_weight(x, sr)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    ms = np.array([np.mean(y[i:i + blk] ** 2, axis=0).sum() for i in range(0, len(y) - blk, hop)])
    lk = -0.691 + 10 * np.log10(ms + 1e-20)
    g = ms[lk > -70]
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = ms[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(g2.mean()))


def true_peak_db(x: np.ndarray) -> float:
    pk = 0.0
    for s in range(0, len(x), 1 << 20):
        seg = x[max(0, s - 64):s + (1 << 20) + 64]
        pk = max(pk, float(np.max(np.abs(signal.resample_poly(seg, 4, 1, axis=0)))))
    return 20 * np.log10(pk + 1e-12)


def load(mod: str):
    return pedalboard.load_plugin(BASE + f"Ozone 12 {mod}.vst3")


def chain(max_gain_db: float):
    stab = load("Stabilizer")
    stab.stbl_stereo_main_amount = 40.0
    stab.stbl_stereo_main_smoothing = 60.0
    stab.stbl_stereo_main_sensitivity = 30.0
    cla = load("Clarity")
    cla.cla_stereo_main_amount = 20.0
    cla.cla_stereo_main_tilt_db_oct = 0.0
    lef = load("Low End Focus")
    lef.lef_stereo_main_contrast = 20.0
    lef.lef_stereo_main_gain_db = -1.5           # punch without adding boom (measured +2 dB sub at 30/0)
    mx = load("Maximizer")
    mx.max_output_level = CEILING - 0.3          # margin for inter-sample overs; verified below
    mx.max_character = 3.0
    mx.max_transient_shaping_amt = 60.0
    mx.max_soft_clip_on_off = False
    mx.max_input_gain_db = float(np.clip(max_gain_db, 0.0, 20.0))
    return pedalboard.Pedalboard([stab, cla, lef, mx])


def main(src: str, dst: str) -> None:
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    L0 = lufs(x, sr)
    # bring the bus to -16 LUFS ahead of the chain, then solve the Maximizer drive for the target
    pre = (x * 10 ** ((-16.0 - L0) / 20)).astype(np.float32)
    gain = 2.0
    out = None
    log = []
    for it in range(4):
        out = chain(gain)(pre.T.copy(), sr).T
        Lo = lufs(out, sr)
        log.append({"iter": it, "max_input_gain_db": round(gain, 2), "lufs": round(Lo, 2)})
        if abs(Lo - TARGET_LUFS) < 0.15:
            break
        gain += TARGET_LUFS - Lo
    tp = true_peak_db(out)
    if tp > CEILING:
        out = out * 10 ** ((CEILING - tp - 0.02) / 20)
        tp = true_peak_db(out)
    sf.write(dst, out.astype(np.float32), sr, subtype="PCM_24")
    rep = {"input_lufs": round(L0, 2), "iterations": log, "out_lufs": round(lufs(out, sr), 2),
           "out_true_peak_db": round(tp, 2),
           "chain": "Stabilizer 40 / Clarity 20 / Low End Focus +20 (-1.5 dB) / Maximizer ch3 transient 60"}
    Path(dst).with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
