"""Objective audio QA for masters/stems: the stand-in for listening.

    python -m concert_studio.v8.analyze a.wav [b.wav ...] --png out.png --start 120 --dur 20

Reports per file:
  LUFS-I, LRA (EBU-ish, 3 s windows), true peak, crest, HF air (>10k re 1k octave),
  buzz lines (narrow LTAS peaks >= 6 dB above 1/3-oct smoothed LTAS, 50 Hz-12 kHz),
  hiss/shimmer index (median 10-16 kHz frame-to-frame flux in quiet passages),
  stereo: mean/min short-term L/R correlation, side/mid ratio, and how much the
  image *moves* (std of the 400 ms balance in dB) - V5's "moves around my head".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .dsp import db, smooth_spectrum_oct


def _lufs_and_lra(x, sr):
    import pyloudnorm as pyln
    m = pyln.Meter(sr)
    x2 = x if x.ndim == 2 else np.stack([x, x], 1)
    I = m.integrated_loudness(x2)
    win, hop = int(3 * sr), int(1 * sr)
    st = []
    for s in range(0, len(x2) - win, hop):
        v = m.integrated_loudness(x2[s:s + win]) if np.any(x2[s:s + win]) else -np.inf
        st.append(v)
    st = np.array([v for v in st if np.isfinite(v) and v > I - 20])
    lra = float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) > 5 else float("nan")
    return float(I), lra


def analyze(path: str | Path) -> dict:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    I, lra = _lufs_and_lra(x, sr)
    up = signal.resample_poly(x, 4, 1, axis=0)
    tp = float(db(np.max(np.abs(up))))
    crest = float(db(np.max(np.abs(mono))) - db(np.sqrt(np.mean(mono ** 2))))
    f, P = signal.welch(mono, fs=sr, nperseg=16384, noverlap=8192, average="median")
    Pdb = 10 * np.log10(P + 1e-20)
    sm = smooth_spectrum_oct(Pdb, f, 3.0)
    band = (f > 50) & (f < 12000)
    peaks = np.where(band & (Pdb - sm > 6.0))[0]
    # merge adjacent bins
    lines = []
    for i in peaks:
        if lines and f[i] - lines[-1][0] < 3 * (f[1] - f[0]):
            if Pdb[i] - sm[i] > lines[-1][1]:
                lines[-1] = (float(f[i]), float(Pdb[i] - sm[i]))
            continue
        lines.append((float(f[i]), float(Pdb[i] - sm[i])))
    oct_ = lambda lo, hi: 10 * np.log10(np.sum(P[(f >= lo) & (f < hi)]) + 1e-20)
    air = oct_(10000, 16000) - oct_(700, 1400)
    sub = oct_(30, 60) - oct_(700, 1400)
    low = oct_(60, 250) - oct_(700, 1400)
    lowmid = oct_(250, 500) - oct_(700, 1400)
    pres = oct_(2500, 5000) - oct_(700, 1400)
    # shimmer: HF band flux in 23 ms frames, relative to the HF band level
    fq, tq, Z = signal.stft(mono, fs=sr, nperseg=1024, noverlap=0)
    hf = np.abs(Z[(fq > 10000) & (fq < 16000)])
    hfl = 20 * np.log10(hf + 1e-9)
    flux = np.median(np.abs(np.diff(hfl, axis=1)))
    out = dict(file=str(path), dur_s=len(x) / sr, lufs_i=round(I, 2), lra_lu=round(lra, 2), true_peak_db=round(tp, 2),
               crest_db=round(crest, 2), sub_30_60=round(sub, 1), low_60_250=round(low, 1),
               lowmid_250_500=round(lowmid, 1), presence_2k5_5k=round(pres, 1), air_10k_16k=round(air, 1),
               hf_flux_db=round(float(flux), 2), buzz_lines=[(round(a), round(b, 1)) for a, b in lines[:12]],
               n_buzz_lines=len(lines))
    if x.shape[1] == 2 and np.any(x[:, 0] != x[:, 1]):
        L, R = x[:, 0], x[:, 1]
        w = int(0.4 * sr)
        n = len(L) // w
        Lr, Rr = L[:n * w].reshape(n, w), R[:n * w].reshape(n, w)
        e = (Lr ** 2).sum(1) + (Rr ** 2).sum(1)
        act = e > np.percentile(e, 20)
        corr = (Lr * Rr).sum(1) / np.sqrt((Lr ** 2).sum(1) * (Rr ** 2).sum(1) + 1e-20)
        bal = 10 * np.log10(((Lr ** 2).sum(1) + 1e-12) / ((Rr ** 2).sum(1) + 1e-12))
        M, S = (L + R) / 2, (L - R) / 2
        out.update(stereo_corr_mean=round(float(corr[act].mean()), 3), stereo_corr_p5=round(float(np.percentile(corr[act], 5)), 3),
                   side_mid_db=round(float(db(np.sqrt(np.mean(S ** 2))) - db(np.sqrt(np.mean(M ** 2)))), 1),
                   balance_wander_db=round(float(np.std(bal[act])), 2))
    return out


def spectrogram_png(paths: list[str], png: str, start: float, dur: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(paths), 1, figsize=(14, 3.2 * len(paths)), squeeze=False)
    for ax, p in zip(axes[:, 0], paths):
        info = sf.info(p)
        x, sr = sf.read(p, start=int(start * info.samplerate), frames=int(dur * info.samplerate),
                        dtype="float32", always_2d=True)
        m = x.mean(axis=1)
        f, t, Z = signal.stft(m, fs=sr, nperseg=4096, noverlap=3072)
        S = 20 * np.log10(np.abs(Z) + 1e-7)
        S -= S.max()
        ax.pcolormesh(t + start, f, S, vmin=-90, vmax=0, shading="auto", cmap="magma")
        ax.set_yscale("symlog", linthresh=500)
        ax.set_ylim(30, sr / 2)
        ax.set_title(Path(p).name, fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=80)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--png")
    ap.add_argument("--start", type=float, default=120.0)
    ap.add_argument("--dur", type=float, default=20.0)
    a = ap.parse_args(argv)
    for p in a.files:
        print(json.dumps(analyze(p), default=float))
    if a.png:
        spectrogram_png(a.files, a.png, a.start, a.dur)


if __name__ == "__main__":
    main()
