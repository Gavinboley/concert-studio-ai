"""Phone-recording simulator + scorer: a measurable benchmark for the multi-POV pipeline.

Real crowd footage has no clean reference, so quality claims need a controlled stand-in. `simulate` turns any
studio track into N "phone clips" of one performance: venue reverb at a per-phone distance, a crowd bed (real crowd
audio if given, else shaped noise with cheers), phone mic response, phone AGC pumping, optional clipping, partial
and possibly non-overlapping coverage, clock drift, and a lossy AAC encode (some at 32 kbps, low-passed and mono,
like TikTok's HE-AAC copies). `score` compares a pipeline output with the clean reference:

  coverage     fraction of the reference span the output covers (gaps = missing music)
  wobble_db    fast level error vs. the reference (AGC pumping, switch swells): std of the 0.4 s loudness
               difference after removing its 8 s trend
  tone_db      mean |1/3-octave long-term spectrum difference| after removing overall level (muffled/harsh)
  tone_wander  std over time of the 10 s band-shape difference (morphing timbre)
  unexplained_db  energy in the output not explained by the dry reference (per-band least-squares fit, so EQ
               alone is not penalised): crowd and noise, but also room and added reverb - a diagnostic, not a goal

    python -m concert_studio.v8.simulate make  --ref song.wav --out case/ --n 3 --start 30 --dur 75 --seed 1
        -> case/clips/phoneNN.m4a, case/reference.wav, case/sim.json
    python -m concert_studio.v8.simulate score --ref case/reference.wav --out master.wav
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100


def _load(path: Path, start: float = 0.0, dur: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-i", str(path)]
    if dur:
        cmd += ["-t", f"{dur}"]
    cmd += ["-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def _sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def _peq(x, f0, gain_db, q=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w0), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w0), 1 - al / A]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def _hall_ir(rt60: float, rng, n_s: float = 2.5) -> np.ndarray:
    n = int(n_s * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / rt60)[:, None]
    ir = signal.sosfilt(_sos("low", 6000), ir, axis=0)               # air/HF damping
    ir[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))[:, None]
    return (ir / np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))).astype(np.float32)


def _agc(x: np.ndarray, rng, depth: float) -> np.ndarray:
    """Phone AGC: fast attack, slow release gain riding with overshoot (the 'pumping' of crowd clips)."""
    m = np.sqrt(np.mean(x ** 2, axis=1)) + 1e-9
    hop = int(0.01 * SR)
    e = m[: len(m) // hop * hop].reshape(-1, hop).mean(1)
    g = np.ones_like(e)
    lvl = e[0]
    target = np.percentile(e, 50)
    for i, v in enumerate(e):
        a = 0.35 if v > lvl else 0.012                                # attack ~30 ms, release ~1 s
        lvl += a * (v - lvl)
        g[i] = (target / max(lvl, 1e-6)) ** depth
    g = np.repeat(g, hop)
    g = np.pad(g, (0, len(x) - len(g)), mode="edge")
    return (x * g[:, None]).astype(np.float32)


def _crowd_bed(n: int, rng, crowd_src: np.ndarray | None) -> np.ndarray:
    if crowd_src is not None and len(crowd_src) > n:
        s = int(rng.integers(0, len(crowd_src) - n))
        c = crowd_src[s:s + n].copy()
    else:
        c = rng.standard_normal((n, 2)).astype(np.float32)
        c = signal.sosfilt(_sos("band", [300, 4000]), c, axis=0)
        bursts = (rng.random(n // SR + 1) < 0.15).astype(float)        # occasional cheers
        env = np.repeat(bursts, SR)[:n]
        env = signal.fftconvolve(env, np.hanning(int(0.8 * SR)) / (0.4 * SR), "same")
        c *= (0.3 + 2.0 * env)[:, None]
    return (c / (np.sqrt(np.mean(c ** 2)) + 1e-9)).astype(np.float32)


def _coverage(n: int, dur: float, rng, mode: str) -> list[tuple[float, float]]:
    """Clip spans (s). 'overlap': staggered partial clips that chain together; 'islands': no overlap at all;
    'full': every phone records everything."""
    if mode == "full" or n == 1:
        return [(0.0, dur)] * n
    if mode == "islands":
        cut = np.sort(rng.uniform(0.25, 0.75, n - 1)) * dur
        edges = np.r_[0.0, cut, dur]
        return [(float(a + (2.0 if k else 0.0)), float(b - (2.0 if k < n - 1 else 0.0)))
                for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]   # 4 s gaps between phones
    spans = []
    for k in range(n):
        length = rng.uniform(0.45, 0.8) * dur
        a = (dur - length) * k / max(1, n - 1)
        spans.append((float(a), float(min(dur, a + length))))
    return spans


def make(ref: Path, out: Path, n: int = 3, start: float = 0.0, dur: float = 75.0, seed: int = 0,
         mode: str = "overlap", crowd: Path | None = None, crowd_db: tuple[float, float] = (-14.0, -6.0)) -> dict:
    rng = np.random.default_rng(seed)
    case, out = out, out / "clips"                                    # clips/ holds only the phone files
    out.mkdir(parents=True, exist_ok=True)
    x = _load(ref, start, dur)
    dur = len(x) / SR
    sf.write(case / "reference.wav", x, SR, subtype="FLOAT")
    crowd_src = _load(crowd) if crowd else None
    spans = _coverage(n, dur, rng, mode)
    meta = {"ref": str(ref), "start": start, "dur": dur, "seed": seed, "mode": mode, "clips": []}
    rt60 = rng.uniform(1.6, 2.4)
    rms_x = np.sqrt(np.mean(x ** 2))
    for k, (a, b) in enumerate(spans):
        # every position hears its own room response (same hall, different reflections) and its own PA delay
        wet = signal.fftconvolve(x, _hall_ir(rt60, rng), axes=0)[: len(x)].astype(np.float32)
        delay = int(rng.uniform(0.0, 0.03) * SR)
        drr = rng.uniform(-9.0, 3.0)                                   # direct-to-reverb (dB): pit .. balcony
        y = x * 10 ** (drr / 20) + wet * (rms_x / (np.sqrt(np.mean(wet ** 2)) + 1e-9))
        y = np.roll(y, delay, axis=0)
        y = _peq(y, rng.uniform(80, 140), rng.uniform(2, 6), 0.8)      # PA sub / room boom
        c = _crowd_bed(len(y), rng, crowd_src)
        cdb = rng.uniform(*crowd_db) + (8.0 if k == 1 and n > 2 else 0.0)   # phone 2 stands next to a screamer
        y = y + c * np.sqrt(np.mean(y ** 2)) * 10 ** (cdb / 20)
        y = signal.sosfilt(_sos("high", rng.uniform(110, 200), 2), y, axis=0)   # phone mic
        y = _peq(y, rng.uniform(2500, 5000), rng.uniform(2, 5), 1.2)
        y = _agc(y, rng, depth=rng.uniform(0.4, 0.8))
        y = y / (np.max(np.abs(y)) + 1e-9) * (1.6 if k == n - 1 and n > 1 else 0.95)
        y = np.clip(y, -1.0, 1.0)                                      # the last phone clips
        drift = rng.uniform(-40, 40) * 1e-6                            # phone clock (ppm)
        i0, i1 = int(a * SR), int(b * SR)
        seg = y[i0:i1]
        if abs(drift) > 0:
            seg = signal.resample(seg, int(round(len(seg) * (1 + drift))), axis=0).astype(np.float32)
        noise = rng.standard_normal(seg.shape).astype(np.float32) * 10 ** (-62 / 20)
        seg = seg + noise
        low = (k == n - 1 and n > 2) or (n == 2 and k == 1 and seed % 2 == 0)
        wav = out / f"_tmp_{k}.wav"
        sf.write(wav, seg, SR, subtype="FLOAT")
        dst = out / f"phone{k + 1:02d}.m4a"
        codec = (["-ac", "1", "-af", "lowpass=f=7000", "-b:a", "32k"] if low else ["-b:a", "128k"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac", *codec, str(dst)], check=True)
        wav.unlink()
        meta["clips"].append({"file": dst.name, "span": [a, b], "drr_db": round(drr, 1), "crowd_db": round(cdb, 1),
                              "drift_ppm": round(drift * 1e6, 1), "low_bitrate": low})
    (case / "sim.json").write_text(json.dumps(meta, indent=1))
    return meta


# ------------------------------------------------------------------------------------------------- scoring
def _mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def _align(ref: np.ndarray, out: np.ndarray) -> int:
    """Lag (samples) of out vs. ref via envelope + fine xcorr: ref[i] ~ out[i - lag]."""
    def env(z, h):
        z = np.abs(signal.sosfilt(_sos("high", 60), z))
        return z[: len(z) // h * h].reshape(-1, h).mean(1)
    h = 441
    a, b = env(ref, h), env(out, h)
    a = np.diff(a, prepend=a[0]).clip(0)
    b = np.diff(b, prepend=b[0]).clip(0)
    cc = signal.fftconvolve(a, b[::-1], mode="full")
    coarse = (int(np.argmax(cc)) - (len(b) - 1)) * h
    # refine within +-2 hops on the raw signal (2 s excerpt)
    c0 = max(0, coarse) + len(ref) // 3
    seg_r = ref[c0:c0 + 2 * SR]
    seg_o = out[max(0, c0 - coarse - 2 * h):max(0, c0 - coarse - 2 * h) + 2 * SR + 4 * h]
    if len(seg_r) < SR or len(seg_o) < len(seg_r):
        return coarse
    cc2 = signal.correlate(seg_o, seg_r, mode="valid")
    k = int(np.argmax(np.abs(cc2)))
    return coarse + 2 * h - k


def _bands(lo=40.0, hi=16000.0):
    c = 1000.0 * 2.0 ** (np.arange(-15, 15) / 3.0)
    return c[(c >= lo) & (c <= hi)]


def score(ref_path: Path, out_path: Path) -> dict:
    ref, sr1 = sf.read(ref_path, dtype="float32", always_2d=True)
    out, sr2 = sf.read(out_path, dtype="float32", always_2d=True)
    if sr2 != SR:
        out = signal.resample_poly(out, SR, sr2, axis=0).astype(np.float32)
    r, o = _mono(ref), _mono(out)
    lag = _align(r, o)
    # o shifted onto the reference timeline
    oo = np.zeros_like(r)
    src0 = max(0, -lag)
    dst0 = max(0, lag)
    n = min(len(r) - dst0, len(o) - src0)
    if n > 0:
        oo[dst0:dst0 + n] = o[src0:src0 + n]
    hop = int(0.1 * SR)
    def lvl(z):
        e = z[: len(z) // hop * hop].reshape(-1, hop)
        p = np.convolve((e ** 2).mean(1), np.ones(4) / 4, "same")
        return 10 * np.log10(p + 1e-12)
    Lr, Lo = lvl(r), lvl(oo)
    active_r = Lr > np.percentile(Lr, 95) - 30
    present = Lo > Lr - 25                                            # output has audio where the music is
    coverage = float((present & active_r).sum() / max(1, active_r.sum()))
    m = present & active_r
    d = Lo - Lr
    from scipy import ndimage
    dm = np.where(m, d, 0.0)
    w = ndimage.uniform_filter1d(m.astype(float), 80, mode="nearest") + 1e-9
    trend = ndimage.uniform_filter1d(dm, 80, mode="nearest") / w
    wobble = float(np.std((d - trend)[m])) if m.sum() > 50 else float("nan")
    # tone: long-term 1/3-octave spectra over the covered region
    mm = np.repeat(m, hop)
    mm = np.pad(mm, (0, max(0, len(r) - len(mm))))[: len(r)]
    f, Pr = signal.welch(r * mm, SR, nperseg=8192)
    _, Po = signal.welch(oo * mm, SR, nperseg=8192)
    c = _bands()
    def b3(P):
        return np.array([P[(f >= x * 2 ** (-1 / 6)) & (f < x * 2 ** (1 / 6))].sum() for x in c]) + 1e-20
    Br, Bo = b3(Pr), b3(Po)
    diff = 10 * np.log10(Bo / Br)
    diff -= np.average(diff, weights=Br / Br.sum())
    tone = float(np.mean(np.abs(diff)))
    # tone wander: per 10 s window band-shape difference
    W = 10 * SR
    shapes = []
    for i in range(0, len(r) - W, W // 2):
        if mm[i:i + W].mean() < 0.9:
            continue
        _, pr = signal.welch(r[i:i + W], SR, nperseg=4096)
        _, po = signal.welch(oo[i:i + W], SR, nperseg=4096)
        fw = np.fft.rfftfreq(4096, 1 / SR)
        def b3w(P):
            return np.array([P[(fw >= x * 2 ** (-1 / 6)) & (fw < x * 2 ** (1 / 6))].sum() for x in c[2:-3]]) + 1e-20
        dd = 10 * np.log10(b3w(po) / b3w(pr))
        shapes.append(dd - dd.mean())
    wander = float(np.mean(np.std(np.array(shapes), axis=0))) if len(shapes) > 2 else float("nan")
    # unexplained energy: per octave band, least-squares fit of output by reference (gain per 2 s block)
    resid, total = 0.0, 0.0
    for lo_, hi_ in ((60, 250), (250, 1000), (1000, 4000), (4000, 12000)):
        sos = _sos("band", [lo_, hi_], 4)
        rb, ob = signal.sosfilt(sos, r * mm), signal.sosfilt(sos, oo * mm)
        B = 2 * SR
        for i in range(0, len(rb) - B, B):
            x_, y_ = rb[i:i + B], ob[i:i + B]
            if mm[i:i + B].mean() < 0.9:
                continue
            # allow a small lag mismatch: best of +-3 ms
            best = None
            for s_ in range(-132, 133, 22):
                xs = np.roll(x_, s_)
                g = float(np.dot(xs, y_) / (np.dot(xs, xs) + 1e-12))
                e = float(np.sum((y_ - g * xs) ** 2))
                best = e if best is None or e < best else best
            resid += best
            total += float(np.sum(y_ ** 2))
    crowd_db = float(10 * np.log10(resid / (total + 1e-12) + 1e-12))
    return {"coverage": round(coverage, 3), "wobble_db": round(wobble, 2), "tone_db": round(tone, 2),
            "tone_wander_db": round(wander, 2), "unexplained_db": round(crowd_db, 1), "lag_s": round(lag / SR, 3)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=75.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="overlap", choices=["overlap", "islands", "full"])
    ap.add_argument("--crowd", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        print(json.dumps(make(a.ref, a.out, a.n, a.start, a.dur, a.seed, a.mode, a.crowd), indent=1))
    else:
        print(json.dumps(score(a.ref, a.out)))


if __name__ == "__main__":
    main()
