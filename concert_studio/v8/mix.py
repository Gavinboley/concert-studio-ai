"""Stem mix + master for V8.

Design rules (from the user's feedback history on V1-V7):
* everything is summed to mono first; stereo is synthesised and is exactly mono-compatible
  (M/S: L = M + S, R = M - S with S a decorrelated, high-passed copy) - no L/R wander
* no wideband squashing: per-stem gentle compression only where it fixes a real problem
  (vocal level riding, bass note evenness); bus glue <= 2 dB GR
* every nonlinearity is 4x oversampled so it cannot alias into buzz
* reverb is a synthesized smooth stereo hall/plate (filtered decaying noise), not a comb-based
  Freeverb (metallic ringing)
* no synthetic drum or bass triggers (those produced the V6 clicking)
"""

from __future__ import annotations

import numpy as np
from scipy import signal, ndimage

from .dsp import db, rms, apply_fir, linear_phase_eq_fir, highpass, lowpass, smooth_spectrum_oct, ltas_db

SR = 44100


# --------------------------------------------------------------------------- EQ primitives
def _biquad(kind: str, f0: float, gain_db: float, q: float, sr: int = SR) -> np.ndarray:
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sA = np.sqrt(A)
        b = [A * ((A + 1) - (A - 1) * cw + 2 * sA * alpha), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * sA * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * sA * alpha, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * sA * alpha]
    elif kind == "highshelf":
        sA = np.sqrt(A)
        b = [A * ((A + 1) + (A - 1) * cw + 2 * sA * alpha), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * sA * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * sA * alpha, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * sA * alpha]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return signal.tf2sos(b, a)


def eq(x: np.ndarray, bands: list[tuple], sr: int = SR) -> np.ndarray:
    """bands: (kind, f0, gain_db, q). Minimum-phase IIR, like a console EQ."""
    sos = np.concatenate([_biquad(k, f, g, q, sr) for k, f, g, q in bands]) if bands else None
    if sos is None:
        return x
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- dynamics
def envelope(x: np.ndarray, attack_ms: float, release_ms: float, sr: int = SR, mode: str = "peak") -> np.ndarray:
    """One-pole attack/release envelope follower (vectorised via lfilter on the two regimes)."""
    d = np.abs(x) if mode == "peak" else x ** 2
    a_att = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    a_rel = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    # decimate for speed: envelopes are smooth
    hop = 32
    dd = d[: len(d) // hop * hop].reshape(-1, hop).max(axis=1)
    e = np.empty_like(dd)
    aa, ar = a_att ** hop, a_rel ** hop
    prev = 0.0
    for i, v in enumerate(dd):
        a = aa if v > prev else ar
        prev = a * prev + (1 - a) * v
        e[i] = prev
    e = np.repeat(e, hop)
    e = np.pad(e, (0, len(d) - len(e)), mode="edge")
    return np.sqrt(e) if mode == "rms" else e


def compress(x: np.ndarray, thresh_db: float, ratio: float, attack_ms: float, release_ms: float,
             knee_db: float = 6.0, makeup_db: float = 0.0, mode: str = "rms", sidechain: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    sc = x if sidechain is None else sidechain
    e_db = db(envelope(sc, attack_ms, release_ms, mode=mode))
    over = e_db - thresh_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    g = 10 ** ((makeup_db - gr) / 20.0)
    g = ndimage.uniform_filter1d(g, 64)  # de-zipper
    if x.ndim == 2:
        g = g[:, None]
    return (x * g).astype(np.float32), gr


def level_rider(x: np.ndarray, target_db: float, max_boost: float = 6.0, max_cut: float = 8.0,
                window_s: float = 0.4, smooth_s: float = 0.6, gate_db: float = -45.0) -> np.ndarray:
    """Slow vocal rider (fader automation), not a compressor: follows phrases, ignores transients."""
    hop = int(0.05 * SR)
    win = int(window_s * SR)
    n = len(x) // hop
    e = np.array([rms(x[i * hop:i * hop + win]) for i in range(n)])
    edb = db(e)
    active = edb > gate_db
    gain = np.where(active, np.clip(target_db - edb, -max_cut, max_boost), 0.0)
    # hold through gaps so breaths/tails are not pumped up
    idx = np.where(active)[0]
    if len(idx):
        gain = np.interp(np.arange(n), idx, gain[idx])
    gain = ndimage.uniform_filter1d(gain, max(1, int(smooth_s / 0.05)))
    g = np.interp(np.arange(len(x)), np.arange(n) * hop + win // 2, gain)
    return (x * 10 ** (g / 20.0)).astype(np.float32)


def deess(x: np.ndarray, lo: float = 5500.0, hi: float = 10000.0, thresh_rel_db: float = 6.0,
          ratio: float = 4.0) -> np.ndarray:
    """Split-band de-esser: only the sibilance band is reduced, only when it pokes out."""
    h = signal.firwin(511, [lo, hi], pass_zero=False, fs=SR).astype(np.float32)
    band = apply_fir(x, h, linear_phase=True)
    rest = x - band
    eb = db(envelope(band, 1.0, 60.0, mode="rms"))
    er = db(envelope(rest, 5.0, 120.0, mode="rms"))
    over = eb - (er + thresh_rel_db - 12.0)
    gr = np.maximum(0.0, over) * (1 - 1 / ratio)
    gr = np.minimum(gr, 12.0)
    g = 10 ** (-ndimage.uniform_filter1d(gr, 32) / 20.0)
    return (rest + band * g).astype(np.float32)


def saturate(x: np.ndarray, drive_db: float, mix: float = 1.0, asym: float = 0.0) -> np.ndarray:
    """4x-oversampled soft saturation (no aliasing -> no buzz)."""
    up = signal.resample_poly(x, 4, 1, axis=0)
    k = 10 ** (drive_db / 20.0)
    y = np.tanh(k * (up + asym)) - np.tanh(k * asym)
    y /= k * (1 - np.tanh(k * asym) ** 2 + 1e-9)
    y = signal.resample_poly(y, 1, 4, axis=0)[: len(x)]
    return ((1 - mix) * x + mix * y).astype(np.float32)


# --------------------------------------------------------------------------- space
def decorrelator(seed: int, length_ms: float = 25.0, sr: int = SR) -> np.ndarray:
    """Flat-magnitude random-phase FIR (all-pass-ish) for synthetic side signals."""
    rng = np.random.default_rng(seed)
    n = int(length_ms * 1e-3 * sr)
    nfft = 1 << int(np.ceil(np.log2(4 * n)))
    ph = rng.uniform(-np.pi, np.pi, nfft // 2 + 1)
    ph[0] = ph[-1] = 0.0
    # smooth phase across frequency to avoid audible smearing of transients
    ph = np.unwrap(ph)
    ph = ndimage.gaussian_filter1d(ph, 3.0)
    h = np.fft.irfft(np.exp(1j * ph), nfft)
    h = np.roll(h, n // 2)[:n] * np.hanning(n)
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)


def stereoize(m: np.ndarray, width: float, seed: int, hp_hz: float = 250.0) -> np.ndarray:
    """Mono-compatible width: L = M + S, R = M - S, S = width * decorrelate(HPF(M)). Mono sum == 2M."""
    if width <= 0:
        return np.stack([m, m], axis=1)
    s = highpass(m, SR, hp_hz, order=2)
    s = apply_fir(s, decorrelator(seed)) * width
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


def synth_reverb_ir(rt60: float, predelay_ms: float, sr: int = SR, seed: int = 7, hf_damp: float = 0.45,
                    lf_rt_mult: float = 1.15, er: bool = True) -> np.ndarray:
    """Stereo IR: band-split decaying noise with frequency-dependent RT60, optional early reflections."""
    rng = np.random.default_rng(seed)
    n = int((rt60 * 1.3 + predelay_ms / 1000) * sr)
    t = np.arange(n) / sr
    pd = int(predelay_ms * 1e-3 * sr)
    edges = [0, 250, 1000, 4000, 9000, sr / 2]
    rts = [rt60 * lf_rt_mult, rt60, rt60 * 0.85, rt60 * hf_damp + 0.1, rt60 * hf_damp * 0.6 + 0.05]
    ir = np.zeros((n, 2))
    for ch in range(2):
        noise = rng.standard_normal(n)
        acc = np.zeros(n)
        for (lo, hi), rt in zip(zip(edges[:-1], edges[1:]), rts):
            if lo == 0:
                sos = signal.butter(4, hi, "low", fs=sr, output="sos")
            elif hi >= sr / 2:
                sos = signal.butter(4, lo, "high", fs=sr, output="sos")
            else:
                sos = signal.butter(4, [lo, hi], "band", fs=sr, output="sos")
            band = signal.sosfilt(sos, noise)
            acc += band * np.exp(-6.91 * t / rt)
        # soft onset (no click), predelay
        ramp = np.minimum(1.0, t / 0.012)
        acc *= ramp
        ir[pd:, ch] = acc[: n - pd]
        if er:
            for k in range(8):
                d = pd + int(rng.uniform(3, 35) * 1e-3 * sr)
                if d < n:
                    ir[d, ch] += rng.uniform(0.15, 0.45) * (1 if rng.random() > 0.5 else -1)
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir.astype(np.float32)


def convolve_stereo(m: np.ndarray, ir: np.ndarray) -> np.ndarray:
    L = signal.oaconvolve(m, ir[:, 0])[: len(m)]
    R = signal.oaconvolve(m, ir[:, 1])[: len(m)]
    return np.stack([L, R], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- loudness + limiting
def lufs(x: np.ndarray) -> float:
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    return float(meter.integrated_loudness(x if x.ndim == 2 else np.stack([x, x], 1)))


def true_peak_db(x: np.ndarray) -> float:
    up = signal.resample_poly(x, 4, 1, axis=0)
    return float(db(np.max(np.abs(up))))


def limiter(x: np.ndarray, ceiling_dbtp: float = -1.0, lookahead_ms: float = 5.0, release_ms: float = 80.0
            ) -> np.ndarray:
    """Look-ahead brickwall limiter on the 4x-oversampled peak; smooth gain (no distortion clicks)."""
    c = 10 ** (ceiling_dbtp / 20.0)
    up = signal.resample_poly(x, 4, 1, axis=0)
    pk = np.max(np.abs(up), axis=1) if up.ndim == 2 else np.abs(up)
    pk = pk[: len(pk) // 4 * 4].reshape(-1, 4).max(axis=1)
    pk = np.pad(pk, (0, len(x) - len(pk)), mode="edge")
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    la = int(lookahead_ms * 1e-3 * SR)
    # min over look-ahead window, then attack smoothing (hann over la) and exponential release
    g = ndimage.minimum_filter1d(need, size=2 * la + 1, origin=0)
    rel = np.exp(-1.0 / (release_ms * 1e-3 * SR / 32))
    hop = 32
    gd = g[: len(g) // hop * hop].reshape(-1, hop).min(axis=1)
    out = np.empty_like(gd)
    prev = 1.0
    for i, v in enumerate(gd):
        prev = v if v < prev else rel * prev + (1 - rel) * v
        out[i] = prev
    g2 = np.repeat(out, hop)
    g2 = np.pad(g2, (0, len(g) - len(g2)), mode="edge")
    g2 = np.minimum(g2, g)
    win = np.hanning(2 * la + 1)
    win /= win.sum()
    g2 = np.minimum(signal.oaconvolve(g2, win, mode="same"), 1.0)
    g2 = np.minimum(g2, ndimage.minimum_filter1d(g2, size=3))
    y = x * (g2[:, None] if x.ndim == 2 else g2)
    # final safety: any residual inter-sample over is tiny; scale globally if needed
    tp = true_peak_db(y)
    if tp > ceiling_dbtp:
        y *= 10 ** ((ceiling_dbtp - tp - 0.02) / 20.0)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- tonal targets
# 1/3-octave band power (dB) relative to the band's own reference, for a finished studio stem.
# Deliberately conservative; corrections are limited and smoothed.
TARGETS = {
    "vocals": {90: -14, 125: -7, 200: -1, 315: 0, 500: 0, 800: -1, 1250: -2.5, 2000: -3.5, 3150: -4.5,
               5000: -6.5, 8000: -9, 12500: -13, 16000: -18},
    "guitar": {63: -14, 100: -5, 160: -1, 250: 0, 500: 0, 1000: 0, 2000: -1.5, 3150: -3.5, 5000: -8,
               8000: -16, 12500: -24, 16000: -30},
    "bass":   {31: -3, 45: 0, 63: 0, 100: 0, 160: -3, 250: -6, 500: -11, 1000: -15, 2000: -19, 4000: -26},
    "drums":  {40: -3, 63: 0, 100: -1, 160: -3, 250: -4, 500: -6, 1000: -7, 2000: -7, 4000: -7, 8000: -8,
               12500: -10, 16000: -15},
    "keys":   {63: -10, 125: -3, 250: 0, 500: 0, 1000: -1, 2000: -3, 4000: -5.5, 8000: -10, 16000: -18},
}


def third_octave_levels(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    f, P = signal.welch(x, fs=sr, nperseg=16384, noverlap=8192)
    centers = 1000.0 * 2.0 ** (np.arange(-16, 14) / 3.0)   # 25 Hz .. 20 kHz
    lv = []
    for c in centers:
        m = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        lv.append(10 * np.log10(np.sum(P[m]) + 1e-20) if m.any() else -200.0)
    return centers, np.array(lv)


def tonal_correct(x: np.ndarray, kind: str, max_boost: float = 6.0, max_cut: float = 9.0,
                  ref_band: tuple[float, float] = (250.0, 2000.0), top_hz: float | None = None,
                  strength: float = 1.0, no_boost_below: float | None = None,
                  boost_caps: list[tuple[float, float, float]] | None = None) -> tuple[np.ndarray, dict]:
    """Minimum-phase EQ moving the stem's 1/3-oct balance toward TARGETS[kind]."""
    from .dsp import min_phase_eq_fir
    c, lv = third_octave_levels(x)
    tf = np.array(sorted(TARGETS[kind]))
    tv = np.array([TARGETS[kind][k] for k in tf])
    tgt = np.interp(np.log2(c), np.log2(tf), tv, left=tv[0] - 12, right=tv[-1] - 12)
    m = (c >= ref_band[0]) & (c <= ref_band[1])
    off = np.mean(lv[m] - tgt[m])
    corr = (tgt + off) - lv
    # don't try to resurrect bands that are essentially empty (noise floor / codec cut)
    floor = np.max(lv) - 55.0
    corr = np.where(lv < floor, np.minimum(corr, 0.0), corr)
    if top_hz:
        corr = np.where(c > top_hz, np.minimum(corr, 0.0), corr)
    if no_boost_below:
        corr = np.where(c < no_boost_below, np.minimum(corr, 0.0), corr)
    for lo_hz, hi_hz, cap in boost_caps or []:
        corr = np.where((c >= lo_hz) & (c < hi_hz), np.minimum(corr, cap), corr)
    corr = np.clip(corr * strength, -max_cut, max_boost)
    # 1/2-octave smoothing on the correction itself
    corr = ndimage.gaussian_filter1d(corr, 0.75, mode="nearest")
    corr -= np.mean(corr[m])  # keep level roughly where it was
    fgrid = np.concatenate([[0.0], c, [SR / 2]])
    ggrid = np.concatenate([[corr[0]], corr, [corr[-1]]])
    h = min_phase_eq_fir(fgrid, ggrid, SR, n_taps=4096)
    y = apply_fir(x, h)
    return y, {f"{int(round(a))}": round(float(b), 1) for a, b in zip(c, corr)}


def match_loudness(x: np.ndarray, target_lufs: float) -> np.ndarray:
    L = lufs(x)
    if not np.isfinite(L):
        return x
    return (x * 10 ** ((target_lufs - L) / 20.0)).astype(np.float32)


def tonal_stabilizer(x: np.ndarray, sr: int, strength: float = 0.6, max_db: float = 4.0, smooth_s: float = 12.0,
                     ref_range: tuple[float, float] | None = None, f_lo: float = 35.0, f_hi: float = 16000.0,
                     gate_db: float = 6.0, presence_deadband_db: float = 2.5, report: dict | None = None,
                     label: str = "bus") -> np.ndarray:
    """Slow time-varying EQ that keeps the mix's tonal balance consistent across the song.

    Phone recordings morph over a show: in loud passages their limiters thin the low end and push harsh mids
    (measured on Early Sunsets: after 3:00 the master was -5..-2 dB below 500 Hz and +2..+4 dB at 0.5-4 kHz
    vs. its best section). Per STFT frame, 1/3-octave band shape (level-normalised) is smoothed over `smooth_s`
    and pulled `strength` of the way toward the target shape (median over the song, or the mean over
    `ref_range` seconds), capped at +-max_db, gains smoothed across frequency, overall level preserved.
    Same gains on both channels (stereo image untouched). Gains move on a ~10 s scale: no pumping.
    Level gate: full strength where the smoothed level is within `gate_db` of the song's body level, fading to 0
    over 3 dB below that, so sparse sections (intro, outro, a stem's silent stretches) keep their natural tone.
    Bands outside [f_lo, f_hi] are left alone (gain tapers to 0 one octave beyond).
    Presence/air (>= 1 kHz) is asymmetric: dull stretches are lifted fully, bright ones are only trimmed beyond
    `presence_deadband_db` above the target - pulling clear sections toward a darker median made them duller
    (Audiobox PQ -0.4 there on Early Sunsets), while lifting the dull ones raised PQ by up to +0.9.
    """
    from scipy import ndimage, signal as sg
    st = x if x.ndim == 2 else x[:, None]
    nper, hop = 4096, 1024
    f, _, Z = sg.stft(st.T, fs=sr, nperseg=nper, noverlap=nper - hop, boundary="even", padded=True)
    P = (np.abs(Z) ** 2).mean(axis=0).astype(np.float64)               # (F, T) power, channel-averaged
    centers = 1000.0 * 2.0 ** (np.arange(-15, 15) / 3.0)
    centers = centers[(centers >= f_lo) & (centers <= f_hi)]
    edges_lo, edges_hi = centers * 2 ** (-1 / 6), centers * 2 ** (1 / 6)
    B = np.stack([P[(f >= lo) & (f < hi)].sum(axis=0) for lo, hi in zip(edges_lo, edges_hi)])   # (nb, T)
    fr = sr / hop
    k = max(1, int(smooth_s * fr))
    floor = max(float(B.sum(axis=0).max()) * 1e-12, 1e-30)
    Bs = np.maximum(ndimage.uniform_filter1d(B, k, axis=1, mode="nearest"), 0.0) + floor   # running sums can dip < 0
    tot = Bs.sum(axis=0, keepdims=True)
    shape = 10 * np.log10(Bs / tot)                                    # level-normalised band shape
    loud = 10 * np.log10(tot[0])
    active = loud > np.percentile(loud, 95) - 25.0
    body = float(np.median(loud[active]))
    wgate = np.clip((loud - (body - gate_db - 3.0)) / 3.0, 0.0, 1.0)
    full = wgate >= 1.0
    if full.sum() > 10:
        active = full
    if ref_range is not None:
        t = np.arange(shape.shape[1]) / fr
        sel = active & (t >= ref_range[0]) & (t < ref_range[1])
        target = shape[:, sel].mean(axis=1) if sel.sum() > 10 else np.median(shape[:, active], axis=1)
    else:
        target = np.median(shape[:, active], axis=1)
    dev = shape - target[:, None]                                      # > 0: band too strong
    hi_b = (centers >= 1000.0)[:, None]
    dev = np.where(hi_b & (dev > 0), np.maximum(dev - presence_deadband_db, 0.0), dev)
    g = -strength * dev
    g = ndimage.uniform_filter1d(g, 3, axis=0, mode="nearest")        # ~1-octave smoothing across bands
    g = np.nan_to_num(np.clip(g, -max_db, max_db), nan=0.0) * wgate[None, :]
    g = ndimage.uniform_filter1d(g, max(1, int(2.0 * fr)), axis=1, mode="nearest")
    # keep loudness: remove the loudness-weighted mean gain per frame (K-weighting: ~+4 dB shelf above 1.5 kHz,
    # low cut below ~60 Hz), so brightening a dull stretch does not make it louder to the ear
    kw_db = 4.0 / (1.0 + (1500.0 / centers) ** 2) - 10.0 * np.log10(1.0 + (60.0 / centers) ** 4)
    w = Bs * (10 ** (kw_db / 10.0))[:, None]
    w = w / w.sum(axis=0, keepdims=True)
    g = g - (w * g).sum(axis=0, keepdims=True)
    lc = np.log2(np.concatenate([[centers[0] / 2], centers, [min(centers[-1] * 2, sr / 2)]]))
    lf = np.log2(np.maximum(f, 1.0))
    gf = np.empty((len(f), g.shape[1]), np.float32)
    for j in range(g.shape[1]):
        gf[:, j] = np.interp(lf, lc, np.concatenate([[0.0], g[:, j], [0.0]]))
    Z = Z * (10 ** (gf / 20.0))[None]
    _, y = sg.istft(Z, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=True)
    y = y.T[: len(st)].astype(np.float32)
    if report is not None:
        report.setdefault("stabilizer", {})[label] = {
            "strength": strength, "max_db": max_db, "ref": ref_range or "median", "f": [f_lo, f_hi],
            "gated_frac": round(float((wgate < 1).mean()), 3),
            "gain_p95_abs_db": round(float(np.percentile(np.abs(g[:, active]), 95)), 2)}
    return y if x.ndim == 2 else y[:, 0]


def reference_tone_match(x: np.ndarray, ref: np.ndarray, strength: float = 0.7, max_db: float = 6.0,
                         report: dict | None = None) -> np.ndarray:
    """Optional: pull the master's long-term 1/3-octave balance toward a studio recording of the same song.

    The stem targets are generic (tuned on bright live rock); a dark lo-fi or a bass-heavy production has its own
    intended tone that phone audio cannot reveal. Given the studio version, this restores it: smoothed, capped
    (+-max_db), level-neutral over 200 Hz-5 kHz, and it never boosts a band the reference itself barely has.
    """
    from .dsp import min_phase_eq_fir
    xm = x.mean(axis=1) if x.ndim == 2 else x
    rm = ref.mean(axis=1) if ref.ndim == 2 else ref
    c, lx = third_octave_levels(xm)
    _, lr = third_octave_levels(rm)
    corr = lr - lx
    mid = (c >= 200.0) & (c <= 5000.0)
    corr -= np.mean(corr[mid])
    corr = np.where(lr < np.max(lr) - 60.0, np.minimum(corr, 0.0), corr)
    corr = np.where((c < 30.0) | (c > 18000.0), 0.0, corr)
    corr = ndimage.gaussian_filter1d(corr, 1.0, mode="nearest")
    corr = np.clip(strength * corr, -max_db, max_db)
    corr -= np.mean(corr[mid])
    h = min_phase_eq_fir(np.concatenate([[0.0], c, [SR / 2]]), np.concatenate([[corr[0]], corr, [corr[-1]]]),
                         SR, n_taps=4096)
    y = apply_fir(x, h)
    if report is not None:
        report["tone_ref_eq"] = {f"{int(round(a))}": round(float(b), 1) for a, b in zip(c, corr)}
    return y.astype(np.float32)
