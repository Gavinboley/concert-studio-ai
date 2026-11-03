"""Shared DSP helpers for V8 (numpy/scipy only)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse to mono. Parametric-stereo and phone-pair stereo are not trusted."""
    if x.ndim == 1:
        return x.astype(np.float32)
    return x.mean(axis=1).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def sos_filter(x: np.ndarray, sos: np.ndarray, zero_phase: bool = True) -> np.ndarray:
    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sos_filter(x, sos)


def highpass(x: np.ndarray, sr: int, fc: float, order: int = 2, zero_phase: bool = False) -> np.ndarray:
    sos = signal.butter(order, fc, btype="high", fs=sr, output="sos")
    return sos_filter(x, sos, zero_phase=zero_phase)


def lowpass(x: np.ndarray, sr: int, fc: float, order: int = 2, zero_phase: bool = False) -> np.ndarray:
    sos = signal.butter(order, fc, btype="low", fs=sr, output="sos")
    return sos_filter(x, sos, zero_phase=zero_phase)


def db(x: float | np.ndarray, floor: float = 1e-12) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """RMS per frame (mono input)."""
    n = 1 + max(0, (len(x) - win) // hop)
    out = np.empty(n, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(np.square(x, dtype=np.float64))])
    for i in range(n):
        s = i * hop
        out[i] = (c[s + win] - c[s]) / win
    return np.sqrt(out + 1e-20)


def novelty(x: np.ndarray, sr: int, hop_s: float = 0.005, fmin: float = 100.0, fmax: float = 5000.0,
            n_bands: int = 40) -> tuple[np.ndarray, float]:
    """Log-band spectral flux onset envelope. Returns (envelope, frame_rate).

    Band-limited to fmax so HE-AAC SBR (synthetic >~6 kHz) cannot drive alignment.
    """
    hop = int(round(hop_s * sr))
    n_fft = 2048
    f, t, Z = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    mag = np.abs(Z)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    idx = np.searchsorted(f, edges)
    bands = np.stack([mag[idx[i]:max(idx[i] + 1, idx[i + 1])].sum(axis=0) for i in range(n_bands)])
    lb = np.log1p(100.0 * bands / (bands.mean() + 1e-12))
    flux = np.maximum(0.0, np.diff(lb, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    # remove slow trend, normalise
    k = int(1.0 / hop_s)
    trend = np.convolve(flux, np.ones(k) / k, mode="same")
    env = flux - trend
    env /= (np.std(env) + 1e-9)
    return env.astype(np.float32), sr / hop


def fft_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation c[k] = sum a[n+k] b[n], lags -len(b)+1 .. len(a)-1."""
    n = len(a) + len(b) - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    c = np.fft.irfft(A * np.conj(B), nfft)
    # rearrange so index 0 == lag -(len(b)-1)
    return np.concatenate([c[-(len(b) - 1):], c[:len(a)]])


def gcc_phat(x: np.ndarray, y: np.ndarray, max_lag: int, beta: float = 0.8) -> tuple[float, float]:
    """Lag (samples, fractional) of x relative to y, i.e. x[n] ~ y[n - lag], and PHAT peak height."""
    n = len(x) + len(y)
    nfft = 1 << int(np.ceil(np.log2(n)))
    X = np.fft.rfft(x * np.hanning(len(x)), nfft)
    Y = np.fft.rfft(y * np.hanning(len(y)), nfft)
    R = X * np.conj(Y)
    R /= np.abs(R) ** beta + 1e-12
    cc = np.fft.irfft(R, nfft)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    i = int(np.argmax(cc))
    peak = cc[i]
    frac = 0.0
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-12:
            frac = 0.5 * (y0 - y2) / d
    lag = i - max_lag + frac
    # normalise peak against the correlation noise floor
    floor = np.median(np.abs(cc)) + 1e-12
    return float(lag), float(peak / floor)


def fractional_delay(x: np.ndarray, delay: float) -> np.ndarray:
    """Delay signal by a (possibly fractional) number of samples via FFT phase shift."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n + abs(int(delay)) + 16)))
    X = np.fft.rfft(x, nfft, axis=0)
    k = np.arange(X.shape[0])
    ph = np.exp(-2j * np.pi * k * delay / nfft)
    if X.ndim == 2:
        ph = ph[:, None]
    y = np.fft.irfft(X * ph, nfft, axis=0)[:n]
    return y.astype(np.float32)


def equal_power_fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.cos(0.5 * np.pi * t), np.sin(0.5 * np.pi * t)


def smooth_spectrum_oct(mag_db: np.ndarray, freqs: np.ndarray, frac: float = 3.0) -> np.ndarray:
    """1/frac-octave smoothing of a dB spectrum."""
    out = np.empty_like(mag_db)
    for i, f0 in enumerate(freqs):
        if f0 <= 0:
            out[i] = mag_db[i]
            continue
        lo, hi = f0 * 2 ** (-0.5 / frac), f0 * 2 ** (0.5 / frac)
        a, b = np.searchsorted(freqs, [lo, hi])
        b = max(b, a + 1)
        out[i] = np.mean(mag_db[a:b])
    return out


def ltas_db(x: np.ndarray, sr: int, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    f, p = signal.welch(x, fs=sr, nperseg=n_fft, noverlap=n_fft // 2, average="median")
    return f, 10.0 * np.log10(p + 1e-20)


def min_phase_eq_fir(freqs: np.ndarray, gain_db: np.ndarray, sr: int, n_taps: int = 2048) -> np.ndarray:
    """Design a minimum-phase FIR matching a (smoothed) gain curve."""
    nfft = 8 * n_taps
    grid = np.linspace(0, sr / 2, nfft // 2 + 1)
    g = np.interp(grid, freqs, gain_db)
    mag = 10 ** (g / 20.0)
    # real cepstrum -> minimum phase
    logm = np.log(np.maximum(mag, 1e-6))
    cep = np.fft.irfft(logm, nfft)
    w = np.zeros(nfft)
    w[0] = 1.0
    w[1:nfft // 2] = 2.0
    w[nfft // 2] = 1.0
    H = np.exp(np.fft.rfft(cep * w, nfft))
    h = np.fft.irfft(H, nfft)[:n_taps]
    h *= np.hanning(2 * n_taps)[n_taps:]
    return h.astype(np.float32)


def linear_phase_eq_fir(freqs: np.ndarray, gain_db: np.ndarray, sr: int, n_taps: int = 4095) -> np.ndarray:
    grid = np.linspace(0, sr / 2, 2048)
    g = np.interp(grid, freqs, gain_db)
    return signal.firwin2(n_taps, grid, 10 ** (g / 20.0), fs=sr).astype(np.float32)


def apply_fir(x: np.ndarray, h: np.ndarray, linear_phase: bool = False) -> np.ndarray:
    y = signal.oaconvolve(x, h if x.ndim == 1 else h[:, None], mode="full", axes=0)
    if linear_phase:
        d = (len(h) - 1) // 2
        return y[d:d + len(x)].astype(np.float32)
    return y[:len(x)].astype(np.float32)
