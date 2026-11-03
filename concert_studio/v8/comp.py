"""Best-take comping across POVs.

Nothing is stacked: at any instant each band of the output comes from exactly one POV
(apart from short crossfades), so there is no comb filtering and no per-bin musical noise.

* lag tracks   time-varying clip<->timeline mapping refined against coherent neighbours
               (phones move: +-10 ms wander measured on this payload)
* features     per 0.5 s: crowd-to-music ratio (crowd RoFormer), occlusion (HF tilt drop),
               clipping, neighbour coherence, codec trust (HE-AACv2 = synthetic highs)
* viterbi      take path with a switching cost and a minimum dwell
* two bands    < XOVER Hz from the best take overall; > XOVER Hz only from LC (full-band)
               takes, so SBR shimmer can never reach the master
* matching     per-take loudness curve follows the consensus dynamics (kills AGC pumping),
               per-take linear-phase EQ to the spatially averaged LTAS (kills room-mode boom)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

from .dsp import (db, bandpass, ltas_db, linear_phase_eq_fir, apply_fir, resample, smooth_spectrum_oct)
from .ingest import SR, ClipInfo

GRID = 0.5           # feature grid hop (s)
LAG_HOP = 1.0        # lag-track grid hop (s)
LAG_WIN = 4.0        # lag-track window (s)
FSR = 16000          # lag measurement rate
XOVER = 4500.0       # two-band comp crossover (Hz): HE-AACv2 SBR starts ~5 kHz on this payload
XFADE = 0.25         # crossfade length (s)


@dataclass
class Track:
    name: str
    info: ClipInfo
    offset: float
    drift: float
    music: np.ndarray
    crowd: np.ndarray
    raw: np.ndarray
    lag_t: np.ndarray | None = None     # timeline grid (s)
    lag_e: np.ndarray | None = None     # correction (s): clip_t = base(t) + e(t)
    lag_ok: np.ndarray | None = None    # had a coherent measurement nearby

    @property
    def dur(self) -> float:
        return len(self.music) / SR

    def base_pos(self, t):
        return (np.asarray(t) - self.offset) / (1.0 + self.drift * 1e-6)

    def pos(self, t):
        t = np.asarray(t, dtype=np.float64)
        p = self.base_pos(t)
        if self.lag_t is not None:
            p = p + np.interp(t, self.lag_t, self.lag_e)
        return p

    def span(self) -> tuple[float, float]:
        return self.offset, self.offset + self.dur * (1.0 + self.drift * 1e-6)


# --------------------------------------------------------------------------- lag tracks
def _batched_gcc_np(a: np.ndarray, b: np.ndarray, max_lag: int):
    W = a.shape[1]
    win = np.hanning(W).astype(np.float32)
    nfft = 1 << int(np.ceil(np.log2(2 * W)))
    R = np.fft.rfft(a * win, nfft) * np.conj(np.fft.rfft(b * win, nfft))
    R = R / (np.abs(R) ** 0.8 + 1e-12)
    cc = np.fft.irfft(R, nfft)
    cc = np.concatenate([cc[:, -max_lag:], cc[:, :max_lag + 1]], axis=1)
    i = np.argmax(cc, axis=1)
    rows = np.arange(len(cc))
    pk = cc[rows, i]
    y0 = cc[rows, np.clip(i - 1, 0, cc.shape[1] - 1)]
    y2 = cc[rows, np.clip(i + 1, 0, cc.shape[1] - 1)]
    d = y0 - 2 * pk + y2
    frac = np.where(np.abs(d) > 1e-12, 0.5 * (y0 - y2) / np.where(d == 0, 1, d), 0.0)
    lag = (i - max_lag) + frac
    score = pk / (np.median(np.abs(cc), axis=1) + 1e-12)
    return lag, score


def _batched_gcc(a: np.ndarray, b: np.ndarray, max_lag: int):
    """a, b: [B, W] windows. Returns (lag samples, score) per row; lag>0: b's content is later in a."""
    try:
        import torch
    except ImportError:                                   # CPU-only install: same maths in numpy
        return _batched_gcc_np(a, b, max_lag)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.as_tensor(a, device=dev, dtype=torch.float32)
    Bt = torch.as_tensor(b, device=dev, dtype=torch.float32)
    W = A.shape[1]
    win = torch.hann_window(W, device=dev)
    nfft = 1 << int(np.ceil(np.log2(2 * W)))
    FA = torch.fft.rfft(A * win, nfft)
    FB = torch.fft.rfft(Bt * win, nfft)
    R = FA * torch.conj(FB)
    R = R / (R.abs() ** 0.8 + 1e-12)
    cc = torch.fft.irfft(R, nfft)
    cc = torch.cat([cc[:, -max_lag:], cc[:, :max_lag + 1]], dim=1)
    i = torch.argmax(cc, dim=1)
    pk = cc.gather(1, i[:, None])[:, 0]
    il = torch.clamp(i - 1, 0, cc.shape[1] - 1)
    ir = torch.clamp(i + 1, 0, cc.shape[1] - 1)
    y0, y2 = cc.gather(1, il[:, None])[:, 0], cc.gather(1, ir[:, None])[:, 0]
    d = y0 - 2 * pk + y2
    frac = torch.where(d.abs() > 1e-12, 0.5 * (y0 - y2) / d, torch.zeros_like(d))
    lag = (i - max_lag).float() + frac
    floor = cc.abs().median(dim=1).values + 1e-12
    return lag.cpu().numpy(), (pk / floor).cpu().numpy()


def build_lag_tracks(tracks: dict[str, Track], coherence: dict[tuple[str, str], float], anchor: str | list[str],
                     iters: int = 4, log=print) -> None:
    anchors = {anchor} if isinstance(anchor, str) else set(anchor)
    fine = {n: bandpass(resample(t.music, SR, FSR), FSR, 120.0, 5000.0) for n, t in tracks.items()}
    t_end = max(t.span()[1] for t in tracks.values())
    grid = np.arange(0.0, t_end, LAG_HOP)
    W = int(LAG_WIN * FSR)
    max_lag = int(0.025 * FSR)
    for t in tracks.values():
        t.lag_t, t.lag_e, t.lag_ok = grid.copy(), np.zeros(len(grid)), np.zeros(len(grid), bool)

    def windows(n: str, times: np.ndarray):
        tr = tracks[n]
        p = np.asarray(tr.pos(times), dtype=np.float64).reshape(-1)
        bad = ~np.isfinite(p)
        start = np.round((np.where(bad, -1e9, p) - LAG_WIN / 2) * FSR).astype(np.int64)
        ok = (start >= 0) & (start + W <= len(fine[n])) & ~bad
        hi = len(fine[n]) - W
        if hi < 0 or len(start) != len(times):
            log(f"  WARN lag windows {n}: len {len(fine[n])} W {W} starts {len(start)} vs times {len(times)}")
            return np.zeros((len(times), W), np.float32), np.zeros(len(times), bool)
        idx = np.clip(start, 0, hi)[:, None] + np.arange(W)[None, :]
        return fine[n][idx], ok

    names = list(tracks)
    for it in range(iters):
        moved = []
        for n in names:
            if n in anchors:
                tracks[n].lag_ok[:] = True
                continue
            tr = tracks[n]
            s0, s1 = tr.span()
            inside = (grid >= s0 + LAG_WIN) & (grid <= s1 - LAG_WIN)
            nb = sorted([m for m in names if m != n], key=lambda m: -coherence.get((n, m), 0.0))[:5]
            meas = np.full((len(nb), len(grid)), np.nan)
            wts = np.zeros((len(nb), len(grid)))
            for k, m in enumerate(nb):
                ms0, ms1 = tracks[m].span()
                both = inside & (grid >= ms0 + LAG_WIN) & (grid <= ms1 - LAG_WIN)
                if both.sum() == 0:
                    continue
                tt = grid[both]
                a, oka = windows(m, tt)
                b, okb = windows(n, tt)
                lag, sc = _batched_gcc(a, b, max_lag)
                good = oka & okb & (sc > 7.5) & tracks[m].lag_ok[both]
                # neighbour m sees n's content `lag` samples later -> n's position is too early
                meas[k, np.where(both)[0][good]] = -lag[good] / FSR
                wts[k, np.where(both)[0][good]] = sc[good]
            # weighted median per grid point
            delta = np.zeros(len(grid))
            have = np.zeros(len(grid), bool)
            for j in np.where(inside)[0]:
                v = ~np.isnan(meas[:, j])
                if v.sum() == 0:
                    continue
                vals, w = meas[v, j], wts[v, j]
                o = np.argsort(vals)
                cw = np.cumsum(w[o])
                delta[j] = vals[o][np.searchsorted(cw, 0.5 * cw[-1])]
                have[j] = True
            if have.sum() < 3:
                tr.lag_ok[:] = False
                continue
            # robust smoothing of the correction, then fill gaps by interpolation
            idx = np.where(have)[0]
            dd = ndimage.median_filter(delta[idx], size=min(7, len(idx) | 1), mode="nearest")
            full = np.interp(np.arange(len(grid)), idx, dd)
            full = ndimage.uniform_filter1d(full, size=5, mode="nearest")
            tr.lag_e = tr.lag_e + full
            near = ndimage.binary_dilation(have, iterations=2)
            tr.lag_ok = near & inside
            moved.append(np.median(np.abs(full[have])) * 1000)
        log(f"  lag iter {it + 1}: median |update| {np.median(moved) if moved else 0:.3f} ms over {len(moved)} clips")
    _trust_solo_regions(tracks, grid, anchors, log)


def _trust_solo_regions(tracks: dict[str, Track], grid: np.ndarray, anchors: set, log=print,
                        edge_s: float = 0.5) -> None:
    """Where no other clip covers a moment, nothing can confirm or contradict a clip's timing: trust its global
    alignment there (lag carried from its nearest validated point). Without this, sparse payloads lose every
    stretch that only one phone filmed (measured: 3 staggered phones -> 74% of the song kept, not 100%).
    Moments where other clips exist but disagree stay invalid (splices, a different song)."""
    spans = {n: tr.span() for n, tr in tracks.items()}
    total = 0.0
    for n, tr in tracks.items():
        if n in anchors:
            continue
        s0, s1 = spans[n]
        inside = (grid >= s0 + edge_s) & (grid <= s1 - edge_s)
        others = np.zeros(len(grid), int)
        for m, (m0, m1) in spans.items():
            if m != n:
                others += ((grid >= m0) & (grid <= m1)).astype(int)
        solo = inside & (others == 0) & ~tr.lag_ok
        if not solo.any():
            continue
        ok = np.where(tr.lag_ok)[0]
        if len(ok):
            tr.lag_e = np.where(solo, np.interp(np.arange(len(grid)), ok, tr.lag_e[ok]), tr.lag_e)
        tr.lag_ok = tr.lag_ok | solo
        total += solo.sum() * (grid[1] - grid[0] if len(grid) > 1 else 0.0)
    if total:
        log(f"  solo stretches trusted (no other clip there): {total:.1f} s")


# --------------------------------------------------------------------------- features
def _band_energy(x: np.ndarray, sr: int, lo: float, hi: float, win: int, hop: int) -> np.ndarray:
    f, t, Z = signal.stft(x, fs=sr, nperseg=win, noverlap=win - hop, boundary=None, padded=False)
    m = (f >= lo) & (f < hi)
    return (np.abs(Z[m]) ** 2).sum(axis=0)


def compute_features(tracks: dict[str, Track], t_end: float) -> dict:
    grid = np.arange(0.0, t_end, GRID)
    feats = {}
    win, hop = 2048, 1024
    for n, tr in tracks.items():
        # per-clip frame features on the clip's own clock, then sampled onto the timeline grid
        m_e = _band_energy(tr.music, SR, 60, 16000, win, hop) + 1e-12
        c_e = _band_energy(tr.crowd, SR, 60, 16000, win, hop) + 1e-12
        lo = _band_energy(tr.music, SR, 200, 1000, win, hop) + 1e-12
        hi = _band_energy(tr.music, SR, 3000, 6000, win, hop) + 1e-12
        hf_m = _band_energy(tr.music, SR, XOVER, 15000, win, hop) + 1e-12
        hf_c = _band_energy(tr.crowd, SR, XOVER, 15000, win, hop) + 1e-12
        fr = SR / hop
        k = int(2.0 * fr)
        sm = lambda v: ndimage.uniform_filter1d(v, k, mode="nearest")
        cr = 10 * np.log10(sm(c_e) / sm(m_e))                    # crowd-to-music (dB)
        tilt = 10 * np.log10(sm(hi) / sm(lo))
        occl = np.maximum(0.0, np.median(tilt) - tilt)           # dB of HF lost vs this clip's norm
        hcr = 10 * np.log10(sm(hf_c) / sm(hf_m))
        loud = 10 * np.log10(sm(m_e))
        loud_hf = 10 * np.log10(sm(hf_m))
        if tr.raw is not None:
            a = np.abs(tr.raw)
        else:                                                    # lazy: keeps 57-take comps under the RAM cap
            from .ingest import load_mono
            a = np.abs(load_mono(tr.info))
        clip_fr = ndimage.uniform_filter1d((a[:len(a) // hop * hop].reshape(-1, hop) > 0.985).mean(axis=1),
                                           k, mode="nearest")
        pos = tr.pos(grid)
        fi = np.clip((pos * fr).astype(int), 0, len(cr) - 1)
        # TikTok/phone exports carry end cards and transition sounds at the very edges
        inside = (pos >= 0.5) & (pos <= tr.dur - 2.0)
        ok = inside & (np.interp(grid, tr.lag_t, tr.lag_ok.astype(float)) > 0.5)
        lg, crg = loud[fi], cr[fi]
        ext = _extend_edges(ok, inside, lg, crg)
        feats[n] = dict(cr=crg, occl=occl[fi], hcr=hcr[fi], loud=lg, loud_hf=loud_hf[fi],
                        clip=clip_fr[np.clip(fi, 0, len(clip_fr) - 1)], ok=ok | ext, ext=ext)
    return {"grid": grid, "f": feats}


def _extend_edges(ok: np.ndarray, inside: np.ndarray, loud: np.ndarray, cr: np.ndarray,
                  max_s: float = 15.0, drop_db: float = 12.0) -> np.ndarray:
    """Grow a take's validated region outward over stretches no neighbour could confirm.

    Only while the take is still clearly music (level within `drop_db` of its validated median and
    not crowd-dominated), and at most `max_s` per edge. Marked separately so scoring can penalise it.
    """
    ext = np.zeros_like(ok)
    if ok.sum() < 10:
        return ext
    ref = np.median(loud[ok])
    musical = inside & (loud >= ref - drop_db) & (cr < 0.0)
    idx = np.where(ok)[0]
    lim = int(max_s / GRID)
    for start, step in ((idx[0] - 1, -1), (idx[-1] + 1, 1)):
        j, k = start, 0
        while 0 <= j < len(ok) and k < lim and musical[j] and not ok[j]:
            ext[j] = True
            j += step
            k += 1
    return ext


def reject_level_anomalies(tracks: dict[str, Track], feats: dict, consensus_db: float = 9.0,
                           burst_db: float = 18.0, log=print) -> None:
    """Invalidate windows whose level is implausible: far off the cross-take consensus (when the
    band gets loud every phone gets loud), or a burst far above the take's own running median
    (end-card swooshes, mic knocks; also covers single-coverage stretches)."""
    names = list(tracks)
    L = np.stack([np.where(feats["f"][n]["ok"], feats["f"][n]["loud"], np.nan) for n in names])
    with np.errstate(all="ignore"):
        med_t = np.nanmedian(L, axis=0)
        off = np.array([np.nanmedian(L[i] - med_t) if np.isfinite(L[i]).any() else 0.0 for i in range(len(names))])
        cons = np.nanmedian(L - off[:, None], axis=0)
        cnt = np.sum(np.isfinite(L), axis=0)
    k = int(10.0 / GRID)
    before = np.stack([feats["f"][n]["ok"].copy() for n in names])
    devs = np.full(L.shape, np.inf)
    for i, n in enumerate(names):
        f = feats["f"][n]
        dev = (L[i] - off[i]) - cons
        devs[i] = np.where(np.isfinite(dev), np.abs(dev), np.inf)
        bad_c = np.isfinite(dev) & (cnt >= 3) & (np.abs(dev) > consensus_db)
        # self-relative bursts only where no consensus exists (a real crescendo lifts every take)
        lo = f["loud"]
        run_med = ndimage.median_filter(lo, size=k | 1, mode="nearest")
        bad_b = f["ok"] & (cnt < 3) & (lo - run_med > burst_db)
        bad = ndimage.binary_dilation(bad_c | bad_b, iterations=2)
        f["ok"] = f["ok"] & ~bad
    # never open a gap: where rejection left nothing, restore the least-anomalous take
    after = np.stack([feats["f"][n]["ok"] for n in names])
    hole = before.any(axis=0) & ~after.any(axis=0)
    for t in np.where(hole)[0]:
        cand = np.where(before[:, t])[0]
        i = int(cand[np.argmin(np.where(np.isfinite(devs[cand, t]), devs[cand, t], 0.0))])
        feats["f"][names[i]]["ok"][t] = True
    total = int((before & ~np.stack([feats["f"][n]["ok"] for n in names])).sum())
    for n in names:
        feats["f"][n]["ext"] = feats["f"][n]["ext"] & feats["f"][n]["ok"]
    log(f"  gap guard restored {int(hole.sum()) * GRID:.1f}s")
    log(f"  level-anomaly rejection: {total * GRID:.1f} take-seconds invalidated")


def vocal_band_coherence(tracks: dict[str, Track], feats: dict, win_s: float = 5.0, hop_s: float = 2.5,
                         lo: float = 300.0, hi: float = 3000.0, log=print) -> dict[str, np.ndarray]:
    """Time-varying vocal-band coherence of each take against the others, on the feature grid.

    Every phone hears the PA; what only one phone hears (a fan singing or talking next to it, its own
    handling noise, a long reverberant path) is incoherent with the rest. Median magnitude-squared
    coherence (300-3000 Hz, 5 s windows) against all other covering takes; NaN where < 3 takes cover.
    """
    from scipy.signal import coherence as msc
    fs = 16000
    grid = feats["grid"]
    names = list(tracks)
    low = {n: resample(t.music, SR, fs) for n, t in tracks.items()}
    centers = np.arange(grid[0] + win_s / 2, grid[-1] - win_s / 2, hop_s)
    C = np.full((len(names), len(centers)), np.nan)
    L = int(win_s * fs)
    for j, c in enumerate(centers):
        segs = {}
        for i, n in enumerate(names):
            f = feats["f"][n]
            k = int(round(c / GRID))
            if k >= len(grid) or not f["ok"][k] or f["ext"][k]:
                continue
            p0 = float(tracks[n].pos(c - win_s / 2))
            a = int(p0 * fs)
            if a < 0 or a + L > len(low[n]):
                continue
            segs[i] = low[n][a:a + L]
        if len(segs) < 3:
            continue
        ks = sorted(segs)
        pair = {}
        for x_ in range(len(ks)):
            for y_ in range(x_ + 1, len(ks)):
                f_, cxy = msc(segs[ks[x_]], segs[ks[y_]], fs, nperseg=1024)
                pair[(ks[x_], ks[y_])] = pair[(ks[y_], ks[x_])] = float(np.mean(cxy[(f_ > lo) & (f_ < hi)]))
        for i in ks:
            C[i, j] = np.median([pair[(i, m)] for m in ks if m != i])
    out = {}
    for i, n in enumerate(names):
        r = C[i]
        ok = np.isfinite(r)
        out[n] = np.interp(grid, centers[ok], r[ok], left=np.nan, right=np.nan) if ok.sum() >= 2 else \
            np.full(len(grid), np.nan)
    log(f"  vocal-band coherence: {np.isfinite(C).sum()} take-windows, median {np.nanmedian(C):.3f}")
    return out


FAST_H = 0.25


def fast_level_curves(tracks: dict[str, Track], feats: dict):
    """1 s loudness curves (0.25 s hop) of every take on the timeline + robust consensus. Returns T, curves, cons, cnt."""
    grid = feats["grid"]
    H = FAST_H
    T = np.arange(grid[0], grid[-1] + GRID, H)
    names = list(tracks)
    h = int(H * SR)
    curves = np.full((len(names), len(T)), np.nan)
    for i, n in enumerate(names):
        x = tracks[n].music
        m = len(x) // h
        e = ndimage.uniform_filter1d(np.mean(x[:m * h].reshape(m, h) ** 2, axis=1), 4, mode="nearest")
        lv = 10 * np.log10(e + 1e-12)
        p = np.asarray(tracks[n].pos(T), dtype=float)
        k = np.round(p / H - 0.5).astype(int)
        okp = (p >= 0.5) & (p <= tracks[n].dur - 2.0) & (k >= 0) & (k < m)
        okg = np.interp(T, grid, feats["f"][n]["ok"].astype(float)) > 0.5
        v = np.where(okp & okg, lv[np.clip(k, 0, m - 1)], np.nan)
        if np.isfinite(v).sum() > 40:
            v = np.where(v > np.nanmax(v) - 40, v, np.nan)
        curves[i] = v
    off = np.zeros(len(names))
    with np.errstate(all="ignore"):
        for _ in range(4):
            cons = np.nanmedian(curves - off[:, None], axis=0)
            off = np.array([np.nanmedian(c - cons) if np.isfinite(c - cons).any() else 0.0 for c in curves])
        cons = np.nanmedian(curves - off[:, None], axis=0)
    cnt = np.sum(np.isfinite(curves), axis=0)
    return T, curves, cons, cnt


def fast_consensus_lock(x: np.ndarray, T: np.ndarray, cons: np.ndarray, cnt: np.ndarray, strength: float = 0.8,
                        smooth_s: float = 1.0, max_db: float = 4.0, min_takes: int = 4, log=print) -> np.ndarray:
    """Second-stage lock at ~1 s: remove most of the residual fast level wobble (phone AGC/limiter pumping of
    the chosen takes) by pulling the render toward the cross-POV 1 s consensus. Split-half validated on the
    56-take Early Sunsets comp: held-out fast deviation 1.15 -> 0.98 dB at strength 0.8 (mid/slow also improve).
    strength < 1 keeps part of a close phone's genuine extra punch. x is on the timeline from t=0."""
    h = int(FAST_H * SR)
    m = len(x) // h
    e = ndimage.uniform_filter1d(np.mean(x[:m * h].reshape(m, h) ** 2, axis=1), 4, mode="nearest")
    tx = (np.arange(m) + 0.5) * FAST_H
    xl = np.interp(T, tx, 10 * np.log10(e + 1e-12), left=np.nan, right=np.nan)
    good = (cnt >= min_takes) & np.isfinite(cons) & np.isfinite(xl) & (xl > np.nanmax(xl) - 40)
    if good.sum() < 40:
        return x
    d = np.where(good, cons - xl, np.nan)
    d -= np.nanmedian(d)
    idx = np.where(good)[0]
    full = np.interp(np.arange(len(d)), idx, d[idx])
    k = max(3, int(smooth_s / FAST_H))
    full = ndimage.median_filter(full, size=k | 1, mode="nearest")
    full = ndimage.uniform_filter1d(full, size=k, mode="nearest")
    w = ndimage.uniform_filter1d(good.astype(float), size=4 * k, mode="nearest")
    full = np.clip(full * w * strength, -max_db, max_db)
    log(f"  fast consensus lock ({smooth_s:.0f} s, x{strength}): correction p50 {np.median(np.abs(full[good])):.2f} dB, "
        f"p95 {np.percentile(np.abs(full[good]), 95):.2f} dB")
    g = 10 ** (np.interp(np.arange(len(x)) / SR, T, full) / 20.0)
    return (x * g.astype(np.float32)).astype(np.float32)


def take_pumping(tracks: dict[str, Track], feats: dict, win_s: float = 10.0, log=print) -> dict[str, np.ndarray]:
    """Time-varying AGC pumping per take: incoherent fast (<3 s) level wobble against the cross-POV consensus.

    1 s loudness curves (0.25 s hop) of every take's de-crowded music on the timeline; consensus = median with
    robust per-take offsets. Per take, the fast part of its curve is regressed on the consensus's fast part:
    the slope keeps genuine extra dynamics (a close phone, slope > 1), the residual is wobble no other phone
    heard (limiter/AGC pumping). Returns the rolling RMS residual (dB) on the feature grid (NaN = no data).
    Measured: the top-PQ take on the Early Sunsets payload had residual 1.53 dB vs a 0.58 dB median.
    """
    grid = feats["grid"]
    names = list(tracks)
    T, curves, cons, cnt = fast_level_curves(tracks, feats)
    H = FAST_H
    k3 = int(3.0 / H)

    def fast(v, m):
        idx = np.where(m)[0]
        if len(idx) < 20:
            return None
        d = np.interp(np.arange(len(v)), idx, v[idx])
        return d - ndimage.uniform_filter1d(d, k3, mode="nearest")

    good_c = cnt >= 4
    cf = fast(cons, good_c & np.isfinite(cons))
    out = {}
    for i, n in enumerate(names):
        m = good_c & np.isfinite(curves[i])
        tf = fast(curves[i], m)
        if tf is None or cf is None:
            out[n] = np.full(len(grid), np.nan)
            continue
        slope = float(np.sum(tf[m] * cf[m]) / max(np.sum(cf[m] ** 2), 1e-9))
        r2 = np.where(m, (tf - slope * cf) ** 2, np.nan)
        num = ndimage.uniform_filter1d(np.nan_to_num(r2), int(win_s / H), mode="nearest")
        den = ndimage.uniform_filter1d(m.astype(float), int(win_s / H), mode="nearest")
        roll = np.where(den > 0.3, np.sqrt(num / np.maximum(den, 1e-9)), np.nan)
        out[n] = np.interp(grid, T, roll, left=np.nan, right=np.nan)
    allv = np.concatenate([v[np.isfinite(v)] for v in out.values()]) if out else np.array([np.nan])
    log(f"  take pumping (fast incoherent level wobble): median {np.nanmedian(allv):.2f} dB, p90 {np.nanpercentile(allv, 90):.2f} dB")
    return out


def quality_on_grid(q: dict, names: list[str], grid: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Interpolate take_quality.npz (window centers on the timeline) onto the feature grid."""
    qn = list(q["names"])
    out = {}
    for n in names:
        if n not in qn:
            continue
        i = qn.index(n)
        d = {}
        for k in ("PQ", "CE"):
            r = q[k][i].astype(float)
            ok = np.isfinite(r)
            d[k] = np.interp(grid, q["centers"][ok], r[ok], left=np.nan, right=np.nan) if ok.sum() >= 2 else \
                np.full(len(grid), np.nan)
        out[n] = d
    return out


def take_scores(tracks: dict[str, Track], feats: dict, coherence_best: dict[str, float],
                quality: dict | None = None, vcoh: dict | None = None,
                w_pq: float = 0.0, w_ce: float = 0.0, w_coh: float = 0.0,
                pump: dict | None = None, w_pump: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (score_lowband[N,T], score_highband[N,T]) with -inf where unusable.

    Units: dB-like per 0.5 s step (crowd-to-music ratio is the base term). Optional perceptual terms:
    w_pq / w_ce per point of Audiobox PQ / CE above the all-take median, w_coh per 0.1 of vocal-band
    coherence above the median. Where a take has no score (edges), its own median is used.
    """
    names = list(tracks)
    T = len(feats["grid"])
    S = np.full((len(names), T), -np.inf)
    H = np.full((len(names), T), -np.inf)

    def _rel(series: dict, key: str | None):
        vals = [(series[n][key] if key else series[n]) for n in names if n in series]
        allv = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.array([0.0])
        ref = float(np.median(allv)) if len(allv) else 0.0
        out = {}
        for n in names:
            if n not in series:
                out[n] = np.zeros(T)
                continue
            v = series[n][key] if key else series[n]
            own = float(np.nanmedian(v)) if np.isfinite(v).any() else ref
            out[n] = np.where(np.isfinite(v), v, own) - ref
        return out

    pq = _rel(quality, "PQ") if (quality and w_pq) else None
    ce = _rel(quality, "CE") if (quality and w_ce) else None
    vc = _rel(vcoh, None) if (vcoh and w_coh) else None
    pm = _rel(pump, None) if (pump and w_pump) else None
    for i, n in enumerate(names):
        f = feats["f"][n]
        info = tracks[n].info
        s = -1.0 * f["cr"] - 1.5 * f["occl"] - 400.0 * f["clip"]
        s += 0.15 * min(coherence_best.get(n, 10.0), 30.0)        # closer / drier POVs cohere better
        if info.sbr:
            s -= 4.0                                               # 32 kbps HE-AACv2 core is coarse
        s = s - 8.0 * f["ext"]                                     # unconfirmed edge: gap filler only
        perc = np.zeros(T)
        if pq is not None:
            perc += w_pq * pq[n]
        if ce is not None:
            perc += w_ce * ce[n]
        if vc is not None:
            perc += w_coh * 10.0 * vc[n]
        if pm is not None:
            perc -= w_pump * np.maximum(pm[n], 0.0)      # only penalise wobble above the typical take
        s = np.where(f["ok"], s + perc, -np.inf)
        S[i] = s
        if not info.sbr:
            h = -1.0 * f["hcr"] - 1.5 * f["occl"] - 400.0 * f["clip"] - 8.0 * f["ext"] + perc
            H[i] = np.where(f["ok"], h, -np.inf)
    return S, H


def viterbi(S: np.ndarray, switch_cost: float, min_dwell: int) -> np.ndarray:
    """Max-score take path. S[N,T] with -inf where a take is unusable. Returns take per step (-1 = gap)."""
    N, T = S.shape
    path = np.full(T, -1)
    avail = np.isfinite(S).any(axis=0)
    t = 0
    while t < T:
        if not avail[t]:
            t += 1
            continue
        r0 = t
        while t < T and avail[t]:
            t += 1
        r1 = t
        L = r1 - r0
        acc = np.where(np.isfinite(S[:, r0]), S[:, r0], -np.inf)
        back = np.zeros((L, N), dtype=np.int32)
        back[0] = np.arange(N)
        for k in range(1, L):
            col = S[:, r0 + k]
            bp = int(np.argmax(acc))
            sw = acc[bp] - switch_cost
            stay_better = acc >= sw
            prev = np.where(stay_better, acc, sw)
            back[k] = np.where(stay_better, np.arange(N), bp)
            acc = np.where(np.isfinite(col), prev + col, -np.inf)
        i = int(np.argmax(acc))
        for k in range(L - 1, -1, -1):
            path[r0 + k] = i
            i = int(back[k][i])
    # minimum dwell: absorb short runs into the better neighbour that is usable there
    for _ in range(50):
        runs, s0 = [], 0
        for k in range(1, T + 1):
            if k == T or path[k] != path[s0]:
                runs.append((s0, k, int(path[s0])))
                s0 = k
        changed = False
        for idx, (a, b, c) in enumerate(runs):
            if c < 0 or b - a >= min_dwell:
                continue
            cands = []
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(runs) and runs[j][2] >= 0 and np.isfinite(S[runs[j][2], a:b]).all():
                    cands.append((float(S[runs[j][2], a:b].sum()), runs[j][2]))
            if cands:
                path[a:b] = max(cands)[1]
                changed = True
                break
        if not changed:
            break
    return path


# --------------------------------------------------------------------------- matching + render
def eq_match_filters(tracks: dict[str, Track], feats: dict, max_db: float = 9.0) -> dict[str, np.ndarray]:
    """Linear-phase FIR per take mapping its LTAS onto the spatial-average LTAS of the LC takes."""
    grid = feats["grid"]
    lt = {}
    for n, tr in tracks.items():
        ok = feats["f"][n]["ok"]
        if ok.sum() < 20:
            continue
        pos = tr.pos(grid[ok])
        # sample ~60 s of valid material spread through the clip
        sel = pos[:: max(1, len(pos) // 60)]
        chunks = [tr.music[int(p * SR):int(p * SR) + SR] for p in sel if 0 <= p * SR < len(tr.music) - SR]
        if not chunks:
            continue
        f, L = ltas_db(np.concatenate(chunks), SR)
        lt[n] = L
    lc = [n for n in lt if not tracks[n].info.sbr]
    target = np.median(np.stack([lt[n] for n in (lc or list(lt))]), axis=0)
    filt = {}
    for n, L in lt.items():
        g = smooth_spectrum_oct(target - L, f, frac=3.0)
        g -= np.interp(1000.0, f, g)                     # loudness matching is done separately
        g = np.clip(g, -max_db, max_db)
        g[f < 40] = g[np.searchsorted(f, 40)]
        if tracks[n].info.sbr:
            g = np.where(f > 6000, np.minimum(g, 0.0), g)  # never boost synthetic SBR highs
        else:
            g = np.where(f > 15000, np.minimum(g, 0.0), g)
        filt[n] = linear_phase_eq_fir(f, g, SR, n_taps=4095)
    return filt


def loudness_gains(tracks: dict[str, Track], feats: dict, key: str = "loud") -> dict[str, np.ndarray]:
    """Per take gain curve (dB, on the feature grid) so every take follows the consensus dynamics."""
    names = list(tracks)
    L = np.stack([np.where(feats["f"][n]["ok"], feats["f"][n][key], np.nan) for n in names])
    # each take's own level offset vs the others (robust, over shared time)
    med_t = np.nanmedian(L, axis=0)
    off = np.array([np.nanmedian(L[i] - med_t) if np.isfinite(L[i]).any() else 0.0 for i in range(len(names))])
    Ln = L - off[:, None]
    consensus = np.nanmedian(Ln, axis=0)
    consensus = ndimage.uniform_filter1d(np.nan_to_num(consensus, nan=np.nanmedian(consensus)), 3, mode="nearest")
    gains = {}
    k = int(6.0 / GRID)
    for i, n in enumerate(names):
        d = consensus - L[i]
        d = np.where(np.isfinite(d), d, np.nan)
        if not np.isfinite(d).any():
            gains[n] = np.zeros(len(consensus))
            continue
        idx = np.where(np.isfinite(d))[0]
        full = np.interp(np.arange(len(d)), idx, d[idx])
        full = ndimage.median_filter(full, size=k | 1, mode="nearest")
        full = ndimage.uniform_filter1d(full, size=k, mode="nearest")
        gains[n] = np.clip(full, -18.0, 18.0)
    return gains


def _sinc_read(x: np.ndarray, pos: np.ndarray, taps: int = 16) -> np.ndarray:
    """Windowed-sinc fractional read x(pos) (pos in samples)."""
    out = np.zeros(len(pos), np.float32)
    half = taps // 2
    k = np.arange(-half + 1, half + 1)
    win_beta = 8.0
    for s in range(0, len(pos), 1 << 18):
        p = pos[s:s + (1 << 18)]
        i0 = np.floor(p).astype(np.int64)
        fr = p - i0
        d = fr[:, None] - k[None, :]
        w = np.sinc(d) * np.kaiser(taps, win_beta)[None, :]
        w /= w.sum(axis=1, keepdims=True)
        idx = i0[:, None] + k[None, :]
        valid = (idx >= 0) & (idx < len(x))
        g = x[np.clip(idx, 0, len(x) - 1)] * valid
        out[s:s + len(p)] = (g * w).sum(axis=1)
    return out


def render_path(tracks: dict[str, Track], names: list[str], path: np.ndarray, grid: np.ndarray, src: str,
                filt: dict[str, np.ndarray], gains: dict[str, np.ndarray], n_out: int,
                band: tuple[float, float] | None = None, power: bool = False, match_s: float = 1.5,
                glide_s: float = 6.0, log=None) -> tuple[np.ndarray, np.ndarray]:
    """Render the take path onto the timeline with Hann crossfades centred on each switch.

    Two passes: (1) render every used take, aligned, with a margin around its active region;
    (2) at each switch measure both takes (in `band`) either side of the crossfade and glide each
    by half the residual level difference over `glide_s`, so the fade meets in the middle.
    `power=True` normalises by sqrt(sum w^2) (uncorrelated content, e.g. the >7 kHz band).
    """
    t_out = np.arange(n_out) / SR
    used = sorted(set(int(c) for c in path if c >= 0))
    xf = int(XFADE * SR)
    hop = int(GRID * SR)
    ker = np.hanning(xf + 2)[1:-1]
    ker /= ker.sum()
    margin = int((match_s + XFADE) * SR)
    W, Y, R = {}, {}, {}
    for c in used:
        n = names[c]
        tr = tracks[n]
        act = np.repeat((path == c).astype(np.float32), hop)[:n_out]
        act = np.pad(act, (0, max(0, n_out - len(act))))
        w = np.clip(signal.oaconvolve(act, ker, mode="same"), 0.0, 1.0).astype(np.float32)
        nz = np.where(w > 1e-4)[0]
        if len(nz) == 0:
            continue
        lo, hi = max(0, nz[0] - margin), min(n_out, nz[-1] + 1 + margin)
        x = getattr(tr, src)
        if n in filt:
            x = apply_fir(x, filt[n], linear_phase=True)
        pos = tr.pos(t_out[lo:hi]) * SR
        y = _sinc_read(x, pos)
        valid = (pos >= 0) & (pos < len(x) - 1)
        g = 10 ** (np.interp(t_out[lo:hi], grid, gains[names[c]]) / 20.0)
        W[c], Y[c], R[c] = w, (g.astype(np.float32) * y * valid), (lo, hi)
    # switch residual matching
    corr = {c: np.zeros(n_out, np.float32) for c in W}
    sos = None
    if band is not None:
        lo_f, hi_f = band
        if lo_f <= 0:
            sos = signal.butter(4, hi_f, "low", fs=SR, output="sos")
        elif hi_f >= SR / 2:
            sos = signal.butter(4, lo_f, "high", fs=SR, output="sos")
        else:
            sos = signal.butter(4, [lo_f, hi_f], "band", fs=SR, output="sos")
    sw = [k for k in range(1, len(path)) if path[k] != path[k - 1] and path[k] >= 0 and path[k - 1] >= 0]
    residuals = []
    for k in sw:
        c1, c2 = int(path[k - 1]), int(path[k])
        if c1 not in Y or c2 not in Y:
            continue
        ts = k * hop
        a0, a1 = ts - int(match_s * SR), ts + int(match_s * SR)
        (l1, h1), (l2, h2) = R[c1], R[c2]
        a0 = max(a0, l1, l2)
        a1 = min(a1, h1, h2)
        if a1 - a0 < int(0.5 * SR):
            continue
        y1 = Y[c1][a0 - l1:a1 - l1]
        y2 = Y[c2][a0 - l2:a1 - l2]
        if sos is not None:
            y1, y2 = signal.sosfilt(sos, y1), signal.sosfilt(sos, y2)
        r = float(db(np.sqrt(np.mean(y1 ** 2)) + 1e-12) - db(np.sqrt(np.mean(y2 ** 2)) + 1e-12))
        r = float(np.clip(r, -12.0, 12.0))
        residuals.append(r)
        gl = int(glide_s * SR)
        # incoming take: +r/2 at the switch, gliding to 0 after; outgoing: -r/2 at the switch, from 0 before
        i0, i1 = ts, min(n_out, ts + gl)
        ramp_in = np.cos(np.linspace(0, np.pi / 2, i1 - i0)) ** 2
        corr[c2][i0:i1] += (r / 2) * ramp_in
        pre0 = max(0, ts - gl)
        ramp_out = np.sin(np.linspace(0, np.pi / 2, ts - pre0)) ** 2
        corr[c1][pre0:ts] += (-r / 2) * ramp_out
        # hold the half-way levels through the crossfade itself
        h0, h1 = max(0, ts - xf), min(n_out, ts + xf)
        corr[c2][h0:ts] += r / 2
        corr[c1][ts:h1] += -r / 2
    if log and residuals:
        ra = np.abs(residuals)
        log(f"    switch residuals ({src}, band={band}): median {np.median(ra):.2f} dB, p90 {np.percentile(ra, 90):.2f} dB")
    out = np.zeros(n_out, np.float32)
    weight = np.zeros(n_out, np.float32)
    for c in W:
        lo, hi = R[c]
        w = W[c][lo:hi]
        g = (10 ** (corr[c][lo:hi] / 20.0)).astype(np.float32)
        out[lo:hi] += w * g * Y[c]
        weight[lo:hi] += w ** 2 if power else w
    norm = np.sqrt(weight) if power else weight
    full = np.zeros(n_out, np.float32)
    for c in W:
        full += W[c]
    return out / np.maximum(norm, 1e-3), np.clip(full, 0.0, 1.0)


def split_bands(x: np.ndarray, fc: float = XOVER) -> tuple[np.ndarray, np.ndarray]:
    h = signal.firwin(1023, fc, fs=SR).astype(np.float32)
    lo = apply_fir(x, h, linear_phase=True)
    return lo, x - lo


# --------------------------------------------------------------------------- time-varying consensus EQ
TV_NFFT = 4096
TV_HOP = 1024
TV_BANDS = 1000.0 * 2.0 ** (np.arange(-15, 14) / 3.0)      # 31.5 Hz .. 20 kHz centres


def _band_matrix(freqs: np.ndarray) -> np.ndarray:
    """Triangular 1/3-octave weights [bands, bins]."""
    lf = np.log2(np.maximum(freqs, 1.0))
    lc = np.log2(TV_BANDS)
    W = np.maximum(0.0, 1.0 - np.abs(lf[None, :] - lc[:, None]) * 3.0)
    return W / (W.sum(axis=1, keepdims=True) + 1e-12)


def timevarying_consensus_eq(tracks: dict[str, Track], feats: dict, used: set[str], win_s: float = 8.0,
                             max_db: float = 9.0, log=print) -> None:
    """Give every used take the consensus *spectral shape* of all takes valid at that moment.

    Smooth in time (~win_s) and frequency (1/3 oct), applied as STFT magnitude gains: no per-bin
    decisions, so no musical noise; zero-phase, so alignment is untouched. Sets `tr.music_eq`.
    """
    grid = feats["grid"]
    names = list(tracks)
    f = np.fft.rfftfreq(TV_NFFT, 1.0 / SR)
    Bm = _band_matrix(f)
    win = np.hanning(TV_NFFT).astype(np.float32)
    shapes = {}
    for n in names:
        tr = tracks[n]
        _, _, Z = signal.stft(tr.music, fs=SR, window=win, nperseg=TV_NFFT, noverlap=TV_NFFT - TV_HOP,
                              boundary=None, padded=False)
        P = (np.abs(Z) ** 2).astype(np.float32)
        E = 10 * np.log10(Bm @ P + 1e-14)                       # [bands, frames] dB
        fr = SR / TV_HOP
        k = max(1, int(2.0 * fr))
        E = ndimage.uniform_filter1d(E, k, axis=1, mode="nearest")
        pos = tr.pos(grid)
        fi = np.clip((pos * fr).astype(int), 0, E.shape[1] - 1)
        S = E[:, fi]                                            # [bands, grid]
        S = S - S[(TV_BANDS >= 250) & (TV_BANDS <= 4000)].mean(axis=0, keepdims=True)
        ok = feats["f"][n]["ok"]
        shapes[n] = np.where(ok[None, :], S, np.nan)
        del Z, P, E
    stack = np.stack([shapes[n] for n in names])                # [takes, bands, grid]
    lc_mask = np.array([not tracks[n].info.sbr for n in names])
    with np.errstate(all="ignore"):
        cons_all = np.nanmedian(stack, axis=0)
        cons_lc = np.nanmedian(stack[lc_mask], axis=0) if lc_mask.any() else cons_all
    hi_b = TV_BANDS > 6000
    cons = np.where(hi_b[:, None] & np.isfinite(cons_lc), cons_lc, cons_all)
    kt = int(win_s / GRID)
    for n in used:
        tr = tracks[n]
        S = shapes[n]
        G = cons - S                                            # [bands, grid]
        valid = np.isfinite(G)
        # fill invalid times by holding the nearest valid correction, then smooth time + frequency
        for b in range(G.shape[0]):
            v = np.where(valid[b])[0]
            G[b] = np.interp(np.arange(G.shape[1]), v, G[b, v]) if len(v) else 0.0
        G = ndimage.median_filter(G, size=(1, kt | 1), mode="nearest")
        G = ndimage.uniform_filter1d(G, kt, axis=1, mode="nearest")
        G = ndimage.gaussian_filter1d(G, 1.0, axis=0, mode="nearest")
        G = np.clip(G, -max_db, max_db)
        if tr.info.sbr:
            G[hi_b] = np.minimum(G[hi_b], 0.0)                  # never boost synthetic SBR highs
        G[TV_BANDS > 16000] = np.minimum(G[TV_BANDS > 16000], 0.0)
        # apply on the clip's own STFT frames
        _, _, Z = signal.stft(tr.music, fs=SR, window=win, nperseg=TV_NFFT, noverlap=TV_NFFT - TV_HOP,
                              boundary="zeros", padded=True)
        t_clip = np.arange(Z.shape[1]) * TV_HOP / SR
        # clip time -> timeline time (invert pos by interpolation on the grid)
        pos = tr.pos(grid)
        order = np.argsort(pos)
        t_tl = np.interp(t_clip, pos[order], grid[order])
        gi = np.clip(np.round(t_tl / GRID).astype(int), 0, G.shape[1] - 1)
        Gdb = G[:, gi]                                          # [bands, frames]
        # bands -> bins (log-frequency interpolation)
        lf = np.log2(np.maximum(f, 20.0))
        lcb = np.log2(TV_BANDS)
        gains = np.empty((len(f), Gdb.shape[1]), np.float32)
        for j in range(0, Gdb.shape[1], 2048):
            blk = Gdb[:, j:j + 2048]
            gains[:, j:j + 2048] = np.stack([np.interp(lf, lcb, blk[:, c]) for c in range(blk.shape[1])], axis=1)
        Z *= 10 ** (gains / 20.0)
        _, y = signal.istft(Z, fs=SR, window=win, nperseg=TV_NFFT, noverlap=TV_NFFT - TV_HOP, boundary=True)
        tr.music_eq = y[:len(tr.music)].astype(np.float32)
        del Z, gains
    log(f"  time-varying consensus EQ applied to {len(used)} takes")


# --------------------------------------------------------------------------- consensus loudness lock
def consensus_loudness_lock(x: np.ndarray, feats: dict, max_db: float = 6.0, smooth_s: float = 3.0,
                            min_takes: int = 3, log=print) -> np.ndarray:
    """Make the render's slow (~2 s) loudness follow the cross-POV consensus curve.

    Every phone heard the same performance; the median of their de-crowded loudness curves is the
    performance's real dynamics with each phone's AGC pumping averaged out. Switch glides and per-take
    gains can only add swells on top of that, so they are removed here. x is on the timeline from t=0.
    """
    grid = feats["grid"]
    names = list(feats["f"])
    L = np.stack([np.where(feats["f"][n]["ok"], feats["f"][n]["loud"], np.nan) for n in names])
    with np.errstate(all="ignore"):
        med_t = np.nanmedian(L, axis=0)
        off = np.array([np.nanmedian(L[i] - med_t) if np.isfinite(L[i]).any() else 0.0 for i in range(len(names))])
        cons = np.nanmedian(L - off[:, None], axis=0)
    cnt = np.sum(np.isfinite(L), axis=0)
    e = _band_energy(x, SR, 60, 16000, 2048, 1024) + 1e-12
    fr = SR / 1024
    e = ndimage.uniform_filter1d(e, int(2.0 * fr), mode="nearest")
    xl = 10 * np.log10(e)[np.clip((grid * fr).astype(int), 0, len(e) - 1)]
    good = (cnt >= min_takes) & np.isfinite(cons) & (xl > np.nanmax(xl) - 40)
    if good.sum() < 20:
        return x
    d = np.where(good, cons - xl, np.nan)
    d -= np.nanmedian(d)
    idx = np.where(good)[0]
    full = np.interp(np.arange(len(d)), idx, d[idx])
    k = max(3, int(smooth_s / GRID))
    full = ndimage.median_filter(full, size=k | 1, mode="nearest")
    full = ndimage.uniform_filter1d(full, size=k, mode="nearest")
    # taper the correction to zero where the consensus is unreliable
    w = ndimage.uniform_filter1d(good.astype(float), size=k, mode="nearest")
    full = np.clip(full * w, -max_db, max_db)
    log(f"  consensus loudness lock: correction p50 {np.median(np.abs(full[good])):.2f} dB, "
        f"p95 {np.percentile(np.abs(full[good]), 95):.2f} dB")
    g = 10 ** (np.interp(np.arange(len(x)) / SR, grid, full) / 20.0)
    return (x * g.astype(np.float32)).astype(np.float32)
