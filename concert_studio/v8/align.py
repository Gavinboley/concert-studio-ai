"""Multi-POV alignment with per-window validity.

Coarse: log-band onset novelty (100-5000 Hz, so HE-AAC SBR never drives it) cross-correlated
against a growing union reference on a shared timeline.
Fine:   8 s windows, GCC-PHAT at 16 kHz within +-60 ms of the coarse lag; Theil-Sen fit of
        lag(t) gives sub-sample offset + clock drift; windows that do not correlate (talking,
        spliced footage, a different song) are marked invalid and never used.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import stats

from .dsp import bandpass, fft_xcorr, gcc_phat, novelty, resample
from .ingest import SR

FINE_SR = 16000
WIN_S = 8.0
HOP_S = 2.0
MARGIN_S = 150.0
ISLAND_GAP_S = 3.0      # silence between islands (groups of clips that share no audio with each other)
CONFIRM_SCORE = 8.0     # fine GCC-PHAT score a coarse placement must reach (wrong-offset null p99 ~6.5-7)
CONFIRM_SCORE_ALT = 10.0  # stricter bar for a runner-up coarse peak (several are tested: guard multiple chances)
ISLAND_MAX_PLACED = 6   # with this many placed clips, an unplaceable clip is foreign, not a separate island


def _fine_confirm(w: np.ndarray, U: np.ndarray, C: np.ndarray, off_s: float, win_s: float = 4.0) -> float:
    """Median GCC-PHAT score of up to 5 windows of clip `w` against the union at timeline offset `off_s`."""
    b = int(round(off_s * FINE_SR))
    win = int(win_s * FINE_SR)
    lo, hi = max(0, b), min(len(U), b + len(w))
    starts = [t for t in range(lo, hi - win, max(win, (hi - lo) // 6)) if C[t:t + win].mean() > 0.95][:5]
    keep = [t for t in starts if np.std(w[t - b:t - b + win]) > 1e-6]
    res = [gcc_phat(U[t:t + win], w[t - b:t - b + win], int(0.06 * FINE_SR)) for t in keep]
    if len(res) < 3:
        return 0.0
    tt = np.array(keep, float) / FINE_SR
    lags = np.array([r[0] for r in res], float) / FINE_SR * 1000.0
    # a true placement: the fine lags lie on one line (offset + phone clock drift, |slope| <= 0.3 ms/s);
    # a chance match scatters over +-60 ms. Robust: judge the median residual, so one noisy window is fine.
    slope, icpt = np.polyfit(tt - tt.mean(), lags, 1)
    resid = np.abs(lags - (icpt + slope * (tt - tt.mean())))
    if abs(slope) > 0.3 or np.median(resid) > 3.0:
        return 0.0
    return float(np.median([r[1] for r in res]))



@dataclass
class Window:
    clip_t: float        # window centre, clip seconds
    lag_ms: float        # measured fine lag relative to coarse placement
    score: float         # PHAT peak / median |cc|
    valid: bool


@dataclass
class Placement:
    name: str
    offset_s: float                 # timeline time of clip sample 0 (after fine fit)
    drift_ppm: float                # timeline_t = offset + clip_t * (1 + drift*1e-6)
    coarse_z: float                 # novelty xcorr peak z-score
    duration: float
    windows: list[Window] = field(default_factory=list)
    island: int = 0                 # clips in different islands share no audio (sparse payloads)

    def timeline_t(self, clip_t):
        return self.offset_s + np.asarray(clip_t) * (1.0 + self.drift_ppm * 1e-6)

    def clip_t(self, timeline_t):
        return (np.asarray(timeline_t) - self.offset_s) / (1.0 + self.drift_ppm * 1e-6)

    def valid_fraction(self) -> float:
        return float(np.mean([w.valid for w in self.windows])) if self.windows else 0.0

    def to_dict(self):
        d = asdict(self)
        return d


def _fine_signal(x: np.ndarray) -> np.ndarray:
    y = resample(x, SR, FINE_SR)
    return bandpass(y, FINE_SR, 120.0, 5000.0)


def align_clips(names: list[str], monos: dict[str, np.ndarray], ref: str, log=print) -> list[Placement]:
    """Place every clip on a timeline anchored at the reference clip (ref sample 0 == MARGIN_S)."""
    nov = {}
    fr = None
    for n in names:
        nov[n], fr = novelty(monos[n], SR)
    fine = {n: _fine_signal(monos[n]) for n in names}

    total_s = max(len(monos[n]) for n in names) / SR + 2 * MARGIN_S + 60
    U_nov = np.zeros(int(total_s * fr), np.float32)
    U_cov = np.zeros(len(U_nov), bool)
    U_fine = np.zeros(int(total_s * FINE_SR), np.float32)
    F_cov = np.zeros(len(U_fine), bool)

    def grow(end_s: float):
        nonlocal U_nov, U_cov, U_fine, F_cov
        need = int((end_s + 60.0) * fr)
        if need > len(U_nov):
            U_nov = np.concatenate([U_nov, np.zeros(need - len(U_nov), np.float32)])
            U_cov = np.concatenate([U_cov, np.zeros(need - len(U_cov), bool)])
        need = int((end_s + 60.0) * FINE_SR)
        if need > len(U_fine):
            U_fine = np.concatenate([U_fine, np.zeros(need - len(U_fine), np.float32)])
            F_cov = np.concatenate([F_cov, np.zeros(need - len(F_cov), bool)])

    def paste(name: str, off_s: float):
        grow(off_s + len(monos[name]) / SR)
        a = int(round(off_s * fr))
        v = nov[name]
        lo, hi = max(0, a), min(len(U_nov), a + len(v))
        seg = slice(lo, hi)
        fill = ~U_cov[seg]
        U_nov[seg][fill] = v[lo - a:hi - a][fill]
        U_cov[seg] = True
        b = int(round(off_s * FINE_SR))
        w = fine[name]
        lo, hi = max(0, b), min(len(U_fine), b + len(w))
        seg = slice(lo, hi)
        fill = ~F_cov[seg]
        U_fine[seg][fill] = w[lo - b:hi - b][fill]
        F_cov[seg] = True

    placements: dict[str, Placement] = {}
    ref_p = Placement(ref, MARGIN_S, 0.0, 99.0, len(monos[ref]) / SR)
    placements[ref] = ref_p
    paste(ref, MARGIN_S)

    order = sorted([n for n in names if n != ref], key=lambda n: -len(monos[n]))
    pending = list(order)
    # Few clips: repeated sections can fake an overlap, and a wrong overlap costs coverage, so coarse peaks must be
    # confirmed on the waveform. Many clips: placements reinforce each other and the original rule is kept.
    sparse = len(names) < ISLAND_MAX_PLACED
    island = 0

    def place_round(pending: list[str]) -> list[str]:
        still = []
        for n in pending:
            v = nov[n]
            cc = fft_xcorr(U_nov * U_cov, v)
            # cc index i -> lag = i - (len(v)-1)  (v placed starting at frame lag)
            z = (cc - np.median(cc)) / (np.std(cc) + 1e-9)
            if not sparse:                                  # big payloads: the original rule, validated on the
                i = int(np.argmax(z))                       # 82-clip payload (its fine windows are crowd-noisy)
                lag_frames = i - (len(v) - 1)
                ov = U_cov[max(0, lag_frames):max(0, lag_frames + len(v))].sum() / fr
                zc = float(z[i])
                if zc < 8.0 or ov < 10.0:
                    still.append(n)
                    continue
                off = lag_frames / fr
                placements[n] = Placement(n, off, 0.0, zc, len(monos[n]) / SR, island=island)
                paste(n, off)
                log(f"  coarse {n:36s} off={off - MARGIN_S:8.3f}s z={zc:6.1f} overlap={ov:6.1f}s")
                continue
            # Repeated sections (loops, backing tracks, chorus 2 vs 1) give several coarse peaks, and the raw
            # correlation favours offsets with more overlap. Check the strongest few with the fine waveform test
            # and keep the best-confirmed one.
            cands, zz = [], z.copy()
            sep = int(2.0 * fr)
            for _k in range(5):
                i = int(np.argmax(zz))
                if zz[i] < 8.0:
                    break
                zz[max(0, i - sep):i + sep] = -np.inf
                lag_frames = i - (len(v) - 1)
                ov = U_cov[max(0, lag_frames):max(0, lag_frames + len(v))].sum() / fr
                if ov >= 10.0:
                    cands.append((float(z[i]), lag_frames / fr, ov))
            if not cands:
                still.append(n)
                continue
            scored = []
            for k_, (zc_, o, ov_) in enumerate(cands):     # cands[0] = strongest coarse peak
                f_ = _fine_confirm(fine[n], U_fine, F_cov, o)
                need = CONFIRM_SCORE if k_ == 0 else CONFIRM_SCORE_ALT
                scored.append((f_ if f_ >= need else 0.0, zc_, o, ov_))
            fs_, zc, off, ov = max(scored)
            if fs_ < CONFIRM_SCORE:
                # repeated song sections (chorus 1 vs chorus 2) fool onset novelty: no shared waveform here
                log(f"  reject {n:36s} coarse z={max(c[0] for c in cands):5.1f} but best fine score {fs_:4.1f}")
                still.append(n)
                continue
            placements[n] = Placement(n, off, 0.0, zc, len(monos[n]) / SR, island=island)
            paste(n, off)
            log(f"  coarse {n:36s} off={off - MARGIN_S:8.3f}s z={zc:6.1f} overlap={ov:6.1f}s")
        return still

    while True:
        for _round in range(3):
            n_before = len(pending)
            pending = place_round(pending)
            if not pending or len(pending) == n_before:
                break
        if not pending:
            break
        # Sparse payloads: whatever shares no audio with the timeline so far is a separate island (another part
        # of the song). It starts after a short gap; its own longest clip anchors it. (Order of islands is the
        # order they were found; a studio reference can reorder them later.)
        # With many clips covering the song, a real clip of it almost surely overlaps one of them: one that
        # shares audio with none is another song or talking. With few clips, gaps between phones are expected.
        # (Chroma cannot decide this: real same-song phone clips correlate 0.07-0.66, a foreign one 0.36.)
        if len(placements) >= ISLAND_MAX_PLACED:
            for n in pending:
                log(f"  UNPLACED {n} (shares no audio with any of {len(placements)} placed clips)")
            break
        island += 1
        new_ref = pending.pop(0)
        end = max(p.offset_s + p.duration for p in placements.values())
        off = end + ISLAND_GAP_S
        placements[new_ref] = Placement(new_ref, off, 0.0, 0.0, len(monos[new_ref]) / SR, island=island)
        paste(new_ref, off)
        log(f"  ISLAND {island}: {new_ref} shares no audio with the earlier clips; placed after a {ISLAND_GAP_S:.0f} s gap")

    # ---- fine alignment against the leave-self-out union reference (2 passes) ----
    for _pass in range(2):
        _fine_pass(placements, fine, len(U_fine) + int(MARGIN_S * FINE_SR), log if _pass == 1 else (lambda *a: None))

    # normalise timeline so the earliest valid sample is t=0
    t0 = min(p.offset_s for p in placements.values())
    for p in placements.values():
        p.offset_s -= t0
    return list(placements.values())


def _fine_pass(placements: dict, fine: dict, n_union: int, log) -> None:
    """Each clip is validated against a union of the *other* clips, so it cannot validate itself."""
    U_fine = np.zeros(n_union, np.float32)
    for n, p in placements.items():
        w = fine[n]
        win = int(WIN_S * FINE_SR)
        hop = int(HOP_S * FINE_SR)
        max_lag = int(0.06 * FINE_SR)
        others = [m for m in placements if m != n]
        # local union of others (first-come fill, longest first)
        Uo = np.zeros_like(U_fine)
        Co = np.zeros(len(U_fine), bool)
        for m in sorted(others, key=lambda m: -len(fine[m])):
            b = int(round(placements[m].offset_s * FINE_SR))
            ww = fine[m]
            lo, hi = max(0, b), min(len(Uo), b + len(ww))
            seg = slice(lo, hi)
            fill = ~Co[seg]
            Uo[seg][fill] = ww[lo - b:hi - b][fill]
            Co[seg] = True
        wins = []
        for s in range(0, max(1, len(w) - win + 1), hop):
            seg = w[s:s + win]
            t_tl = int(round(p.offset_s * FINE_SR)) + s
            lo, hi = t_tl, t_tl + win
            if lo < 0 or hi > len(Uo) or Co[lo:hi].mean() < 0.9 or np.std(seg) < 1e-5:
                wins.append(Window((s + win / 2) / FINE_SR, 0.0, 0.0, False))
                continue
            lag, score = gcc_phat(Uo[lo:hi], seg, max_lag)
            wins.append(Window((s + win / 2) / FINE_SR, 1000.0 * lag / FINE_SR, score, False))
        p.windows = wins
        scores = np.array([x.score for x in wins])
        good = scores > 12.0
        if good.sum() >= 3:
            t = np.array([x.clip_t for x in wins])[good]
            lg = np.array([x.lag_ms for x in wins])[good]
            slope, icpt, _, _ = stats.theilslopes(lg, t)
            resid = np.abs(np.array([x.lag_ms for x in wins]) - (icpt + slope * np.array([x.clip_t for x in wins])))
            for x, g, r in zip(wins, good, resid):
                x.valid = bool(g and r < 1.0)
            # lag>0 means the reference is later than the clip window -> clip sits later on the timeline
            p.offset_s += icpt / 1000.0
            p.drift_ppm = float(slope * 1000.0)  # ms per s -> ppm
            for x in wins:
                x.lag_ms -= (icpt + slope * x.clip_t)
        log(f"  fine   {n:36s} off={p.offset_s - MARGIN_S:9.4f}s drift={p.drift_ppm:7.2f}ppm "
            f"valid={p.valid_fraction():5.1%} medscore={np.median(scores):5.1f}")



def pairwise_coherence(names: list[str], monos: dict[str, np.ndarray], placements: dict[str, Placement],
                       step_s: float = 12.0, win_s: float = 4.0) -> np.ndarray:
    """Median GCC-PHAT peak score for every overlapping clip pair (0 where no overlap).

    Calibrated on this payload: wrong-offset null p99 ~= 6.5-7, real distinct POVs 9-30,
    re-uploads of the same recording 150-270.
    """
    import itertools
    F = {n: _fine_signal(monos[n]) for n in names}
    win = int(win_s * FINE_SR)
    M = np.zeros((len(names), len(names)))
    for i, j in itertools.combinations(range(len(names)), 2):
        a, b = names[i], names[j]
        pa, pb = placements[a], placements[b]
        t0 = max(pa.offset_s, pb.offset_s)
        t1 = min(pa.offset_s + len(F[a]) / FINE_SR, pb.offset_s + len(F[b]) / FINE_SR) - win_s
        sc = []
        for t in np.arange(t0 + 1, t1, step_s):
            sa, sb = int((t - pa.offset_s) * FINE_SR), int((t - pb.offset_s) * FINE_SR)
            sc.append(gcc_phat(F[a][sa:sa + win], F[b][sb:sb + win], int(0.06 * FINE_SR))[1])
        if sc:
            M[i, j] = M[j, i] = float(np.median(sc))
    return M


def overlap_matrix(names: list[str], placements: dict[str, Placement], min_s: float = 8.0) -> np.ndarray:
    """True where two placed clips share at least `min_s` seconds of timeline."""
    O = np.zeros((len(names), len(names)), bool)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                pa, pb = placements[a], placements[b]
                ov = min(pa.offset_s + pa.duration, pb.offset_s + pb.duration) - max(pa.offset_s, pb.offset_s)
                O[i, j] = ov >= min_s
    return O


def select_clips(names: list[str], M: np.ndarray, clip_rank: dict[str, tuple], dup_thresh: float = 60.0,
                 orphan_thresh: float = 8.0, overlap: np.ndarray | None = None) -> dict:
    """Group re-uploads (keep the best encode) and drop clips that cohere with nothing they overlap.

    A clip that overlaps no other clip cannot be checked against anything: it is kept (sparse payloads, where
    each phone filmed a different part of the song), not called an orphan."""
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if M[i, j] > dup_thresh:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(names)):
        groups.setdefault(find(i), []).append(i)
    keep, dups, orphans = [], {}, []
    for members in groups.values():
        best = max(members, key=lambda i: clip_rank[names[i]])
        if len(members) > 1:
            dups[names[best]] = [names[i] for i in members if i != best]
        # coherence with clips outside its own duplicate group
        outside = [k for k in range(len(names)) if k not in members]
        coh = max((M[i, k] for i in members for k in outside), default=0.0)
        overlaps = (overlap is None and bool(outside)) or (
            overlap is not None and any(overlap[i, k] for i in members for k in outside))
        if coh < orphan_thresh and overlaps:
            orphans.append(names[best])
        else:
            keep.append(names[best])
    return {"keep": keep, "duplicates": dups, "orphans": orphans}
