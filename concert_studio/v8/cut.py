"""The multi-camera cut, in every mode.

    format     vertical (1080x1920, TikTok / YouTube Shorts; captions and labels kept clear of the apps' buttons)
               or landscape (YouTube, 2560x1440)
    multi      one camera at a time, or 1-4 perspectives on screen at once (--min-cams / --max-cams)
    cinematic  stronger stabilisation, slow push-ins, a filmic grade with highlight bloom, grain and vignette,
               dissolves between quiet sections, title and credit fades

What the director does, whatever the mode:
* follows the music: cuts and layout changes land on downbeats and vocal entrances; quiet passages hold fewer,
  longer shots, choruses open up to more panels and faster swaps; a jump in energy can "slam" to one full frame;
* follows the band: the stems say who is featured (singing, a guitar part, a fill); CLIP-tagged performers in each
  phone's footage (vision.py) let the hero panel show that member, framed tight, while other panels show the stage,
  the other players or a second angle;
* only real footage: every frame is a phone's own pixels, cropped, stabilised, colour-matched and scaled; nothing is
  generated or inpainted. Each panel's footage is mapped to the master clock with the measured offset/drift and its
  audio is cross-correlated against the master to verify sync;
* consistency: each phone's colour and exposure are pulled toward what the other phones saw at the same moment
  (the picture's consensus, like the audio's loudness lock), so a cut does not jump in white balance;
* credits the filmers: the @handle fades in when a phone's shot starts, and the end card lists everyone.

CLI:  python -m concert_studio.v8.cut --job workspace_ui/<job> --format vertical --multi --cinematic \
          --min-cams 1 --max-cams 4 [--highlight 60 | --start S --end E] [--title "..."] [--audio native,ozone]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import video as V
from . import vision as VI

FPS = V.FPS
HANDLE = 0.5              # s of extra footage rendered on both sides of every panel run (dissolves)
DISSOLVE = 0.5
MIN_RUN = 2.0
BG = "0x0b0b0d"
# TikTok and YouTube Shorts overlay tabs (top), buttons (right) and caption/music rows (bottom) at 1080x1920
SAFE = {"top": 170, "bottom": 430, "left": 40, "right": 150}
STRINGS = ("guitar", "bass")


@dataclass
class Style:
    fmt: str = "vertical"
    multi: bool = False
    cinematic: bool = False
    min_cams: int = 1
    max_cams: int = 4
    max_zoom: float = V.MAX_ZOOM
    labels: bool = True
    credits: bool = True
    title: str = ""
    subtitle: str = ""
    stabilize: bool = True
    seed: int = 0

    def norm(self) -> "Style":
        self.fmt = "landscape" if str(self.fmt).lower().startswith(("land", "yt", "16")) else "vertical"
        self.min_cams = int(np.clip(self.min_cams, 1, 4))
        self.max_cams = int(np.clip(self.max_cams, self.min_cams, 4))
        if not self.multi:
            self.min_cams = self.max_cams = 1
        return self


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


# ----------------------------------------------------------------------------------------------- geometry
def canvas(style: Style, roster: dict) -> tuple[int, int]:
    """Vertical 1080x1920. Landscape 2560x1440: YouTube encodes 1440p uploads at a much higher bitrate (VP9/AV1), which
    keeps phone footage cleaner even for viewers watching at 1080p."""
    return (1080, 1920) if style.fmt == "vertical" else (2560, 1440)


def layout(fmt: str, n: int, W: int, H: int) -> list[tuple[int, int, int, int]]:
    """Panel rects (x, y, w, h), hero first. Landscape: equal columns (three 9:16-ish columns suit phones).
    Vertical: full / stacked / one over two / 2x2 (each quarter is 9:16 like the phones)."""
    g = _even(W * (0.006 if fmt == "vertical" else 0.004))
    if n <= 1:
        return [(0, 0, W, H)]
    if fmt == "landscape":
        w = _even((W - (n - 1) * g) / n)
        xs = [round(i * (W - w) / (n - 1)) // 2 * 2 for i in range(n)]
        rects = [(x, 0, w, H) for x in xs]
        order = {2: [0, 1], 3: [1, 0, 2], 4: [1, 2, 0, 3]}[n]
        return [rects[i] for i in order]
    h2, w2 = _even((H - g) / 2), _even((W - g) / 2)
    top, bottom = (0, 0, W, h2), (0, H - h2, W, h2)
    bl, br = (0, H - h2, w2, h2), (W - w2, H - h2, w2, h2)
    if n == 2:
        return [top, bottom]
    if n == 3:
        return [top, bl, br]
    return [(0, 0, w2, h2), (W - w2, 0, w2, h2), bl, br]


def fit(cw: int, ch: int, pw: int, ph: int, max_zoom: float) -> tuple[float, float, float, float]:
    """Largest window of the panel's aspect inside a cw x ch picture, upscale capped at max_zoom.
    Returns (w, h, scale, fill): fill < 1 leaves a blurred margin."""
    s = min(max(pw / cw, ph / ch), max_zoom)
    w, h = min(cw, pw / s), min(ch, ph / s)
    return w, h, s, (w * s * h * s) / (pw * ph)


# ----------------------------------------------------------------------------------------------- per-camera facts
def _to_master(cam: dict, c: float) -> float:
    return (c - cam["av_offset_s"]) * (1.0 + cam["drift_ppm"] * 1e-6) + cam["offset_s"] - cam["trim_start_s"]


def _samples(vis: dict, cam_id: str, cam: dict, t0: float, t1: float) -> list[dict]:
    smp = (vis.get(cam_id) or {}).get("samples") or []
    if not smp:
        return []
    c0, c1 = float(V.master_to_container(cam, t0)), float(V.master_to_container(cam, t1))
    sel = [s for s in smp if c0 - 0.8 <= s["t"] <= c1 + 0.8]
    if not sel:
        mid = (c0 + c1) / 2
        sel = sorted(smp, key=lambda s: abs(s["t"] - mid))[:2]
    return sel


def _match(p: list, want: str) -> float:
    if len(p) < 6:
        return 0.0
    if want in ("guitar", "bass", "strings"):
        return p[6] * (1.0 if p[5] in STRINGS else 0.0) + (0.1 if p[5] == want else 0.0)
    return p[6] if p[5] == want else 0.0


def content_score(samples: list[dict], want: str) -> float:
    """0..1: how well this footage shows what the slot wants ('vocals', 'strings', 'drums', 'keys', 'wide', 'any')."""
    if not samples or want == "any":
        return 0.0
    vals = []
    for s in samples:
        ps = s.get("persons") or []
        if want in ("wide", "band"):
            hs = [p[3] - p[1] for p in ps]
            n = len(ps) + s.get("n_small", 0)
            vals.append(min(1.0, n / 4.0) * (1.0 if (not hs or max(hs) < 0.4) else 0.4))
        else:
            vals.append(max([_match(p, want) * min(1.0, (p[3] - p[1]) / 0.35) for p in ps] or [0.0]))
    return float(np.mean(vals))


def subject(samples: list[dict]) -> str:
    """Who this footage is mostly about: a band role when one performer dominates the frame, else 'wide'."""
    score: dict[str, float] = {}
    for s in samples:
        for p in s.get("persons") or []:
            if len(p) > 6 and p[3] - p[1] >= 0.2:
                r = "strings" if p[5] in STRINGS else p[5]
                score[r] = score.get(r, 0.0) + p[6] * (p[3] - p[1])
    if not score:
        return "wide"
    r = max(score, key=score.get)
    return r if score[r] >= 0.15 * max(1, len(samples)) else "wide"


def captions_of(v: dict | None) -> list[list[float]]:
    """Burned-in caption bands of a phone (vision.caption_bands), else the older single caption box."""
    v = v or {}
    if v.get("captions"):
        return v["captions"]
    ov = v.get("overlay")
    return [ov] if ov and ov[3] - ov[1] <= 0.45 else []


def caption_free(bands: list[list[float]], ch: float, h_min: float) -> tuple[float, float] | None:
    """Tallest horizontal stretch of the picture (y0, y1 in px) clear of every caption band, if a window of at
    least h_min fits in it; None if the captions cannot be framed out."""
    if not bands:
        return 0.0, ch
    edges, y = [], 0.0
    for b in sorted(bands, key=lambda q: q[1]):
        edges.append((y, b[1] * ch))
        y = max(y, b[3] * ch)
    edges.append((y, ch))
    best = max(edges, key=lambda r: r[1] - r[0])
    return best if best[1] - best[0] >= h_min else None


def _hits(bands: list[list[float]], ch: float, y: float, h: float) -> bool:
    return any(y < b[3] * ch and y + h > b[1] * ch for b in bands)


def shot_size(samples: list[dict]) -> str:
    hs = [max([p[3] - p[1] for p in s.get("persons") or []] or [0.0]) for s in samples]
    h = float(np.median(hs)) if hs else 0.0
    return "close" if h >= 0.4 else ("mid" if h >= 0.2 else "wide")


def color_corrections(roster: dict, vis: dict, strength: float = 0.7, bin_s: float = 3.0) -> dict:
    """Per camera: RGB gains + luma midpoint mapping toward the consensus of the phones filming the same moment."""
    rows = []                                                      # (bin, cam, log r, log g, log b, luma p50)
    for n, cam in roster["cams"].items():
        for s in (vis.get(n) or {}).get("samples") or []:
            tm = _to_master(cam, s["t"])
            if not (np.isfinite(tm) and min(s["rgb"]) > 1e-5 and s["luma"][1] > 0.01):
                continue
            rows.append((int(tm // bin_s), n, *np.log(s["rgb"]), s["luma"][1]))
    out = {}
    if not rows:
        return out
    bins: dict[int, dict[str, list]] = {}
    for b, n, lr, lg, lb, y in rows:
        bins.setdefault(b, {}).setdefault(n, []).append((lr, lg, lb, y))
    deltas: dict[str, list] = {}
    for b, per in bins.items():
        if len(per) < 3:
            continue
        mean = {n: np.mean(v, axis=0) for n, v in per.items()}
        cons = np.median(np.stack(list(mean.values())), axis=0)
        for n, m in mean.items():
            deltas.setdefault(n, []).append((cons[:3] - m[:3], m[3], cons[3]))
    for n, d in deltas.items():
        if len(d) < 2:
            continue
        dl = np.median(np.stack([x[0] for x in d]), axis=0)
        wb = dl - dl.mean()                                         # white balance only; exposure via the curve
        gains = np.clip(np.exp(strength * wb) ** (1 / 2.2), 0.85, 1.18)
        y_cam = float(np.median([x[1] for x in d]))
        y_tgt = float(np.median([x[2] for x in d]))
        y_new = float(np.clip(y_cam + strength * (y_tgt - y_cam), y_cam * 0.78, y_cam * 1.35))
        out[n] = {"gains": [round(float(g), 4) for g in gains], "mid": [round(y_cam, 4), round(y_new, 4)]}
    return out


def _corr_filter(c: dict | None) -> str:
    if not c:
        return ""
    r, g, b = c["gains"]
    p, q = c["mid"]
    p, q = float(np.clip(p, 0.04, 0.9)), float(np.clip(q, 0.04, 0.92))
    f = f"colorchannelmixer=rr={r:.4f}:gg={g:.4f}:bb={b:.4f}"
    if abs(q - p) > 0.005:
        f += f",curves=master='0/0 {p:.4f}/{q:.4f} 1/1'"
    return f


# ----------------------------------------------------------------------------------------------- music
def _phrases(mu: dict, start: float, end: float) -> list[tuple[float, float]]:
    downs = np.asarray(mu["downbeats"], float)
    downs = downs[(downs > start + 1.0) & (downs < end - 1.5)]
    bar = float(np.median(np.diff(mu["downbeats"]))) if len(mu["downbeats"]) > 2 else 2.0
    per = 4 if bar <= 3.2 else 2
    marks = list(downs[::per])
    for e in mu.get("vocal_entries", []):                          # vocal entrances start phrases
        if start + 1.0 < e < end - 1.5:
            near = downs[np.argmin(np.abs(downs - e))] if len(downs) else e
            marks.append(float(near) if abs(near - e) < 0.6 * bar else float(e))
    marks = sorted(set(round(m, 3) for m in marks))
    kept: list[float] = []
    for m in marks:
        if not kept or m - kept[-1] >= 1.5 * bar:
            kept.append(m)
        elif m in [round(float(x), 3) for x in mu.get("vocal_entries", [])]:
            kept[-1] = m
    b = [start] + [m for m in kept if m - start >= MIN_RUN and end - m >= MIN_RUN] + [end]
    return list(zip(b[:-1], b[1:]))


def _energy_rank(mu: dict, spans: list[tuple[float, float]]) -> list[float]:
    e = np.asarray(mu.get("energy_10hz") or [0.0])
    vals = [float(np.mean(e[int(a * 10):max(int(a * 10) + 1, int(b * 10))])) if len(e) > 1 else 0.0 for a, b in spans]
    order = np.argsort(np.argsort(vals))
    return [float(o) / max(1, len(vals) - 1) for o in order]


def _feat(ft: dict, t0: float, t1: float) -> tuple[str, dict]:
    hop = ft.get("hop_s", 0.5)
    f = ft.get("featured") or []
    a, b = int(t0 / hop), max(int(t0 / hop) + 1, int(t1 / hop))
    seg = f[a:b]
    main = max(set(seg), key=seg.count) if seg else "vocals"
    act = {r: float(np.mean(v[a:b])) if len(v[a:b]) else 0.0 for r, v in (ft.get("activity") or {}).items()}
    return main, act


def highlight(mu: dict, length: float, roster: dict | None = None) -> tuple[float, float]:
    """Best `length` s for a Short: loud, sung, well covered; starts on a phrase boundary."""
    dur = float(mu["duration"])
    if length >= dur - 1:
        return 0.0, dur
    e = np.asarray(mu["energy_10hz"])
    starts = [a for a, _ in _phrases(mu, 0.0, dur) if a + length <= dur - 0.5]
    cov = None
    if roster:
        grid = np.arange(0, dur, 1.0)
        cov = np.array([sum(V._covers(c, t, t + 1.0) for c in roster["cams"].values()) for t in grid])
    best, bs = 0.0, -1e9
    for s in starts:
        seg = e[int(s * 10):int((s + length) * 10)]
        sc = float(np.mean(seg)) + 0.5 * float(np.percentile(seg, 20))
        sc += 2.0 * float(np.mean(e[int(s * 10):int((s + 2) * 10)]) > np.percentile(e, 50))   # opens strong
        if cov is not None:
            sc += 1.5 * float(np.mean(np.minimum(cov[int(s):int(s + length)], 4))) / 4
        if sc > bs:
            best, bs = s, sc
    return round(best, 3), round(min(dur, best + length), 3)


# ----------------------------------------------------------------------------------------------- planner
def _wants(feat: str, act: dict, n: int) -> list[str]:
    hero = {"vocals": "vocals", "guitar": "strings", "bass": "strings", "drums": "drums", "keys": "keys"}.get(feat, "wide")
    others = sorted([r for r in ("vocals", "guitar", "drums", "keys", "bass") if act.get(r, 0) > 0.45],
                    key=lambda r: -act[r])
    others = [("strings" if r in STRINGS else r) for r in others]
    out = [hero]
    for w in others + ["wide", "any", "any"]:
        if len(out) >= n:
            break
        if w not in out or w == "any":
            out.append(w)
    return out[:n]


class Director:
    def __init__(self, roster: dict, vis: dict, ft: dict, style: Style, W: int, H: int):
        self.r, self.cams, self.vis, self.ft, self.st = roster, roster["cams"], vis, ft, style
        self.W, self.H = W, H
        self.rng = np.random.default_rng(style.seed)
        sh = [s["sharp"] for v in vis.values() for s in (v or {}).get("samples", [])]
        self.sh_mu, self.sh_sd = (float(np.median(sh)), float(np.std(sh)) + 1e-6) if sh else (0.0, 1.0)
        self.history: list[str] = []
        self.left_at: dict[str, float] = {}

    def _quality(self, n: str, t0: float, t1: float, pw: int, ph: int) -> float:
        cam = self.cams[n]
        smp = _samples(self.vis, n, cam, t0, t1)
        if smp:
            sharp = (float(np.median([s["sharp"] for s in smp])) - self.sh_mu) / self.sh_sd
            luma = float(np.median([s["luma"][1] for s in smp]))
            expo = -2.0 * max(0.0, 0.10 - luma) / 0.10 - 2.0 * max(0.0, luma - 0.75) / 0.25
        else:
            sharp, expo = V._video_score(cam, t0, t1) - 1.0, 0.0
        cw, ch = V._dims(cam)
        _, _, s, fill = fit(cw, ch, pw, ph, self.st.max_zoom)
        up = -0.8 * max(0.0, np.log2(s / 1.25))
        bands = captions_of(self.vis.get(n))
        cap = 0.0
        if bands:
            h_min = ph / self.st.max_zoom                          # tightest window the zoom cap allows
            free = caption_free(bands, ch, h_min)
            area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in bands)
            cap = -0.3 if free else -2.5 * min(1.0, 0.25 + 3.0 * area)   # framed out (costs zoom) / stays visible
        return 0.6 * sharp + expo + up - 2.0 * (1.0 - fill) + cap

    def pick(self, t0: float, t1: float, rect, want: str, onscreen: list[dict], hero: bool,
             exclude: frozenset = frozenset()) -> tuple[str, float] | None:
        """Best camera for one panel over [t0, t1); returns (cam, run end) — the run may end early at the end of
        that phone's validated footage."""
        if exclude:
            got = self.pick(t0, t1, rect, want, onscreen + [{"cam": c} for c in exclude], hero)
            if got is not None:
                return got
        pw, ph = rect[2], rect[3]
        busy = {o["cam"] for o in onscreen}
        need = min(MIN_RUN, t1 - t0) - 1e-6
        cand = []
        for n, cam in self.cams.items():
            if n in busy:
                continue
            ce = V._cover_end(cam, t0)
            if ce - t0 >= need:
                cand.append((n, min(ce, t1)))
                continue
            for ua, ub in cam["usable"]:                           # validated footage starts within 1.5 s
                if t0 < ua <= t0 + 1.5 and ub - t0 >= need:
                    cand.append((n, min(ub, t1)))
                    break
        if not cand:                                               # no validated audio here: physical footage
            for n, cam in self.cams.items():
                if n in busy:
                    continue
                c0 = float(V.master_to_container(cam, t0))
                if V.EDGE_IN <= c0 <= cam["dur"] - V.EDGE_OUT - 1.0:
                    cand.append((n, min(t1, V._phys_end(cam, t0))))
        if not cand:
            return None
        best, bs = None, -1e9
        for n, end in cand:
            smp = _samples(self.vis, n, self.cams[n], t0, end)
            s = self._quality(n, t0, end, pw, ph)
            cs = content_score(smp, want)
            s += (3.0 if hero else 1.2) * cs
            subj = subject(smp) if smp else "wide"
            if cs < 0.15 and want not in ("any", "wide"):
                s += 0.5 * content_score(smp, "wide")              # nobody shows that member: a different view
            same = sum(1 for o in onscreen if o.get("subject") == subj and subj != "wide")
            s -= (1.4 if subj != want else 0.6) * same            # three singer close-ups side by side is dull
            s -= (1.0 - (0.7 * cs if hero else 0.0)) * self.history[-8:].count(n)   # the featured member may recur
            if t0 - self.left_at.get(n, -99) < 4.0:
                s -= 1.0
            s += 0.4 * min(1.0, (end - t0) / max(1e-6, t1 - t0))
            heard = np.asarray(self.cams[n].get("heard") or [])
            if len(heard) and np.mean((heard >= t0) & (heard < end)) > 0:
                s += 0.2
            s += 0.15 * self.rng.standard_normal()
            if s > bs:
                best, bs = (n, end), s
        return best

    def plan(self, start: float, end: float) -> list[dict]:
        st, mu = self.st, self.r["music"]
        spans = _phrases(mu, start, end)
        ranks = _energy_rank(mu, spans)
        bar = float(np.median(np.diff(mu["downbeats"]))) if len(mu["downbeats"]) > 2 else 2.0
        downs = np.asarray(mu["downbeats"], float)
        # split phrases for "slams": one full frame for a bar where the energy jumps
        plan_spans = []
        prev = None
        for (a, b), e in zip(spans, ranks):
            feat, act = _feat(self.ft, a, b)
            if st.multi and st.min_cams == 1 and st.max_cams > 1 and prev is not None and e - prev >= 0.35 \
                    and b - a >= 2 * bar + MIN_RUN:
                plan_spans.append((a, a + bar, 1, e, feat, act, "slam"))
                a = a + bar
            if st.multi:
                n = st.min_cams + int(round((st.max_cams - st.min_cams) * e))
                if feat not in ("vocals", "band"):
                    n = max(st.min_cams, min(n, 3 if st.fmt == "landscape" else 2))
            else:
                n = 1
            plan_spans.append((a, b, n, e, feat, act, "phrase"))
            prev = e
        runs: list[dict] = []
        active: list[dict] = []
        cur_n = None
        for a, b, n, e, feat, act, why in plan_spans:
            avail = sum(1 for c in self.cams.values()
                        if sum(max(0.0, min(ub, b) - max(ua, a)) for ua, ub in c["usable"]) >= 0.6 * (b - a))
            n = int(min(n, max(1, avail))) if st.multi else 1
            n = max(n, min(st.min_cams, avail)) if st.multi else 1
            if st.multi and st.fmt == "landscape" and n < 3 <= min(st.max_cams, avail):
                wide_src = sum(1 for c in self.cams.values() if V._dims(c)[0] / V._dims(c)[1] >= 1.2
                               and sum(max(0.0, min(ub, b) - max(ua, a)) for ua, ub in c["usable"]) >= 0.6 * (b - a))
                if wide_src < n:                                    # phones are vertical: 3 columns fit them
                    n = 3
            rects = layout(st.fmt, n, self.W, self.H)
            wants = _wants(feat, act, n)
            # swap cadence inside the phrase
            if st.multi:                                           # one panel changes every ~2.5 s / ~5 s / never
                every = max(1, round(2.5 / bar)) if e >= 0.66 else (max(1, round(5.0 / bar)) if e >= 0.33 else 0)
            else:                                                  # shots of ~4 s loud, ~5.5 s mid, ~7.5 s quiet
                every = max(1, round((4.0 if e >= 0.66 else 5.5 if e >= 0.33 else 7.5) / bar))
            after = downs[(downs > a + 1e-3) & (downs < b - MIN_RUN + 1e-6)]
            swaps = [float(t) for i, t in enumerate(after, start=1)
                     if every and i % every == 0 and t - a >= MIN_RUN - 1e-6]
            if n != cur_n or not active or why == "slam":
                for o in active:
                    o["t1"] = a
                    self.left_at[o["cam"]] = a
                active = []
                for k, rect in enumerate(rects):
                    self._assign(a, b, k, rect, wants[k], active, runs, why)
                cur_n = n
            else:                                                   # same layout: follow who is featured now
                for o in sorted(active, key=lambda q: q["slot"]):
                    if wants[o["slot"]] != o["want"] and a - o["t0"] >= MIN_RUN - 1e-6:
                        self._replace(o, a, b, active, runs, want=wants[o["slot"]])
            for ts in swaps + [b]:
                # slots whose footage ends before ts must change at their end
                for _ in range(64):
                    short = [o for o in active if o["end_cov"] < min(ts, b) - 1e-6 and o["end_cov"] > o["t0"] + 1e-6]
                    if not short:
                        break
                    o = min(short, key=lambda q: q["end_cov"])
                    self._replace(o, o["end_cov"], b, active, runs)
                if ts >= b - 1e-6:
                    break
                if st.multi:
                    old = [o for o in active if ts - o["t0"] >= MIN_RUN]
                    if old:
                        o = min(old, key=lambda q: q["t0"])
                        self._replace(o, float(ts), b, active, runs)
                else:
                    o = active[0]
                    if ts - o["t0"] >= MIN_RUN:
                        mix = [wants[0], wants[0], "wide"] if wants[0] != "wide" else ["wide", "any"]
                        nxt = mix[len([r for r in runs if r["t0"] >= a]) % len(mix)]
                        self._replace(o, float(ts), b, active, runs, want=nxt)
        for o in active:
            o["t1"] = end
        runs = [r for r in runs if r["t1"] - r["t0"] > 1e-3]
        for k, r in enumerate(runs):
            r["id"] = k
        return runs

    def _assign(self, a, b, slot, rect, want, active, runs, why, exclude: frozenset = frozenset()):
        got = self.pick(a, b, rect, want, active, hero=(slot == 0), exclude=exclude)
        if got is None:                  # nobody films this instant: the phone whose footage reaches furthest
            busy = {o["cam"] for o in active}
            free = [n for n in self.cams if n not in busy] or list(self.cams)
            got = (max(free, key=lambda n: V._phys_end(self.cams[n], a)), b)
        n, cov = got
        cam = self.cams[n]
        smp = _samples(self.vis, n, cam, a, cov) if self.vis.get(n) else []
        run = {"cam": n, "label": cam["label"], "slot": slot, "rect": list(rect), "t0": round(a, 3), "t1": round(b, 3),
               "want": want, "why": why, "end_cov": cov, "subject": subject(smp) if smp else "wide",
               "size": shot_size(smp) if smp else cam.get("closeness")}
        active.append(run)
        runs.append(run)
        self.history.append(n)

    def _replace(self, o, t, b, active, runs, want: str | None = None):
        o["t1"] = round(t, 3)
        active.remove(o)
        if t >= b - 1e-6:
            self.left_at[o["cam"]] = t
            return
        want = want or o["want"]
        same_ok = self.st.multi and want == o["want"] and t - o["t0"] < 16.0     # a panel may hold, a cut may not
        self._assign(t, b, o["slot"], o["rect"], want, active, runs, "swap",
                     exclude=frozenset() if same_ok else frozenset({o["cam"]}))
        new = runs[-1]
        if same_ok and new["cam"] == o["cam"] and new["rect"] == o["rect"]:
            runs.pop()                                             # still the best: the shot simply continues
            active.remove(new)
            o["t1"], o["end_cov"] = new["t1"], new["end_cov"]
            active.append(o)
            self.history.pop()
        else:
            self.left_at[o["cam"]] = t
        active.sort(key=lambda q: q["slot"])


# ----------------------------------------------------------------------------------------------- framing
def frame_run(run: dict, cam: dict, vis: dict, style: Style) -> dict:
    """Window (w, h, x, y) in picture pixels + output size, framing the wanted band member or the performers."""
    cw, ch = V._dims(cam)
    pw, ph = run["rect"][2], run["rect"][3]
    w0, h0, s0, _ = fit(cw, ch, pw, ph, style.max_zoom)
    smp = _samples(vis, run["cam"], cam, run["t0"], run["t1"])
    focus = None
    want = run["want"]
    if smp and want not in ("wide", "any", "band"):
        boxes = []
        for s in smp:
            ps = [p for p in s.get("persons") or [] if _match(p, want) >= 0.35]
            if ps:
                boxes.append(max(ps, key=lambda p: p[3] - p[1])[:4])
        if len(boxes) >= max(1, 0.4 * len(smp)):
            focus = np.median(np.asarray(boxes), axis=0) * [cw, ch, cw, ch]
            spread = float(np.std([(b[0] + b[2]) / 2 for b in boxes])) * cw
    z = 1.0
    if focus is not None:
        fh = focus[3] - focus[1]
        frac = 0.6 if run["slot"] == 0 else 0.5
        zmax = max(1.0, style.max_zoom / s0)
        z = float(np.clip(h0 * frac / max(fh, 1.0), 1.0, zmax))
        if spread > 0.25 * w0 / z:                                  # a moving performer: keep room
            z = max(1.0, z * 0.7)
    w, h = _even(w0 / z), _even(h0 / z)
    s = s0 * z
    if focus is not None:
        cx = (focus[0] + focus[2]) / 2
        head = focus[1] + 0.08 * (focus[3] - focus[1])
        x, y = cx - w / 2, head - h * (0.22 if style.fmt == "vertical" else 0.18)
    else:
        boxes = [[(p[0] * cw, p[1] * ch, p[2] * cw, p[3] * ch, p[4]) for p in s.get("persons") or []] for s in smp]
        got = V._roi_from_people(boxes, cw, ch, w, h) if any(boxes) else None
        x, y = got if got else ((cw - w) / 2, (ch - h) * 0.4)
    bands = captions_of(vis.get(run["cam"]))
    if _hits(bands, ch, y, h):                                      # a burned-in caption is in the window
        free = caption_free(bands, ch, ph / style.max_zoom)
        if free:
            fy0, fy1 = free[0] + 4.0, free[1] - 4.0                # a few px of margin survive rounding
            if h > fy1 - fy0:                                      # zoom in until the window fits the clean band
                k = (fy1 - fy0) / h
                cx_ = x + w / 2
                w, h = _even(w * k), _even(h * k)
                s = s * (1.0 / k)
                z = z / k
                x = cx_ - w / 2
            y = float(np.clip(y, fy0, fy1 - h))
    x = int(np.clip(x, 0, cw - w)) // 2 * 2
    y = int(np.ceil(np.clip(y, 0, ch - h) / 2.0)) * 2 if bands else int(np.clip(y, 0, ch - h)) // 2 * 2
    y = min(y, (ch - h) // 2 * 2)
    ow, oh = min(pw, _even(w * s)), min(ph, _even(h * s))
    if pw - ow <= 6 and ph - oh <= 6:                                 # rounding: fill the panel exactly
        ow, oh = pw, ph
    return {"win": [w, h, x, y], "out": [ow, oh], "zoom": round(z, 3), "focus": focus is not None,
            "caption_visible": _hits(bands, ch, y, h)}


# ----------------------------------------------------------------------------------------------- render
def _font(bold: bool) -> str:
    env = os.environ.get("CS_FONT_BOLD" if bold else "CS_FONT")
    if env and Path(env).exists():
        return env
    for q in (["Inter:bold", "Montserrat:bold", "DejaVu Sans:bold"] if bold else ["Inter", "Montserrat", "DejaVu Sans"]):
        r = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout and Path(r.stdout).exists():
            return r.stdout
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _esc(p: Path | str) -> str:
    """Path for a quoted filter option (quoted text is literal; a quote cannot be escaped inside it)."""
    s = str(p)
    if "'" in s:
        raise ValueError(f"path with a quote cannot be passed to an ffmpeg filter: {s}")
    return s


def _run_job(run: dict, cam: dict, style: Style, corr: dict | None, tmp: Path, enc_threads: int) -> Path:
    """Render one panel run (with handles) to an intermediate clip of the panel's size."""
    cw, ch = V._dims(cam)
    cx, cy = (cam["crop"][2], cam["crop"][3]) if cam.get("crop") else (0, 0)
    t0, t1 = run["t0"] - HANDLE, run["t1"] + HANDLE
    nfr = int(round((t1 - t0) * FPS))
    src = float(V.master_to_container(cam, t0))
    head = [f"crop={cw}:{ch}:{cx}:{cy}"]
    if src < 0:
        head.append(f"tpad=start_mode=clone:start_duration={-src:.4f}")
    head.append("tpad=stop_mode=clone:stop=-1")
    pre = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, src):.4f}", "-i", cam["video"]]
    f = list(head)
    cf = _corr_filter(corr)
    if cf:                          # colour first: ffmpeg 8.0.1 segfaults on vidstabtransform -> colorchannelmixer
        f.append(cf + ",format=yuv420p")
    if style.stabilize:
        trf = tmp / f"r{run['id']:04d}.trf"
        det = ",".join(head + [f"vidstabdetect=shakiness=6:accuracy=12:stepsize=6:mincontrast=0.2:result='{_esc(trf)}'"])
        subprocess.run(pre + ["-vf", det, "-frames:v", str(nfr), "-f", "null", "-"], capture_output=True)
        if trf.exists() and trf.stat().st_size > 0:
            sm = 24 if style.cinematic else 14
            f.append(f"vidstabtransform=input='{_esc(trf)}':smoothing={sm}:optzoom=1:maxshift={int(0.035 * cw)}"
                     f":maxangle=0.035:interpol=bicubic:crop=black")
    w, h, x, y = run["win"]
    ow, oh = run["out"]
    pw, ph = run["rect"][2], run["rect"][3]
    f.append(f"crop={w}:{h}:{x}:{y}")
    f.append("hqdn3d=1.2:1.2:3:3")
    if style.cinematic:                                             # slow push-in (alternating pull-out), sub-pixel
        z = 0.022 if run["id"] % 3 else -0.018
        a, bb = (0.0, z) if z > 0 else (-z, 0.0)
        e = f"({a}+({bb}-{a})*on/{nfr})"
        f.append(f"perspective=x0='W*{e}':y0='H*{e}':x1='W*(1-{e})':y1='H*{e}':x2='W*{e}':y2='H*(1-{e})'"
                 f":x3='W*(1-{e})':y3='H*(1-{e})':interpolation=cubic:sense=source:eval=frame")
    f.append(f"scale={ow}:{oh}:flags=lanczos")
    up = ow / max(1, w)
    f.append(f"cas={min(0.55, 0.2 + 0.2 * max(0.0, up - 1.0)):.2f}")
    vf = ",".join(f)
    if (ow, oh) != (pw, ph):                                        # leftover margin: blurred, darkened picture
        vf = (f"{vf},split[a][b];[b]scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph},"
              f"gblur=sigma={max(12, pw // 40)},eq=brightness=-0.1:saturation=0.8[bg];[bg][a]overlay=(W-w)/2:(H-h)/2")
    out = tmp / f"r{run['id']:04d}.mp4"
    cmd = pre + ["-filter_complex", f"[0:v]{vf},fps={FPS},format=yuv420p[v]", "-map", "[v]", "-frames:v", str(nfr),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-g", "15", "-bf", "0",
                 "-threads", str(enc_threads), "-an", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"run {run['id']} ({cam['label']}) failed (rc {r.returncode}): {r.stderr[-500:]}")
    return out


def _grade(W: int, H: int) -> str:
    """Filmic look: lifted blacks + highlight roll-off, gentle split tone, light bloom on stage lights, vignette,
    fine temporal grain. Deterministic per frame, so pieces join without seams."""
    sig = max(8, W // 110)
    return (f"curves=master='0/0.02 0.22/0.19 0.5/0.5 0.8/0.83 1/0.965',"
            f"colorbalance=rs=-0.02:bs=0.03:rh=0.03:bh=-0.02,eq=saturation=0.96,"
            f"split[g0][g1];[g1]curves=master='0/0 0.72/0 1/0.9',gblur=sigma={sig}[gl];"
            f"[g0][gl]blend=all_mode=screen:all_opacity=0.32,"
            f"vignette=angle=0.42,noise=alls=3:allf=t")


def _text_layers(style: Style, W: int, H: int, piece_t0: float, start: float, end: float, runs_here: list[dict],
                 tmp: Path, cams: dict, credit_names: list[str]) -> list[str]:
    """drawtext filters (timed by global time g = t + piece_t0)."""
    out = []
    fb, fr = _font(True), _font(False)
    vert = style.fmt == "vertical"
    g = f"(t+{piece_t0:.4f})"
    if style.labels:
        fs = 28 if vert else max(24, H // 52)
        for r in runs_here:
            lab = cams[r["cam"]]["label"]
            if not lab.startswith("@"):
                continue
            x, y, w, h = r["rect"]
            lx = x + (SAFE["left"] if vert and x == 0 else max(18, w // 30))
            ly = y + max(18, h // 40)
            if vert:
                ly = max(ly, SAFE["top"] + 10)
            tf = tmp / f"lab{r['id']}.txt"
            u = f"({g}-{r['t0']:.3f})"
            alpha = f"if(lt({u},0),0,if(lt({u},0.4),{u}/0.4,if(lt({u},2.6),1,if(lt({u},3.2),(3.2-{u})/0.6,0))))"
            out.append(f"drawtext=fontfile='{_esc(fb)}':textfile='{_esc(tf)}':expansion=none:fontsize={fs}:"
                       f"fontcolor=white:alpha='0.92*{alpha}':shadowcolor=black@0.6:shadowx=2:shadowy=2:"
                       f"box=1:boxcolor=black@0.28:boxborderw=12:x={lx}:y={ly}")
    if style.title and piece_t0 < start + 6.0:
        u = f"({g}-{start + 0.6:.3f})"
        alpha = f"if(lt({u},0),0,if(lt({u},0.8),{u}/0.8,if(lt({u},4.2),1,if(lt({u},5.2),(5.2-{u}),0))))"
        tf, sf = tmp / "title.txt", tmp / "subtitle.txt"
        if vert:
            fs1, fs2, x, y1 = 54, 32, "(w-text_w)/2", SAFE["top"] + 120
        else:
            fs1, fs2, x, y1 = H // 20, H // 36, f"{W // 20}", H - H // 5
        out.append(f"drawtext=fontfile='{_esc(fb)}':textfile='{_esc(tf)}':expansion=none:fontsize={fs1}:"
                   f"fontcolor=white:alpha='{alpha}':shadowcolor=black@0.7:shadowx=3:shadowy=3:x={x}:y={y1}")
        if style.subtitle:
            out.append(f"drawtext=fontfile='{_esc(fr)}':textfile='{_esc(sf)}':expansion=none:fontsize={fs2}:"
                       f"fontcolor=white:alpha='0.9*{alpha}':shadowcolor=black@0.7:shadowx=2:shadowy=2:x={x}:"
                       f"y={y1}+{int(fs1 * 1.35)}")
    if style.credits and credit_names:
        c0 = max(start, end - 9.0)
        u = f"({g}-{c0:.3f})"
        alpha = f"if(lt({u},0),0,if(lt({u},1),{u},1))"
        n_lines = len(list(tmp.glob("credits_*.txt")))
        fs = 30 if vert else max(30, H // 42)
        lh = int(fs * 1.55)
        y0 = SAFE["top"] + 40 if vert else (H - n_lines * lh) // 2
        for i in range(n_lines):
            tf = tmp / f"credits_{i:02d}.txt"
            big = i == 0
            out.append(f"drawtext=fontfile='{_esc(fb if big else fr)}':textfile='{_esc(tf)}':expansion=none:"
                       f"fontsize={int(fs * 1.25) if big else fs}:fontcolor=white:alpha='{alpha}':"
                       f"shadowcolor=black@0.8:shadowx=2:shadowy=2:x=(w-text_w)/2:y={y0 + i * lh}")
    return out


def _write_texts(style: Style, runs: list[dict], cams: dict, credit_names: list[str], tmp: Path,
                 n_total: int = 0) -> None:
    for r in runs:
        lab = cams[r["cam"]]["label"]
        (tmp / f"lab{r['id']}.txt").write_text("".join(ch for ch in lab if ch.isprintable())[:40])
    (tmp / "title.txt").write_text(style.title or "")
    (tmp / "subtitle.txt").write_text(style.subtitle or "")
    cols = 2 if style.fmt == "vertical" else 5
    n_total = max(n_total, len(credit_names))
    lines = [f"Filmed and recorded by {n_total} fans in the crowd", ""]
    for i in range(0, len(credit_names), cols):
        lines.append("    ".join(credit_names[i:i + cols]))
    if n_total > len(credit_names):
        lines += ["", f"and {n_total - len(credit_names)} more"]
    for i, ln in enumerate(lines):
        (tmp / f"credits_{i:02d}.txt").write_text(ln if ln.strip() else " ")


def _piece_job(k: int, p: dict, runs: list[dict], files: dict, style: Style, W: int, H: int, start: float,
               end: float, tmp: Path, cams: dict, credit_names: list[str], enc: list[str]) -> Path:
    a, b = p["t0"], p["t1"]
    nfr = int(round(b * FPS)) - int(round(a * FPS))
    dur = nfr / FPS
    args, chains = [], []
    idx = 0
    layer_out = []
    for li, layer in enumerate(p["layers"]):
        base = f"bg{li}"
        chains.append(f"color=c={BG}:s={W}x{H}:r={FPS}:d={dur + 0.1:.4f}[{base}]")
        cur = base
        for rid in layer:
            r = runs[rid]
            off = a - r["t0"] + HANDLE
            args += ["-ss", f"{max(0.0, off):.4f}", "-i", str(files[rid])]
            x, y = r["rect"][0], r["rect"][1]
            chains.append(f"[{idx}:v]setpts=PTS-STARTPTS[p{idx}]")
            nxt = f"l{li}_{idx}"
            chains.append(f"[{cur}][p{idx}]overlay={x}:{y}:eof_action=repeat[{nxt}]")
            cur = nxt
            idx += 1
        layer_out.append(cur)
    if len(layer_out) == 2:
        chains.append(f"[{layer_out[0]}][{layer_out[1]}]xfade=transition=fade:duration={dur:.4f}:offset=0[mix]")
        cur = "mix"
    else:
        cur = layer_out[0]
    post = []
    here = [runs[rid] for layer in p["layers"] for rid in layer]
    post += _text_layers(style, W, H, a, start, end, here, tmp, cams, credit_names)
    if style.cinematic:
        post.append(_grade(W, H))
    fi, fo = (1.2, 2.0) if style.cinematic else (0.4, 1.0)
    K = fi + fo + 1.0
    blacks = []
    if a - start < fi:
        blacks.append(f"fade=t=out:st={start - a + K:.4f}:d={fi}:alpha=1")       # from black at the start
    if end - b < fo:
        blacks.append(f"fade=t=in:st={end - fo - a + K:.4f}:d={fo}:alpha=1")     # to black at the end
    graph = ";".join(chains) + f";[{cur}]" + (",".join(post) + "," if post else "") + "format=yuv420p[c0]"
    for j, fd in enumerate(blacks):
        graph += (f";color=c=black:s={W}x{H}:r={FPS}:d={dur + K + 0.1:.4f},format=rgba,{fd},"
                  f"trim=start={K:.4f},setpts=PTS-STARTPTS[k{j}];[c{j}][k{j}]overlay=0:0,format=yuv420p[c{j + 1}]")
    graph += f";[c{len(blacks)}]null[v]"
    out = tmp / f"p{k:05d}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", graph, "-map", "[v]", "-frames:v", str(nfr),
           "-r", str(FPS), *enc, "-an", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"piece {k} failed (rc {r.returncode}): {r.stderr[-600:]}")
    return out


def pieces(runs: list[dict], style: Style, start: float, end: float) -> list[dict]:
    """Cut the timeline at every run boundary; in cinematic mode, quiet layout changes become dissolves."""
    cuts = sorted({start, end, *[r["t0"] for r in runs], *[r["t1"] for r in runs]})
    cuts = [c for c in cuts if start - 1e-6 <= c <= end + 1e-6]
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1.0 / FPS:
            continue
        on = [r["id"] for r in runs if r["t0"] <= a + 1e-6 and r["t1"] >= b - 1e-6]
        segs.append({"t0": a, "t1": b, "layers": [sorted(on, key=lambda i: runs[i]["slot"])]})
    if not style.cinematic:
        return segs
    out: list[dict] = []
    h = DISSOLVE / 2
    for i, s in enumerate(segs):
        out.append(s)
        if i + 1 >= len(segs):
            continue
        nx = segs[i + 1]
        A = {runs[r]["slot"] for r in s["layers"][0]}
        B = {runs[r]["slot"] for r in nx["layers"][0]}
        layout_change = len(s["layers"][0]) != len(nx["layers"][0])
        quiet = all(runs[r]["why"] != "slam" for r in nx["layers"][0] + s["layers"][0])
        if layout_change and quiet and A and B and s["t1"] - s["t0"] > DISSOLVE + 0.2 and nx["t1"] - nx["t0"] > DISSOLVE + 0.2:
            T = s["t1"]
            out[-1] = dict(s, t1=T - h)
            out.append({"t0": T - h, "t1": T + h, "layers": [s["layers"][0], nx["layers"][0]]})
            segs[i + 1] = dict(nx, t0=T + h)
    return out


def export(job_dir: Path, style: Style, start: float | None = None, end: float | None = None,
           highlight_s: float | None = None, audio_keys=("native", "ozone"), verify: bool = True,
           workers: int = 2, progress=None, log=print) -> dict:
    style.norm()
    roster = json.loads((job_dir / "director_roster.json").read_text())
    cams = roster["cams"]
    for n, c in cams.items():
        c.setdefault("crop", None)
    dur = float(roster["music"]["duration"])
    vis_path = job_dir / "director_vision.json"
    vis = json.loads(vis_path.read_text()) if vis_path.exists() else {}
    if not vis:
        try:
            vis = VI.analyze(job_dir, roster, log=log)
        except Exception as e:                                      # no torch: plan from audio + keyframes only
            log(f"  vision unavailable ({type(e).__name__}): framing by light/motion only")
    vis = {n: (v if isinstance(v, dict) else {"samples": v, "overlay": None}) for n, v in vis.items()}
    for v in vis.values():
        ov = v.get("overlay")
        if ov and ov[3] - ov[1] > 0.45:                             # a static bright scene, not a caption
            v["overlay"] = None
    ft = VI.featured_track(job_dir)
    if highlight_s:
        s0, s1 = highlight(roster["music"], float(highlight_s), roster)
    else:
        s0 = 0.0 if start is None else max(0.0, float(start))
        s1 = dur if end is None else min(dur, float(end))
    W, H = canvas(style, roster)
    log(f"  cut: {style.fmt} {W}x{H}, {'multi ' + str(style.min_cams) + '-' + str(style.max_cams) if style.multi else 'single'}"
        f"{', cinematic' if style.cinematic else ''}, {s0:.1f}-{s1:.1f} s")
    d = Director(roster, vis, ft, style, W, H)
    runs = d.plan(s0, s1)
    corr = color_corrections(roster, vis)
    for r in runs:
        r.update(frame_run(r, cams[r["cam"]], vis, style))
    segs = pieces(runs, style, s0, s1)
    log(f"  {len(runs)} panel runs, {len(segs)} pieces, {len({r['cam'] for r in runs})} phones on screen")
    tag = f"cut_{style.fmt}" + ("_multi" if style.multi else "") + ("_cinematic" if style.cinematic else "")
    if s0 > 0.05 or s1 < dur - 0.05:
        tag += f"_{int(s0 // 60)}m{s0 % 60:04.1f}-{int(s1 // 60)}m{s1 % 60:04.1f}".replace(".", "s")
    tmp_root = Path(os.environ.get("CS_TMP") or (job_dir / ".cut_tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cut_", dir=str(tmp_root)))
    enc = ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-profile:v", "high", "-g", str(FPS), "-threads", "6"]
    # everyone whose phone is in the remaster (the audio comp uses phones that never appear on screen too)
    credit = sorted({c["label"] for c in cams.values() if c["label"].startswith("@")}, key=str.lower)
    try:
        _write_texts(style, runs, cams, credit, tmp, n_total=len(cams))
        files: dict[int, Path] = {}
        threads = max(2, (os.cpu_count() or 8) // workers)
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_run_job, r, cams[r["cam"]], style, corr.get(r["cam"]), tmp, threads): r["id"] for r in runs}
            for k, fu in enumerate(futs):
                files[futs[fu]] = fu.result()
                if progress:
                    progress(55.0 * (k + 1) / len(runs), f"panel {k + 1}/{len(runs)}")
        with ThreadPoolExecutor(workers) as ex:
            jobs = [ex.submit(_piece_job, k, p, runs, files, style, W, H, s0, s1, tmp, cams, credit, enc)
                    for k, p in enumerate(segs)]
            outs = []
            for k, fu in enumerate(jobs):
                outs.append(fu.result())
                if progress:
                    progress(55.0 + 35.0 * (k + 1) / len(jobs), f"piece {k + 1}/{len(jobs)}")
        (tmp / "list.txt").write_text("".join(f"file '{o.name}'\n" for o in outs))
        vid = tmp / "video.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy",
                        str(vid)], check=True, cwd=str(tmp))
        files_a = {"native": job_dir / "studio_hd_master.wav", "ozone": job_dir / "studio_hd_ozone_master.wav"}
        audio = {k: files_a[k] for k in audio_keys if files_a[k].exists()}
        span = (int(round(s1 * FPS)) - int(round(s0 * FPS))) / FPS
        written = []
        for name, wav in audio.items():
            af = []
            if s0 > 0.05:
                af.append("afade=t=in:st=0:d=0.3")
            if s1 < dur - 0.05:
                af.append(f"afade=t=out:st={max(0.0, span - 1.5):.3f}:d=1.5")
            dst = job_dir / f"{tag}_{name}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-ss", f"{s0:.4f}", "-t", f"{span:.4f}",
                            "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                            "-ar", "48000", *(["-af", ",".join(af)] if af else []), "-movflags", "+faststart",
                            "-shortest", str(dst)], check=True)
            written.append(dst.name)
            log(f"  wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
        if progress:
            progress(100.0, "done")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sync = []
    if verify:
        pseudo = [{"shot_index": r["id"], "camera": r["cam"], "camera_name": r["label"], "start_sec": r["t0"],
                   "duration_sec": r["t1"] - r["t0"], "src_in_sec": float(V.master_to_container(cams[r["cam"]], r["t0"]))}
                  for r in runs if float(V.master_to_container(cams[r["cam"]], r["t0"])) >= 0]
        sync = V.verify_sync(roster, pseudo, job_dir / "studio_hd_master.wav")
    good = [x for x in sync if x["score"] >= 8]
    rep = {"style": asdict(style), "canvas": [W, H], "start": s0, "end": s1, "files": written,
           "runs": [{k: v for k, v in r.items() if k != "end_cov"} for r in runs], "pieces": len(segs),
           "color": corr, "sync": sync,
           "sync_median_abs_ms": float(np.median([abs(x["offset_ms"]) for x in good])) if good else None,
           "sync_max_abs_ms": float(max(abs(x["offset_ms"]) for x in good)) if good else None,
           "credits": credit}
    (job_dir / f"{tag}_report.json").write_text(json.dumps(rep, indent=1, default=float))
    log(f"  sync: median {rep['sync_median_abs_ms']} ms, max {rep['sync_max_abs_ms']} ms over {len(good)} checked panels")
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="ConcertStudio multi-camera cut (all modes)")
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--format", default="vertical", choices=["vertical", "landscape"])
    ap.add_argument("--multi", action="store_true", help="show 1-4 perspectives at once")
    ap.add_argument("--cinematic", action="store_true")
    ap.add_argument("--min-cams", type=int, default=1)
    ap.add_argument("--max-cams", type=int, default=4)
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--highlight", type=float, help="auto-pick the best N seconds (Shorts: <= 60 recommended)")
    ap.add_argument("--audio", default="native,ozone")
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-credits", action="store_true")
    ap.add_argument("--no-stabilize", action="store_true")
    ap.add_argument("--max-zoom", type=float, default=V.MAX_ZOOM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2, help="parallel panel renders (each ~1-1.5 GB at 1440p)")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    st = Style(fmt=a.format, multi=a.multi, cinematic=a.cinematic, min_cams=a.min_cams, max_cams=a.max_cams,
               max_zoom=a.max_zoom, labels=not a.no_labels, credits=not a.no_credits, title=a.title,
               subtitle=a.subtitle, stabilize=not a.no_stabilize, seed=a.seed)
    rep = export(a.job, st, a.start, a.end, a.highlight, tuple(a.audio.split(",")), verify=not a.no_verify,
                 workers=a.workers, progress=lambda p, d: print(f"  {p:5.1f}% {d}", flush=True))
    print(json.dumps({k: rep[k] for k in ("files", "canvas", "pieces", "sync_median_abs_ms", "sync_max_abs_ms")}))


if __name__ == "__main__":
    main()
