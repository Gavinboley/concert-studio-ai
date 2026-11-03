"""Curate a multi-POV payload: master list of unique recordings, trash list, cleaned per-clip WAVs.

Runs after `align` (dedup groups + orphans), `crowd`, `quality.py` and a plan-only comp
(`comp --set plan_only=1`, which dumps per-take validity on the timeline grid to comp<name>/scores_dump.npz).

Cleaning is conservative and never touches the music itself:
  * trim each clip to the span where it verifiably matches the other phones (lag-confirmed or a musical edge
    extension), which drops TikTok intro/outro cards, transition sounds and uploader talk at the edges;
  * mute (50 ms fades) internal stretches of foreign audio: the clip is loud while >= 3 other takes agree with
    each other but not with it (licensed music laid over the video, voice-overs, re-cut segments);
  * DC/sub-20 Hz removal, scalar gain to -18 LUFS (peak-safe), 24-bit WAV; HE-AACv2 sources (parametric,
    synthetic stereo) are written mono, true-stereo sources stay stereo.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
GRID = 0.5
TARGET_LUFS = -18.0


def _lufs(x: np.ndarray) -> float:
    import pyloudnorm as pyln
    v = pyln.Meter(SR).integrated_loudness(x if x.ndim == 1 else x)
    return float(v) if np.isfinite(v) else -70.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def curate(work: Path, dest: Path, comp_name: str, records: list[dict] | None = None,
           min_usable_s: float = 20.0, min_ok_frac: float = 0.25, log=print) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    man = json.loads((work / "manifest.json").read_text())
    man = man["clips"] if isinstance(man, dict) else man
    info = {c["name"]: c for c in man}
    sel = json.loads((work / "selection.json").read_text())
    pls = {p["name"]: p for p in json.loads((work / "placements.json").read_text())}
    D = np.load(work / f"comp{comp_name}" / "scores_dump.npz", allow_pickle=True)
    names = list(D["names"])
    ok = np.isfinite(D["S0"])
    grid, pos, loud, PQ, CE, vco = D["grid"], D["pos"], D["loud"], D["PQ"], D["CE"], D["vcoh"]
    cnt = ok.sum(axis=0)
    t_origin = float(grid[np.argmax(cnt > 0)])
    by_rank = {int(r["rank"]): r for r in (records or [])}

    def meta(n: str) -> dict:
        if n.startswith("new") and n[3:5].isdigit() and int(n[3:5]) in by_rank:
            r = by_rank[int(n[3:5])]
            return {"set": "new", "uploader": r["uploader"], "url": r["url"], "original_file": r.get("best_audio_file")
                    or r["video_file"], "title": r.get("title", ""), "note_dl": r.get("note", "")}
        return {"set": "old" if n.startswith("old_") else "?", "uploader": "", "url": "",
                "original_file": Path(info[n]["src"]).name.removeprefix("old_"), "title": "", "note_dl": ""}

    def src_kind(n: str) -> str:
        c = info[n]
        prof = str(c.get("profile") or "")
        if c.get("codec") == "mp3":
            return f"MP3 {c['bitrate'] // 1000}k (TikTok sound, full-band)"
        return f"AAC {prof or 'LC'} {c['bitrate'] // 1000}k"

    fs = 16000
    _low: dict[int, np.ndarray] = {}

    def low(j: int) -> np.ndarray:
        if j not in _low:
            y, _ = sf.read(info[names[j]]["wav"], dtype="float32", always_2d=True)
            _low[j] = signal.resample_poly(y.mean(axis=1), 160, 441).astype(np.float32)
        return _low[j]

    def _coh(i: int, j: int, k0: int, k1: int) -> float:
        a, b = int(pos[i][k0] * fs), int(pos[i][k1] * fs)
        c = int(pos[j][k0] * fs)
        n_ = b - a
        if n_ < fs * 2 or a < 0 or c < 0 or b > len(low(i)) or c + n_ > len(low(j)):
            return float("nan")
        f_, cxy = signal.coherence(low(i)[a:b], low(j)[c:c + n_], fs, nperseg=1024)
        return float(np.mean(cxy[(f_ > 300) & (f_ < 3000)]))

    def _seg_coherence(i: int, k0: int, k1: int) -> tuple[float, float]:
        ctx0, ctx1 = max(0, k0 - 50), min(len(grid) - 1, k1 + 50)
        refs = [j for j in np.argsort(-ok[:, ctx0:ctx1 + 1].sum(axis=1)) if j != i
                and ok[j, k0:k1].mean() > 0.9 and ok[j, ctx0:ctx1 + 1].mean() > 0.9][:3]
        if not refs:
            return float("nan"), float("nan")
        inside = np.nanmedian([_coh(i, j, k0, k1 - 1) for j in refs])
        ctx = []
        for a_, b_ in ((ctx0, k0 - 4), (k1 + 4, ctx1)):
            if b_ - a_ >= 10 and ok[i, a_:b_].mean() > 0.5:
                ctx += [_coh(i, j, a_, b_) for j in refs]
        return float(inside), float(np.nanmedian(ctx)) if ctx else float("nan")

    dup_of = {d: k for k, v in sel["duplicates"].items() for d in v}
    keep_rows, trash_rows = [], []
    for n in sel["orphans"]:
        trash_rows.append({"name": n, **meta(n), "source": src_kind(n), "reason":
                           "orphan: does not overlap the performance (post-song talk/applause or other audio)"})
    for d, k in dup_of.items():
        trash_rows.append({"name": d, **meta(d), "source": src_kind(d), "reason": f"duplicate re-upload of {k}"})

    order = sorted([n for n in sel["keep"] if n in names], key=lambda n: pls[n]["offset_s"])
    idx = 0
    for n in order:
        i = names.index(n)
        m = meta(n)
        okc = ok[i]
        if not okc.any():
            trash_rows.append({"name": n, **m, "source": src_kind(n), "reason": "never matches the other phones"})
            continue
        g0, g1 = np.where(okc)[0][[0, -1]]
        span_grid = np.zeros_like(okc)
        span_grid[g0:g1 + 1] = True
        # foreign audio: inside the span, not ok, loud, while >= 3 other takes are ok
        others = cnt - okc.astype(int)
        lmed = float(np.median(loud[i][okc]))
        foreign = span_grid & ~okc & (others >= 3) & (loud[i] > lmed - 10.0)
        cand = [(a, b) for a, b in _runs(foreign) if (b - a) * GRID >= 3.0]
        # only audio that stops matching the other phones is foreign; many takes also lose lag/level validation
        # during loud crowd moments while still carrying the concert (coherence unchanged) -> keep + flag those
        fr, unverified = [], []
        for a, b in cand:
            r_in, r_ctx = _seg_coherence(i, a, b)
            if np.isfinite(r_in) and np.isfinite(r_ctx) and r_in < 0.5 * r_ctx:
                fr.append((a, b, r_in, r_ctx))
            else:
                unverified.append((a, b, r_in, r_ctx))
        usable_s = float(okc.sum() * GRID)
        ok_frac = usable_s / max((g1 - g0 + 1) * GRID, 1e-9)
        dur = float(D["dur"][i])
        c0 = max(0.0, float(pos[i][g0]) - GRID / 2)
        c1 = min(dur, float(pos[i][g1]) + GRID / 2)
        pq = float(np.nanmean(np.where(okc, PQ[i], np.nan))) if np.isfinite(PQ[i]).any() else float("nan")
        ce = float(np.nanmean(np.where(okc, CE[i], np.nan))) if np.isfinite(CE[i]).any() else float("nan")
        vc = float(np.nanmean(np.where(okc, vco[i], np.nan))) if np.isfinite(vco[i]).any() else float("nan")
        dups = sel["duplicates"].get(n, [])
        row = {"name": n, **m, "source": src_kind(n), "timeline_start_s": round(float(grid[g0]) - t_origin, 1),
               "timeline_end_s": round(float(grid[g1]) - t_origin, 1), "clip_in_s": round(c0, 2),
               "clip_out_s": round(c1, 2), "orig_dur_s": round(dur, 1), "usable_s": round(usable_s, 1),
               "trim_head_s": round(c0, 1), "trim_tail_s": round(dur - c1, 1),
               "muted": "; ".join(f"{float(pos[i][a]):.1f}-{float(pos[i][b - 1]) + GRID:.1f}s (coh {ri:.2f} vs {rc:.2f})"
                                  for a, b, ri, rc in fr),
               "unverified_kept": "; ".join(f"{float(pos[i][a]):.1f}-{float(pos[i][b - 1]) + GRID:.1f}s (coh {ri:.2f} vs {rc:.2f})"
                                            for a, b, ri, rc in unverified),
               "PQ": round(pq, 2), "CE": round(ce, 2), "vocal_coherence": round(vc, 3),
               "duplicates_merged": len(dups), "duplicate_names": " ".join(dups)}
        if usable_s < min_usable_s or ok_frac < min_ok_frac:
            row["reason"] = (f"too little matching audio ({usable_s:.0f} s usable, {100 * ok_frac:.0f}% of its span)")
            trash_rows.append(row)
            continue
        # ---- clean + export
        x, sr = sf.read(info[n]["wav"], dtype="float32", always_2d=True)
        assert sr == SR
        stereo_real = x.shape[1] == 2 and "HE-AACv2" not in str(info[n].get("profile"))
        x = x if stereo_real else x.mean(axis=1, keepdims=True)
        a_, b_ = int(c0 * SR), int(c1 * SR)
        y = x[a_:b_].copy()
        sos = signal.butter(2, 20.0, "highpass", fs=SR, output="sos")
        y = signal.sosfiltfilt(sos, y, axis=0).astype(np.float32)
        env = np.ones(len(y), np.float32)
        fl = int(0.05 * SR)
        for a, b, _ri, _rc in fr:
            s0 = int((float(pos[i][a]) - c0) * SR)
            s1 = int((float(pos[i][b - 1]) + GRID - c0) * SR)
            s0, s1 = max(0, s0), min(len(y), s1)
            env[s0:s1] = 0.0
            env[max(0, s0 - fl):s0] = np.minimum(env[max(0, s0 - fl):s0], np.linspace(1, 0, s0 - max(0, s0 - fl)))
            env[s1:s1 + fl] = np.minimum(env[s1:s1 + fl], np.linspace(0, 1, len(env[s1:s1 + fl])))
        fe = int(0.02 * SR)
        env[:fe] *= np.linspace(0, 1, fe)
        env[-fe:] *= np.linspace(1, 0, fe)
        y *= env[:, None]
        L = _lufs(y if y.shape[1] == 2 else y[:, 0])
        g = 10 ** ((TARGET_LUFS - L) / 20.0)
        pk = float(np.max(np.abs(y))) * g
        if pk > 10 ** (-1.0 / 20):
            g *= 10 ** (-1.0 / 20) / pk
        y *= g
        idx += 1
        label = (m["uploader"] or n.removeprefix("old_")[-10:]).replace(".", "_")
        fname = f"{idx:02d}_t{row['timeline_start_s']:05.0f}s_{label}.wav"
        sf.write(dest / fname, y if y.shape[1] == 2 else y[:, 0], SR, subtype="PCM_24")
        row.update({"no": idx, "file": fname, "channels": "stereo" if y.shape[1] == 2 else "mono (HE-AACv2 PS)",
                    "gain_db": round(20 * np.log10(g), 1)})
        keep_rows.append(row)
        log(f"  {fname:52s} {usable_s:6.0f}s usable  PQ {pq:.2f}  muted {row['muted'] or '-'}"
            f"  unverified {row['unverified_kept'] or '-'}")

    cols = ["no", "file", "set", "uploader", "url", "original_file", "source", "channels", "timeline_start_s",
            "timeline_end_s", "clip_in_s", "clip_out_s", "orig_dur_s", "usable_s", "trim_head_s", "trim_tail_s",
            "muted", "unverified_kept", "gain_db", "PQ", "CE", "vocal_coherence", "duplicates_merged", "duplicate_names",
            "note_dl", "title", "name"]
    with open(dest / "MASTER_LIST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep_rows)
    tcols = ["name", "set", "uploader", "url", "original_file", "source", "reason", "note_dl"]
    with open(dest / "TRASH_LIST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tcols, extrasaction="ignore")
        w.writeheader()
        w.writerows(trash_rows)
    return {"keep": keep_rows, "trash": trash_rows, "t_origin": t_origin}
