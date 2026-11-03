"""V8 stage runner. Each stage caches into the work dir so stages can be re-run independently.

    python -m concert_studio.v8.run align   --src <clips dir> --work ~/cstudio_work
    python -m concert_studio.v8.run crowd   --work ~/cstudio_work
    python -m concert_studio.v8.run comp    --work ~/cstudio_work
    python -m concert_studio.v8.run stems   --work ~/cstudio_work
    python -m concert_studio.v8.run mix     --work ~/cstudio_work --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _soften_gaps(sig: np.ndarray, w: np.ndarray, sr: int, fade_s: float = 0.25, min_gap_s: float = 1.0) -> None:
    """In place: raised-cosine fade-out before and fade-in after every gap (>= min_gap_s) inside the song."""
    on = np.asarray(w)[: len(sig)] > 0.5
    idx = np.where(on)[0]
    if len(idx) < 2:
        return
    a, b = idx[0], idx[-1]
    d = np.diff(np.r_[0, (~on[a:b]).astype(np.int8), 0])
    starts, ends = np.where(d == 1)[0] + a, np.where(d == -1)[0] + a
    n = int(fade_s * sr)
    for s0, e0 in zip(starts, ends):
        if e0 - s0 < min_gap_s * sr:
            continue
        lo, hi = max(0, s0 - n), min(len(sig), e0 + n)
        out_r = (np.cos(np.linspace(0, np.pi / 2, s0 - lo)) ** 2).astype(np.float32)
        in_r = (np.sin(np.linspace(0, np.pi / 2, hi - e0)) ** 2).astype(np.float32)
        sig[lo:s0] *= out_r if sig.ndim == 1 else out_r[:, None]
        sig[e0:hi] *= in_r if sig.ndim == 1 else in_r[:, None]


def _models_dir(work: Path) -> Path:
    """Separation model cache: $CS_MODELS if set (share one cache across work dirs), else <work>/models."""
    env = os.environ.get("CS_MODELS")
    return Path(env).expanduser() if env else work / "models"


def stage_align(src: Path, work: Path, ref: str | None) -> None:
    from .align import align_clips
    from .ingest import ingest_folder, load_mono, save_manifest

    clips = ingest_folder(src, work / "raw")
    save_manifest(clips, work / "manifest.json")
    monos = {c.name: load_mono(c) for c in clips}
    if ref is None:
        # longest LC (true-stereo, full-band) clip anchors the timeline
        lc = [c for c in clips if not c.sbr] or clips
        ref = max(lc, key=lambda c: c.duration).name
    print(f"reference: {ref}")
    pl = align_clips([c.name for c in clips], monos, ref)
    (work / "placements.json").write_text(json.dumps([p.to_dict() for p in pl], indent=1))
    print(f"wrote {work / 'placements.json'}")

    from .align import overlap_matrix, pairwise_coherence, select_clips
    pmap = {p.name: p for p in pl}
    names = [p.name for p in pl]
    M = pairwise_coherence(names, monos, pmap)
    info = {c.name: c for c in clips}
    rank = {n: (not info[n].sbr, info[n].bitrate >= 96000, info[n].duration) for n in names}
    sel = select_clips(names, M, rank, overlap=overlap_matrix(names, pmap))
    sel["islands"] = {n: pmap[n].island for n in names}
    if not sel["keep"] and names:                       # never end with nothing: keep the best-ranked clip
        best = max(names, key=lambda n: rank[n])
        sel["keep"] = [best]
        sel["orphans"] = [n for n in sel["orphans"] if n != best]
        print(f"WARNING: every clip looked like an orphan; keeping the best one ({best})")
    sel["coherence"] = {n: float(np.max(M[i])) for i, n in enumerate(names)}
    (work / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"keep {len(sel['keep'])}, duplicate groups {sel['duplicates']}, orphans {sel['orphans']}")


def stage_crowd(work: Path) -> None:
    """De-crowd every kept clip (mono). Outputs work/crowd/<name>_music.wav and _crowd.wav."""
    import soundfile as sf
    from .ingest import load_manifest, load_mono, SR
    from .neural import Runner
    clips = {c.name: c for c in load_manifest(work / "manifest.json")}
    sel = json.loads((work / "selection.json").read_text())
    out = work / "crowd"
    out.mkdir(exist_ok=True)
    r = Runner(_models_dir(work), work / "tmp_sep")
    for i, n in enumerate(sorted(sel["keep"], key=lambda n: -clips[n].duration)):
        if (out / f"{n}_crowd.wav").exists():
            continue
        t = time.time()
        x = load_mono(clips[n])
        res = r.run("crowd", x, SR, n)
        sf.write(out / f"{n}_music.wav", res["other"], SR, subtype="FLOAT")
        sf.write(out / f"{n}_crowd.wav", res["crowd"], SR, subtype="FLOAT")
        print(f"  [{i + 1}/{len(sel['keep'])}] {n} {len(x) / SR:.0f}s in {time.time() - t:.0f}s", flush=True)


def stage_comp(work: Path, switch_cost: float = 20.0, min_dwell_s: float = 6.0, name: str = "", opts: dict | None = None,
               tv_eq: bool = True) -> None:
    import soundfile as sf
    from .align import Placement, pairwise_coherence
    from .comp import (GRID, Track, build_lag_tracks, compute_features, reject_level_anomalies, take_scores, viterbi,
                       timevarying_consensus_eq, consensus_loudness_lock,
                       eq_match_filters, loudness_gains, render_path, split_bands)
    from .ingest import load_manifest, load_mono, SR

    clips = {c.name: c for c in load_manifest(work / "manifest.json")}
    pls = {d["name"]: d for d in json.loads((work / "placements.json").read_text())}
    sel = json.loads((work / "selection.json").read_text())
    tracks = {}
    missing = [n for n in sel["keep"] if not (work / "crowd" / f"{n}_crowd.wav").exists()]
    if missing:
        print(f"WARNING: {len(missing)} kept clips have no crowd stems yet; comping without them")
    for n in [k for k in sel["keep"] if k not in missing]:
        m, _ = sf.read(work / "crowd" / f"{n}_music.wav", dtype="float32")
        c, _ = sf.read(work / "crowd" / f"{n}_crowd.wav", dtype="float32")
        tracks[n] = Track(n, clips[n], pls[n]["offset_s"], pls[n]["drift_ppm"], m, c, None)
    names = list(tracks)
    print(f"comp over {len(names)} takes")

    # coherence on the de-crowded music (cleaner than raw)
    plc = {n: Placement(n, tracks[n].offset, tracks[n].drift, 0.0, tracks[n].dur) for n in names}
    M = pairwise_coherence(names, {n: tracks[n].music for n in names}, plc)
    coh = {(a, b): float(M[i, j]) for i, a in enumerate(names) for j, b in enumerate(names) if i != j}
    best = {n: float(M[i].max()) for i, n in enumerate(names)}
    # lag validity only spreads where the anchor is: it must span the song. Among full-band takes at least 90%
    # as long as the longest, take the best-connected one (a short, highly coherent end clip once won here).
    isl = {n: int(pls[n].get("island", 0)) for n in names}
    anchors = []
    for k in sorted(set(isl.values())):                    # one anchor per island (sparse payloads)
        members = [n for n in names if isl[n] == k]
        fb = [n for n in members if not clips[n].sbr] or members
        longest = max(tracks[n].dur for n in fb)
        cand = [n for n in fb if tracks[n].dur >= 0.9 * longest]
        anchors.append(max(cand, key=lambda n: float(np.sort(M[names.index(n)])[-4:].sum())))
    anchor = anchors[0]
    print(f"anchor {anchor}" + (f" (+ {len(anchors) - 1} island anchors: {anchors[1:]})" if len(anchors) > 1 else ""))
    build_lag_tracks(tracks, coh, anchors)

    t_end = max(t.span()[1] for t in tracks.values())
    feats = compute_features(tracks, t_end)
    reject_level_anomalies(tracks, feats)
    grid = feats["grid"]
    from .comp import vocal_band_coherence, quality_on_grid
    opts = opts or {}
    w_pq, w_ce, w_coh = float(opts.get("w_pq", 0.0)), float(opts.get("w_ce", 0.0)), float(opts.get("w_coh", 0.0))
    switch_cost = float(opts.get("switch_cost", switch_cost))
    min_dwell_s = float(opts.get("min_dwell_s", min_dwell_s))
    qf = work / "take_quality.npz"
    quality = quality_on_grid(dict(np.load(qf, allow_pickle=True)), names, grid) if qf.exists() else None
    vcoh = vocal_band_coherence(tracks, feats)
    from .comp import take_pumping
    pump = take_pumping(tracks, feats)
    w_pump = float(opts.get("w_pump", 0.0))
    print(f"  perceptual weights: w_pq {w_pq} w_ce {w_ce} w_coh {w_coh} (quality file: {qf.exists()}), "
          f"switch_cost {switch_cost}, min_dwell {min_dwell_s}s")
    S, H = take_scores(tracks, feats, best, quality, vcoh, w_pq, w_ce, w_coh, pump, w_pump)
    out = work / f"comp{name}"
    out.mkdir(exist_ok=True)
    S0, H0 = take_scores(tracks, feats, best)
    np.savez(out / "scores_dump.npz", S0=S0, H0=H0, grid=grid, names=np.array(names),
             sbr=np.array([clips[n].sbr for n in names]),
             cr=np.stack([feats["f"][n]["cr"] for n in names]),
             vcoh=np.stack([vcoh[n] for n in names]),
             PQ=np.stack([quality[n]["PQ"] if quality and n in quality else np.full(len(grid), np.nan) for n in names]),
             CE=np.stack([quality[n]["CE"] if quality and n in quality else np.full(len(grid), np.nan) for n in names]),
             ext=np.stack([feats["f"][n]["ext"] for n in names]),
             pump=np.stack([pump[n] for n in names]),
             loud=np.stack([feats["f"][n]["loud"] for n in names]),
             pos=np.stack([tracks[n].pos(grid) for n in names]),
             dur=np.array([tracks[n].dur for n in names]))
    if opts.get("plan_only"):
        print(f"plan only: wrote {out / 'scores_dump.npz'}")
        return
    dwell = int(min_dwell_s / GRID)
    pathL = viterbi(S, switch_cost, dwell)
    pathH = viterbi(H, switch_cost, dwell)
    SB = S.copy()
    for t_, c_ in enumerate(pathL):
        if c_ >= 0:
            SB[c_, t_] = -np.inf
    for i_, n_ in enumerate(names):
        if clips[n_].sbr:
            SB[i_] -= 6.0                     # B feeds the stereo side incl. highs: prefer full-band takes
    pathB = viterbi(SB, switch_cost, dwell)
    # where no LC take exists for the highs, fall back to the low-band take (HF band is then muted >7k)
    print(f"low-band switches {int((np.diff(pathL) != 0).sum())}, high-band switches {int((np.diff(pathH) != 0).sum())}")
    used = {names[i] for i in set(pathL.tolist()) | set(pathH.tolist()) | set(pathB.tolist()) if i >= 0}
    if tv_eq:
        timevarying_consensus_eq(tracks, feats, used)
        filt, src = {}, "music_eq"
    else:
        filt, src = eq_match_filters(tracks, feats), "music"
    gains = loudness_gains(tracks, feats)
    gains_hf = loudness_gains(tracks, feats, key="loud_hf")
    n_out = int(t_end * SR)
    musL, wL = render_path(tracks, names, pathL, grid, src, filt, gains, n_out, band=(0, 7000), log=print)
    musH, wH = render_path(tracks, names, pathH, grid, src, filt, gains_hf, n_out, band=(7000, 16000),
                           power=True, log=print)
    crowd, _ = render_path(tracks, names, pathL, grid, "crowd", {}, gains, n_out, power=True)
    lo, _ = split_bands(musL)
    _, hi = split_bands(musH)
    hi *= np.clip(wH, 0.0, 1.0)
    cons_lock = float(opts.get("cons_lock", 1.0)) > 0
    comp = consensus_loudness_lock(lo + hi, feats) if cons_lock else lo + hi
    from .comp import fast_level_curves, fast_consensus_lock
    fast_strength = float(opts.get("fast_lock", 0.8))
    if fast_strength > 0:
        T_, _, cons_, cnt_ = fast_level_curves(tracks, feats)
        comp = fast_consensus_lock(comp, T_, cons_, cnt_, strength=fast_strength)
    musB, wB = render_path(tracks, names, pathB, grid, src, filt, gains, n_out, band=(0, 16000), log=print)
    # B only exists to provide a real second microphone position; where it has no take, fall back to A
    musB = musB * np.clip(wB, 0, 1) + comp * (1 - np.clip(wB, 0, 1))
    compB = consensus_loudness_lock(musB, feats) if cons_lock else musB
    if fast_strength > 0:
        compB = fast_consensus_lock(compB, T_, cons_, cnt_, strength=fast_strength)
    # islands / unusable stretches inside the song: fade around the silence so it plays like an edit, not a dropout
    for sig_ in (comp, compB, crowd):
        _soften_gaps(sig_, wL, SR)
    # trim leading/trailing silence where nothing was usable
    act = np.where(wL > 0.5)[0]
    a, b = int(act[0]), int(act[-1]) + 1
    sf.write(out / "comp_music.wav", comp[a:b], SR, subtype="FLOAT")
    sf.write(out / "comp_crowd.wav", crowd[a:b], SR, subtype="FLOAT")
    sf.write(out / "comp_music_B.wav", compB[a:b], SR, subtype="FLOAT")
    usage = {}
    for i, n in enumerate(names):
        usage[n] = {"low_s": float((pathL == i).sum() * GRID), "high_s": float((pathH == i).sum() * GRID),
                    "profile": clips[n].profile}
    (out / "comp_path.json").write_text(json.dumps({
        "grid_s": GRID, "trim_start_s": a / SR, "names": names, "pathL": pathL.tolist(), "pathH": pathH.tolist(),
        "pathB": pathB.tolist(), "opts": opts,
        "usage": usage, "anchor": anchor,
        "lag_ms": {n: (tracks[n].lag_e * 1000).round(3).tolist() for n in names},
    }))
    for n, u in sorted(usage.items(), key=lambda kv: -kv[1]["low_s"]):
        if u["low_s"] or u["high_s"]:
            print(f"  {n:36s} {u['profile']:9s} low {u['low_s']:6.1f}s  high {u['high_s']:6.1f}s")
    print(f"comp length {(b - a) / SR:.1f}s")


VOCAL_ENSEMBLE = ["mel_band_roformer_kim_ft3_unwa.ckpt", "melband_roformer_big_beta6x.ckpt",
                  "mel_band_roformer_vocals_becruily.ckpt"]


def stage_stems(work: Path, name: str = "") -> None:
    """Vocal ensemble on the comp, 6-stem SW on the instrumental, de-reverb on the vocal."""
    import soundfile as sf
    from .ingest import SR
    from .neural import Runner, MODELS
    comp, _ = sf.read(work / f"comp{name}" / "comp_music.wav", dtype="float32")
    out = work / f"stems{name}"
    out.mkdir(exist_ok=True)
    r = Runner(_models_dir(work), work / "tmp_sep")
    vox = []
    for i, m in enumerate(VOCAL_ENSEMBLE):
        f = out / f"vocals_m{i}.wav"
        if not f.exists():
            MODELS[f"v{i}"] = m
            t = time.time()
            res = r.run(f"v{i}", comp, SR, f"comp_v{i}")
            v = res.get("vocals")
            if v is None:
                raise RuntimeError(f"{m}: stems {list(res)}")
            sf.write(f, v, SR, subtype="FLOAT")
            print(f"  vocal model {m} in {time.time() - t:.0f}s", flush=True)
        vox.append(sf.read(f, dtype="float32")[0])
    v = np.mean(np.stack(vox), axis=0)
    sf.write(out / "vocals_ens.wav", v, SR, subtype="FLOAT")
    inst = comp - v
    sf.write(out / "instrumental.wav", inst, SR, subtype="FLOAT")
    t = time.time()
    res = r.run("stems6", inst, SR, "comp_inst")
    for k, y in res.items():
        sf.write(out / f"sw_{k}.wav", y, SR, subtype="FLOAT")
    print(f"  SW 6-stem on instrumental in {time.time() - t:.0f}s: {sorted(res)}", flush=True)
    fB = work / f"comp{name}" / "comp_music_B.wav"
    if fB.exists():
        t = time.time()
        cB, _ = sf.read(fB, dtype="float32")
        resB = r.run("stems6", cB, SR, "comp_B")
        for k, y in resB.items():
            sf.write(out / f"swB_{k}.wav", y, SR, subtype="FLOAT")
        print(f"  SW 6-stem on comp B in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    res = r.run("dereverb", v, SR, "comp_vox")
    for k, y in res.items():
        sf.write(out / f"vocals_{k}.wav", y, SR, subtype="FLOAT")
    print(f"  vocal de-reverb in {time.time() - t:.0f}s: {sorted(res)}", flush=True)


def stage_dereverb(work: Path, name: str = "") -> None:
    import soundfile as sf
    from .ingest import SR
    from .neural import Runner
    out = work / f"stems{name}"
    v, _ = sf.read(out / "vocals_ens.wav", dtype="float32")
    r = Runner(_models_dir(work), work / "tmp_sep")
    res = r.run("dereverb", v, SR, "comp_vox")
    for k, y in res.items():
        sf.write(out / f"vocals_{k}.wav", y, SR, subtype="FLOAT")
    print(f"  dereverb: " + ", ".join(f"{k} {20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12):.1f} dB" for k, y in res.items()))


def stage_mix(work: Path, out: Path, tag: str = "v8", overrides: str | None = None, name: str = "") -> None:
    import soundfile as sf
    from .master import MixParams, load_stems, mix_and_master
    p = MixParams()
    if overrides:
        for kv in overrides.split(","):
            k, v = kv.split("=")
            setattr(p, k.strip(), type(getattr(p, k.strip()))(v))
    stems = load_stems(work / f"stems{name}")
    crowd, _ = sf.read(work / f"comp{name}" / "comp_crowd.wav", dtype="float32")
    rep = mix_and_master(stems, crowd, p, out, tag=tag)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_eq") and k != "params"}, indent=1))
    for k, v in rep.items():
        if k.endswith("_eq"):
            print(f"  {k}: " + " ".join(f"{f}:{g:+.1f}" for f, g in v.items() if abs(g) >= 1.0))


def stage_publish(work: Path, out: Path, tag: str, repo: Path, name: str = "") -> None:
    """Stage a V8 render as a Web UI job (same schema the UI already reads) + A/B against V7."""
    import shutil
    from .analyze import analyze
    n_takes = len(json.loads((work / "selection.json").read_text())["keep"])
    job = repo / "workspace_ui" / (f"tiktok_31_{tag}" if n_takes <= 31 else f"tiktok_{n_takes}_{tag}")
    job.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / f"{tag}_master_48k24.wav", job / "studio_hd_master.wav")
    oz = out / f"{tag}_ozone_master_44k24.wav"
    if oz.exists():
        shutil.copy2(oz, job / "studio_hd_ozone_master.wav")
    import soundfile as _sf
    for f in sorted(out.glob(f"{tag}_stem_*.wav")):
        if _sf.info(f).frames and not np.any(_sf.read(f, frames=_sf.info(f).frames, dtype="float32")[0]):
            continue                                   # skip silent stems (e.g. folded piano)
        shutil.copy2(f, job / f.name.replace(f"{tag}_stem_", "stem_"))
    v7 = repo / "outputs" / "tiktok_concert" / "studio_hd_master.wav"
    if v7.exists():
        shutil.copy2(v7, job / "compare_V7_agy_master.wav")
    base = repo / "outputs" / "tiktok_concert" / "mastering_report.json"
    rep = json.loads(base.read_text()) if base.exists() else {}
    m = analyze(job / "studio_hd_master.wav")
    sel = json.loads((work / "selection.json").read_text())
    path = json.loads((work / f"comp{name}" / "comp_path.json").read_text())
    mix = json.loads((out / f"{tag}_mix_report.json").read_text())
    rep.update({
        "quality_mode": f"{tag}_best_take_comp", "true_data_mode": True,
        "num_povs_fused": len([n for n, u in path["usage"].items() if u["low_s"] or u["high_s"]]),
        "duplicates_removed": sum(len(v) for v in sel["duplicates"].values()),
        "orphans_excluded": sel["orphans"], "duplicate_groups_v8": sel["duplicates"],
        "integrated_lufs": m["lufs_i"], "true_peak_dbtp": m["true_peak_db"], "crest_factor_db": m["crest_db"],
        "loudness_range_lu": m["lra_lu"], "master_duration_sec": m["dur_s"],
        "take_usage_sec": path["usage"], "v8_mix": {k: v for k, v in mix.items() if not k.endswith("_eq")},
    })
    (job / "mastering_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"published {job}")


def _parse_opts(spec: str | None) -> dict:
    out = {}
    for kv in (spec or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--src", type=Path)
    ap.add_argument("--work", type=Path, default=Path.home() / "cstudio_work")
    ap.add_argument("--ref")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--set", dest="overrides")
    ap.add_argument("--name", default="", help="variant suffix: comp<name>/, stems<name>/")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.stage == "align":
        stage_align(a.src, a.work, a.ref)
    elif a.stage == "crowd":
        stage_crowd(a.work)
    elif a.stage == "comp":
        stage_comp(a.work, name=a.name, opts=_parse_opts(a.overrides))
    elif a.stage == "stems":
        stage_stems(a.work, name=a.name)
    elif a.stage == "publish":
        stage_publish(a.work, a.out, a.tag, Path(__file__).resolve().parents[2], a.name)
    elif a.stage == "dereverb":
        stage_dereverb(a.work, a.name)
    elif a.stage == "curate":
        from .curate import curate
        recs = json.loads(Path(a.ref).read_text())["records"] if a.ref else None
        res = curate(a.work, a.out, a.name, recs)
        print(f"kept {len(res['keep'])}, trash {len(res['trash'])}")
    elif a.stage == "mix":
        stage_mix(a.work, a.out, a.tag, a.overrides, a.name)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    print(f"[{a.stage}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
