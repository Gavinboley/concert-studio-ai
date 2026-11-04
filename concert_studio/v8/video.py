"""Real multi-cam director + MP4 export for V8 jobs.

The V2 director (`concert_studio/director.py`) invented four cameras. This one cuts between the actual phone
videos behind a V8 remaster:

* roster: every kept recording's video file, its measured place on the master timeline (placements + comp
  trim), the MP3-sound -> MP4 offset (MP3 encoder delay, measured: -25.1 / -50.1 ms), and the stretches where its
  audio verifiably matched the other phones (comp validity), i.e. no edits/splices;
* director: cuts on the master's downbeats (2 or 4 bars by energy) and on vocal entrances; per shot it picks the
  sharpest, best-exposed, highest-resolution camera that covers the whole shot, never the previous one, avoids
  recent repeats, alternates close/wide (closeness = vocal-band coherence), with a bonus for the phone we hear;
* render: frame-exact per-shot H.264 segments (1080x1920, 30 fps; wide clips are auto-cropped to 9:16 around the
  performers found by a person detector, falling back to stage light/motion, blurred fill only past --max-zoom), concat,
  then one MP4 per master (native / Ozone) with AAC 320k 48 kHz; optional trim (start/end on the master clock)
  renders only the shots inside it, with fades at the trimmed edges;
* verify_sync: each shot's own clip audio (from the video file, at the computed in-point) is cross-correlated
  with the master -> per-shot offset in ms.

Build the roster once (needs librosa for beats): `python -m concert_studio.v8.video roster --work W --name 3 --job J`.
Export (numpy + ffmpeg; torchvision optional for performer framing): `python -m concert_studio.v8.video export
--job J [--start S --end E] [--audio native,ozone] [--max-zoom 2.7 | --no-reframe]`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FPS = 30
OUT_W, OUT_H = 1080, 1920
EDGE_IN, EDGE_OUT = 0.6, 2.2       # keep clear of TikTok intro/outro cards (container seconds)
MIN_SHOT, MAX_SHOT = 2.4, 8.5


# ----------------------------------------------------------------------------------------------- roster
def _ffprobe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                       capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    if "streams" not in d:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr.strip()[:300]}")
    return d


def _content_box(video: str, w: int, h: int) -> list[int] | None:
    """Picture area inside baked-in black bars (a landscape clip uploaded in a 9:16 frame), or None.

    Keyframes only; the box must stay put over the clip and cut at least 8% of a side to count."""
    r = subprocess.run(["ffmpeg", "-v", "info", "-skip_frame", "nokey", "-i", video, "-vf",
                        "cropdetect=limit=24:round=2:reset=0", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    boxes = [m for m in r.stderr.split() if m.startswith("crop=")]
    if not boxes:
        return None
    cw, ch, cx, cy = (int(v) for v in boxes[-1][5:].split(":"))
    if cw <= 0 or ch <= 0 or (cw >= 0.92 * w and ch >= 0.92 * h):
        return None
    return [cw, ch, cx, cy]


def _dims(cam: dict) -> tuple[int, int]:
    """Effective picture size (inside baked-in bars)."""
    c = cam.get("crop")
    return (c[0], c[1]) if c else (cam["w"], cam["h"])


MAX_ZOOM = 2.7      # max upscale when auto-cropping wide footage to 9:16 (720p landscape fills the frame at 2.67x)


def _reframe_geometry(cam: dict, max_zoom: float = MAX_ZOOM) -> tuple[int, int, float] | None:
    """Auto-crop window (w, h, upscale) inside the picture for a frame-filling 9:16 shot, or None if the picture is
    already vertical. Full picture height unless that needs more than `max_zoom` upscale; never wider than the picture."""
    cw, ch = _dims(cam)
    if cw / ch <= OUT_W / OUT_H * 1.03:
        return None
    s = max(min(OUT_H / ch, max_zoom), OUT_W / cw)
    w = min(cw, int(round(OUT_W / s)) // 2 * 2)
    h = min(ch, int(round(OUT_H / s)) // 2 * 2)
    return w, h, s


def _fill_and_scale(cam: dict, max_zoom: float = MAX_ZOOM) -> tuple[float, float]:
    """(share of the 9:16 frame the picture fills, upscale factor) as rendered."""
    g = _reframe_geometry(cam, max_zoom)
    cw, ch = _dims(cam)
    if g is None:
        s = OUT_W / cw                                   # vertical: fit width
        return min(1.0, ch * s / OUT_H), s
    w, h, s = g
    return min(1.0, (w * s * h * s) / (OUT_W * OUT_H)), s


def _roi_from_frames(frames: np.ndarray, w_frac: float, h_frac: float) -> tuple[float, float]:
    """Top-left (fractions of the picture) of the crop window that holds the action.

    Saliency = performer motion (frame differences) + stage light (bright areas, squared), mild centre prior."""
    def norm(a):
        a = a - a.min()
        return a / (a.max() + 1e-9)
    lum = frames.mean(axis=0)
    mot = np.abs(np.diff(frames, axis=0)).mean(axis=0) if len(frames) > 1 else np.zeros_like(lum)
    sal = 1.5 * norm(mot) + norm(lum ** 2)
    stage = sal[: max(1, int(0.8 * sal.shape[0]))]       # bottom rows are the crowd's heads and raised phones

    def place(profile: np.ndarray, frac: float, centre: float) -> float:
        n = len(profile)
        win = int(round(frac * n))
        if win >= n - 1:
            return 0.0
        prof = np.convolve(profile, np.ones(5) / 5, mode="same")
        sums = np.convolve(prof, np.ones(win), mode="valid")
        mid = np.arange(len(sums)) + win / 2
        prior = 1.0 - np.abs(mid - centre * n) / n
        score = sums / (sums.max() + 1e-9) + 0.25 * prior
        return float(np.argmax(score)) / n
    return place(stage.mean(axis=0), w_frac, 0.5), place(sal.mean(axis=1), h_frac, 0.42)


_DET = None


def _person_detector():
    """COCO person detector (torchvision Faster R-CNN v2), GPU when present; None if torchvision is unavailable."""
    global _DET
    if _DET is None:
        try:
            import torch
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().to(dev)
            _DET = (m, dev, torch)
        except Exception:                                 # no torchvision / weights: saliency framing only
            _DET = False
    return _DET or None


def _detect_people(frames: np.ndarray, min_score: float = 0.5) -> list[list[tuple]] | None:
    """Per frame: (x0, y0, x1, y1, score) of each person in uint8 RGB frames (N, H, W, 3)."""
    det = _person_detector()
    if det is None:
        return None
    m, dev, torch = det
    out = []
    with torch.inference_mode():
        for f in frames:
            r = m([torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().div(255.0).to(dev)])[0]
            keep = (r["labels"] == 1) & (r["scores"] >= min_score)
            out.append([(*map(float, bx), float(sc)) for bx, sc in zip(r["boxes"][keep].cpu().numpy(),
                                                                       r["scores"][keep].cpu().numpy())])
    return out


def _roi_from_people(dets: list[list[tuple]], W: int, H: int, win_w: int, win_h: int) -> tuple[int, int] | None:
    """Crop offset that frames the performers: whole bodies, heads in the top third, centre stage (where the lead
    vocal usually stands) preferred. None when no performer is visible."""
    boxes = [b for fr in dets for b in fr
             if not (b[3] >= 0.97 * H and b[1] >= 0.6 * H)]   # crowd heads poking up from the bottom edge
    if not boxes:
        return None
    b = np.asarray(boxes, float)
    bw = np.maximum(b[:, 2] - b[:, 0], 1.0)
    area = bw * (b[:, 3] - b[:, 1]) / (W * H)
    central = 1.0 - np.abs((b[:, 0] + b[:, 2]) / 2 - W / 2) / (W / 2)
    wt = b[:, 4] * np.sqrt(area) * (1.0 + 0.6 * central)
    xs = np.arange(0, W - win_w + 1)
    ov = np.clip(np.minimum(b[None, :, 2], xs[:, None] + win_w) - np.maximum(b[None, :, 0], xs[:, None]), 0, None) / bw
    score = (wt * ov ** 2).sum(axis=1)                    # a person cut in half counts a quarter
    prior = 1.0 - np.abs(xs + win_w / 2 - W / 2) / W
    x = int(xs[np.argmax(score / (score.max() + 1e-12) + 0.05 * prior)])
    y = 0
    if win_h < H:
        inside = ov[x] > 0.5
        heads = b[inside, 1] + 0.12 * (b[inside, 3] - b[inside, 1])
        if len(heads):
            y = int(np.clip(np.average(heads, weights=wt[inside]) - win_h / 3, 0, H - win_h))
    return x, y


def _decode(video: str, t_in: float, dur: float, vf: str, w: int, h: int, pix: str, n_max: int) -> np.ndarray:
    fps = max(0.5, min(2.0, n_max / max(dur, 0.5)))
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_in):.3f}", "-t", f"{max(0.5, dur):.3f}",
                        "-i", video, "-vf", f"{vf},fps={fps:.3f},format={pix}", "-f", "rawvideo", "-"],
                       capture_output=True)
    ch = 3 if pix == "rgb24" else 1
    n = min(n_max, len(r.stdout) // (w * h * ch))
    return np.frombuffer(r.stdout[: n * w * h * ch], np.uint8).reshape((n, h, w, ch) if ch == 3 else (n, h, w))


def _roi_crop(cam: dict, t_in: float, dur: float, w: int, h: int, detect: bool = True) -> tuple[int, int]:
    """Crop offset (x, y) inside the picture for one shot: frame the performers (person detector), else the
    stage action (motion + light)."""
    cw, ch = _dims(cam)
    cx, cy = (cam["crop"][2], cam["crop"][3]) if cam.get("crop") else (0, 0)
    if detect and _person_detector() is not None:
        dw = min(cw, 1280) // 2 * 2
        dh = int(round(ch * dw / cw / 2)) * 2
        fr = _decode(cam["video"], t_in, dur, f"crop={cw}:{ch}:{cx}:{cy},scale={dw}:{dh}", dw, dh, "rgb24", 4)
        if len(fr):
            k = dw / cw
            got = _roi_from_people(_detect_people(fr), dw, dh, int(round(w * k)), int(round(h * k)))
            if got is not None:
                return (int(np.clip(round(got[0] / k), 0, cw - w)) // 2 * 2,
                        int(np.clip(round(got[1] / k), 0, ch - h)) // 2 * 2)
    W = 96
    H = max(2, int(round(ch * W / cw / 2)) * 2)
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_in):.3f}", "-t", f"{max(0.5, dur):.3f}",
                        "-i", cam["video"], "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={W}:{H},fps=2,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    n = len(r.stdout) // (W * H)
    if n == 0:
        return (cw - w) // 2 // 2 * 2, (ch - h) // 2 // 2 * 2
    frames = np.frombuffer(r.stdout[: n * W * H], np.uint8).reshape(n, H, W).astype(np.float32) / 255.0
    xf, yf = _roi_from_frames(frames, w / cw, h / ch)
    x = int(np.clip(round(xf * cw), 0, cw - w)) // 2 * 2
    y = int(np.clip(round(yf * ch), 0, ch - h)) // 2 * 2
    return x, y


def _keyframe_stats(video: str, w: int, h: int) -> dict:
    """Sharpness (log Laplacian variance) and mean luma of every keyframe at 180 px width."""
    from scipy import ndimage
    sw = 180
    sh = max(2, int(round(sw * h / w / 2)) * 2)
    p = subprocess.run(["ffmpeg", "-v", "info", "-hide_banner", "-nostats", "-skip_frame", "nokey", "-i", video,
                        "-vf", f"scale={sw}:{sh},format=gray,showinfo", "-an", "-f", "rawvideo", "-"],
                       capture_output=True)
    raw = np.frombuffer(p.stdout, np.uint8)
    n = len(raw) // (sw * sh)
    frames = raw[: n * sw * sh].reshape(n, sh, sw).astype(np.float32)
    times = []
    for line in p.stderr.decode("utf-8", "replace").splitlines():
        if "showinfo" in line and "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except ValueError:
                pass
    n = min(n, len(times))
    sharp = [float(np.log10(np.var(ndimage.laplace(f)) + 1e-3)) for f in frames[:n]]
    luma = [float(f.mean()) for f in frames[:n]]
    return {"t": times[:n], "sharp": sharp, "luma": luma}


def _music_grid(master: Path, vocals: Path | None) -> dict:
    """Beats, downbeats (4/4 phase with the strongest low-band onsets), vocal entrances, 1 s energy."""
    import librosa
    import soundfile as sf
    x, sr = sf.read(master, dtype="float32", always_2d=True)
    y = librosa.resample(x.mean(axis=1), orig_sr=sr, target_sr=22050)
    tempo, beats = librosa.beat.beat_track(y=y, sr=22050, hop_length=512, units="time", tightness=120)
    low = librosa.onset.onset_strength(y=y, sr=22050, hop_length=512, fmax=200.0, n_mels=32)
    ot = librosa.times_like(low, sr=22050, hop_length=512)
    strength = np.interp(beats, ot, low)
    phase = int(np.argmax([strength[ph::4].sum() for ph in range(4)])) if len(beats) >= 8 else 0
    downbeats = beats[phase::4]
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=2205)[0]          # 10 Hz
    energy = np.convolve(20 * np.log10(rms + 1e-6), np.ones(10) / 10, mode="same")
    entries: list[float] = []
    if vocals is not None and vocals.exists():
        v, vsr = sf.read(vocals, dtype="float32", always_2d=True)
        v = v.mean(axis=1)
        hop = vsr // 10
        e = np.sqrt(np.convolve(v[: len(v) // hop * hop].reshape(-1, hop).var(axis=1), np.ones(3) / 3, mode="same"))
        on = e > 0.25 * np.percentile(e, 90)
        quiet = 0
        for i, a in enumerate(on):
            if a and quiet >= 15:
                entries.append(i / 10.0)
            quiet = 0 if a else quiet + 1
    return {"tempo": float(np.atleast_1d(tempo)[0]), "beats": [float(b) for b in beats],
            "downbeats": [float(b) for b in downbeats], "vocal_entries": entries,
            "energy_10hz": [float(e) for e in energy], "duration": len(x) / sr}


def build_roster(work: Path, comp_name: str, job_dir: Path, dl_manifest: Path | None = None, log=print) -> dict:
    """Write job_dir/director_roster.json for the V8 job made from work/comp<comp_name>."""
    man = json.loads((work / "manifest.json").read_text())
    man = man["clips"] if isinstance(man, dict) else man
    info = {c["name"]: c for c in man}
    sel = json.loads((work / "selection.json").read_text())
    pls = {p["name"]: p for p in json.loads((work / "placements.json").read_text())}
    cp = json.loads((work / f"comp{comp_name}" / "comp_path.json").read_text())
    D = np.load(work / f"comp{comp_name}" / "scores_dump.npz", allow_pickle=True)
    names, grid, ok = list(D["names"]), D["grid"], np.isfinite(D["S0"])
    vco = D["vcoh"]
    trim = float(cp["trim_start_s"])
    path = np.array(cp["pathL"])
    recs = {}
    if dl_manifest and dl_manifest.exists():
        recs = {int(r["rank"]): r for r in json.loads(dl_manifest.read_text())["records"]}
    vq = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in vco])
    lo_t, hi_t = np.nanpercentile(vq, 33), np.nanpercentile(vq, 67)
    cams = {}
    for n in sel["keep"]:
        if n not in names:
            continue
        i = names.index(n)
        src = os.path.realpath(info[n]["src"])
        label = Path(src).stem
        if n.startswith("new") and n[3:5].isdigit() and int(n[3:5]) in recs:
            r = recs[int(n[3:5])]
            video = str(dl_manifest.parent / r["video_file"])
            label = "@" + r["uploader"]
        else:
            video = src
        pr = _ffprobe(video)
        vs = [s for s in pr["streams"] if s["codec_type"] == "video"]
        if not vs:
            continue
        v = vs[0]
        w, h, dur = int(v["width"]), int(v["height"]), float(pr["format"]["duration"])
        av = 0.0
        if os.path.realpath(video) != src:
            av = _measure_av_offset(info[n]["wav"], video, dur)
        # usable master-time ranges: audio-validated grid steps, inside the container's safe zone
        tl = grid
        p_audio = (tl - pls[n]["offset_s"]) / (1.0 + pls[n]["drift_ppm"] * 1e-6)
        cont = p_audio + av
        good = ok[i] & (cont >= EDGE_IN) & (cont <= dur - EDGE_OUT)
        ranges = []
        d = np.diff(np.concatenate([[0], good.astype(int), [0]]))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            t0, t1 = float(tl[a]) - trim + 0.25, float(tl[b - 1]) - trim - 0.25
            if t1 - t0 >= 1.5:
                ranges.append([round(t0, 3), round(t1, 3)])
        if not ranges:
            continue
        ci = cp["names"].index(n) if n in cp["names"] else -1
        heard = [[round(float(tl[k]) - trim, 2)] for k in np.where(path == ci)[0]] if ci >= 0 else []
        closeness = "close" if vq[i] >= hi_t else ("wide" if vq[i] <= lo_t else "mid")
        cams[n] = {"label": label, "video": video, "w": w, "h": h, "dur": dur, "av_offset_s": round(av, 5),
                   "offset_s": pls[n]["offset_s"], "drift_ppm": pls[n]["drift_ppm"], "trim_start_s": trim,
                   "usable": ranges, "heard": sorted({x[0] for x in heard}), "closeness": closeness,
                   "vcoh": None if not np.isfinite(vq[i]) else round(float(vq[i]), 3),
                   "kf": _keyframe_stats(video, w, h)}
        cams[n]["crop"] = _content_box(video, w, h)
        log(f"  {label[:28]:28s} {w}x{h} av {1000 * av:+6.1f} ms  usable {sum(b - a for a, b in ranges):6.1f}s  "
            f"{closeness}  keyframes {len(cams[n]['kf']['t'])}")
    master = job_dir / "studio_hd_master.wav"
    grid_info = _music_grid(master, job_dir / "stem_vocals.wav")
    log(f"  tempo {grid_info['tempo']:.1f} BPM, {len(grid_info['downbeats'])} downbeats, "
        f"{len(grid_info['vocal_entries'])} vocal entrances")
    roster = {"cams": cams, "music": grid_info, "fps": FPS}
    (job_dir / "director_roster.json").write_text(json.dumps(roster))
    return roster


def _measure_av_offset(audio_wav: str, video: str, dur: float, fs: int = 16000) -> float:
    """video_time = audio_time + d for an audio file taken from a different container (GCC-PHAT)."""
    t0 = min(20.0, max(0.0, dur / 2 - 30))
    ln = max(5.0, min(60.0, dur - t0 - 1))

    def dec(p):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(ln), "-i", p, "-vn", "-ac", "1",
                              "-ar", str(fs), "-f", "f32le", "-"], capture_output=True).stdout
        return np.frombuffer(raw, np.float32)
    x, y = dec(audio_wav), dec(video)
    m = min(len(x), len(y))
    if m < fs:
        return 0.0
    X, Y = np.fft.rfft(x[:m], 2 * m), np.fft.rfft(y[:m], 2 * m)
    G = Y * np.conj(X)
    c = np.fft.irfft(G / (np.abs(G) + 1e-9))
    half = int(0.5 * fs)
    c = np.concatenate([c[-half:], c[:half]])
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:
        a, b, cc = c[k - 1], c[k], c[k + 1]
        den = a - 2 * b + cc
        frac = 0.5 * (a - cc) / den if den else 0.0
    return float((k + frac - half) / fs)


# ----------------------------------------------------------------------------------------------- director
def master_to_container(cam: dict, t: float | np.ndarray):
    tl = np.asarray(t, dtype=float) + cam["trim_start_s"]
    return (tl - cam["offset_s"]) / (1.0 + cam["drift_ppm"] * 1e-6) + cam["av_offset_s"]


def _covers(cam: dict, t0: float, t1: float) -> bool:
    return any(a <= t0 and t1 <= b for a, b in cam["usable"])


def _cover_end(cam: dict, t0: float) -> float:
    for a, b in cam["usable"]:
        if a <= t0 < b:
            return b
    return t0


def _physical(cam: dict, t0: float, t1: float) -> bool:
    c0, c1 = master_to_container(cam, t0), master_to_container(cam, t1)
    return EDGE_IN <= c0 and c1 <= cam["dur"] - EDGE_OUT


def _phys_end(cam: dict, t0: float) -> float:
    # master time where this camera's safe footage ends
    return float(t0 + (cam["dur"] - EDGE_OUT - master_to_container(cam, t0)) * (1.0 + cam["drift_ppm"] * 1e-6))


def _video_score(cam: dict, t0: float, t1: float, max_zoom: float = MAX_ZOOM) -> float:
    kf = cam["kf"]
    if not kf["t"]:
        return -1.0
    c0, c1 = master_to_container(cam, t0), master_to_container(cam, t1)
    kt = np.asarray(kf["t"])
    sel = (kt >= c0 - 1.0) & (kt <= c1 + 1.0)
    if not sel.any():
        sel = np.abs(kt - (c0 + c1) / 2) == np.min(np.abs(kt - (c0 + c1) / 2))
    sharp = float(np.mean(np.asarray(kf["sharp"])[sel]))
    luma = float(np.mean(np.asarray(kf["luma"])[sel]))
    expo = -abs(luma - 110.0) / 60.0 if (luma < 50 or luma > 200) else 0.0
    fill, up = _fill_and_scale(cam, max_zoom)
    res = np.log2(min(OUT_H, OUT_H / up) / 1280.0)      # detail shown (1280 = 720p vertical); downscaling adds none
    return sharp + expo + 0.5 * res - 0.8 * (1.0 - fill)


def plan_shots(roster: dict, start: float | None = None, end: float | None = None, seed: int = 0,
               max_zoom: float = MAX_ZOOM) -> list[dict]:
    cams, mu = roster["cams"], roster["music"]
    dur = float(mu["duration"])
    start = 0.0 if start is None else max(0.0, float(start))
    end = dur if end is None else min(dur, float(end))
    if end - start < 1.0:
        raise ValueError("trim range shorter than 1 s")
    rng = np.random.default_rng(seed)
    downs = np.asarray(mu["downbeats"])
    bar = float(np.median(np.diff(downs))) if len(downs) > 2 else 2.0
    energy = np.asarray(mu["energy_10hz"])
    e_hi = np.percentile(energy, 60) if len(energy) else 0.0
    entries = np.asarray(mu["vocal_entries"])
    # per-camera stats for normalisation
    names = list(cams)
    shots: list[dict] = []
    t, last, recent, last_type = start, None, [], None
    while t < end - 0.05:
        loud = len(energy) and energy[min(len(energy) - 1, int(t * 10))] > e_hi
        want_len = 4.0 if loud else 7.0                       # seconds; independent of half/double-time tracking
        bars = max(1, int(round(want_len / bar)))
        lo_t, hi_t = t + MIN_SHOT, t + min(MAX_SHOT, bars * bar + 0.5 * bar)
        # candidate cut points: downbeats in range, preferring a vocal entrance
        cand = downs[(downs >= lo_t) & (downs <= hi_t)]
        ent = entries[(entries >= lo_t) & (entries <= hi_t)]
        if len(ent):
            target = float(ent[0])
            reason = "vocal_entrance"
            if len(cand):
                near = cand[np.argmin(np.abs(cand - target))]
                target = float(near) if abs(near - target) < 0.6 * bar else target
        elif len(cand):
            want = t + bars * bar
            target = float(cand[np.argmin(np.abs(cand - want))])
            reason = "downbeat"
        else:
            target, reason = min(t + bars * bar, t + MAX_SHOT), "phrase"
        if end - target < MIN_SHOT:
            target, reason = end, ("trim_end" if end < dur else "song_end")
        target = min(target, end)
        # cameras covering [t, target]; if none, shorten to the longest coverage available
        cover = [n for n in names if _covers(cams[n], t, target)]
        if not cover:
            ends = {n: _cover_end(cams[n], t) for n in names}
            best = max(ends.values()) if ends else t
            if best - t >= 1.2:
                target = min(best, target)
                cover = [n for n in names if _covers(cams[n], t, target)]
                reason = "coverage_limit"
        if not cover:
            # no take's audio validates here (band stop, applause): any camera whose footage physically covers
            # the moment will do, preferring the current one; the video never has holes
            nxt = min([a for n in names for a, _ in cams[n]["usable"] if a > t + 0.5] + [end])
            target = min(nxt if nxt - t >= MIN_SHOT or end - t < MIN_SHOT else t + MIN_SHOT, end)
            phys = [n for n in names if _physical(cams[n], t, target)]
            if not phys:
                for n in names:                                  # shorten to the longest physical coverage
                    c0 = master_to_container(cams[n], t)
                    if EDGE_IN <= c0 <= cams[n]["dur"] - EDGE_OUT - 1.0:
                        phys.append(n)
                if phys:
                    lim = max(_phys_end(cams[n], t) for n in phys)
                    target = min(target, lim)
            if not phys:
                if shots:
                    shots[-1]["end_sec"] = round(min(t + 0.5, end), 3)
                t += 0.5
                continue
            n = last if last in phys else max(phys, key=lambda m: _video_score(cams[m], t, target, max_zoom))
            c = cams[n]
            shots.append({"shot_index": len(shots) + 1, "camera": n, "camera_name": c["label"],
                          "start_sec": round(t, 3), "end_sec": round(target, 3),
                          "duration_sec": round(target - t, 3), "shot_type": c["closeness"],
                          "switch_reason": "unvalidated_hold", "src_in_sec": round(float(master_to_container(c, t)), 4)})
            last, last_type = n, c["closeness"]
            recent.append(n)
            t = target
            continue
        scored = []
        for n in cover:
            c = cams[n]
            s = _video_score(c, t, target, max_zoom)
            if n == last:
                s -= 100.0
            s -= 1.5 * recent[-4:].count(n)
            if last_type and c["closeness"] != last_type:
                s += 0.6
            heard = np.asarray(c["heard"])
            if len(heard) and np.sum((heard >= t) & (heard < target)) * 0.5 > 0.5 * (target - t):
                s += 0.4
            s += 0.15 * rng.standard_normal()                  # seed = re-direct variation
            scored.append((s, n))
        scored.sort(reverse=True)
        n = scored[0][1]
        c = cams[n]
        shots.append({"shot_index": len(shots) + 1, "camera": n, "camera_name": c["label"],
                      "start_sec": round(t, 3), "end_sec": round(target, 3),
                      "duration_sec": round(target - t, 3), "shot_type": c["closeness"], "switch_reason": reason,
                      "src_in_sec": round(float(master_to_container(c, t)), 4)})
        last, last_type = n, c["closeness"]
        recent.append(n)
        t = target
    merged: list[dict] = []
    for s in shots:                                          # same camera back-to-back = one continuous shot
        if merged and merged[-1]["camera"] == s["camera"] and abs(merged[-1]["end_sec"] - s["start_sec"]) < 1e-3:
            merged[-1]["end_sec"] = s["end_sec"]
        else:
            merged.append(dict(s))
    if merged and merged[0]["start_sec"] > start:
        # nothing covered the first instant: the first shot starts at the trim point anyway (the renderer freezes
        # its first frame if the clip begins later), otherwise every later shot would sit early against the audio
        c = cams[merged[0]["camera"]]
        merged[0]["lead_in_sec"] = round(merged[0]["start_sec"] - start, 3)
        merged[0]["start_sec"] = round(start, 3)
        merged[0]["src_in_sec"] = round(float(master_to_container(c, start)), 4)
    for k, s in enumerate(merged):
        s["shot_index"] = k + 1
        s["duration_sec"] = round(s["end_sec"] - s["start_sec"], 3)
    return merged


def export_edl(shots: list[dict], roster: dict, title: str = "ConcertStudio_MultiCam") -> str:
    """CMX 3600 with real source timecodes (clip container time) and record timecodes (master time)."""
    def tc(sec: float) -> str:
        f = int(round(max(0.0, sec) * FPS))
        return f"{f // (FPS * 3600):02d}:{(f // (FPS * 60)) % 60:02d}:{(f // FPS) % 60:02d}:{f % FPS:02d}"
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for s in shots:
        reel = f"AX{s['shot_index']:03d}"
        src_in = s["src_in_sec"]
        lines.append(f"{s['shot_index']:03d}  {reel:<8} V     C        {tc(src_in)} {tc(src_in + s['duration_sec'])} "
                     f"{tc(s['start_sec'])} {tc(s['end_sec'])}")
        lines.append(f"* FROM CLIP NAME: {Path(roster['cams'][s['camera']]['video']).name}")
        lines.append(f"* COMMENT: {s['camera_name']} {s['shot_type']} {s['switch_reason']}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- render
def _fit_filter(w: int, h: int) -> str:
    if abs(w / h - OUT_W / OUT_H) < 0.02:
        return f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1"
    return (f"split[a][b];[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
            f"gblur=sigma=30,eq=brightness=-0.08[bg];[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:"
            f"flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def render(roster: dict, shots: list[dict], out_dir: Path, base: str, audio: dict[str, Path],
           start: float, end: float, encoder: str = "libx264", progress=None, log=print,
           reframe: bool = True, max_zoom: float = MAX_ZOOM) -> list[Path]:
    """Render the cut once (video only), then mux each master. Returns the MP4 paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="mc_", dir=str(out_dir)))
    try:
        f0 = int(round(start * FPS))
        n_total = int(round(end * FPS)) - f0
        lst = []
        trimmed_in, trimmed_out = start > 0.05, end < roster["music"]["duration"] - 0.05
        for k, s in enumerate(shots):
            c = roster["cams"][s["camera"]]
            a = int(round(s["start_sec"] * FPS)) - f0
            b = int(round(s["end_sec"] * FPS)) - f0
            nfr = max(1, b - a)
            g = _reframe_geometry(c, max_zoom) if reframe else None
            ox, oy = (c["crop"][2], c["crop"][3]) if c.get("crop") else (0, 0)
            if g:                                         # auto-crop wide footage to the action, then fill 9:16
                w_, h_, _ = g
                x_, y_ = _roi_crop(c, s["src_in_sec"], s["duration_sec"], w_, h_)
                s["reframe"] = [w_, h_, ox + x_, oy + y_]
                vf = f"crop={w_}:{h_}:{ox + x_}:{oy + y_}," + _fit_filter(w_, h_) + f",fps={FPS},format=yuv420p"
            else:
                ew, eh = _dims(c)
                vf = _fit_filter(ew, eh) + f",fps={FPS},format=yuv420p"
                if c.get("crop"):                         # drop baked-in black bars
                    cw, ch, cx, cy = c["crop"]
                    vf = f"crop={cw}:{ch}:{cx}:{cy}," + vf
            if s["src_in_sec"] < 0:                       # clip starts after this instant: hold its first frame
                vf = f"tpad=start_mode=clone:start_duration={-s['src_in_sec']:.4f}," + vf
            if k == 0 and trimmed_in:
                vf += ",fade=t=in:st=0:d=0.5"
            if k == len(shots) - 1 and trimmed_out:
                vf += f",fade=t=out:st={max(0.0, nfr / FPS - 1.0):.3f}:d=1.0"
            seg = tmp / f"s{k:04d}.mp4"
            enc = (["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high"] if encoder == "libx264"
                   else ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "19", "-b:v", "0"])
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, s['src_in_sec']):.4f}", "-i", c["video"],
                   "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", "-frames:v", str(nfr), "-r", str(FPS),
                   *enc, "-g", str(FPS), "-an", str(seg)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"segment {k} failed: {r.stderr[-400:]}")
            lst.append(f"file '{seg.name}'")
            if progress:
                progress(80.0 * (k + 1) / len(shots), f"shot {k + 1}/{len(shots)}")
        (tmp / "list.txt").write_text("\n".join(lst) + "\n")
        vid = tmp / "video.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                        "-c", "copy", str(vid)], check=True, cwd=str(tmp))
        outs = []
        span = n_total / FPS
        for i, (name, wav) in enumerate(audio.items()):
            af = []
            if trimmed_in:
                af.append("afade=t=in:st=0:d=0.3")
            if trimmed_out:
                af.append(f"afade=t=out:st={max(0.0, span - 1.5):.3f}:d=1.5")
            dst = out_dir / f"{base}_{name}.mp4"
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-ss", f"{start:.4f}", "-t", f"{span:.4f}",
                   "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                   "-ar", "48000", *(["-af", ",".join(af)] if af else []), "-movflags", "+faststart", "-shortest",
                   str(dst)]
            subprocess.run(cmd, check=True)
            outs.append(dst)
            if progress:
                progress(80.0 + 20.0 * (i + 1) / len(audio), f"muxed {dst.name}")
            log(f"  wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
        return outs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify_sync(roster: dict, shots: list[dict], master: Path, fs: int = 16000, max_shots: int = 0) -> list[dict]:
    """Per shot: xcorr of the clip's own audio (from the video file at the computed in-point) vs the master."""
    import soundfile as sf
    x, sr = sf.read(master, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    out = []
    pick = shots if not max_shots else [shots[i] for i in np.linspace(0, len(shots) - 1, max_shots).astype(int)]
    for s in pick:
        c = roster["cams"][s["camera"]]
        L = min(s["duration_sec"], 6.0)
        if L < 1.5:
            continue
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{s['src_in_sec']:.4f}", "-t", f"{L:.3f}", "-i",
                              c["video"], "-vn", "-ac", "1", "-ar", str(fs), "-f", "f32le", "-"],
                             capture_output=True).stdout
        a = np.frombuffer(raw, np.float32)
        m0 = int(s["start_sec"] * sr)
        mseg = x[m0:m0 + int(L * sr)]
        if len(a) < fs or len(mseg) < sr:
            continue
        idx = np.round(np.arange(int(len(mseg) * fs / sr)) * sr / fs).astype(int)
        b = mseg[np.clip(idx, 0, len(mseg) - 1)]
        n = min(len(a), len(b))
        A, B = np.fft.rfft(a[:n], 2 * n), np.fft.rfft(b[:n], 2 * n)
        G = A * np.conj(B)
        cc = np.fft.irfft(G / (np.abs(G) + 1e-9))
        lim = int(0.2 * fs)
        cc = np.concatenate([cc[-lim:], cc[:lim]])
        k = int(np.argmax(cc))
        score = float(cc[k] / (np.median(np.abs(cc)) * 1.4826 + 1e-12))
        out.append({"shot": s["shot_index"], "camera": s["camera_name"], "offset_ms": round(1000 * (k - lim) / fs, 1),
                    "score": round(score, 1)})
    return out


def export(job_dir: Path, start: float | None = None, end: float | None = None, audio_keys=("native", "ozone"),
           seed: int = 0, encoder: str = "libx264", verify: bool = True, progress=None, log=print,
           reframe: bool = True, max_zoom: float = MAX_ZOOM) -> dict:
    rf = job_dir / "director_roster.json"
    roster = json.loads(rf.read_text())
    if any("crop" not in c for c in roster["cams"].values()):
        for n, c in roster["cams"].items():
            if "crop" not in c:
                c["crop"] = _content_box(c["video"], c["w"], c["h"])
                if c["crop"]:
                    log(f"  {c['label']}: baked-in bars, picture {c['crop'][0]}x{c['crop'][1]}")
        rf.write_text(json.dumps(roster, indent=1, default=float))
    dur = float(roster["music"]["duration"])
    s0 = 0.0 if start is None else max(0.0, float(start))
    s1 = dur if end is None else min(dur, float(end))
    shots = plan_shots(roster, s0, s1, seed, max_zoom if reframe else 1e-6)
    files = {"native": job_dir / "studio_hd_master.wav", "ozone": job_dir / "studio_hd_ozone_master.wav"}
    audio = {k: files[k] for k in audio_keys if files[k].exists()}
    trim_tag = "" if (s0 <= 0.05 and s1 >= dur - 0.05) else f"_{int(s0 // 60)}m{s0 % 60:04.1f}-{int(s1 // 60)}m{s1 % 60:04.1f}"
    base = f"multicam{trim_tag}".replace(".", "s")
    outs = render(roster, shots, job_dir, base, audio, s0, s1, encoder, progress, log, reframe, max_zoom)
    (job_dir / f"{base}.edl").write_text(export_edl(shots, roster))
    sync = verify_sync(roster, shots, files["native"]) if verify else []
    good = [r for r in sync if r["score"] >= 8]
    rep = {"start": s0, "end": s1, "shots": shots, "files": [p.name for p in outs], "edl": f"{base}.edl",
           "sync": sync, "sync_median_abs_ms": float(np.median([abs(r["offset_ms"]) for r in good])) if good else None,
           "sync_max_abs_ms": float(max([abs(r["offset_ms"]) for r in good])) if good else None}
    (job_dir / f"{base}_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["roster", "plan", "export"])
    ap.add_argument("--work", type=Path)
    ap.add_argument("--name", default="")
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--dl-manifest", type=Path)
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--audio", default="native,ozone")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default="libx264")
    ap.add_argument("--max-zoom", type=float, default=MAX_ZOOM,
                    help="max upscale when auto-cropping wide footage to fill the 9:16 frame")
    ap.add_argument("--no-reframe", action="store_true", help="fit wide footage over a blurred fill instead")
    a = ap.parse_args(argv)
    mz = a.max_zoom if not a.no_reframe else 1e-6
    if a.cmd == "roster":
        build_roster(a.work, a.name, a.job, a.dl_manifest)
    elif a.cmd == "plan":
        shots = plan_shots(json.loads((a.job / "director_roster.json").read_text()), a.start, a.end, a.seed, mz)
        for s in shots:
            print(f"{s['shot_index']:3d} {s['start_sec']:7.2f}-{s['end_sec']:7.2f} {s['duration_sec']:5.2f}s "
                  f"{s['camera_name'][:24]:24s} {s['shot_type']:5s} {s['switch_reason']}")
    else:
        rep = export(a.job, a.start, a.end, tuple(a.audio.split(",")), a.seed, a.encoder,
                     progress=lambda p, d: print(f"  {p:5.1f}% {d}", flush=True),
                     reframe=not a.no_reframe, max_zoom=a.max_zoom)
        print(json.dumps({k: rep[k] for k in ("files", "edl", "sync_median_abs_ms", "sync_max_abs_ms")}))


if __name__ == "__main__":
    main()
