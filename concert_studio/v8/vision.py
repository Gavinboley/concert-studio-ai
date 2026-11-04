"""What each phone sees, for the director: performers (where, how big, which band member), the kind of shot,
colour and sharpness, sampled every ~1.5 s of footage. Cached in job_dir/director_vision.json.

Both models are discriminative (they recognise, nothing is generated): torchvision Faster R-CNN v2 (COCO) finds
people; CLIP ViT-L/14, zero-shot, names what each performer is doing (singing, guitar, bass, drums, keys) and what
kind of shot the frame is (wide stage, close-up, crowd, dark). Without transformers the roles stay unknown and the
director falls back to framing whoever is on stage.

Also: which band member the music features at each moment (`featured_track`, from the stems), so the director can
show the guitarist in a solo and the drummer in a fill.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROLES = ["vocals", "guitar", "bass", "drums", "keys"]
_ROLE_TEXT = {
    "vocals": ["a photo of a singer singing into a microphone on stage",
               "a photo of a rock singer holding a microphone"],
    "guitar": ["a photo of a guitarist playing an electric guitar on stage",
               "a photo of a musician strumming a guitar"],
    "bass": ["a photo of a bassist playing a bass guitar on stage"],
    "drums": ["a photo of a drummer playing a drum kit on stage",
              "a photo of a drummer behind cymbals and drums"],
    "keys": ["a photo of a musician playing a keyboard synthesizer on stage"],
    "crowd": ["a photo of fans in a concert crowd", "a photo of the back of a person's head in the dark"],
    "person": ["a photo of a person standing on a stage", "a photo of a stage light"],
}
_SCENE_TEXT = {
    "wide": ["a wide photo of a whole band performing on a big concert stage"],
    "close": ["a close-up photo of a musician performing", "a close-up of a singer on a giant video screen"],
    "crowd": ["a photo of a concert crowd with raised phones", "a photo of the audience at a concert"],
    "dark": ["a dark blurry photo", "a black image"],
}
SAMPLE_S = 1.5
EDGE_IN, EDGE_OUT = 0.6, 2.2


def _feat(x):
    """CLIP projected embeddings (transformers 5 wraps them in an output object)."""
    return x if hasattr(x, "norm") else x.pooler_output


class _Models:
    def __init__(self, clip: bool = True):
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().to(self.dev)
        self.clip = None
        if clip:
            try:
                from transformers import CLIPModel, CLIPProcessor
                name = "openai/clip-vit-large-patch14"
                dt = torch.float16 if self.dev == "cuda" else torch.float32
                self.clip = CLIPModel.from_pretrained(name, dtype=dt).eval().to(self.dev)
                self.proc = CLIPProcessor.from_pretrained(name)
                self.role_keys, self.role_emb = self._text(_ROLE_TEXT)
                self.scene_keys, self.scene_emb = self._text(_SCENE_TEXT)
            except Exception as e:                       # no transformers / offline: people only
                print(f"  vision: CLIP unavailable ({type(e).__name__}); roles unknown")
                self.clip = None

    def _text(self, table: dict) -> tuple[list[str], "object"]:
        torch = self.torch
        keys, embs = [], []
        with torch.inference_mode():
            for k, prompts in table.items():
                t = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.dev)
                e = _feat(self.clip.get_text_features(**t))
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(0)
                keys.append(k)
                embs.append(e / e.norm())
        return keys, torch.stack(embs)

    def people(self, frame: np.ndarray, min_score: float = 0.5) -> list[tuple]:
        torch = self.torch
        with torch.inference_mode():
            r = self.det([torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float().div(255.0)
                          .to(self.dev)])[0]
        keep = (r["labels"] == 1) & (r["scores"] >= min_score)
        return [(*map(float, b), float(s)) for b, s in zip(r["boxes"][keep].cpu().numpy(),
                                                           r["scores"][keep].cpu().numpy())]

    def classify(self, images: list[np.ndarray], emb) -> np.ndarray:
        torch = self.torch
        if not images:
            return np.zeros((0, len(emb)))
        with torch.inference_mode():
            px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.dev, self.clip.dtype)
            f = _feat(self.clip.get_image_features(pixel_values=px))
            f = f / f.norm(dim=-1, keepdim=True)
            return (100.0 * f @ emb.T).float().softmax(dim=-1).cpu().numpy()


def _crop_for_role(frame: np.ndarray, box: tuple) -> np.ndarray:
    """Person box widened for the instrument (guitar necks stick out sideways, drums sit in front)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box[:4]
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - 0.45 * bw)), min(W, int(x1 + 0.45 * bw))
    y0, y1 = max(0, int(y0 - 0.08 * bh)), min(H, int(y1 + 0.05 * bh))
    return frame[y0:y1, x0:x1]


def _color_stats(frame: np.ndarray) -> dict:
    f = frame.astype(np.float32) / 255.0
    y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    ok = (y > 0.03) & (f.max(axis=-1) < 0.98)                 # ignore black and clipped stage lights
    lin = f[ok] ** 2.2 if ok.sum() > 100 else f.reshape(-1, 3) ** 2.2
    p5, p50, p95 = np.percentile(y, [5, 50, 95])
    return {"rgb": [round(float(v), 5) for v in lin.mean(axis=0)], "luma": [round(float(p5), 4),
            round(float(p50), 4), round(float(p95), 4)]}


def _sharpness(gray: np.ndarray) -> float:
    from scipy import ndimage
    return float(np.log10(np.var(ndimage.laplace(gray.astype(np.float32))) + 1e-3))


def _frames(cam: dict, every: float = SAMPLE_S, width: int = 720):
    """(container_time, uint8 RGB frame of the picture) every `every` seconds of the clip's safe footage."""
    c = cam.get("crop") or [cam["w"], cam["h"], 0, 0]
    cw, ch, cx, cy = c
    w = min(width, cw) // 2 * 2
    h = int(round(ch * w / cw / 2)) * 2
    t0, t1 = EDGE_IN, max(EDGE_IN + 0.5, cam["dur"] - EDGE_OUT)
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", cam["video"],
                        "-vf", f"crop={cw}:{ch}:{cx}:{cy},fps=1/{every},scale={w}:{h}:flags=area,format=rgb24",
                        "-f", "rawvideo", "-"], capture_output=True)
    n = len(r.stdout) // (w * h * 3)
    arr = np.frombuffer(r.stdout[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    for k in range(n):
        yield t0 + (k + 0.5) * every, arr[k]


def _overlay_box(mins: np.ndarray) -> list[float] | None:
    """Burned-in caption: pixels that stay bright in every sample while the phone moves. Box as picture fractions."""
    from scipy import ndimage
    m = ndimage.binary_dilation(mins > 0.72, iterations=3)
    if m.mean() < 0.002 or m.mean() > 0.25:                # nothing, or a static bright scene rather than text
        return None
    lab, k = ndimage.label(m)
    if not k:
        return None
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    keep = np.isin(lab, 1 + np.where(sizes >= 0.15 * sizes.max())[0])
    ys, xs = np.where(keep)
    H, W = m.shape
    box = [round(xs.min() / W, 3), round(ys.min() / H, 3), round((xs.max() + 1) / W, 3), round((ys.max() + 1) / H, 3)]
    return box if box[3] - box[1] <= 0.45 else None           # captions are bands; a tall box is a steady bright scene


def caption_bands(mins: np.ndarray) -> list[list[float]]:
    """Burned-in text as horizontal bands: rows where bright pixels stay put in every sample while the phone moves.
    Each band [x0, y0, x1, y1] as picture fractions; tall static bright regions (a steady shot of the screen) are
    not text and are skipped."""
    from scipy import ndimage
    m = ndimage.binary_dilation(mins > 0.72, iterations=2)
    H, W = m.shape
    if m.mean() < 0.002:
        return []
    rows = m.mean(axis=1) > 0.035
    rows = ndimage.binary_closing(rows, iterations=max(1, H // 60))
    lab, k = ndimage.label(rows)
    out = []
    for i in range(1, k + 1):
        ys = np.where(lab == i)[0]
        y0, y1 = ys.min(), ys.max() + 1
        if (y1 - y0) / H > 0.4 or (y1 - y0) < 2:
            continue
        band = m[y0:y1]
        if band.mean() < 0.05:
            continue
        xs = np.where(band.any(axis=0))[0]
        out.append([round(xs.min() / W, 3), round(y0 / H, 3), round((xs.max() + 1) / W, 3), round(y1 / H, 3)])
    return out


def overlay_pass(job_dir: Path, roster: dict | None = None, log=print) -> dict:
    """Recompute caption bands for every phone (fast: low-res grey frames, no models) into director_vision.json."""
    out = job_dir / "director_vision.json"
    have = json.loads(out.read_text())
    roster = roster or json.loads((job_dir / "director_roster.json").read_text())
    for n, cam in roster["cams"].items():
        c = cam.get("crop") or [cam["w"], cam["h"], 0, 0]
        cw, ch, cx, cy = c
        w = 180
        h = int(round(ch * w / cw / 2)) * 2
        t0, t1 = EDGE_IN, max(EDGE_IN + 0.5, cam["dur"] - EDGE_OUT)
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", cam["video"],
                            "-vf", f"crop={cw}:{ch}:{cx}:{cy},fps=1/{SAMPLE_S},scale={w}:{h},format=gray",
                            "-f", "rawvideo", "-"], capture_output=True)
        k = len(r.stdout) // (w * h)
        if k < 6 or n not in have:
            continue
        mins = np.frombuffer(r.stdout[: k * w * h], np.uint8).reshape(k, h, w).min(axis=0) / 255.0
        have[n]["captions"] = caption_bands(mins)
        log(f"  captions {cam['label'][:24]:24s} {have[n]['captions']}")
    out.write_text(json.dumps(have))
    return have


def analyze_cam(cam: dict, M: _Models) -> dict:
    out = []
    mins = None
    for t, fr in _frames(cam):
        H, W = fr.shape[:2]
        g = fr[::4, ::4].mean(axis=-1) / 255.0
        mins = g if mins is None else np.minimum(mins, g)
        ppl = [p for p in M.people(fr) if not (p[3] >= 0.97 * H and p[1] >= 0.6 * H)]   # crowd heads at the edge
        rec = {"t": round(t, 2), **_color_stats(fr), "sharp": round(_sharpness(fr.mean(axis=-1)), 3)}
        big = [p for p in ppl if (p[3] - p[1]) >= 0.08 * H]
        roles = M.classify([_crop_for_role(fr, p) for p in big], M.role_emb) if M.clip is not None else None
        persons = []
        for i, p in enumerate(big):
            q = [round(p[0] / W, 4), round(p[1] / H, 4), round(p[2] / W, 4), round(p[3] / H, 4), round(p[4], 3)]
            if roles is not None:
                pr = dict(zip(M.role_keys, roles[i]))
                if pr["crowd"] > 0.5:                     # a fan's head or raised phone, not a performer
                    continue
                band = {k: pr[k] for k in ROLES}
                k = max(band, key=band.get)
                q += [k, round(float(band[k]), 3), round(float(pr["crowd"]), 3)]
            persons.append(q)
        rec["persons"] = persons
        rec["n_small"] = len(ppl) - len(big)
        if M.clip is not None:
            sc = M.classify([fr], M.scene_emb)[0]
            rec["scene"] = {k: round(float(v), 3) for k, v in zip(M.scene_keys, sc)}
        out.append(rec)
    ok = mins is not None and len(out) >= 6
    return {"samples": out, "overlay": _overlay_box(mins) if ok else None, "captions": caption_bands(mins) if ok else []}


def analyze(job_dir: Path, roster: dict | None = None, force: bool = False, log=print) -> dict:
    """job_dir/director_vision.json: {cam: [sample, ...]} (container times, coordinates as picture fractions)."""
    out = job_dir / "director_vision.json"
    have = json.loads(out.read_text()) if out.exists() and not force else {}
    roster = roster or json.loads((job_dir / "director_roster.json").read_text())
    todo = [n for n in roster["cams"] if n not in have]
    if not todo:
        return have
    M = _Models()
    for k, n in enumerate(todo):
        cam = roster["cams"][n]
        have[n] = analyze_cam(cam, M)
        smp = have[n]["samples"]
        roles = [p[5] for s in smp for p in s["persons"] if len(p) > 5 and p[6] > 0.5]
        top = max(set(roles), key=roles.count) if roles else "-"
        log(f"  vision {k + 1}/{len(todo)} {cam['label'][:24]:24s} {len(smp)} samples, "
            f"{sum(len(s['persons']) for s in smp)} performers, mostly {top}, caption {have[n]['overlay']}")
        out.write_text(json.dumps(have))
    return have


# ----------------------------------------------------------------------------------------------- music
def featured_track(job_dir: Path, hop_s: float = 0.5) -> dict:
    """Who the music features, every `hop_s`: 'vocals' while singing, else the instrument standing out of the
    mix relative to its own level through the song (solo, fill, riff), else 'band'. Plus per-role activity 0..1."""
    import soundfile as sf
    stems = {"vocals": "stem_vocals.wav", "guitar": "stem_guitar.wav", "bass": "stem_bass.wav",
             "drums": "stem_drums.wav", "keys": "stem_piano.wav"}
    lev = {}
    n = None
    for role, fn in stems.items():
        p = job_dir / fn
        if not p.exists():
            continue
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        hop = int(hop_s * sr)
        m = len(x) // hop
        rms = np.sqrt((x[: m * hop].reshape(m, hop) ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms)
        k = max(1, int(round(2.0 / hop_s)))
        lev[role] = np.convolve(db, np.ones(k) / k, mode="same")
        n = m if n is None else min(n, m)
    if not lev:
        return {"hop_s": hop_s, "featured": [], "activity": {}}
    lev = {r: v[:n] for r, v in lev.items()}
    loud = max(np.percentile(v, 95) for v in lev.values())
    act, rel = {}, {}
    for r, v in lev.items():
        p95 = np.percentile(v, 95)
        act[r] = np.clip((v - (p95 - 24.0)) / 18.0, 0.0, 1.0) * (p95 > loud - 30.0)     # silent stems never act
        rel[r] = v - np.median(v)                                                         # stands out vs itself
    voc = act.get("vocals", np.zeros(n)) > 0.45
    feat = []
    for i in range(n):
        if voc[i]:
            feat.append("vocals")
            continue
        cand = {r: rel[r][i] + 6.0 * act[r][i] - (3.0 if r == "keys" else 0.0)
                for r in rel if r != "vocals" and act[r][i] > 0.5}
        best = max(cand, key=cand.get) if cand else None
        feat.append(best if best and rel[best][i] > 2.0 else "band")
    # hold decisions for >= 2 s so the picture does not flicker between members
    k = max(1, int(round(2.0 / hop_s)))
    out = list(feat)
    i = 0
    while i < n:
        j = i
        while j < n and feat[j] == feat[i]:
            j += 1
        if j - i < k and i > 0:
            for q in range(i, j):
                out[q] = out[i - 1]
        i = j
    return {"hop_s": hop_s, "featured": out, "activity": {r: [round(float(a), 3) for a in v] for r, v in act.items()}}
