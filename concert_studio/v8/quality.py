"""Per-take perceptual quality on the comp timeline (Audiobox-Aesthetics, Meta 2025).

Audiobox-Aesthetics is a no-reference predictor trained on human ratings: PQ (production quality),
CE (content enjoyment), CU (content usefulness), PC (production complexity). It sees 16 kHz mono in 10 s
windows, so it judges the 0-8 kHz band (vocals, guitars, distortion, noise, room) and not air or stereo.
On this payload it ranks V7 5.69 < v8e 6.32 < v8e+Ozone 6.51 (PQ), and a take's PQ at a given moment agrees
with its vocal-band coherence against the other phones (median within-window r = 0.47): it hears which
phone is muddy, distant, distorted or has a fan singing next to it.

It needs torch + torchaudio + audiobox_aesthetics, kept in their own venv (~/.venvs/aes, CPU is enough),
so this file is a standalone script that only uses numpy/soundfile/torch:

    ~/.venvs/aes/bin/python concert_studio/v8/quality.py --work ~/cstudio_work

Writes <work>/take_quality.npz: names, centers (timeline s), PQ[N, G], CE[N, G] (NaN where not covered).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
WIN_S = 10.0          # the model's training window
HOP_S = 2.5
EDGE_IN, EDGE_OUT = 0.5, 2.0   # same clip-edge trim as the comp (TikTok end cards)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--hop", type=float, default=HOP_S)
    a = ap.parse_args()
    import torch
    from audiobox_aesthetics.infer import initialize_predictor

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    work = a.work
    sel = json.loads((work / "selection.json").read_text())
    pls = {d["name"]: d for d in json.loads((work / "placements.json").read_text())}
    names = [n for n in sel["keep"] if (work / "crowd" / f"{n}_music.wav").exists()]
    t_lo = min(pls[n]["offset_s"] for n in names)
    t_hi = max(pls[n]["offset_s"] + pls[n]["duration"] for n in names)
    starts = np.arange(t_lo, t_hi - WIN_S, a.hop)
    PQ = np.full((len(names), len(starts)), np.nan, np.float32)
    CE = PQ.copy()
    pred = initialize_predictor()
    t0 = time.time()
    for i, n in enumerate(names):
        x, sr = sf.read(work / "crowd" / f"{n}_music.wav", dtype="float32")
        assert sr == SR
        if x.ndim == 2:
            x = x.mean(axis=1)
        dur = len(x) / SR
        off = pls[n]["offset_s"]
        items, idx = [], []
        for j, s in enumerate(starts):
            p0 = s - off
            if p0 >= EDGE_IN and p0 + WIN_S <= dur - EDGE_OUT:
                seg = x[int(p0 * SR):int(p0 * SR) + int(WIN_S * SR)]
                items.append({"path": torch.from_numpy(np.ascontiguousarray(seg[None])), "sample_rate": SR})
                idx.append(j)
        for k in range(0, len(items), 8):
            for j, o in zip(idx[k:k + 8], pred.forward(items[k:k + 8])):
                PQ[i, j], CE[i, j] = o["PQ"], o["CE"]
        print(f"  {n[-10:]} {len(idx):3d} windows  PQ {np.nanmean(PQ[i]) if idx else float('nan'):.2f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(work / "take_quality.npz", names=np.array(names), centers=starts + WIN_S / 2, PQ=PQ, CE=CE,
             win_s=WIN_S, hop_s=a.hop)
    print(f"wrote {work / 'take_quality.npz'}")


if __name__ == "__main__":
    main()
