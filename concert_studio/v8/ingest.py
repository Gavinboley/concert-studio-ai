"""Clip ingest: probe codec profile, decode to float WAV cache, load mono."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import to_mono

SR = 44100  # every TikTok/phone export here is 44.1 kHz AAC; keep it native until the master.
MEDIA_EXT = {".mp4", ".mov", ".m4a", ".aac", ".wav", ".flac", ".mp3", ".mkv", ".webm"}


@dataclass
class ClipInfo:
    name: str
    src: str
    wav: str
    codec: str
    profile: str          # "LC", "HE-AAC", "HE-AACv2", "PCM", ...
    bitrate: int
    duration: float
    has_video: bool

    @property
    def parametric_stereo(self) -> bool:
        """HE-AACv2 synthesises its stereo image from a mono core: never trust L/R."""
        return self.profile.upper() == "HE-AACV2"

    @property
    def sbr(self) -> bool:
        """HE-AAC (v1/v2) synthesises highs above the SBR crossover from the lows."""
        return self.profile.upper().startswith("HE-AAC")


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,profile,bit_rate,sample_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def ingest_folder(src_dir: str | Path, cache_dir: str | Path) -> list[ClipInfo]:
    src_dir, cache_dir = Path(src_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    clips: list[ClipInfo] = []
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in MEDIA_EXT:
            continue
        meta = _probe(p)
        a = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), None)
        if a is None:
            continue
        has_video = any(s.get("codec_type") == "video" for s in meta.get("streams", []))
        wav = cache_dir / f"{p.stem}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-vn", "-ac", "2", "-ar", str(SR),
                            "-c:a", "pcm_f32le", str(wav)], check=True)
        clips.append(ClipInfo(
            name=p.stem, src=str(p), wav=str(wav), codec=a.get("codec_name", "?"),
            profile=a.get("profile", "PCM" if a.get("codec_name", "").startswith("pcm") else "?"),
            bitrate=int(a.get("bit_rate") or 0), duration=float(meta["format"]["duration"]),
            has_video=has_video,
        ))
    return clips


def load_mono(clip: ClipInfo) -> np.ndarray:
    x, sr = sf.read(clip.wav, dtype="float32", always_2d=True)
    assert sr == SR, (clip.name, sr)
    return to_mono(x)


def load_stereo(clip: ClipInfo) -> np.ndarray:
    x, sr = sf.read(clip.wav, dtype="float32", always_2d=True)
    assert sr == SR
    return x


def save_manifest(clips: list[ClipInfo], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(c) for c in clips], indent=1))


def load_manifest(path: str | Path) -> list[ClipInfo]:
    return [ClipInfo(**d) for d in json.loads(Path(path).read_text())]
