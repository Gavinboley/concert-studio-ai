"""Neural separation wrappers (python-audio-separator, RoFormer family, CUDA).

Models (all public UVR/MSST community checkpoints):
  crowd   mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt   -> crowd / other(music)
  stems6  BS-Roformer-SW.ckpt (jarredou)                           -> vocals bass drums guitar piano other
  vocals  mel_band_roformer_kim_ft3_unwa.ckpt                      -> vocals / other
  dereverb dereverb_mel_band_roformer_mono_anvuew.ckpt (mono model) -> noreverb / reverb
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

MODELS = {
    "crowd": "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
    "stems6": "BS-Roformer-SW.ckpt",
    "vocals": "mel_band_roformer_kim_ft3_unwa.ckpt",
    # the stereo anvuew model keys on L/R decorrelation and returns identity for mono input
    "dereverb": "dereverb_mel_band_roformer_mono_anvuew.ckpt",
}


def _stem_label(fname: str) -> str:
    # audio-separator names outputs "<base>_(<stem>)_<model>.wav"
    a = fname.find("_(")
    b = fname.find(")", a)
    return fname[a + 2:b].lower()


class Runner:
    def __init__(self, model_dir: str | Path, tmp_dir: str | Path):
        from audio_separator.separator import Separator
        self.tmp = Path(tmp_dir)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.sep = Separator(model_file_dir=str(model_dir), output_dir=str(self.tmp), output_format="WAV",
                             log_level=30, normalization_threshold=0.98, sample_rate=44100)
        self.loaded = None

    def load(self, key: str):
        if self.loaded != key:
            self.sep.load_model(model_filename=MODELS[key])
            self.loaded = key

    def run(self, key: str, x: np.ndarray, sr: int, name: str) -> dict[str, np.ndarray]:
        """Separate a mono or stereo float array. Returns {stem: array (same channel count)}."""
        self.load(key)
        mono = x.ndim == 1
        xin = np.stack([x, x], axis=1) if mono else x
        peak = float(np.max(np.abs(xin))) + 1e-9
        gain = 0.9 / peak if peak > 0.9 else 1.0
        src = self.tmp / f"{name}.wav"
        sf.write(src, xin * gain, sr, subtype="FLOAT")
        outs = self.sep.separate(str(src))
        res = {}
        for o in outs:
            p = Path(o)
            if not p.is_absolute():
                p = self.tmp / p
            y, _ = sf.read(p, dtype="float32", always_2d=True)
            y = y[:len(xin)] / gain
            if len(y) < len(xin):
                y = np.pad(y, ((0, len(xin) - len(y)), (0, 0)))
            res[_stem_label(p.name)] = y.mean(axis=1) if mono else y
            p.unlink(missing_ok=True)
        src.unlink(missing_ok=True)
        return res


def cleanup(tmp_dir: str | Path) -> None:
    shutil.rmtree(tmp_dir, ignore_errors=True)
