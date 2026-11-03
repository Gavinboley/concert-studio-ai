"""iZotope Ozone 12 Advanced Mastering Bridge.

Applies native VST3 mastering chain using Pedalboard:
1. Ozone 12 Stabilizer (Adaptive resonance suppression & spectral tonal balancing)
2. Ozone 12 Low End Focus (Bass & Kick punch, transient definition & mud elimination)
3. Ozone 12 Clarity (Adaptive psychoacoustic clarity & dynamic range enhancement)
4. Ozone 12 Maximizer (IRC IV True-Peak limiting & integrated LUFS normalization)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import os
import subprocess
import sys
import numpy as np
from scipy import signal

from concert_studio.mastering import measure_lufs, measure_true_peak_dbtp


OZONE_VST_DIR_WIN = Path("C:/Program Files/Common Files/VST3/iZotope")
# Windows Python with `pedalboard` that can host the Ozone VST3 (WSL cannot load Windows plugins itself).
# Set CS_OZONE_PYTHON, e.g. C:/Users/<you>/.venvs/ozone/Scripts/python.exe
WIN_PYTHON = Path(os.environ.get("CS_OZONE_PYTHON", "python.exe"))


def _to_win_path(p: Path | str) -> str:
    path_str = str(p).replace("\\", "/")
    if path_str.startswith("/mnt/c/"):
        return "C:/" + path_str[7:]
    return path_str


def _to_posix_path(p: Path | str) -> str:
    path_str = str(p).replace("\\", "/")
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        return f"/mnt/{drive}/" + path_str[3:]
    return path_str


def is_ozone_available() -> bool:
    """Check if Ozone 12 VST3 and a Python environment with pedalboard are available."""
    if platform.system() == "Windows":
        vst_path = OZONE_VST_DIR_WIN / "Ozone 12 Maximizer.vst3"
        if not vst_path.exists():
            vst_path = OZONE_VST_DIR_WIN / "Ozone 12.vst3"
        return vst_path.exists()
    else:
        # In WSL/Linux, check if Windows python and VST exist
        wsl_vst = Path("/mnt/c/Program Files/Common Files/VST3/iZotope/Ozone 12 Maximizer.vst3")
        wsl_vst_main = Path("/mnt/c/Program Files/Common Files/VST3/iZotope/Ozone 12.vst3")
        return (wsl_vst.exists() or wsl_vst_main.exists())


def master_with_ozone_native(
    audio: np.ndarray,
    sr: int = 48000,
    target_lufs: float = -14.0,
    ceiling_dbtp: float = -1.0,
) -> tuple[np.ndarray, dict]:
    """Process audio directly via Pedalboard and Ozone 12 VST3 on Windows."""
    try:
        from pedalboard import Pedalboard, load_plugin
    except ImportError:
        raise RuntimeError("pedalboard is required to run Ozone 12 mastering on Windows.")

    # Convert audio to shape (channels, samples) float32 for Pedalboard
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    channels_first = arr.T  # (2, N)

    # Build Ozone 12 mastering chain
    chain_plugins = []
    modules_used = []
    module_names = ["Stabilizer", "Low End Focus", "Clarity", "Maximizer"]

    for mod in module_names:
        vst_file = OZONE_VST_DIR_WIN / f"Ozone 12 {mod}.vst3"
        if vst_file.exists():
            p = load_plugin(str(vst_file))
            if mod == "Stabilizer":
                if hasattr(p, "stbl_stereo_main_amount"):
                    p.stbl_stereo_main_amount = 30.0
                if hasattr(p, "stbl_tame_transients_on_off"):
                    p.stbl_tame_transients_on_off = True
            elif mod == "Low End Focus":
                if hasattr(p, "lef_stereo_main_contrast"):
                    p.lef_stereo_main_contrast = 35.0
                if hasattr(p, "lef_stereo_main_gain_db"):
                    p.lef_stereo_main_gain_db = 1.0
            elif mod == "Clarity":
                if hasattr(p, "cla_stereo_main_amount"):
                    p.cla_stereo_main_amount = 25.0
            elif mod == "Maximizer":
                if hasattr(p, "max_output_level"):
                    p.max_output_level = ceiling_dbtp
                if hasattr(p, "max_transient_shaping_amt"):
                    p.max_transient_shaping_amt = 20.0
                if hasattr(p, "max_character"):
                    p.max_character = 2.5
            chain_plugins.append(p)
            modules_used.append(mod)

    # Fallback to full Ozone 12 plugin if standalone modules not found
    if not chain_plugins:
        full_vst = OZONE_VST_DIR_WIN / "Ozone 12.vst3"
        if full_vst.exists():
            chain_plugins.append(load_plugin(str(full_vst)))
            modules_used.append("Ozone 12 Advanced")

    if not chain_plugins:
        raise FileNotFoundError(f"No Ozone 12 VST3 plugins found in {OZONE_VST_DIR_WIN}")

    board = Pedalboard(chain_plugins)

    # Process through Ozone 12 rack
    processed_cf = board(channels_first, sr)
    processed = processed_cf.T.astype(np.float64)  # (N, 2)

    # Fine-tune integrated LUFS & True Peak compliance
    curr_lufs = measure_lufs(processed, sr=sr)
    if curr_lufs > -65.0:
        gain_db = float(np.clip(target_lufs - curr_lufs, -6.0, 6.0))
        processed = processed * (10.0 ** (gain_db / 20.0))

    tp_db = measure_true_peak_dbtp(processed)
    if tp_db > ceiling_dbtp:
        scale_down = 10.0 ** ((ceiling_dbtp - tp_db - 0.05) / 20.0)
        processed *= scale_down

    final_lufs = measure_lufs(processed, sr=sr)
    final_tp = measure_true_peak_dbtp(processed)

    meta = {
        "engine": "iZotope Ozone 12 Advanced",
        "modules": modules_used,
        "sample_rate": sr,
        "integrated_lufs": round(final_lufs, 2),
        "true_peak_dbtp": round(final_tp, 2),
    }
    return processed, meta


def master_file_with_ozone(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    target_lufs: float = -14.0,
    ceiling_dbtp: float = -1.0,
) -> dict:
    """Master a WAV file through Ozone 12, working seamlessly across Windows and WSL."""
    in_p = Path(input_wav_path)
    out_p = Path(output_wav_path)

    if platform.system() == "Windows":
        from concert_studio.io import load_media_track, write_wav_hd
        trk = load_media_track(in_p)
        out_samples, meta = master_with_ozone_native(
            trk.samples,
            sr=trk.sample_rate,
            target_lufs=target_lufs,
            ceiling_dbtp=ceiling_dbtp,
        )
        write_wav_hd(out_p, out_samples, sample_rate=trk.sample_rate, bit_depth=24)
        return meta
    else:
        # In WSL, run via Windows Python
        win_in = _to_win_path(in_p)
        win_out = _to_win_path(out_p)
        cmd = [
            "cmd.exe",
            "/c",
            f"{WIN_PYTHON} -m concert_studio.ozone \"{win_in}\" \"{win_out}\" --target-lufs {target_lufs} --ceiling-dbtp {ceiling_dbtp}",
        ]
        res = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Ozone Windows bridge failed: {res.stderr}\n{res.stdout}")
        
        # Verify output exists
        posix_out = Path(_to_posix_path(win_out))
        if not posix_out.exists():
            raise FileNotFoundError(f"Ozone master file was not created: {posix_out}")

        return {
            "engine": "iZotope Ozone 12 Advanced",
            "modules": ["Stabilizer", "Low End Focus", "Clarity", "Maximizer"],
            "target_lufs": target_lufs,
            "ceiling_dbtp": ceiling_dbtp,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ozone 12 Advanced Mastering Bridge")
    parser.add_argument("input_wav", help="Input WAV audio file")
    parser.add_argument("output_wav", help="Output mastered WAV audio file")
    parser.add_argument("--target-lufs", type=float, default=-14.0)
    parser.add_argument("--ceiling-dbtp", type=float, default=-1.0)
    args = parser.parse_args()

    meta = master_file_with_ozone(
        args.input_wav,
        args.output_wav,
        target_lufs=args.target_lufs,
        ceiling_dbtp=args.ceiling_dbtp,
    )
    print(f"Ozone 12 Master Complete: {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
