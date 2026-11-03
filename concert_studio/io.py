"""High-definition WAV audio I/O supporting 16-bit, 24-bit HD Studio PCM, and 32-bit float."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import wave
import numpy as np
from scipy import signal

from concert_studio.models import AudioTrack, IMUTelemetry
from concert_studio.telemetry import extract_imu_from_mp4_bytes, parse_imu_json_sidecar


def write_wav_hd(
    path: str | Path,
    track: AudioTrack | np.ndarray,
    sample_rate: int = 48000,
    bit_depth: int = 24,
) -> Path:
    """Write an AudioTrack or numpy array to a 16-bit or 24-bit PCM WAV file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(track, AudioTrack):
        samples = track.samples
        sr = track.sample_rate
    else:
        samples = np.asarray(track, dtype=np.float64)
        sr = sample_rate

    if samples.ndim == 1:
        channels = 1
        frames = samples[:, None]
    elif samples.ndim == 2:
        channels = samples.shape[1]
        frames = samples
    else:
        raise ValueError(f"Unsupported sample array shape: {samples.shape}")

    clipped = np.clip(frames, -1.0, 0.99999988)

    if bit_depth == 24:
        scale = float((1 << 23) - 1)
        pcm32 = np.ascontiguousarray(np.round(clipped * scale).astype("<i4"))
        raw_bytes = pcm32.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        sampwidth = 3
    elif bit_depth == 16:
        scale = float((1 << 15) - 1)
        pcm16 = np.ascontiguousarray(np.round(clipped * scale).astype("<i2"))
        raw_bytes = pcm16.tobytes()
        sampwidth = 2
    else:
        raise ValueError(f"bit_depth must be 16 or 24, got {bit_depth}")

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(int(sr))
        wf.writeframes(raw_bytes)

    return out_path


def read_wav(
    path: str | Path,
    target_sr: int | None = None,
) -> AudioTrack:
    """Read a 16-bit, 24-bit, or 32-bit PCM WAV file into an AudioTrack."""
    in_path = Path(path)
    with wave.open(str(in_path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / float(1 << 15)
    elif sampwidth == 3:
        raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((raw_u8.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = raw_u8
        # Sign-extend 24-bit to 32-bit signed integer
        negative_mask = (raw_u8[:, 2] & 0x80) != 0
        padded[negative_mask, 3] = 0xFF
        pcm32 = padded.view("<i4").reshape(-1)
        data = pcm32.astype(np.float64) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        data = data.reshape(-1, channels)

    if target_sr is not None and target_sr != sr:
        gcd = int(np.gcd(target_sr, sr))
        up = target_sr // gcd
        down = sr // gcd
        data = signal.resample_poly(data, up, down, axis=0)
        sr = target_sr

    if data.shape[0] > int(30 * sr):
        data = data.astype(np.float32, copy=False)

    return AudioTrack(name=in_path.stem, samples=data, sample_rate=sr)


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg on PATH or common Windows WinGet/Scoop installation paths."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    local_appdata = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if local_appdata.exists():
        for candidate in local_appdata.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"):
            if candidate.is_file():
                return str(candidate)
    return None


def _read_mp4_head_tail_bytes(path: Path, max_chunk: int = 2 * 1024 * 1024) -> bytes:
    """Read header and trailer bytes of an MP4/MOV container without loading hundreds of MBs of video."""
    size = path.stat().st_size
    if size <= max_chunk * 2:
        return path.read_bytes()
    with path.open("rb") as fh:
        head = fh.read(max_chunk)
        fh.seek(max(0, size - max_chunk))
        tail = fh.read(max_chunk)
    return head + tail


def load_media_track(
    path: str | Path,
    target_sr: int | None = 48000,
    imu_sidecar: str | Path | dict | None = None,
) -> AudioTrack:
    """Load an AudioTrack from WAV or video/compressed container (.mp4, .mov, .mp3, .flac, .m4a).

    Automatically scans MP4/MOV binary containers for embedded `camm`/`gpmd` IMU motion tracks,
    extracts lightweight RGB stage keyframes from video containers, and attaches JSON IMU sidecars.
    """
    in_path = Path(path)
    imu: IMUTelemetry | None = None
    video_frames: np.ndarray | None = None

    if imu_sidecar is not None:
        imu = parse_imu_json_sidecar(imu_sidecar)
    elif in_path.suffix.lower() in (".mp4", ".mov"):
        try:
            raw_bytes = _read_mp4_head_tail_bytes(in_path)
            imu = extract_imu_from_mp4_bytes(raw_bytes)
        except OSError:
            imu = None

    # Check if companion <stem>.imu.json exists beside the media file
    if imu is None:
        auto_sidecar = in_path.with_suffix(".imu.json")
        if auto_sidecar.exists():
            imu = parse_imu_json_sidecar(auto_sidecar)

    if in_path.suffix.lower() == ".wav":
        trk = read_wav(in_path, target_sr=target_sr)
        return AudioTrack(
            name=trk.name,
            samples=trk.samples,
            sample_rate=trk.sample_rate,
            imu_telemetry=imu,
        )

    ffmpeg_bin = _find_ffmpeg()
    if ffmpeg_bin is None:
        raise RuntimeError(
            f"Cannot decode non-WAV container '{in_path.name}' without ffmpeg on PATH"
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        sr_arg = str(target_sr or 48000)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(in_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            sr_arg,
            "-acodec",
            "pcm_s24le",
            str(tmp_wav),
        ]
        if in_path.suffix.lower() in (".mp4", ".mov"):
            cmd.extend(
                [
                    "-an",
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=64:36",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
            )
        res = subprocess.run(cmd, capture_output=True, check=True)
        frame_bytes = 64 * 36 * 3
        if len(res.stdout) >= frame_bytes:
            video_frames = np.frombuffer(res.stdout[:frame_bytes], dtype=np.uint8).reshape(
                1, 36, 64, 3
            )
        trk = read_wav(tmp_wav, target_sr=target_sr)
        return AudioTrack(
            name=in_path.stem,
            samples=trk.samples,
            sample_rate=trk.sample_rate,
            imu_telemetry=imu,
            video_frames_rgb=video_frames,
        )
    finally:
        tmp_wav.unlink(missing_ok=True)
