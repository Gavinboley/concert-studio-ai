"""Stage 4: Multi-stem neural and spectral source separation (Vocals, Drums, Bass, Other, Crowd)."""

from __future__ import annotations

import importlib.util
from typing import Any
import numpy as np
from scipy import ndimage, signal

from concert_studio.models import AudioTrack, StemBundle

_DEMUCS_MODEL: Any = None


def _get_demucs_model():
    """Lazily load and cache the Demucs htdemucs neural separation model."""
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is not None:
        return _DEMUCS_MODEL

    try:
        import torch
        from demucs.pretrained import get_model

        model = get_model(name="htdemucs")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _DEMUCS_MODEL = model
        return _DEMUCS_MODEL
    except Exception as e:
        return None


def _separate_with_demucs(stereo_audio: np.ndarray, sr: int = 48000) -> StemBundle | None:
    """Run Facebook Demucs (htdemucs) neural stem separation in chunked 30s blocks."""
    model = _get_demucs_model()
    if model is None:
        return None

    try:
        import torch
        from demucs.apply import apply_model

        device = next(model.parameters()).device
        model_sr = model.samplerate
        n_samples = stereo_audio.shape[0]

        # Resample to model sample rate (44100 Hz) if needed
        if sr != model_sr:
            gcd = int(np.gcd(sr, model_sr))
            resampled = signal.resample_poly(stereo_audio, model_sr // gcd, sr // gcd, axis=0)
        else:
            resampled = stereo_audio.copy()

        total_res_samples = resampled.shape[0]
        # Process in 30-second overlap-add chunks to preserve GPU VRAM
        chunk_len = int(30.0 * model_sr)
        overlap_len = int(2.0 * model_sr)
        step = chunk_len - overlap_len

        # Model sources: ['drums', 'bass', 'other', 'vocals']
        out_accum = {src: np.zeros((total_res_samples, 2), dtype=np.float64) for src in model.sources}
        weight_acc = np.zeros(total_res_samples, dtype=np.float64)

        for s0 in range(0, total_res_samples, step):
            s1 = min(total_res_samples, s0 + chunk_len)
            blk_len = s1 - s0
            if blk_len < int(0.5 * model_sr):
                break

            blk_data = resampled[s0:s1]
            tensor = torch.from_numpy(blk_data.T[None, :, :]).to(device).float()

            with torch.no_grad():
                chunk_sources = apply_model(model, tensor, device=device, split=True, overlap=0.25)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Create smooth Hann crossfade window for chunk boundaries
            win = np.ones(blk_len, dtype=np.float64)
            if s0 > 0 and blk_len > overlap_len:
                win[:overlap_len] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, overlap_len)))
            if s1 < total_res_samples and blk_len > overlap_len:
                win[-overlap_len:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, overlap_len)))

            for i, src_name in enumerate(model.sources):
                src_np = chunk_sources[0, i].cpu().numpy().T.astype(np.float64)
                out_accum[src_name][s0:s1] += src_np[:blk_len] * win[:, None]

            weight_acc[s0:s1] += win
            if s1 >= total_res_samples:
                break

        valid = weight_acc > 1e-8
        for src_name in model.sources:
            out_accum[src_name][valid] /= weight_acc[valid, None]

        # Resample back to original sr if needed
        final_stems: dict[str, np.ndarray] = {}
        for src_name in model.sources:
            if sr != model_sr:
                gcd = int(np.gcd(sr, model_sr))
                st = signal.resample_poly(out_accum[src_name], sr // gcd, model_sr // gcd, axis=0)
            else:
                st = out_accum[src_name]
            final_stems[src_name] = st[:n_samples]

        vocals = final_stems["vocals"]
        drums = final_stems["drums"]
        bass = final_stems["bass"]
        other = final_stems["other"]

        # Synthesize Crowd Removed stem as the physical acoustic residual
        # mix - (vocals + drums + bass + other), isolating audience cheering and room acoustics
        music_sum = vocals + drums + bass + other
        crowd_residual = stereo_audio - music_sum
        # Gentle highpass on crowd to remove low-frequency rumble below 80 Hz
        sos_crowd = signal.butter(2, 80.0, btype="highpass", fs=sr, output="sos")
        crowd_removed = signal.sosfilt(sos_crowd, crowd_residual, axis=0)

        return StemBundle(
            vocals=vocals,
            drums=drums,
            bass=bass,
            other=other,
            crowd_removed=crowd_removed,
            sample_rate=sr,
        )
    except Exception as e:
        return None


def _hpss_complex_masks(
    mag: np.ndarray,
    harmonic_kernel: int = 17,
    percussive_kernel: int = 17,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute soft Wiener masks for harmonic (horizontal) vs percussive (vertical) energy."""
    harm_est = ndimage.median_filter(mag, size=(1, harmonic_kernel))
    perc_est = ndimage.median_filter(mag, size=(percussive_kernel, 1))
    h_p = harm_est**power
    p_p = perc_est**power
    total = h_p + p_p + 1e-12
    return h_p / total, p_p / total


def _torch_refine_masks(
    freqs: np.ndarray,
    mag_mid: np.ndarray,
    mag_side: np.ndarray,
    harm_mask: np.ndarray,
    perc_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robust 5-stem DSP mask computation with wide vocal presence and clean bass/drum isolation."""
    f = freqs[:, None]

    # 1. Diffuse crowd floor (aperiodic background murmur)
    temporal_min = ndimage.minimum_filter(mag_mid + 0.5 * mag_side, size=(1, 25))
    temporal_med = ndimage.median_filter(mag_mid + 0.5 * mag_side, size=(5, 21))
    flatness = temporal_min / (temporal_med + 1e-9)
    crowd_band = np.exp(-0.5 * ((np.log2(np.maximum(f, 50.0) / 2200.0)) / 1.4) ** 2)
    raw_crowd_mask = np.clip(0.12 * flatness * crowd_band, 0.0, 0.15)
    clean_ratio = 1.0 - raw_crowd_mask

    # 2. Bass mask: strictly low fundamental register (35 Hz - 180 Hz) so vocals and snare do NOT bleed into bass
    bass_freq_weight = 1.0 / (1.0 + (np.maximum(f, 1.0) / 190.0) ** 4)
    mask_bass = clean_ratio * (0.85 * harm_mask + 0.15 * perc_mask) * bass_freq_weight

    # 3. Drums mask: true percussive transients across spectrum + sub-kick punch (<100 Hz), plosive-suppressed
    kick_punch_weight = 1.0 / (1.0 + (np.maximum(f, 1.0) / 100.0) ** 4)
    cymbals_weight = 1.0 / (1.0 + (5000.0 / np.maximum(f, 1.0)) ** 3)
    # Vocal plosives suppression: speech formants (300-2500 Hz) are dampened in drums mask
    vocal_plosive_band = np.exp(-0.5 * ((np.log2(np.maximum(f, 50.0) / 900.0)) / 1.1) ** 2)
    mask_drums = clean_ratio * np.clip(
        perc_mask * (1.0 + 0.50 * kick_punch_weight + 0.30 * cymbals_weight - 0.45 * vocal_plosive_band),
        0.0,
        1.0,
    )

    # 4. Vocals mask: wide, prominent vocal presence (110 Hz – 9000 Hz) with full harmonic body
    mid_dominance = (mag_mid**2) / (mag_mid**2 + 1.8 * (mag_side**2) + 1e-12)
    vocal_presence = (1.0 / (1.0 + (110.0 / np.maximum(f, 1.0)) ** 4)) * (
        1.0 / (1.0 + (np.maximum(f, 1.0) / 9200.0) ** 4)
    )
    # Give vocals full priority in the vocal formant register so they are loud, intelligible, and prominent
    mask_vocals = clean_ratio * harm_mask * mid_dominance * vocal_presence * 1.35

    # 5. Other: electric guitars, rhythm instruments, keyboards, synths, stereo content
    side_dominance = (mag_side**2) / (mag_mid**2 + mag_side**2 + 1e-12)
    inst_harmonic = harm_mask * (1.0 - 0.50 * (mask_vocals / (clean_ratio + 1e-12))) * (1.0 - bass_freq_weight)
    mask_other = clean_ratio * np.maximum(
        inst_harmonic,
        0.22 * harm_mask + 0.35 * side_dominance,
    )

    # Normalize soft masks so sum equals unity
    denom = mask_vocals + mask_drums + mask_bass + mask_other + raw_crowd_mask + 1e-12
    return (
        mask_vocals / denom,
        mask_drums / denom,
        mask_bass / denom,
        mask_other / denom,
        raw_crowd_mask / denom,
    )


def separate_concert_stems(
    track: AudioTrack,
    backend: str = "auto",
    n_fft: int = 2048,
) -> StemBundle:
    """Separate a consensus concert track into Vocals, Drums, Bass, Other, and Crowd Removed stems.

    Supports `backend="auto" | "neural" | "dsp"`.
    In "auto" or "neural" mode, utilizes Facebook Demucs (htdemucs) on CUDA/CPU for
    industry-standard studio stem isolation. Falls back to advanced DSP filterbank
    if Demucs is unavailable.
    """
    if backend not in ("auto", "neural", "dsp"):
        raise ValueError(f"Unsupported separation backend: {backend}")

    sr = track.sample_rate
    stereo = track.to_stereo()

    # 1. Neural Separation via Demucs (CUDA accelerated on RTX 5070 Ti)
    if backend in ("auto", "neural"):
        neural_bundle = _separate_with_demucs(stereo, sr=sr)
        if neural_bundle is not None:
            return neural_bundle

    # 2. High-Fidelity DSP Fallback
    n_samples = stereo.shape[0]
    hop = n_fft // 4

    freqs, _, Z_l = signal.stft(stereo[:, 0], fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z_r = signal.stft(stereo[:, 1], fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    Z_mid = 0.5 * (Z_l + Z_r)
    Z_side = 0.5 * (Z_l - Z_r)

    mag_mid = np.abs(Z_mid)
    mag_side = np.abs(Z_side)
    mag_total = np.sqrt(0.5 * (np.abs(Z_l) ** 2 + np.abs(Z_r) ** 2))

    harm_mask, perc_mask = _hpss_complex_masks(mag_total)

    if importlib.util.find_spec("torch") is not None:
        import torch

        with torch.no_grad():
            t_harm = torch.from_numpy(harm_mask)
            t_perc = torch.from_numpy(perc_mask)
            stacked = torch.stack([t_harm, t_perc], dim=0)
            temperature_sharpened = torch.softmax(stacked * 2.4, dim=0)
            harm_mask = temperature_sharpened[0].numpy()
            perc_mask = temperature_sharpened[1].numpy()

    m_voc, m_drm, m_bas, m_oth, m_crd = _torch_refine_masks(
        freqs, mag_mid, mag_side, harm_mask, perc_mask
    )

    def _reconstruct_stereo(mask: np.ndarray, center_bias: float = 0.0) -> np.ndarray:
        if center_bias > 0.0:
            z_out_l = mask * ((1.0 - center_bias) * Z_l + center_bias * Z_mid)
            z_out_r = mask * ((1.0 - center_bias) * Z_r + center_bias * Z_mid)
        else:
            z_out_l = mask * Z_l
            z_out_r = mask * Z_r
        _, yl = signal.istft(z_out_l, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        _, yr = signal.istft(z_out_r, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        if len(yl) < n_samples:
            yl = np.pad(yl, (0, n_samples - len(yl)))
        if len(yr) < n_samples:
            yr = np.pad(yr, (0, n_samples - len(yr)))
        out = np.zeros((n_samples, 2), dtype=np.float64)
        out[:, 0] = yl[:n_samples]
        out[:, 1] = yr[:n_samples]
        return out

    vocals = _reconstruct_stereo(m_voc, center_bias=0.75)
    drums = _reconstruct_stereo(m_drm, center_bias=0.20)
    bass = _reconstruct_stereo(m_bas, center_bias=0.90)
    other = _reconstruct_stereo(m_oth, center_bias=0.0)
    crowd_removed = _reconstruct_stereo(m_crd, center_bias=0.0)

    return StemBundle(
        vocals=vocals,
        drums=drums,
        bass=bass,
        other=other,
        crowd_removed=crowd_removed,
        sample_rate=sr,
    )
