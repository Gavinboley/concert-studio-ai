"""Unit tests for the Aggressive Studio Instrument & Vocal Recreation Engine."""

import numpy as np
import pytest
from concert_studio.studio_recreation import (
    generate_studio_kick_patch,
    generate_studio_snare_patch,
    recreate_drums_studio,
    recreate_bass_studio,
    recreate_guitars_studio,
    recreate_vocals_studio,
    mix_and_master_studio_recreation,
    apply_ultrasonic_guard,
)


def test_generate_studio_patches():
    sr = 48000
    kick = generate_studio_kick_patch(sr=sr)
    snare = generate_studio_snare_patch(sr=sr)

    assert kick.ndim == 1
    assert snare.ndim == 1
    assert len(kick) == int(0.22 * sr)
    assert len(snare) == int(0.24 * sr)

    # Normalized peaks
    assert np.isclose(np.max(np.abs(kick)), 1.0, atol=1e-3)
    assert np.isclose(np.max(np.abs(snare)), 1.0, atol=1e-3)

    # Smooth attack ramps (Zero clicking)
    assert np.abs(kick[0]) < 1e-4
    assert np.abs(snare[0]) < 1e-4


def test_recreate_drums_studio():
    sr = 48000
    n = int(2.0 * sr)
    t = np.arange(n) / float(sr)

    # Synthetic drum track with 2 kicks and 2 snares
    dummy_drums = np.zeros((n, 2))
    # Kick at 0.2s and 1.2s (60 Hz sine bursts)
    for t_hit in [0.2, 1.2]:
        idx = int(t_hit * sr)
        k_len = int(0.15 * sr)
        dummy_drums[idx : idx + k_len, 0] += 0.8 * np.sin(2.0 * np.pi * 60.0 * t[:k_len])
        dummy_drums[idx : idx + k_len, 1] += 0.8 * np.sin(2.0 * np.pi * 60.0 * t[:k_len])

    # Snare at 0.7s and 1.7s (220 Hz burst)
    for t_hit in [0.7, 1.7]:
        idx = int(t_hit * sr)
        s_len = int(0.15 * sr)
        dummy_drums[idx : idx + s_len, 0] += 0.6 * np.sin(2.0 * np.pi * 220.0 * t[:s_len])
        dummy_drums[idx : idx + s_len, 1] += 0.6 * np.sin(2.0 * np.pi * 220.0 * t[:s_len])

    reinforced = recreate_drums_studio(dummy_drums, sr=sr)
    assert reinforced.shape == (n, 2)
    assert not np.any(np.isnan(reinforced))
    assert np.max(np.abs(reinforced)) > 0.1


def test_recreate_bass_studio():
    sr = 48000
    n = int(2.0 * sr)
    t = np.arange(n) / float(sr)

    # Synthetic bass track: 65 Hz (C2) with higher-frequency bleed at 2 kHz
    clean_bass = 0.5 * np.sin(2.0 * np.pi * 65.0 * t)
    bleed = 0.3 * np.sin(2.0 * np.pi * 2000.0 * t)  # simulated vocal/snare bleed
    dummy_bass = np.column_stack([clean_bass + bleed, clean_bass + bleed])

    recreated = recreate_bass_studio(dummy_bass, sr=sr)
    assert recreated.shape == (n, 2)
    assert not np.any(np.isnan(recreated))

    # Center-locked (L == R)
    assert np.allclose(recreated[:, 0], recreated[:, 1], atol=1e-4)

    # Bleed at 2 kHz should be attenuated by >20 dB
    fft_in = np.abs(np.fft.rfft(dummy_bass[:, 0]))
    fft_out = np.abs(np.fft.rfft(recreated[:, 0]))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    idx_bleed = np.argmin(np.abs(freqs - 2000.0))

    ratio_db = 20.0 * np.log10((fft_out[idx_bleed] + 1e-12) / (fft_in[idx_bleed] + 1e-12))
    assert ratio_db < -15.0


def test_recreate_guitars_studio():
    sr = 48000
    n = int(2.0 * sr)
    t = np.arange(n) / float(sr)

    dummy_guitars = np.column_stack([
        0.4 * np.sin(2.0 * np.pi * 440.0 * t) + 0.1 * np.random.randn(n),
        0.4 * np.sin(2.0 * np.pi * 440.0 * t) + 0.1 * np.random.randn(n),
    ])

    recreated = recreate_guitars_studio(dummy_guitars, sr=sr)
    assert recreated.shape == (n, 2)
    assert not np.any(np.isnan(recreated))

    # Check stereo width (side channel energy should exist)
    side = 0.5 * (recreated[:, 0] - recreated[:, 1])
    assert np.mean(side**2) > 1e-6


def test_recreate_vocals_studio():
    sr = 48000
    n = int(2.0 * sr)
    t = np.arange(n) / float(sr)

    # Vocal with low rumble at 30 Hz and sibilance at 7 kHz
    clean_voc = 0.4 * np.sin(2.0 * np.pi * 350.0 * t)
    rumble = 0.3 * np.sin(2.0 * np.pi * 30.0 * t)
    dummy_voc = np.column_stack([clean_voc + rumble, clean_voc + rumble])

    recreated = recreate_vocals_studio(dummy_voc, sr=sr)
    assert recreated.shape == (n, 2)
    assert not np.any(np.isnan(recreated))

    # Rumble at 30 Hz should be significantly attenuated
    fft_out = np.abs(np.fft.rfft(recreated[:, 0]))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    idx_rumble = np.argmin(np.abs(freqs - 30.0))
    idx_clean = np.argmin(np.abs(freqs - 350.0))
    assert fft_out[idx_clean] > 4.0 * fft_out[idx_rumble]


def test_mix_and_master_studio_recreation():
    sr = 48000
    n = int(2.0 * sr)
    t = np.arange(n) / float(sr)

    v = np.column_stack([0.3 * np.sin(2.0 * np.pi * 440.0 * t), 0.3 * np.sin(2.0 * np.pi * 440.0 * t)])
    o = np.column_stack([0.2 * np.sin(2.0 * np.pi * 330.0 * t), 0.2 * np.sin(2.0 * np.pi * 330.0 * t)])
    b = np.column_stack([0.3 * np.sin(2.0 * np.pi * 65.0 * t), 0.3 * np.sin(2.0 * np.pi * 65.0 * t)])
    d = np.column_stack([0.2 * np.sin(2.0 * np.pi * 100.0 * t), 0.2 * np.sin(2.0 * np.pi * 100.0 * t)])

    mastered, solo_stems = mix_and_master_studio_recreation(
        vocals=v, other=o, bass=b, drums=d, sr=sr, target_lufs=-14.0, ceiling_dbtp=-1.0
    )

    assert mastered.shape == (n, 2)
    assert "vocals" in solo_stems
    assert "other" in solo_stems
    assert "bass" in solo_stems
    assert "drums" in solo_stems

    # Peak complies with -1.0 dBTP ceiling (0.891)
    assert np.max(np.abs(mastered)) <= 0.895
