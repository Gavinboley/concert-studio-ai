"""Tests for arbitrary start/stop partial-clip timeline detection and concert anomaly correction."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.anomalies import (
    cancel_discrete_echoes,
    classify_and_equalize_speaker_zones,
    compute_tf_anomaly_reliability_masks,
    invert_agc_pumping,
)
from concert_studio.models import AudioTrack
from concert_studio.synthetic import _apply_delay_and_drift, generate_synthetic_concert
from concert_studio.timeline import assemble_concert_timeline


def test_partial_clip_timeline_autodetection_and_moving_phone() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=4.5, sr=sr, num_povs=2, seed=101)
    master = scenario.studio_reference.samples

    # Create 1 full anchor track and 2 partial clips starting & stopping at random places
    # plus a moving-phone sinusoidal delay wave on Clip 3
    clip_full = AudioTrack(name="full_anchor", samples=master.copy(), sample_rate=sr)

    s1, e1 = int(0.85 * sr), int(3.25 * sr)
    clip_partial_1 = AudioTrack(
        name="partial_verse",
        samples=master[s1:e1].copy(),
        sample_rate=sr,
    )

    s2, e2 = int(1.60 * sr), int(4.20 * sr)
    moving_slice = _apply_delay_and_drift(
        master[s2:e2].copy(),
        sr=sr,
        offset_sec=0.008,
        drift_ppm=15.0,
        motion_excursion_sec=0.0025,
    )
    clip_partial_2 = AudioTrack(
        name="partial_chorus_moving",
        samples=moving_slice,
        sample_rate=sr,
    )

    placed, ref_idx, cov = assemble_concert_timeline(
        [clip_full, clip_partial_1, clip_partial_2],
        target_sr=sr,
        num_subwindows=14,
        enable_nonrigid_motion=True,
    )

    assert ref_idx == 0
    assert abs(placed[1].timeline_start_sec - 0.85) < 0.035
    assert abs(placed[1].timeline_end_sec - 3.25) < 0.035
    assert abs(placed[2].timeline_start_sec - 1.60) < 0.045
    assert placed[2].max_motion_drift_ms > 0.5
    assert int(np.max(cov)) == 3


def test_discrete_delay_tower_echo_canceller() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=2, seed=33)
    dry = scenario.studio_reference.samples.copy()

    # Inject a discrete 48 ms delay-tower echo with amplitude 0.34
    true_delay_ms = 48.0
    d_samples = int(round(true_delay_ms * 1e-3 * sr))
    echoed = dry.copy()
    echoed[d_samples:] += 0.34 * dry[:-d_samples]

    active_mask = np.ones(len(dry), dtype=np.float64)
    cleaned, det_delay_ms, atten_db, events = cancel_discrete_echoes(
        echoed, active_mask, sr=sr
    )

    assert abs(det_delay_ms - true_delay_ms) < 1.0
    assert atten_db > 1.5
    assert len(events) == 1 and events[0].kind == "delay_echo"
    # Verify cleaned waveform is much closer to dry original than echoed waveform
    err_before = float(np.mean((echoed - dry) ** 2))
    err_after = float(np.mean((cleaned - dry) ** 2))
    assert err_after < err_before * 0.35


def test_mic_bump_and_finger_occlusion_tf_masking() -> None:
    sr = 48000
    n_fft = 2048
    hop = n_fft // 4
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=3, seed=71)
    clean_mono = scenario.studio_reference.to_mono()
    n = len(clean_mono)

    pov1 = clean_mono.copy()
    pov2 = clean_mono.copy()
    pov3 = clean_mono.copy()

    # Inject a heavy mechanical mic bump (42 Hz thump) on POV 2 at t = 0.8s..1.05s
    b0, b1 = int(0.80 * sr), int(1.05 * sr)
    tb = np.arange(b1 - b0, dtype=np.float64) / sr
    pov2[b0:b1] += 0.90 * np.sin(2.0 * np.pi * 42.0 * tb) * np.hanning(b1 - b0)

    # Inject finger occlusion (low-pass at 550 Hz) on POV 3 at t = 1.6s..2.3s
    o0, o1 = int(1.60 * sr), int(2.30 * sr)
    sos_occ = signal.butter(4, 550.0, btype="lowpass", fs=sr, output="sos")
    pov3[o0:o1] = signal.sosfilt(sos_occ, pov3[o0:o1])

    stfts = []
    freqs = None
    frame_times = None
    for p in (pov1, pov2, pov3):
        f_arr, t_arr, Z = signal.stft(p, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        freqs = f_arr
        frame_times = t_arr
        stfts.append(np.abs(Z))
    mags = np.stack(stfts, axis=0)
    act_frames = np.ones((3, len(frame_times)), dtype=np.float64)

    masks, events = compute_tf_anomaly_reliability_masks(
        stft_mags=mags,
        active_frame_masks=act_frames,
        freqs=freqs,
        frame_times_sec=frame_times,
    )

    assert any(ev.kind == "mic_bump" for ev in events[1])
    assert any(ev.kind == "mic_occluded" for ev in events[2])

    # Verify POV 2's sub-bass reliability weight drops sharply during the bump
    bump_f_mask = (frame_times >= 0.85) & (frame_times <= 1.00)
    assert float(np.min(masks[1, freqs < 100.0][:, bump_f_mask])) < 0.25

    # Verify POV 3's high-frequency reliability weight drops sharply during finger occlusion
    occ_f_mask = (frame_times >= 1.70) & (frame_times <= 2.20)
    assert float(np.mean(masks[2, freqs > 2500.0][:, occ_f_mask])) < 0.25


def test_agc_pumping_inversion_and_speaker_zone_classification() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=2, seed=82)
    ref = scenario.studio_reference.samples.copy()
    n = len(ref)
    t = np.arange(n, dtype=np.float64) / sr

    # Simulate smartphone AGC gain ducking by -8 dB in the middle of the track
    agc_curve = 1.0 - 0.60 * np.exp(-0.5 * ((t - 1.25) / 0.35) ** 2)
    pumped = ref * agc_curve[:, None]
    active_mask = np.ones(n, dtype=np.float64)
    ref_env = np.sqrt(
        signal.fftconvolve(np.mean(ref**2, axis=1), np.ones(int(0.35 * sr)) / int(0.35 * sr), mode="same")
        + 1e-12
    )

    restored, events = invert_agc_pumping(pumped, ref_env, active_mask, sr=sr)
    assert len(events) == 1 and events[0].kind == "agc_pumping"
    err_pumped = float(np.mean((pumped - ref) ** 2))
    err_restored = float(np.mean((restored - ref) ** 2))
    assert err_restored < err_pumped * 0.45

    # Test speaker zone classification (sub_proximity vs off_axis_distant vs foh_balanced)
    sos_sub = signal.butter(2, 95.0, btype="lowpass", fs=sr, output="sos")
    sub_heavy = ref + 3.0 * signal.sosfilt(sos_sub, ref, axis=0)
    sos_dist = signal.butter(4, 2800.0, btype="lowpass", fs=sr, output="sos")
    distant = signal.sosfilt(sos_dist, ref, axis=0)

    eq_stack, zones = classify_and_equalize_speaker_zones(
        [ref, sub_heavy, distant],
        [active_mask, active_mask, active_mask],
        sr=sr,
    )
    assert zones[0] == "foh_balanced"
    assert zones[1] == "sub_proximity"
    assert zones[2] == "off_axis_distant"
    assert len(eq_stack) == 3
