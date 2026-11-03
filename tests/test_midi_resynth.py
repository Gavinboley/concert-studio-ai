"""Unit tests for MIDI transcription and studio instrument reconstruction."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from concert_studio.midi_resynth import (
    MidiNoteEvent,
    export_standard_midi,
    transcribe_drums_and_resynthesize,
    transcribe_bass_and_resynthesize,
    reconstruct_studio_multitrack,
)
from concert_studio.models import StemBundle


def test_export_standard_midi():
    events = [
        MidiNoteEvent(instrument="kick", pitch=36, start_sec=0.2, duration_sec=0.15, velocity=110),
        MidiNoteEvent(instrument="snare", pitch=38, start_sec=0.7, duration_sec=0.18, velocity=105),
        MidiNoteEvent(instrument="bass", pitch=36, start_sec=0.2, duration_sec=0.45, velocity=95),
    ]
    with tempfile.TemporaryDirectory() as td:
        midi_path = Path(td) / "test.mid"
        out_p = export_standard_midi(events, midi_path, bpm=120.0)
        assert out_p.exists()
        raw = out_p.read_bytes()
        assert raw.startswith(b"MThd")
        assert b"MTrk" in raw
        assert len(raw) > 30


def test_transcribe_drums_and_resynthesize():
    sr = 48000
    n = sr * 2
    t = np.arange(n) / sr
    drums = np.zeros((n, 2), dtype=np.float64)

    # Synthetic kick at t=0.25s (60 Hz decay)
    k_env = np.exp(-np.maximum(t - 0.25, 0.0) / 0.06) * (t >= 0.25)
    drums[:, 0] += 0.8 * np.sin(2 * np.pi * 60.0 * t) * k_env
    drums[:, 1] += 0.8 * np.sin(2 * np.pi * 60.0 * t) * k_env

    # Synthetic snare at t=0.75s (220 Hz + noise)
    s_env = np.exp(-np.maximum(t - 0.75, 0.0) / 0.08) * (t >= 0.75)
    drums[:, 0] += 0.6 * np.sin(2 * np.pi * 220.0 * t) * s_env
    drums[:, 1] += 0.6 * np.sin(2 * np.pi * 220.0 * t) * s_env

    events, synth_drums = transcribe_drums_and_resynthesize(drums, sr=sr)
    assert len(events) >= 1
    assert synth_drums.shape == (n, 2)
    assert np.max(np.abs(synth_drums)) > 0.01


def test_transcribe_bass_and_resynthesize():
    sr = 48000
    n = sr * 2
    t = np.arange(n) / sr
    bass = np.zeros((n, 2), dtype=np.float64)

    # 82.4 Hz (E2) bass note from 0.3s to 1.2s
    b_env = np.exp(-np.maximum(t - 0.3, 0.0) / 0.5) * (t >= 0.3) * (t <= 1.2)
    bass[:, 0] = 0.7 * np.sin(2 * np.pi * 82.4 * t) * b_env
    bass[:, 1] = 0.7 * np.sin(2 * np.pi * 82.4 * t) * b_env

    events, synth_bass = transcribe_bass_and_resynthesize(bass, sr=sr)
    assert len(events) >= 1
    assert synth_bass.shape == (n, 2)
    assert np.max(np.abs(synth_bass)) > 0.01


def test_reconstruct_studio_multitrack_modes():
    sr = 48000
    n = sr * 2
    t = np.arange(n) / sr

    stems = StemBundle(
        vocals=np.sin(2 * np.pi * 440.0 * t)[:, None] * np.ones((1, 2)),
        drums=np.sin(2 * np.pi * 60.0 * t)[:, None] * np.ones((1, 2)),
        bass=np.sin(2 * np.pi * 82.0 * t)[:, None] * np.ones((1, 2)),
        other=np.sin(2 * np.pi * 1000.0 * t)[:, None] * np.ones((1, 2)),
        crowd_removed=np.random.normal(0, 0.01, (n, 2)),
        sample_rate=sr,
    )

    # True audio mode: exact pass-through
    true_stems, true_events = reconstruct_studio_multitrack(stems, quality_mode="true_audio")
    assert len(true_events) == 0
    np.testing.assert_allclose(true_stems.drums, stems.drums)

    # Studio reconstructed mode: blends studio definition
    recon_stems, recon_events = reconstruct_studio_multitrack(stems, quality_mode="studio_reconstructed")
    assert recon_stems.drums.shape == stems.drums.shape
    assert recon_stems.bass.shape == stems.bass.shape
