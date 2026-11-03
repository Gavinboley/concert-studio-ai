"""Tests for Crowdsource Ingestion, Deduplication Ledger & Continuous Versioning."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from concert_studio.crowdsource import CrowdsourceLibrary
from concert_studio.models import AudioTrack


def test_crowdsource_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_library.db"
    lib = CrowdsourceLibrary(db_path=db_file)
    concert_id = "oasis_wembley_2025"

    lib.register_concert(
        concert_id=concert_id,
        artist="Oasis",
        venue="Wembley Stadium",
        date="2025-07-26",
        initial_version="v1.0",
    )

    sr = 48000
    t = np.linspace(0, 5.0, 5 * sr, endpoint=False)
    sig_clean = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    track1 = AudioTrack(name="fan_standing_pit.mp4", sample_rate=sr, samples=sig_clean)

    # 1. First submission should be ACCEPTED_NOVEL and bump version to v1.1
    res1 = lib.submit_pov(concert_id, track1)
    assert res1.action == "ACCEPTED_NOVEL"
    assert res1.is_helpful is True
    assert res1.triggers_remaster is True
    assert res1.new_master_version == "v1.1"

    # 2. Exact duplicate submission should be REJECTED_EXACT_DUPLICATE
    res2 = lib.submit_pov(concert_id, track1)
    assert res2.action == "REJECTED_EXACT_DUPLICATE"
    assert res2.is_helpful is False
    assert res2.triggers_remaster is False

    # 3. Acoustically redundant submission (same notes, lower quality)
    sig_inferior = sig_clean + 0.15 * np.random.randn(len(t))
    track_inf = AudioTrack(name="fan_reupload_lowbitrate.mp4", sample_rate=sr, samples=sig_inferior)
    res3 = lib.submit_pov(concert_id, track_inf)
    assert res3.action in ("REJECTED_REDUNDANT", "REJECTED_EXACT_DUPLICATE")
    assert res3.is_helpful is False

    # 4. Novel song / different audio
    sig_diff = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t)
    track2 = AudioTrack(name="fan_bleachers_angle.mp4", sample_rate=sr, samples=sig_diff)
    res4 = lib.submit_pov(concert_id, track2)
    assert res4.action == "ACCEPTED_NOVEL"
    assert res4.triggers_remaster is True
    assert res4.new_master_version == "v1.2"

    accepted = lib.get_accepted_submissions(concert_id)
    assert len(accepted) == 2
