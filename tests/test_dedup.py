"""Tests for Duplicate & Re-Upload File Detection and Best-Version Selection."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.dedup import detect_and_deduplicate_tracks, score_track_fidelity
from concert_studio.models import AudioTrack
from concert_studio.pipeline import ConcertStudioPipeline, PipelineConfig
from concert_studio.synthetic import generate_synthetic_concert


def test_duplicate_and_reupload_detection_picks_best_version() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=3, seed=501)
    pov1_orig, pov2_orig, pov3_orig = scenario.pov_tracks

    # 1. Create a lossy transcoded re-upload of POV 1 (brickwalled at 9.5 kHz + mono-collapsed)
    sos_lp = signal.butter(4, 9500.0, btype="lowpass", fs=sr, output="sos")
    pov1_lp_mono = signal.sosfiltfilt(sos_lp, pov1_orig.to_mono())
    pov1_transcode = AudioTrack(
        name="pov_01_tiktok_transcode_reupload",
        samples=np.column_stack([pov1_lp_mono, pov1_lp_mono]),
        sample_rate=sr,
    )

    # 2. Create a heavily brickwall-clipped re-upload of POV 1
    pov1_clipped_samples = np.clip(pov1_orig.to_stereo() * 2.4, -0.98, 0.98)
    pov1_clipped = AudioTrack(
        name="pov_01_boosted_clipped_reupload",
        samples=pov1_clipped_samples,
        sample_rate=sr,
    )

    # 3. Create a trimmed 1.6s sub-clip re-upload of POV 2 (starting at t=0.5s)
    s0 = int(0.5 * sr)
    s1 = int(2.1 * sr)
    pov2_subclip = AudioTrack(
        name="pov_02_trimmed_highlight_reupload",
        samples=pov2_orig.to_stereo()[s0:s1] * 0.85,
        sample_rate=sr,
    )

    # Verify individual fidelity scores rank originals strictly above degraded re-uploads
    s_orig1 = score_track_fidelity(pov1_orig)
    s_trans1 = score_track_fidelity(pov1_transcode)
    s_clip1 = score_track_fidelity(pov1_clipped)
    assert s_orig1.quality_score > s_trans1.quality_score
    assert s_orig1.quality_score > s_clip1.quality_score

    # Shuffle/combine all 6 files (3 unique POVs + 3 re-uploads)
    all_uploads = [
        pov1_transcode,
        pov2_subclip,
        pov1_orig,
        pov3_orig,
        pov1_clipped,
        pov2_orig,
    ]

    res = detect_and_deduplicate_tracks(all_uploads)

    # Exactly 3 unique physical POVs should survive, and 3 re-uploads should be removed
    assert len(res.unique_tracks) == 3
    assert res.duplicates_removed_count == 3
    assert len(res.duplicate_groups) == 2

    winning_names = {t.name for t in res.unique_tracks}
    assert winning_names == {pov1_orig.name, pov2_orig.name, pov3_orig.name}

    # Verify classifications in the duplicate groups
    group_by_winner = {g.best_track_name: g for g in res.duplicate_groups}
    g1 = group_by_winner[pov1_orig.name]
    assert set(g1.discarded_tracks) == {pov1_transcode.name, pov1_clipped.name}
    assert g1.duplicate_classifications[pov1_clipped.name] == "clipped_reupload"

    g2 = group_by_winner[pov2_orig.name]
    assert g2.discarded_tracks == [pov2_subclip.name]
    assert g2.duplicate_classifications[pov2_subclip.name] == "subclip_reupload"


def test_tail_grafting_and_pipeline_deduplication_integration() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=2, seed=602)
    full_pov1 = scenario.pov_tracks[0]
    full_pov2 = scenario.pov_tracks[1]

    # Suppose the high-resolution original of POV 1 was cut off at 2.0s,
    # while a lower-bitrate 9 kHz transcode of POV 1 runs the full 3.0s:
    hi_res_short = AudioTrack(
        name="pov1_hires_2sec",
        samples=full_pov1.to_stereo()[: int(2.0 * sr)],
        sample_rate=sr,
    )
    sos_lp = signal.butter(4, 8500.0, btype="lowpass", fs=sr, output="sos")
    lo_res_long_samples = np.clip(
        signal.sosfiltfilt(sos_lp, full_pov1.to_stereo(), axis=0) * 2.1, -0.98, 0.98
    )
    lo_res_long = AudioTrack(
        name="pov1_lores_clipped_3sec",
        samples=lo_res_long_samples,
        sample_rate=sr,
    )

    res = detect_and_deduplicate_tracks([hi_res_short, lo_res_long], graft_non_overlapping_tails=True)
    assert len(res.unique_tracks) == 1
    assert res.unique_tracks[0].name == "pov1_hires_2sec"
    # Verify the 1.0s non-overlapping tail was grafted onto the 2.0s high-res winner!
    assert res.unique_tracks[0].duration_sec > 2.85
    assert res.duplicate_groups[0].tail_grafted_sec > 0.85

    # Verify end-to-end pipeline integration with duplicate uploads
    pipeline = ConcertStudioPipeline(PipelineConfig(quality_tier="draft", enable_deduplication=True))
    _, _, report = pipeline.run([full_pov1, lo_res_long, full_pov2])
    assert report.num_povs_fused == 2
    assert report.duplicates_removed == 1
    assert len(report.duplicate_groups) == 1
    assert report.duplicate_groups[0]["best_track_name"] == full_pov1.name
