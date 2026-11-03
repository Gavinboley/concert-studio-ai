"""End-to-end tests for ConcertStudioPipeline and CLI."""

from __future__ import annotations

from pathlib import Path

from concert_studio.cli import main as cli_main
from concert_studio.pipeline import ConcertStudioPipeline, PipelineConfig
from concert_studio.synthetic import generate_synthetic_concert


def test_concert_studio_pipeline_end_to_end() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=5, seed=42)
    pipeline = ConcertStudioPipeline(
        PipelineConfig(target_sample_rate=sr, target_lufs=-14.0, true_peak_limit_db=-1.0)
    )
    mastered, stems, report = pipeline.run(
        scenario.pov_tracks,
        reference_studio_track=scenario.studio_reference,
    )

    assert mastered.samples.shape == scenario.studio_reference.samples.shape
    assert report.num_povs_fused == 5
    assert report.clipped_samples_repaired > 0
    assert report.crowd_suppression_db > 6.0
    assert report.dereverb_tail_reduction_db > 0.2
    assert report.hf_bandwidth_extension_db > 0.1
    assert report.true_peak_dbtp <= -0.95
    assert abs(report.integrated_lufs - (-14.0)) < 1.2


def test_cli_demo_and_mix_commands(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo_run"
    rc = cli_main(
        [
            "demo",
            "--out-dir",
            str(demo_dir),
            "--duration",
            "2.0",
            "--num-povs",
            "3",
        ]
    )
    assert rc == 0
    assert (demo_dir / "studio_hd_master.wav").exists()
    assert (demo_dir / "mastering_report.json").exists()
    assert (demo_dir / "studio_stems" / "vocals.wav").exists()

    # Now test `mix` command using the raw POV files written by `demo`
    mix_dir = tmp_path / "mix_run"
    pov_wavs = [str(p) for p in sorted((demo_dir / "raw_povs").glob("*.wav"))]
    rc_mix = cli_main(
        [
            "mix",
            *pov_wavs,
            "--reference",
            str(demo_dir / "ground_truth_studio_ref.wav"),
            "--out-dir",
            str(mix_dir),
        ]
    )
    assert rc_mix == 0
    assert (mix_dir / "studio_hd_master.wav").exists()
    assert (mix_dir / "mastering_report.json").exists()
