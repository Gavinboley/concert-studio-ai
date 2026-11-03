"""Command-Line Interface for ConcertStudio AI v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from concert_studio.auto_tuner import AutoSettingOptimizer
from concert_studio.estimator import QUALITY_TIERS, estimate_processing_plan
from concert_studio.io import load_media_track, write_wav_hd
from concert_studio.ozone import is_ozone_available, master_file_with_ozone
from concert_studio.pipeline import ConcertStudioPipeline, PipelineConfig
from concert_studio.synthetic import generate_synthetic_concert
from concert_studio.web import create_http_server


def _cmd_mix(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".mov", ".ogg"}
    resolved_inputs: list[Path] = []
    for raw_in in args.inputs:
        p = Path(raw_in)
        if p.is_dir():
            resolved_inputs.extend(sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts))
        else:
            resolved_inputs.append(p)

    pov_tracks = [load_media_track(p, target_sr=args.sample_rate) for p in resolved_inputs]
    ref_track = load_media_track(args.reference, target_sr=args.sample_rate) if args.reference else None

    est = estimate_processing_plan(
        num_files=len(pov_tracks),
        durations_sec=[t.duration_sec for t in pov_tracks],
        sample_rate=args.sample_rate,
        quality_tier=args.quality,
    )
    print(
        f"[Estimator] Quality Tier: {est.quality_tier} | Files: {est.num_files} | "
        f"Est. Time: ~{est.estimated_processing_sec:.2f}s | Expected SNR Gain: +{est.expected_snr_gain_db:.2f} dB"
    )

    cfg = PipelineConfig(
        target_sample_rate=args.sample_rate,
        quality_tier=args.quality,
        true_data_mode=not args.generative_stems,
        target_lufs=args.target_lufs,
        true_peak_limit_db=args.true_peak_db,
        separation_backend=args.backend,
    )

    if getattr(args, "auto_tune", True):
        auto_res = AutoSettingOptimizer.analyze_and_optimize(
            pov_tracks, sample_rate=args.sample_rate, base_config=cfg
        )
        print("=== Stage 0: Dynamic Acoustic Auto-Tuning ===")
        for f in auto_res.diagnostic_findings:
            print(f"  * {f}")
        print(
            f"Suggested Curve: {auto_res.suggested_mastering_curve} | "
            f"Expected Improvement: +{auto_res.expected_quality_improvement_db:.1f} dB"
        )
        cfg = auto_res.optimized_config
    def _cli_progress(pct: float, stage: str, detail: str) -> None:
        filled = int(round((pct / 100.0) * 24))
        bar = "#" * filled + "-" * (24 - filled)
        print(f"[{bar}] {pct:5.1f}% | {stage} — {detail}", flush=True)

    pipeline = ConcertStudioPipeline(cfg)
    mastered, stems, report = pipeline.run(
        pov_tracks, reference_studio_track=ref_track, progress_callback=_cli_progress
    )

    master_path = write_wav_hd(out_dir / "studio_hd_master.wav", mastered, bit_depth=24)
    stems_dir = out_dir / "stems"
    for stem_name, stem_arr in stems.as_dict().items():
        write_wav_hd(stems_dir / f"{stem_name}.wav", stem_arr, sample_rate=stems.sample_rate, bit_depth=24)

    ozone_meta = None
    if not getattr(args, "no_ozone", False) and is_ozone_available():
        ozone_path = out_dir / "studio_hd_ozone_master.wav"
        print("[Ozone] Mastering wavefront through iZotope Ozone 12 Advanced VST3 (Stabilizer + Low End Focus + Clarity + Maximizer)...")
        try:
            ozone_meta = master_file_with_ozone(
                master_path,
                ozone_path,
                target_lufs=args.target_lufs,
                ceiling_dbtp=args.true_peak_db,
            )
            print(f"[Ozone] Mastered HD track written to: {ozone_path}")
        except Exception as e:
            print(f"[Ozone] Warning: Ozone 12 mastering step failed: {e}")

    report_dict = report.to_dict()
    if ozone_meta:
        report_dict["ozone_master"] = ozone_meta
    report_path = out_dir / "mastering_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    print(f"Mastered HD track written to: {master_path}")
    print(f"Report written to:            {report_path}")
    print(
        f"LUFS: {report.integrated_lufs:.2f} LUFS | True Peak: {report.true_peak_dbtp:.2f} dBTP | "
        f"Crowd Suppression: {report.crowd_suppression_db:.2f} dB | "
        f"Duplicates Filtered: {report.duplicates_removed} | Anomalies Mitigated: {report.total_anomalies_mitigated}"
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = generate_synthetic_concert(
        duration_sec=args.duration,
        sr=args.sample_rate,
        num_povs=args.num_povs,
        include_partial_clips=args.partial_clips,
        include_physical_anomalies=args.anomalies,
    )

    pov_dir = out_dir / "raw_povs"
    for trk in scenario.pov_tracks:
        write_wav_hd(pov_dir / f"{trk.name}.wav", trk, bit_depth=16)
    write_wav_hd(out_dir / "ground_truth_studio_ref.wav", scenario.studio_reference, bit_depth=24)

    pipeline = ConcertStudioPipeline(
        PipelineConfig(
            target_sample_rate=args.sample_rate,
            quality_tier=args.quality,
            true_data_mode=True,
            target_lufs=-14.0,
        )
    )
    mastered, stems, report = pipeline.run(
        scenario.pov_tracks,
        reference_studio_track=scenario.studio_reference,
    )

    master_path = write_wav_hd(out_dir / "studio_hd_master.wav", mastered, bit_depth=24)
    stems_dir = out_dir / "studio_stems"
    for stem_name, stem_arr in stems.as_dict().items():
        write_wav_hd(stems_dir / f"{stem_name}.wav", stem_arr, sample_rate=stems.sample_rate, bit_depth=24)

    report_path = out_dir / "mastering_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print("=== ConcertStudio AI v2 End-to-End Demo Complete ===")
    print(f"Fused POVs:               {report.num_povs_fused} (Anchor: {report.reference_pov})")
    print(f"Quality Tier / True-Data: {report.quality_tier} (true_data_mode={report.true_data_mode})")
    print(f"Clipped Samples Repaired: {report.clipped_samples_repaired}")
    print(f"Anomalies Mitigated:      {report.total_anomalies_mitigated} (Bumps={report.bumps_masked}, Occlusions={report.occlusions_masked}, Echoes={report.echoes_cancelled})")
    print(f"Crowd Suppression:        {report.crowd_suppression_db:.2f} dB")
    print(f"Dereverb Tail Reduction:  {report.dereverb_tail_reduction_db:.2f} dB")
    print(f"Integrated Loudness:      {report.integrated_lufs:.2f} LUFS (Target: -14.00 LUFS)")
    print(f"True Peak Ceiling:        {report.true_peak_dbtp:.2f} dBTP (Limit: -1.00 dBTP)")
    print(f"HD 24-bit Master:         {master_path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    server, _ = create_http_server(host=args.host, port=args.port, workspace_dir=args.workspace_dir)
    print(f"ConcertStudio AI v2 Web UI listening at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concert-studio",
        description="ConcertStudio AI v2: Multi-POV True-Data Live Capture Reconstruction Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_mix = sub.add_parser("mix", help="Fuse and master multiple POV audio/video clips into an HD live master")
    p_mix.add_argument("inputs", nargs="+", help="Input POV WAV/MP4/MOV/MP3/FLAC files")
    p_mix.add_argument("--reference", help="Optional reference WAV track for spectral matchering")
    p_mix.add_argument("--out-dir", default="output_studio", help="Directory for mastered HD outputs")
    p_mix.add_argument("--quality", choices=list(QUALITY_TIERS.keys()), default="archival_max")
    p_mix.add_argument("--auto-tune", action="store_true", default=True, help="Auto-probe acoustic payload and auto-select optimal DSP parameters (default: True)")
    p_mix.add_argument("--no-auto-tune", dest="auto_tune", action="store_false", help="Disable acoustic auto-tuning")
    p_mix.add_argument("--generative-stems", action="store_true", help="Opt into Stage 5 harmonic synthesis (off by default)")
    p_mix.add_argument("--sample-rate", type=int, default=48000, help="Target sample rate (default: 48000)")
    p_mix.add_argument("--target-lufs", type=float, default=-14.0, help="Target integrated LUFS")
    p_mix.add_argument("--true-peak-db", type=float, default=-1.0, help="True-peak ceiling in dBTP")
    p_mix.add_argument("--backend", choices=["auto", "neural", "dsp"], default="auto")
    p_mix.add_argument("--no-ozone", action="store_true", help="Disable iZotope Ozone 12 mastering step")
    p_mix.set_defaults(func=_cmd_mix)

    p_demo = sub.add_parser("demo", help="Synthesize a multi-POV concert and run the v2 HD live reconstruction")
    p_demo.add_argument("--out-dir", default="demo_output", help="Output folder")
    p_demo.add_argument("--duration", type=float, default=4.0, help="Duration in seconds")
    p_demo.add_argument("--sample-rate", type=int, default=48000, help="Sample rate in Hz")
    p_demo.add_argument("--num-povs", type=int, default=5, help="Number of fan POVs to simulate")
    p_demo.add_argument("--quality", choices=list(QUALITY_TIERS.keys()), default="archival_max")
    p_demo.add_argument("--partial-clips", action="store_true", help="Simulate random start/stop partial clips")
    p_demo.add_argument("--anomalies", action="store_true", help="Simulate bumps, occlusion, moving phone & delay echo")
    p_demo.set_defaults(func=_cmd_demo)

    p_serve = sub.add_parser("serve", help="Launch the interactive ConcertStudio AI v2 Web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--workspace-dir", default=None)
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
