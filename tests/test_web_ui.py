"""Tests for the Interactive Web Studio UI and HTTP REST API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

from concert_studio.io import write_wav_hd
from concert_studio.synthetic import generate_synthetic_concert
from concert_studio.web import create_http_server


def test_web_ui_and_rest_api_endpoints(tmp_path: Path) -> None:
    server, app = create_http_server(host="127.0.0.1", port=0, workspace_dir=tmp_path)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{port}"

        # 1. GET / serves the Studio Web UI
        with urllib.request.urlopen(f"{base_url}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "CONCERTSTUDIO AI v2" in html
            assert "Quality Scaling &amp; Processing Time Estimator" in html

        # 2. POST /api/estimate returns dynamic processing time & resolution estimate
        req_est = urllib.request.Request(
            f"{base_url}/api/estimate",
            data=json.dumps({"num_files": 12, "avg_duration_sec": 90.0, "quality_tier": "ultra"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_est) as resp:
            assert resp.status == 200
            est_data = json.loads(resp.read().decode("utf-8"))
            assert est_data["quality_tier"] == "ultra"
            assert est_data["num_files"] == 12
            assert est_data["estimated_processing_sec"] > 0.5
            assert est_data["expected_snr_gain_db"] > 8.0

        # 3. POST /api/demo runs the multi-POV partial-clip + anomaly showcase
        req_demo = urllib.request.Request(
            f"{base_url}/api/demo",
            data=json.dumps({"quality_tier": "draft", "num_povs": 3, "duration_sec": 2.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_demo) as resp:
            assert resp.status == 200
            demo_data = json.loads(resp.read().decode("utf-8"))
            assert demo_data["report"]["num_povs_fused"] == 3
            assert len(demo_data["report"]["clip_placements"]) == 3
            master_url = demo_data["master_audio_url"]

        # 4. GET /api/audio/<job_id>/studio_hd_master.wav streams the 24-bit WAV
        with urllib.request.urlopen(f"{base_url}{master_url}") as resp:
            assert resp.status == 200
            wav_bytes = resp.read()
            assert wav_bytes[:4] == b"RIFF"

        # 5. Test multipart upload via app.run_reconstruct_job
        scenario = generate_synthetic_concert(duration_sec=2.0, sr=48000, num_povs=2, seed=17)
        uploaded: list[tuple[str, bytes]] = []
        for idx, trk in enumerate(scenario.pov_tracks):
            p = write_wav_hd(tmp_path / f"upload_{idx}.wav", trk, bit_depth=16)
            uploaded.append((p.name, p.read_bytes()))

        job_res = app.run_reconstruct_job(
            uploaded,
            {"quality_tier": "standard", "true_data_mode": "true", "target_lufs": "-14.0"},
        )
        assert job_res["report"]["num_povs_fused"] == 2
        assert job_res["report"]["true_data_mode"] is True

        # 6. Verify /api/latest_job returns the most recent completed job bundle
        with urllib.request.urlopen(f"{base_url}/api/latest_job") as resp:
            assert resp.status == 200
            latest_data = json.loads(resp.read().decode("utf-8"))
            assert latest_data["job_id"] == job_res["job_id"]
            assert latest_data["report"]["num_povs_fused"] == 2

        # 7. Verify /api/reconstruct_folder processes a local directory of clips
        folder_dir = tmp_path / "local_clips"
        folder_dir.mkdir(parents=True, exist_ok=True)
        for idx, trk in enumerate(scenario.pov_tracks):
            write_wav_hd(folder_dir / f"clip_{idx}.wav", trk, bit_depth=16)
        req_folder = urllib.request.Request(
            f"{base_url}/api/reconstruct_folder",
            data=json.dumps({"folder_path": str(folder_dir), "quality_tier": "draft"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_folder) as resp:
            assert resp.status == 200
            folder_data = json.loads(resp.read().decode("utf-8"))
            assert folder_data["report"]["num_povs_fused"] == 2

        # 8. Verify path traversal protection on get_audio_file
        assert app.get_audio_file("..", "upload_0.wav") is None
        assert app.get_audio_file("job_test", "..") is None
        assert app.get_audio_file("../..", "etc/passwd") is None

        # 9. Verify new API endpoints: auto_tune, crowdsource, setlist, and director
        req_at = urllib.request.Request(
            f"{base_url}/api/auto_tune",
            data=json.dumps({"num_povs": 2, "duration_sec": 3.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_at) as resp:
            assert resp.status == 200
            at_data = json.loads(resp.read().decode("utf-8"))
            assert "acoustic_profile" in at_data
            assert at_data["optimized_config"]["quality_tier"] == "archival_max"

        req_cs = urllib.request.Request(
            f"{base_url}/api/crowdsource/submit",
            data=json.dumps({"concert_id": "test_gig", "filename": "fan1.mp4", "duration_sec": 4.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_cs) as resp:
            assert resp.status == 200
            cs_data = json.loads(resp.read().decode("utf-8"))
            assert cs_data["action"] == "ACCEPTED_NOVEL"

        with urllib.request.urlopen(f"{base_url}/api/crowdsource/library?concert_id=test_gig") as resp:
            assert resp.status == 200
            lib_data = json.loads(resp.read().decode("utf-8"))
            assert len(lib_data["accepted_povs"]) == 1

        req_sl = urllib.request.Request(
            f"{base_url}/api/setlist",
            data=json.dumps({"artist": "Oasis", "venue": "Wembley", "timeline_duration_sec": 120.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_sl) as resp:
            assert resp.status == 200
            sl_data = json.loads(resp.read().decode("utf-8"))
            assert len(sl_data["songs"]) >= 3
            assert "cue_sheet" in sl_data

        req_dir = urllib.request.Request(
            f"{base_url}/api/director/plan",
            data=json.dumps({"total_duration_sec": 60.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_dir) as resp:
            assert resp.status == 200
            dir_data = json.loads(resp.read().decode("utf-8"))
            assert dir_data["num_shots"] >= 4
            assert "cmx3600_edl" in dir_data
    finally:
        server.shutdown()
        server.server_close()
