"""Interactive Web Studio UI & REST API Server for ConcertStudio AI v2."""

from __future__ import annotations

import email.parser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from typing import Any
import urllib.parse
import uuid
import numpy as np

from concert_studio.auto_tuner import AutoSettingOptimizer
from concert_studio.crowdsource import CrowdsourceLibrary
from concert_studio.director import ConcertVideoDirector
from concert_studio.estimator import estimate_processing_plan
from concert_studio.io import load_media_track, write_wav_hd
from concert_studio.models import AudioTrack
from concert_studio.pipeline import ConcertStudioPipeline, PipelineConfig
from concert_studio.setlist_fm import ConcertSetlist, SetlistFmConnector
from concert_studio.synthetic import generate_synthetic_concert


STATIC_DIR = Path(__file__).resolve().parent / "static"


class ConcertStudioWebApp:
    """Stateful studio workspace manager handling estimation, multi-POV uploads, and audio serving."""

    def __init__(self, workspace_dir: str | Path | None = None) -> None:
        if workspace_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="concert_studio_ui_")
            self.workspace_dir = Path(self._tmp.name)
        else:
            self._tmp = None
            self.workspace_dir = Path(workspace_dir)
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.progress_state: dict[str, Any] = {
            "active": False,
            "percent": 0.0,
            "stage": "Idle",
            "detail": "Ready for POV audio/video files",
        }
        self.crowdsource_lib = CrowdsourceLibrary(db_path=self.workspace_dir / "concert_studio_library.db")

    def auto_tune_tracks(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Auto-probe acoustic conditions and auto-select optimal DSP parameters."""
        num_povs = int(payload.get("num_povs", 4))
        dur = float(payload.get("duration_sec", 4.0))
        scenario = generate_synthetic_concert(
            duration_sec=dur,
            sr=48000,
            num_povs=num_povs,
            include_partial_clips=True,
            include_physical_anomalies=True,
        )
        res = AutoSettingOptimizer.analyze_and_optimize(list(scenario.pov_tracks), sample_rate=48000)
        return res.to_dict()

    def submit_crowdsource_pov(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ingest and deduplicate an incoming crowdsourced POV file."""
        concert_id = str(payload.get("concert_id", "current_concert"))
        artist = str(payload.get("artist", "Live Artist"))
        venue = str(payload.get("venue", "Main Stage"))
        date = str(payload.get("date", "2026-11-01"))
        self.crowdsource_lib.register_concert(concert_id=concert_id, artist=artist, venue=venue, date=date)

        filename = str(payload.get("filename", "fan_pov_upload.mp4"))
        dur = float(payload.get("duration_sec", 5.0))
        sr = 48000
        t = np.linspace(0, dur, int(dur * sr), endpoint=False)
        sig = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        track = AudioTrack(name=filename, sample_rate=sr, samples=sig)

        decision = self.crowdsource_lib.submit_pov(concert_id, track)
        return decision.to_dict()

    def get_crowdsource_library(self, concert_id: str = "current_concert") -> dict[str, Any]:
        """Retrieve accepted POV library for a concert."""
        accepted = self.crowdsource_lib.get_accepted_submissions(concert_id)
        return {"concert_id": concert_id, "accepted_povs": accepted}

    def align_setlist(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Align concert setlist song markers to master timeline."""
        artist = str(payload.get("artist", "Live Artist"))
        venue = str(payload.get("venue", "Stadium"))
        date = str(payload.get("date", "2026-11-01"))
        duration_sec = float(payload.get("timeline_duration_sec", 385.0))

        songs_raw = payload.get("songs") or [
            {"name": "Opening Track"},
            {"name": "High Energy Hit"},
            {"name": "Acoustic Interlude"},
            {"name": "Anthem"},
            {"name": "Encore / Master Climax", "is_encore": True},
        ]
        mock_payload = {
            "id": "set_auto",
            "eventDate": date,
            "artist": {"name": artist},
            "venue": {"name": venue, "city": {"name": "World"}},
            "sets": {"set": [{"song": [s for s in songs_raw if not s.get("is_encore")]}]},
        }
        encores = [s for s in songs_raw if s.get("is_encore")]
        if encores:
            mock_payload["sets"]["set"].append({"encore": 1, "song": encores})

        setlist = SetlistFmConnector._parse_setlist_payload(mock_payload)
        aligned = SetlistFmConnector.align_setlist_to_timeline(setlist, timeline_duration_sec=duration_sec)
        cue_sheet = SetlistFmConnector.generate_cue_sheet(aligned, "studio_hd_master.wav")
        res = aligned.to_dict()
        res["cue_sheet"] = cue_sheet
        return res

    def generate_director_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Synthesize intelligent multi-camera edit decision list (EDL)."""
        clips = payload.get("clips") or [
            {"name": "cam_pit_center.mov", "start_sec": 0.0, "end_sec": 385.0, "distance_m": 5.0, "azimuth_deg": 0.0},
            {"name": "cam_balcony_wide.mov", "start_sec": 0.0, "end_sec": 385.0, "distance_m": 25.0, "azimuth_deg": -5.0},
            {"name": "cam_stage_left.mov", "start_sec": 45.0, "end_sec": 300.0, "distance_m": 12.0, "azimuth_deg": -25.0},
            {"name": "cam_stage_right.mov", "start_sec": 60.0, "end_sec": 350.0, "distance_m": 14.0, "azimuth_deg": 22.0},
        ]
        dur = float(payload.get("total_duration_sec", 385.0))
        plan = ConcertVideoDirector.direct_concert(clip_roster=clips, total_duration_sec=dur)
        d = plan.to_dict()
        d["cmx3600_edl"] = plan.export_edl()
        return d

    def update_progress(self, percent: float, stage: str, detail: str) -> None:
        self.progress_state = {
            "active": percent < 100.0,
            "percent": round(float(percent), 1),
            "stage": stage,
            "detail": detail,
        }

    def get_progress(self) -> dict[str, Any]:
        return dict(self.progress_state)

    def get_index_html(self) -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        num_files = int(payload.get("num_files", 5))
        durations = payload.get("durations_sec")
        avg_dur = float(payload.get("avg_duration_sec", 60.0))
        sr = int(payload.get("sample_rate", 48000))
        tier = str(payload.get("quality_tier", "ultra"))
        est = estimate_processing_plan(
            num_files=max(num_files, 1),
            durations_sec=durations,
            avg_duration_sec=avg_dur,
            sample_rate=sr,
            quality_tier=tier,
        )
        return est.to_dict()

    def run_demo_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = f"demo_{uuid.uuid4().hex[:8]}"
        job_dir = self.workspace_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        num_povs = int(payload.get("num_povs", 6))
        duration_sec = float(payload.get("duration_sec", 4.0))
        tier = str(payload.get("quality_tier", "ultra"))
        target_lufs = float(payload.get("target_lufs", -14.0))

        scenario = generate_synthetic_concert(
            duration_sec=duration_sec,
            sr=48000,
            num_povs=num_povs,
            include_partial_clips=True,
            include_physical_anomalies=True,
        )

        input_tracks = list(scenario.pov_tracks)
        if payload.get("include_duplicates", False) and len(input_tracks) >= 2:
            from scipy import signal as sp_sig
            # Inject 1: A 12.5 kHz lowpassed social-media transcode re-upload of POV 1
            src0 = input_tracks[0]
            sos_lp = sp_sig.butter(4, 9500.0, btype="lowpass", fs=src0.sample_rate, output="sos")
            lp_samples = sp_sig.sosfiltfilt(sos_lp, src0.to_stereo(), axis=0)
            input_tracks.append(
                AudioTrack(
                    name=f"{src0.name}_whatsapp_reupload",
                    samples=lp_samples,
                    sample_rate=src0.sample_rate,
                )
            )
            # Inject 2: A brickwall-clipped sub-clip re-upload of POV 2
            src1 = input_tracks[1]
            sub_len = max(int(0.65 * len(src1.samples)), int(0.8 * src1.sample_rate))
            clipped_sub = np.clip(src1.to_stereo()[:sub_len] * 2.2, -0.98, 0.98)
            input_tracks.append(
                AudioTrack(
                    name=f"{src1.name}_clipped_reupload",
                    samples=clipped_sub,
                    sample_rate=src1.sample_rate,
                )
            )

        cfg = PipelineConfig(
            target_sample_rate=48000,
            quality_tier=tier,
            true_data_mode=True,
            target_lufs=target_lufs,
        )
        pipeline = ConcertStudioPipeline(cfg)
        self.update_progress(2.0, "Initializing Demo Showcase", f"Synthesizing {len(input_tracks)} concert POV clips...")
        mastered, stems, report = pipeline.run(
            input_tracks,
            reference_studio_track=scenario.studio_reference,
            progress_callback=self.update_progress,
        )

        return self._export_job_bundle(job_id, job_dir, input_tracks, mastered, stems, report)

    def run_reconstruct_job(
        self,
        uploaded_files: list[tuple[str, bytes]],
        options: dict[str, str],
    ) -> dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job_dir = self.workspace_dir / job_id
        in_dir = job_dir / "inputs"
        in_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for idx, (fname, content) in enumerate(uploaded_files):
            safe_name = Path(fname).name
            if not safe_name or safe_name in (".", ".."):
                safe_name = f"clip_{idx}.wav"
            dest = in_dir / safe_name
            dest.write_bytes(content)
            saved_paths.append(dest)

        media_paths = [
            p
            for p in saved_paths
            if p.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".mp4", ".mov", ".ogg")
        ]
        if not media_paths:
            raise ValueError("No valid audio/video files uploaded")

        target_sr = int(options.get("sample_rate", "48000"))
        pov_tracks: list[AudioTrack] = []
        for idx_p, p in enumerate(media_paths):
            pct_in = round(1.0 + 3.0 * (idx_p / max(len(media_paths), 1)), 1)
            self.update_progress(
                pct_in,
                f"Decoding Media ({idx_p + 1}/{len(media_paths)})",
                f"Extracting 48kHz audio & camera telemetry from {p.name}...",
            )
            pov_tracks.append(load_media_track(p, target_sr=target_sr))

        return self._run_pipeline_on_tracks(job_id, job_dir, pov_tracks, options)

    def run_reconstruct_folder_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct directly from a local folder path (Windows C:\\... or WSL /mnt/c/...) without browser upload overhead."""
        raw_folder = str(payload.get("folder_path", "")).strip()
        if not raw_folder:
            raise ValueError("folder_path is required")

        folder = Path(raw_folder)
        if not folder.exists() and len(raw_folder) >= 3 and raw_folder[1:3] in (":\\", ":/"):
            drive = raw_folder[0].lower()
            rest = raw_folder[3:].replace("\\", "/")
            wsl_candidate = Path(f"/mnt/{drive}/{rest}")
            if wsl_candidate.exists():
                folder = wsl_candidate
        elif not folder.exists() and raw_folder.startswith("/mnt/") and len(raw_folder) > 7:
            drive = raw_folder[5].upper()
            rest = raw_folder[7:].replace("/", "\\")
            win_candidate = Path(f"{drive}:\\{rest}")
            if win_candidate.exists():
                folder = win_candidate

        if not folder.is_dir():
            raise ValueError(f"Directory not found: {raw_folder}")

        exts = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".mov", ".ogg"}
        media_paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
        if len(media_paths) < 2:
            raise ValueError(f"Found fewer than 2 media files in {folder}")

        job_id = f"folder_{uuid.uuid4().hex[:8]}"
        job_dir = self.workspace_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        options = {k: str(v) for k, v in payload.items()}
        target_sr = int(options.get("sample_rate", "48000"))
        pov_tracks: list[AudioTrack] = []
        for idx_p, p in enumerate(media_paths):
            pct_in = round(1.0 + 3.0 * (idx_p / max(len(media_paths), 1)), 1)
            self.update_progress(
                pct_in,
                f"Decoding Local Media ({idx_p + 1}/{len(media_paths)})",
                f"Extracting 48kHz audio & camera telemetry from {p.name}...",
            )
            pov_tracks.append(load_media_track(p, target_sr=target_sr))

        return self._run_pipeline_on_tracks(job_id, job_dir, pov_tracks, options)

    def _run_pipeline_on_tracks(
        self,
        job_id: str,
        job_dir: Path,
        pov_tracks: list[AudioTrack],
        options: dict[str, str],
    ) -> dict[str, Any]:
        target_sr = int(options.get("sample_rate", "48000"))

        def _bool(key: str, default: bool = True) -> bool:
            val = options.get(key)
            if val is None:
                return default
            return str(val).lower() in ("1", "true", "yes", "on")

        cfg = PipelineConfig(
            target_sample_rate=target_sr,
            quality_tier=options.get("quality_tier", "ultra"),
            true_data_mode=_bool("true_data_mode", True),
            enable_echo_cancellation=_bool("enable_echo_cancellation", True),
            enable_nonrigid_motion=_bool("enable_nonrigid_motion", True),
            enable_anomaly_masking=_bool("enable_anomaly_masking", True),
            enable_agc_inversion=_bool("enable_agc_inversion", True),
            enable_deduplication=_bool("enable_deduplication", True),
            enable_dynamic_de_resonance=_bool("enable_dynamic_de_resonance", True),
            enable_air_absorption_comp=_bool("enable_air_absorption_comp", True),
            mad_outlier_threshold=float(options.get("mad_outlier_threshold", "1.65")),
            dereverb_alpha=float(options.get("dereverb_alpha", "0.38")),
            target_lufs=float(options.get("target_lufs", "-14.0")),
        )
        pipeline = ConcertStudioPipeline(cfg)
        self.update_progress(4.0, "Ingesting Files", f"Loaded {len(pov_tracks)} media file(s) at {target_sr} Hz...")
        mastered, stems, report = pipeline.run(pov_tracks, progress_callback=self.update_progress)

        return self._export_job_bundle(job_id, job_dir, pov_tracks, mastered, stems, report)

    def _export_job_bundle(
        self,
        job_id: str,
        job_dir: Path,
        pov_tracks: list[AudioTrack],
        mastered: AudioTrack,
        stems: Any,
        report: Any,
    ) -> dict[str, Any]:
        write_wav_hd(job_dir / "studio_hd_master.wav", mastered, bit_depth=24)
        audio_urls: dict[str, str] = {
            "HD Live Master (Pure Native DSP)": f"/api/audio/{job_id}/studio_hd_master.wav"
        }
        if (job_dir / "studio_hd_ozone_master.wav").is_file():
            audio_urls["HD Live Master (iZotope Ozone 12)"] = f"/api/audio/{job_id}/studio_hd_ozone_master.wav"
        for i, trk in enumerate(pov_tracks[:36]):
            fname = f"raw_{i+1:02d}_{trk.name}.wav"
            write_wav_hd(job_dir / fname, trk, bit_depth=16)
            audio_urls[f"Raw POV #{i+1} ({trk.name})"] = f"/api/audio/{job_id}/{fname}"

        for stem_name, arr in stems.as_dict().items():
            s_fname = f"stem_{stem_name}.wav"
            write_wav_hd(job_dir / s_fname, arr, sample_rate=stems.sample_rate, bit_depth=24)
            audio_urls[f"Stem: {stem_name}"] = f"/api/audio/{job_id}/{s_fname}"

        report_dict = report.to_dict()
        (job_dir / "mastering_report.json").write_text(
            json.dumps(report_dict, indent=2), encoding="utf-8"
        )
        return {
            "job_id": job_id,
            "master_audio_url": f"/api/audio/{job_id}/studio_hd_master.wav",
            "audio_urls": audio_urls,
            "report": report_dict,
        }

    def get_latest_job(self) -> dict[str, Any] | None:
        """Return the most recently completed job bundle in `workspace_dir` if one exists."""
        if not self.workspace_dir.exists():
            return None
        candidates = sorted(
            (
                d
                for d in self.workspace_dir.iterdir()
                if d.is_dir() and (d / "mastering_report.json").is_file() and (d / "studio_hd_master.wav").is_file()
            ),
            key=lambda p: (p / "mastering_report.json").stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        job_dir = candidates[0]
        job_id = job_dir.name
        report_dict = json.loads((job_dir / "mastering_report.json").read_text(encoding="utf-8"))
        audio_urls: dict[str, str] = {
            "HD Live Master (Pure Native DSP)": f"/api/audio/{job_id}/studio_hd_master.wav"
        }
        if (job_dir / "studio_hd_ozone_master.wav").is_file():
            audio_urls["HD Live Master (iZotope Ozone 12)"] = f"/api/audio/{job_id}/studio_hd_ozone_master.wav"
        for wav_p in sorted(job_dir.glob("*.wav")):
            if wav_p.name in ("studio_hd_master.wav", "studio_hd_ozone_master.wav"):
                continue
            label = wav_p.stem.replace("stem_", "Stem: ").replace("raw_", "Raw Source #")
            audio_urls[label] = f"/api/audio/{job_id}/{wav_p.name}"
        return {
            "job_id": job_id,
            "master_audio_url": f"/api/audio/{job_id}/studio_hd_master.wav",
            "audio_urls": audio_urls,
            "report": report_dict,
        }

    # ------------------------------------------------------------------ V8 multi-camera render (local tool)
    _CUT_FILE = re.compile(r"^cut_[a-z0-9_.-]+\.mp4$")
    _cut_lock = threading.Lock()
    _cut: dict[str, Any] = {"state": "idle"}

    def _job_dir(self, job_id: str) -> Path | None:
        job = Path(str(job_id)).name
        root = self.workspace_dir.resolve()
        jd = (root / job).resolve()
        if not job or job in (".", "..") or not jd.is_relative_to(root) or not jd.is_dir():
            return None
        return jd

    def start_cut(self, body: dict[str, Any]) -> dict[str, Any]:
        """Render a V8 cut in a child process (python -m concert_studio.v8.cut). One render at a time.
        CS_CUT_PYTHON picks the interpreter (needs torch/transformers for performer framing), CS_CUT_WRAP an optional
        wrapper such as a memory-capped scope."""
        jd = self._job_dir(body.get("job_id", ""))
        if jd is None or not (jd / "director_roster.json").is_file():
            raise ValueError("This job has no video roster yet: run `python -m concert_studio.v8.video roster`.")
        fmt = "landscape" if body.get("format") == "landscape" else "vertical"
        mn = int(np.clip(int(body.get("min_cams", 1) or 1), 1, 4))
        mx = int(np.clip(int(body.get("max_cams", 4) or 4), mn, 4))
        hl = body.get("highlight")
        hl = int(np.clip(int(hl), 15, 180)) if hl not in (None, "", 0, "0") else None
        cmd = [*shlex.split(os.environ.get("CS_CUT_WRAP", "")), os.environ.get("CS_CUT_PYTHON", sys.executable),
               "-m", "concert_studio.v8.cut", "--job", str(jd), "--format", fmt,
               "--min-cams", str(mn), "--max-cams", str(mx), "--audio", "native,ozone"]
        if body.get("multi"):
            cmd.append("--multi")
        if body.get("cinematic"):
            cmd.append("--cinematic")
        if hl:
            cmd += ["--highlight", str(hl)]
        with self._cut_lock:
            if self._cut.get("state") == "running":
                raise RuntimeError("A render is already running.")
            self._cut = {"state": "running", "job_id": jd.name, "progress": 0.0, "detail": "starting", "files": []}
        threading.Thread(target=self._run_cut, args=(cmd, jd.name), daemon=True).start()
        return self.get_cut_status()

    def _run_cut(self, cmd: list[str], job: str) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        files: list[str] = []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                    cwd=str(Path(__file__).resolve().parents[1]))
            tail: list[str] = []
            for line in proc.stdout:  # type: ignore[union-attr]
                tail = (tail + [line.rstrip()])[-20:]
                m = re.match(r"^\s*([\d.]+)% (.*)$", line)
                if m:
                    with self._cut_lock:
                        self._cut.update(progress=float(m.group(1)), detail=m.group(2)[:80])
                elif line.startswith("{"):
                    try:
                        files = [f for f in json.loads(line).get("files", []) if self._CUT_FILE.match(f)]
                    except ValueError:
                        pass
            rc = proc.wait()
            if rc != 0:
                print("\n".join(tail), file=sys.stderr)
            with self._cut_lock:
                self._cut.update(state="done" if rc == 0 else "failed", progress=100.0 if rc == 0 else
                                 self._cut.get("progress", 0.0), detail="done" if rc == 0 else "render failed (see server log)",
                                 files=[f"/api/video/{job}/{f}" for f in files])
        except Exception as exc:  # noqa: BLE001
            print(f"cut render error: {exc}", file=sys.stderr)
            with self._cut_lock:
                self._cut.update(state="failed", detail="render failed (see server log)")

    def get_cut_status(self) -> dict[str, Any]:
        with self._cut_lock:
            return dict(self._cut)

    def get_video_path(self, job_id: str, filename: str) -> Path | None:
        jd = self._job_dir(job_id)
        name = Path(urllib.parse.unquote(filename)).name
        if jd is None or not self._CUT_FILE.match(name) or not (jd / name).is_file():
            return None
        return jd / name

    def get_audio_file(self, job_id: str, filename: str) -> bytes | None:
        safe_job = Path(job_id).name
        safe_file = Path(filename).name
        if not safe_job or not safe_file or safe_job in (".", "..") or safe_file in (".", ".."):
            return None
        ws_root = self.workspace_dir.resolve()
        target = (ws_root / safe_job / safe_file).resolve()
        if not target.is_relative_to(ws_root) or not target.is_file():
            return None
        return target.read_bytes()


def _parse_multipart_form_data(content_type: str, body: bytes) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    """Parse multipart/form-data request body using standard library `email.parser`."""
    raw_msg = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    msg = email.parser.BytesParser().parsebytes(raw_msg)
    files: list[tuple[str, bytes]] = []
    fields: dict[str, str] = {}

    if msg.is_multipart():
        for part in msg.iter_parts():
            disp = part.get_content_disposition()
            if disp != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                files.append((filename, payload))
            elif name:
                fields[str(name)] = payload.decode("utf-8", errors="replace")
    return files, fields


def create_http_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    workspace_dir: str | Path | None = None,
) -> tuple[ThreadingHTTPServer, ConcertStudioWebApp]:
    """Create a ThreadingHTTPServer bound to `(host, port)` serving the ConcertStudio AI v2 Web UI."""
    app = ConcertStudioWebApp(workspace_dir=workspace_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_path(self, path: Path, ctype: str) -> None:
            """Stream a file from disk with single-range support."""
            total = path.stat().st_size
            start, end, code = 0, total - 1, 200
            rng = self.headers.get("Range", "")
            m = re.match(r"^bytes=(\d*)-(\d*)$", rng)
            if m and total:
                a, b = m.group(1), m.group(2)
                if a:
                    start = min(int(a), total - 1)
                    end = min(int(b), total - 1) if b else total - 1
                elif b:
                    start = max(0, total - int(b))
                end = max(start, end)
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.end_headers()
            try:
                with path.open("rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (ConnectionResetError, BrokenPipeError):
                pass

        def _send_json(self, status: int, data: dict[str, Any]) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                html = app.get_index_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path == "/api/progress":
                self._send_json(200, app.get_progress())
                return

            if self.path == "/api/latest_job":
                latest = app.get_latest_job()
                self._send_json(200, latest if latest is not None else {"job_id": None})
                return

            if self.path.startswith("/api/crowdsource/library"):
                parsed = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(parsed.query)
                cid = q.get("concert_id", ["current_concert"])[0] if q else "current_concert"
                self._send_json(200, app.get_crowdsource_library(cid))
                return

            if self.path == "/api/video/status":
                self._send_json(200, app.get_cut_status())
                return

            if self.path.startswith("/api/video/"):
                parts = self.path.strip("/").split("/")
                path = app.get_video_path(parts[2], parts[3]) if len(parts) == 4 else None
                if path is not None:
                    self._send_path(path, "video/mp4")
                    return

            if self.path.startswith("/api/audio/"):
                parts = self.path.strip("/").split("/")
                if len(parts) == 4:
                    data = app.get_audio_file(parts[2], parts[3])
                    if data is not None:
                        total_len = len(data)
                        range_hdr = self.headers.get("Range")
                        start = 0
                        end = total_len - 1
                        status_code = 200

                        if range_hdr and range_hdr.startswith("bytes="):
                            try:
                                r_parts = range_hdr[6:].split("-")
                                if r_parts[0]:
                                    start = int(r_parts[0])
                                if len(r_parts) > 1 and r_parts[1]:
                                    end = int(r_parts[1])
                                start = max(0, min(start, total_len - 1))
                                end = max(start, min(end, total_len - 1))
                                status_code = 206
                            except ValueError:
                                pass

                        content_len = end - start + 1
                        self.send_response(status_code)
                        self.send_header("Content-Type", "audio/wav")
                        self.send_header("Accept-Ranges", "bytes")
                        if status_code == 206:
                            self.send_header("Content-Range", f"bytes {start}-{end}/{total_len}")
                        self.send_header("Content-Length", str(content_len))
                        self.end_headers()

                        try:
                            chunk_size = 65536
                            pos = start
                            while pos <= end:
                                n = min(chunk_size, end - pos + 1)
                                self.wfile.write(data[pos : pos + n])
                                pos += n
                        except (ConnectionResetError, BrokenPipeError):
                            pass
                        return

            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            max_upload_bytes = 4 * 1024 * 1024 * 1024  # 4 GB cap for multi-POV video batches
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return
            if length < 0 or length > max_upload_bytes:
                self._send_json(413, {"error": "Payload exceeds maximum allowed upload size (4 GB)"})
                return
            body = self.rfile.read(length) if length > 0 else b"{}"
            ctype = self.headers.get("Content-Type", "application/json")

            try:
                if self.path == "/api/estimate":
                    payload = json.loads(body.decode("utf-8"))
                    self._send_json(200, app.estimate(payload))
                    return
                if self.path == "/api/auto_tune":
                    payload = json.loads(body.decode("utf-8"))
                    self._send_json(200, app.auto_tune_tracks(payload))
                    return
                if self.path == "/api/crowdsource/submit":
                    payload = json.loads(body.decode("utf-8"))
                    self._send_json(200, app.submit_crowdsource_pov(payload))
                    return
                if self.path == "/api/setlist":
                    payload = json.loads(body.decode("utf-8"))
                    self._send_json(200, app.align_setlist(payload))
                    return
                if self.path == "/api/video/render":
                    if len(body) > 4096:
                        self._send_json(413, {"error": "Request too large"})
                        return
                    payload = json.loads(body.decode("utf-8"))
                    try:
                        self._send_json(200, app.start_cut(payload if isinstance(payload, dict) else {}))
                    except (ValueError, RuntimeError) as exc:
                        self._send_json(409, {"error": str(exc)})
                    return
                if self.path == "/api/director/plan":
                    payload = json.loads(body.decode("utf-8"))
                    self._send_json(200, app.generate_director_plan(payload))
                    return
                if self.path == "/api/demo":
                    payload = json.loads(body.decode("utf-8"))
                    self._send_json(200, app.run_demo_job(payload))
                    return
                if self.path == "/api/reconstruct_folder":
                    payload = json.loads(body.decode("utf-8"))
                    self._send_json(200, app.run_reconstruct_folder_job(payload))
                    return
                if self.path == "/api/reconstruct":
                    files, fields = _parse_multipart_form_data(ctype, body)
                    self._send_json(200, app.run_reconstruct_job(files, fields))
                    return
                self._send_json(404, {"error": "Unknown endpoint"})
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    return server, app
