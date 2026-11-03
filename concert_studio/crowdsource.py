"""Crowdsource Ingestion, Deduplication Ledger & Continuous Versioning for ConcertStudio AI v2.

Allows fans and concert attendees to upload novel POVs. Discards redundant or inferior
re-uploads via SHA-256 and acoustic fingerprint hashing. Ingests high-value, novel POVs
into the persistent SQLite concert library and triggers new master revisions automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import sqlite3
import time
from typing import Any
import numpy as np
from scipy import signal

from concert_studio.models import AudioTrack


@dataclass(frozen=True)
class CrowdsourceDecision:
    """Evaluation result for an incoming user-submitted concert POV file."""

    submission_id: str
    concert_id: str
    filename: str
    action: str              # "ACCEPTED_NOVEL" | "ACCEPTED_UPGRADE" | "REJECTED_EXACT_DUPLICATE" | "REJECTED_REDUNDANT" | "REJECTED_INFERIOR"
    rejection_reason: str | None
    quality_score: float     # [0.0, 100.0] composite SNR, dynamic range, and frequency extension
    is_helpful: bool         # True if submission adds novel coverage or superior acoustic clarity
    triggers_remaster: bool  # True if master revision should be built
    new_master_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrowdsourceLibrary:
    """Persistent SQLite database tracking all concert projects, submissions, and master revisions."""

    def __init__(self, db_path: str | Path = "concert_studio_library.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS concerts (
                    concert_id TEXT PRIMARY KEY,
                    artist TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    master_version TEXT NOT NULL,
                    total_povs_accepted INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pov_submissions (
                    submission_id TEXT PRIMARY KEY,
                    concert_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    acoustic_hash TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    snr_est_db REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    action TEXT NOT NULL,
                    rejection_reason TEXT,
                    is_helpful INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(concert_id) REFERENCES concerts(concert_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS master_revisions (
                    revision_id TEXT PRIMARY KEY,
                    concert_id TEXT NOT NULL,
                    version_name TEXT NOT NULL,
                    num_povs INTEGER NOT NULL,
                    trigger_submission_id TEXT,
                    master_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(concert_id) REFERENCES concerts(concert_id)
                )
                """
            )
            conn.commit()

    @classmethod
    def compute_file_sha256(cls, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def compute_acoustic_fingerprint(cls, samples: np.ndarray, sr: int = 48000) -> str:
        """Compute an acoustic peak constellation hash invariant to slight volume shifts."""
        mono = np.mean(samples, axis=1) if samples.ndim == 2 else samples
        if len(mono) < sr:
            return "too_short"

        # Downsample to 8000 Hz for robust perceptual landmark extraction
        down_sr = 8000
        step = max(sr // down_sr, 1)
        mono_sub = mono[::step]

        # 512-point STFT
        f, t, Z = signal.stft(mono_sub, fs=down_sr, nperseg=256, noverlap=128)
        mag = np.abs(Z)

        # Extract top 3 peak frequencies per 1-second time window
        hop_sec = 1.0
        frames_per_sec = int(down_sr / 128)
        landmarks: list[str] = []
        for sec_idx in range(int(t[-1])):
            s = sec_idx * frames_per_sec
            e = s + frames_per_sec
            if e > mag.shape[1]:
                break
            sec_mag = np.mean(mag[:, s:e], axis=1)
            top_bins = np.argsort(sec_mag)[-3:]
            landmarks.append("-".join(str(b) for b in sorted(top_bins)))

        raw_str = "|".join(landmarks)
        return hashlib.md5(raw_str.encode("utf-8")).hexdigest()

    @classmethod
    def evaluate_quality(cls, samples: np.ndarray, sr: int = 48000) -> tuple[float, float]:
        """Compute SNR estimate (dB) and a 0-100 composite audio quality score."""
        mono = np.mean(samples, axis=1) if samples.ndim == 2 else samples
        if len(mono) < int(0.5 * sr):
            return 0.0, 0.0

        peak = float(np.max(np.abs(mono)) + 1e-9)
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-9)
        crest_db = float(np.clip(20.0 * np.log10(peak / rms), 3.0, 30.0))

        # Estimate background noise floor from 10th percentile
        p10 = float(np.percentile(np.abs(mono), 10)) + 1e-9
        snr_est_db = float(np.clip(20.0 * np.log10(rms / p10), 3.0, 45.0))

        # White noise / hiss penalty (spectral flatness in time domain)
        sub_abs = np.abs(mono[::8]) + 1e-7
        geom_mean = float(np.exp(np.mean(np.log(sub_abs))))
        arith_mean = float(np.mean(sub_abs))
        flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

        # Clipping penalty
        clip_ratio = float(np.sum(np.abs(mono) >= 0.95)) / float(len(mono))

        # Composite Quality Score [0, 100]
        score = 20.0 + 1.8 * snr_est_db + 1.0 * crest_db - 30.0 * flatness - clip_ratio * 300.0
        quality_score = float(np.clip(score, 5.0, 98.0))
        return snr_est_db, quality_score

    def register_concert(
        self,
        concert_id: str,
        artist: str,
        venue: str,
        date: str,
        initial_version: str = "v1.0",
    ) -> None:
        """Register or verify an active concert project in the library."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO concerts (concert_id, artist, venue, date, created_at, master_version, total_povs_accepted)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (concert_id, artist, venue, date, time.time(), initial_version),
            )
            conn.commit()

    def submit_pov(
        self,
        concert_id: str,
        track_or_path: AudioTrack | Path | str,
        filename: str | None = None,
    ) -> CrowdsourceDecision:
        """Ingest and evaluate a newly uploaded POV against all existing POVs in the concert database."""
        # Resolve track / samples
        if isinstance(track_or_path, (str, Path)):
            p = Path(track_or_path)
            fn = filename or p.name
            file_sha = self.compute_file_sha256(p)
            from concert_studio.io import load_media_track
            trk = load_media_track(p)
            samples = trk.samples
            sr = trk.sample_rate
        else:
            fn = filename or track_or_path.name
            samples = track_or_path.samples
            sr = track_or_path.sample_rate
            # Synthetic SHA from waveform bytes
            file_sha = hashlib.sha256(samples.tobytes()).hexdigest()

        submission_id = f"sub_{int(time.time() * 1000)}_{fn[:12]}"
        acoustic_hash = self.compute_acoustic_fingerprint(samples, sr=sr)
        dur = float(len(samples) / sr)
        snr_db, quality_score = self.evaluate_quality(samples, sr=sr)

        # Check for duplicates in SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # 1. Exact Binary SHA-256 Duplicate Check
            cur.execute(
                "SELECT * FROM pov_submissions WHERE concert_id = ? AND file_sha256 = ? AND action IN ('ACCEPTED_NOVEL', 'ACCEPTED_UPGRADE')",
                (concert_id, file_sha),
            )
            exact_dup = cur.fetchone()
            if exact_dup:
                decision = CrowdsourceDecision(
                    submission_id=submission_id,
                    concert_id=concert_id,
                    filename=fn,
                    action="REJECTED_EXACT_DUPLICATE",
                    rejection_reason=f"Identical binary file already accepted (Submission {exact_dup['submission_id']}). Thrown out.",
                    quality_score=quality_score,
                    is_helpful=False,
                    triggers_remaster=False,
                    new_master_version=None,
                )
                self._record_submission(conn, decision, file_sha, acoustic_hash, dur, sr, snr_db)
                return decision

            # 2. Acoustic Fingerprint Duplicate / Redundancy Check
            cur.execute(
                "SELECT * FROM pov_submissions WHERE concert_id = ? AND acoustic_hash = ? AND action IN ('ACCEPTED_NOVEL', 'ACCEPTED_UPGRADE')",
                (concert_id, acoustic_hash),
            )
            acoustic_dup = cur.fetchone()
            if acoustic_dup:
                prev_score = float(acoustic_dup["quality_score"])
                prev_snr = float(acoustic_dup["snr_est_db"])
                # If incoming file has noticeably superior fidelity (> +5 quality points AND higher SNR), upgrade!
                if (quality_score > prev_score + 5.0 and snr_db >= prev_snr) or (snr_db > prev_snr + 4.0):
                    action = "ACCEPTED_UPGRADE"
                    reason = None
                    is_helpful = True
                    triggers_remaster = True
                    # Increment master version (e.g. v1.1, v1.2)
                    new_ver = self._bump_version(conn, concert_id)
                else:
                    action = "REJECTED_REDUNDANT"
                    reason = f"Acoustically redundant to existing POV ({acoustic_dup['filename']}) without fidelity improvement. Thrown out."
                    is_helpful = False
                    triggers_remaster = False
                    new_ver = None

                decision = CrowdsourceDecision(
                    submission_id=submission_id,
                    concert_id=concert_id,
                    filename=fn,
                    action=action,
                    rejection_reason=reason,
                    quality_score=quality_score,
                    is_helpful=is_helpful,
                    triggers_remaster=triggers_remaster,
                    new_master_version=new_ver,
                )
                self._record_submission(conn, decision, file_sha, acoustic_hash, dur, sr, snr_db)
                return decision

            # 3. Novel POV Submission
            # Determine if helpful: duration >= 3.0s and quality_score >= 20.0
            if dur >= 3.0 and quality_score >= 20.0:
                action = "ACCEPTED_NOVEL"
                reason = None
                is_helpful = True
                triggers_remaster = True
                new_ver = self._bump_version(conn, concert_id)
                conn.execute(
                    "UPDATE concerts SET total_povs_accepted = total_povs_accepted + 1 WHERE concert_id = ?",
                    (concert_id,),
                )
            else:
                action = "REJECTED_INFERIOR"
                reason = f"Audio quality too low (score {quality_score:.1f}/100) or clip too brief ({dur:.1f}s) to benefit master wavefront."
                is_helpful = False
                triggers_remaster = False
                new_ver = None

            decision = CrowdsourceDecision(
                submission_id=submission_id,
                concert_id=concert_id,
                filename=fn,
                action=action,
                rejection_reason=reason,
                quality_score=quality_score,
                is_helpful=is_helpful,
                triggers_remaster=triggers_remaster,
                new_master_version=new_ver,
            )
            self._record_submission(conn, decision, file_sha, acoustic_hash, dur, sr, snr_db)
            return decision

    def _bump_version(self, conn: sqlite3.Connection, concert_id: str) -> str:
        cur = conn.cursor()
        cur.execute("SELECT master_version FROM concerts WHERE concert_id = ?", (concert_id,))
        row = cur.fetchone()
        cur_ver = row[0] if row else "v1.0"
        try:
            # Parse v1.X
            major, minor = cur_ver.lstrip("v").split(".")
            new_ver = f"v{major}.{int(minor) + 1}"
        except Exception:
            new_ver = "v1.1"

        conn.execute("UPDATE concerts SET master_version = ? WHERE concert_id = ?", (new_ver, concert_id))
        conn.commit()
        return new_ver

    def _record_submission(
        self,
        conn: sqlite3.Connection,
        decision: CrowdsourceDecision,
        file_sha: str,
        acoustic_hash: str,
        dur: float,
        sr: int,
        snr_db: float,
    ) -> None:
        conn.execute(
            """
            INSERT INTO pov_submissions (
                submission_id, concert_id, filename, file_sha256, acoustic_hash,
                duration_sec, sample_rate, snr_est_db, quality_score, action,
                rejection_reason, is_helpful, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.submission_id,
                decision.concert_id,
                decision.filename,
                file_sha,
                acoustic_hash,
                dur,
                sr,
                round(snr_db, 2),
                round(decision.quality_score, 2),
                decision.action,
                decision.rejection_reason,
                1 if decision.is_helpful else 0,
                time.time(),
            ),
        )
        conn.commit()

    def get_accepted_submissions(self, concert_id: str) -> list[dict[str, Any]]:
        """List all valid accepted POVs contributing to the concert master."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT submission_id, filename, duration_sec, quality_score, snr_est_db, action, created_at
                FROM pov_submissions
                WHERE concert_id = ? AND action IN ('ACCEPTED_NOVEL', 'ACCEPTED_UPGRADE')
                ORDER BY created_at ASC
                """,
                (concert_id,),
            )
            return [dict(r) for r in cur.fetchall()]
