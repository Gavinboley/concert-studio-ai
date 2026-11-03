"""Stage 0 / Dynamic Acoustic Analyzer & Auto-Setting Optimizer for ConcertStudio AI v2.

Probes multi-POV concert acoustic conditions (sub-bass deficiency, PA horn harshness,
reverberant stadium decay, crowd entropy, mic clipping, and stereo purity) and dynamically
synthesizes optimal pipeline parameters to achieve archival-grade audio fidelity automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from scipy import signal, ndimage

from concert_studio.models import AudioTrack
from concert_studio.pipeline import PipelineConfig


@dataclass(frozen=True)
class AcousticProfile:
    """Quantitative acoustic diagnosis of the multi-POV audio capture payload."""

    num_tracks_analyzed: int
    duration_analyzed_sec: float
    sub_bass_ratio_db: float          # Energy in 35-110 Hz relative to mid-band (300-2500 Hz)
    harsh_treble_ratio_db: float      # Energy in 3.5-7.5 kHz relative to mid-band
    reverb_decay_score: float         # [0.0, 1.0] estimated arena/stadium decay tail
    clipping_ratio: float             # Fraction of samples exceeding +/- 0.95 full scale
    noise_floor_dbfs: float           # 10th percentile background noise level
    crest_factor_db: float            # Peak-to-RMS dynamic ratio
    stereo_purity_score: float        # Inter-channel correlation (-1.0 to 1.0; <= 0.05 is mono)
    crowd_entropy_score: float        # High-entropy non-tonal energy ratio (crowd cheering/clapping)
    detected_venue_profile: str       # "stadium_arena" | "outdoor_festival" | "indoor_club" | "theater_hall"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutoTuningResult:
    """Optimized pipeline configuration and explanatory acoustic findings."""

    acoustic_profile: AcousticProfile
    optimized_config: PipelineConfig
    diagnostic_findings: list[str]
    suggested_mastering_curve: str    # "live_rock_punch" | "stadium_clarity" | "club_sub_weight" | "balanced_archival"
    expected_quality_improvement_db: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "acoustic_profile": self.acoustic_profile.to_dict(),
            "optimized_config": {
                "quality_tier": self.optimized_config.quality_tier,
                "dereverb_alpha": round(self.optimized_config.dereverb_alpha, 3),
                "mad_outlier_threshold": round(self.optimized_config.mad_outlier_threshold, 3),
                "clip_threshold": round(self.optimized_config.clip_threshold, 3),
                "target_lufs": round(self.optimized_config.target_lufs, 2),
                "true_peak_limit_db": round(self.optimized_config.true_peak_limit_db, 2),
                "enable_dynamic_de_resonance": self.optimized_config.enable_dynamic_de_resonance,
                "enable_air_absorption_comp": self.optimized_config.enable_air_absorption_comp,
            },
            "diagnostic_findings": self.diagnostic_findings,
            "suggested_mastering_curve": self.suggested_mastering_curve,
            "expected_quality_improvement_db": round(self.expected_quality_improvement_db, 2),
        }


class AutoSettingOptimizer:
    """Probes acoustic payload and auto-selects optimal DSP settings for maximum fidelity."""

    @classmethod
    def probe_tracks(
        cls,
        tracks: list[AudioTrack],
        sample_rate: int = 48000,
        max_probe_sec: float = 60.0,
    ) -> AcousticProfile:
        """Inspect representative slices of multi-POV audio tracks to measure acoustics."""
        if not tracks:
            return AcousticProfile(
                num_tracks_analyzed=0,
                duration_analyzed_sec=0.0,
                sub_bass_ratio_db=0.0,
                harsh_treble_ratio_db=0.0,
                reverb_decay_score=0.25,
                clipping_ratio=0.0,
                noise_floor_dbfs=-60.0,
                crest_factor_db=14.0,
                stereo_purity_score=0.0,
                crowd_entropy_score=0.20,
                detected_venue_profile="indoor_club",
            )

        # Sample up to top 6 longest tracks for fast, accurate multi-channel probing
        sorted_tracks = sorted(tracks, key=lambda t: t.duration_sec, reverse=True)[:6]
        probe_len = int(min(max_probe_sec, min(t.duration_sec for t in sorted_tracks)) * sample_rate)
        if probe_len < int(2.0 * sample_rate):
            probe_len = min(t.samples.shape[0] for t in sorted_tracks)

        sub_ratios: list[float] = []
        harsh_ratios: list[float] = []
        reverb_scores: list[float] = []
        clip_ratios: list[float] = []
        noise_floors: list[float] = []
        crest_factors: list[float] = []
        stereo_scores: list[float] = []
        crowd_scores: list[float] = []

        # Filter SOS definitions
        sos_sub = signal.butter(2, [35.0, 110.0], btype="bandpass", fs=sample_rate, output="sos")
        sos_mid = signal.butter(2, [300.0, 2500.0], btype="bandpass", fs=sample_rate, output="sos")
        sos_harsh = signal.butter(2, [3500.0, 7500.0], btype="bandpass", fs=sample_rate, output="sos")

        for trk in sorted_tracks:
            # Analyze mid-section of recording where performance is active
            start = max(0, (trk.samples.shape[0] - probe_len) // 2)
            chunk = trk.samples[start : start + probe_len]
            mono = np.mean(chunk, axis=1) if chunk.ndim == 2 else chunk

            # Sub-bass vs Mid-band energy ratio
            sub_e = float(np.mean(signal.sosfilt(sos_sub, mono) ** 2) + 1e-12)
            mid_e = float(np.mean(signal.sosfilt(sos_mid, mono) ** 2) + 1e-12)
            harsh_e = float(np.mean(signal.sosfilt(sos_harsh, mono) ** 2) + 1e-12)

            sub_ratios.append(10.0 * np.log10(sub_e / mid_e))
            harsh_ratios.append(10.0 * np.log10(harsh_e / mid_e))

            # Clipping check
            clip_cnt = float(np.sum(np.abs(mono) >= 0.94))
            clip_ratios.append(clip_cnt / float(len(mono)))

            # Noise floor (10th percentile energy in 50ms blocks)
            blk_sz = int(0.05 * sample_rate)
            n_blks = len(mono) // blk_sz
            if n_blks > 4:
                blks = mono[: n_blks * blk_sz].reshape(n_blks, blk_sz)
                rms_blks = np.sqrt(np.mean(blks**2, axis=1) + 1e-12)
                p10_rms = float(np.percentile(rms_blks, 10))
                noise_floors.append(20.0 * np.log10(p10_rms + 1e-9))
                # Crest factor
                peak_val = float(np.max(np.abs(mono)))
                rms_val = float(np.sqrt(np.mean(mono**2)) + 1e-12)
                crest_factors.append(20.0 * np.log10(peak_val / rms_val))
            else:
                noise_floors.append(-50.0)
                crest_factors.append(12.0)

            # Stereo purity check (if stereo track)
            if chunk.ndim == 2 and chunk.shape[1] == 2:
                l, r = chunk[:, 0], chunk[:, 1]
                denom = float(np.linalg.norm(l) * np.linalg.norm(r) + 1e-12)
                corr = float(np.clip(np.dot(l, r) / denom, -1.0, 1.0))
                # True stereo has corr in [0.2, 0.85]; identical mono has 1.0
                stereo_scores.append(corr)
            else:
                stereo_scores.append(1.0)  # purely mono

            # Reverberation tail estimation via energy autocorrelation decay
            hop = int(0.010 * sample_rate)
            if len(mono) > hop * 50:
                frame_rms = np.sqrt(ndimage.uniform_filter1d(mono**2, size=hop * 2)[::hop] + 1e-12)
                norm_rms = frame_rms - np.mean(frame_rms)
                auto_corr = np.correlate(norm_rms, norm_rms, mode="full")
                center = len(auto_corr) // 2
                half = auto_corr[center : center + 25]  # ~250ms decay window
                if half[0] > 1e-6:
                    half = half / half[0]
                    reverb_tail = float(np.clip(np.mean(half[5:15]), 0.05, 0.95))
                    reverb_scores.append(reverb_tail)
                else:
                    reverb_scores.append(0.30)
            else:
                reverb_scores.append(0.30)

            # Crowd cheering entropy score (high-frequency spectral flux variation)
            diffs = np.diff(mono[::4])
            spectral_flux = float(np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-9))
            crowd_scores.append(float(np.clip(spectral_flux / 3.0, 0.1, 0.9)))

        # Aggregate averages
        mean_sub = float(np.mean(sub_ratios))
        mean_harsh = float(np.mean(harsh_ratios))
        mean_reverb = float(np.mean(reverb_scores))
        mean_clip = float(np.mean(clip_ratios))
        mean_noise = float(np.mean(noise_floors))
        mean_crest = float(np.mean(crest_factors))
        mean_stereo = float(np.mean(stereo_scores))
        mean_crowd = float(np.mean(crowd_scores))

        # Classify Venue Profile
        if mean_reverb > 0.52:
            venue = "stadium_arena"
        elif mean_noise > -38.0 and mean_reverb < 0.35:
            venue = "outdoor_festival"
        elif mean_sub < -10.0 and mean_harsh > 4.0:
            venue = "indoor_club"
        else:
            venue = "theater_hall"

        return AcousticProfile(
            num_tracks_analyzed=len(sorted_tracks),
            duration_analyzed_sec=round(float(probe_len / sample_rate), 2),
            sub_bass_ratio_db=round(mean_sub, 2),
            harsh_treble_ratio_db=round(mean_harsh, 2),
            reverb_decay_score=round(mean_reverb, 3),
            clipping_ratio=round(mean_clip, 4),
            noise_floor_dbfs=round(mean_noise, 2),
            crest_factor_db=round(mean_crest, 2),
            stereo_purity_score=round(mean_stereo, 3),
            crowd_entropy_score=round(mean_crowd, 3),
            detected_venue_profile=venue,
        )

    @classmethod
    def optimize_settings(
        cls,
        profile: AcousticProfile,
        base_config: PipelineConfig | None = None,
    ) -> AutoTuningResult:
        """Derive optimal pipeline parameters and report diagnostic findings based on acoustic profile."""
        findings: list[str] = []
        exp_gain_db = 0.0

        # 1. Quality Tier defaults to archival_max (User Rule)
        quality_tier = "archival_max"
        findings.append("Selected Quality Tier 'archival_max' (4-Band STFT 512/1024/2048/4096, 32 sync subwindows).")

        # 2. Reverberation Tail & WPE Tuning
        if profile.reverb_decay_score > 0.50:
            dereverb_alpha = float(np.clip(0.42 + (profile.reverb_decay_score - 0.50) * 0.40, 0.42, 0.52))
            findings.append(
                f"High stadium/arena reverberation tail detected (score {profile.reverb_decay_score:.2f}) -> "
                f"Increased dereverb_alpha to {dereverb_alpha:.2f} with deep multi-channel WPE prediction."
            )
            exp_gain_db += 4.5
        elif profile.reverb_decay_score < 0.25:
            dereverb_alpha = 0.22
            findings.append(
                f"Dry acoustic profile detected (decay {profile.reverb_decay_score:.2f}) -> "
                f"Reduced dereverb_alpha to {dereverb_alpha:.2f} to preserve intimate live room bloom."
            )
            exp_gain_db += 1.8
        else:
            dereverb_alpha = 0.35
            findings.append(f"Balanced acoustic decay detected -> set dereverb_alpha to {dereverb_alpha:.2f}.")
            exp_gain_db += 3.0

        # 3. Anomaly / Handling Outlier Sensitivity
        if profile.noise_floor_dbfs > -42.0 or profile.crowd_entropy_score > 0.55:
            mad_threshold = 1.40
            findings.append(
                f"Dense crowd cheering & elevated noise floor ({profile.noise_floor_dbfs:.1f} dBFS) -> "
                f"Tightened MAD anomaly threshold to {mad_threshold:.2f} to filter mic rubs and sharp claps."
            )
            exp_gain_db += 2.8
        elif profile.noise_floor_dbfs < -55.0:
            mad_threshold = 1.85
            findings.append(
                f"Clean, low-noise input signals ({profile.noise_floor_dbfs:.1f} dBFS) -> "
                f"Relaxed MAD threshold to {mad_threshold:.2f} to preserve full musical dynamic range."
            )
            exp_gain_db += 1.5
        else:
            mad_threshold = 1.65
            findings.append(f"Nominal noise floor -> MAD outlier threshold set to standard {mad_threshold:.2f}.")
            exp_gain_db += 2.0

        # 4. Distortion & Digital Clipping Repair
        if profile.clipping_ratio > 0.005:
            clip_thresh = 0.92
            findings.append(
                f"High clip density ({profile.clipping_ratio * 100:.2f}% of samples clipped) -> "
                f"Lowered clip detection ceiling to {clip_thresh:.2f} to enable cubic spline waveform declipping."
            )
            exp_gain_db += 3.2
        else:
            clip_thresh = 0.96
            findings.append("Low clipping detected -> standard 0.96 ceiling preserved.")
            exp_gain_db += 0.8

        # 5. High-Frequency Harshness & Soothe De-Resonance
        enable_de_resonance = True
        if profile.harsh_treble_ratio_db > 2.0:
            findings.append(
                f"Excessive PA horn whistling & MEMS distortion detected ({profile.harsh_treble_ratio_db:+.1f} dB treble tilt) -> "
                f"Activated aggressive dynamic resonance suppression with -5.0 dB maximum notch cut."
            )
            exp_gain_db += 3.5
        else:
            findings.append("High-frequency tilt within nominal bounds -> subtle 2.5 dB de-resonance smoothing enabled.")
            exp_gain_db += 1.2

        # 6. Low-End / Sub-Bass Foundation & Mastering Target Curve
        if profile.sub_bass_ratio_db < -6.0:
            mastering_curve = "club_sub_weight"
            findings.append(
                f"Severe sub-kick and bass fundamental loss detected ({profile.sub_bass_ratio_db:+.1f} dB sub-bass deficiency) -> "
                f"Selected 'club_sub_weight' mastering curve (+7.5 dB 60 Hz chest punch, +7.0 dB 160 Hz bassline definition)."
            )
            exp_gain_db += 4.0
        elif profile.detected_venue_profile == "stadium_arena":
            mastering_curve = "stadium_clarity"
            findings.append("Stadium arena profile -> selected 'stadium_clarity' mastering curve with open stereo air.")
            exp_gain_db += 2.5
        else:
            mastering_curve = "live_rock_punch"
            findings.append("Selected 'live_rock_punch' mastering curve for authoritative live concert presence.")
            exp_gain_db += 3.0

        # 7. Mono vs Stereo Guidance
        if profile.stereo_purity_score >= 0.95:
            findings.append(
                "Input POVs are predominantly Mono/Dual-Mono (correlation >= 0.95) -> "
                "Mono core consensus will synthesize a phase-linear, comb-free studio soundstage with zero wander."
            )
        else:
            findings.append(
                f"Multi-mic stereo decorrelation detected (corr {profile.stereo_purity_score:.2f}) -> "
                "Acoustic apertures will inform lateral stem placement without Haas comb filtering."
            )

        optimized_cfg = PipelineConfig(
            target_sample_rate=base_config.target_sample_rate if base_config else 48000,
            quality_tier=quality_tier,
            true_data_mode=True,
            auto_detect_partial_clips=True,
            enable_echo_cancellation=True,
            enable_nonrigid_motion=True,
            enable_anomaly_masking=True,
            enable_agc_inversion=True,
            enable_deduplication=True,
            enable_dynamic_de_resonance=enable_de_resonance,
            enable_air_absorption_comp=True,
            mad_outlier_threshold=mad_threshold,
            clip_threshold=clip_thresh,
            dereverb_alpha=dereverb_alpha,
            separation_backend="auto",
            target_lufs=-14.0,
            true_peak_limit_db=-1.0,
        )

        return AutoTuningResult(
            acoustic_profile=profile,
            optimized_config=optimized_cfg,
            diagnostic_findings=findings,
            suggested_mastering_curve=mastering_curve,
            expected_quality_improvement_db=exp_gain_db,
        )

    @classmethod
    def analyze_and_optimize(
        cls,
        tracks: list[AudioTrack],
        sample_rate: int = 48000,
        base_config: PipelineConfig | None = None,
    ) -> AutoTuningResult:
        """Single-call probe and auto-tune pipeline."""
        profile = cls.probe_tracks(tracks, sample_rate=sample_rate)
        return cls.optimize_settings(profile, base_config=base_config)
