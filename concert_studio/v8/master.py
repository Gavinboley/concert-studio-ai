"""V8 live-album mix + master from comp stems.

Signal flow (all stems mono in, stereo out):

  vocals  = dereverbed lead (+ a little of its own room) -> HPF -> tonal EQ -> rider -> 2.5:1
            -> de-ess -> gentle 4x-oversampled exciter on >3 kHz -> centre; plate + hall sends
  guitar  -> HPF -> tonal EQ (never boosts above 5.5 kHz: cab fizz) -> light comp -> M/S width
  bass    -> HPF -> tonal EQ -> 3:1 note evener -> parallel oversampled harmonics (<1.2 kHz) -> mono
  drums   -> HPF -> tonal EQ -> parallel compression -> mono core + width on cymbals; hall send
  keys/other -> HPF -> tonal EQ -> M/S width
  crowd   -> band-limited, wide, low; it is the room, not a performer
  bus     -> glue 1.5:1 (<= ~2 dB GR) -> final tonal tilt check -> true-peak limiter
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import mix as M
from .dsp import highpass, lowpass, db, rms

SR = 44100


@dataclass
class MixParams:
    target_lufs: float = -14.0
    ceiling_dbtp: float = -1.0
    # stem loudness (LUFS) before the bus; relative balance is what matters
    vocals_lufs: float = -14.5
    guitar_lufs: float = -16.5
    bass_lufs: float = -20.5
    drums_lufs: float = -19.0
    keys_lufs: float = -25.0
    other_lufs: float = -29.0
    crowd_lufs: float = -33.0
    vocal_room_keep: float = 0.05     # fraction of the separated venue reverb kept on the vocal
    vocal_plate_send: float = 0.04
    hall_send_vocal: float = 0.025
    hall_send_guitar: float = 0.015
    hall_send_drums: float = 0.012
    guitar_width: float = 0.55
    keys_width: float = 0.45
    other_width: float = 0.45
    drums_width: float = 0.22
    crowd_width: float = 0.9
    exciter_mix: float = 0.06
    bass_harmonics_mix: float = 0.18
    drum_parallel: float = 0.2
    drum_attack: float = 0.5
    glue_ratio: float = 1.2
    # real two-POV stereo (side = (B - A)/2 from a second phone position); 0 disables -> decorrelator
    real_width_guitar: float = 0.7
    real_width_keys: float = 0.6
    real_width_drums: float = 0.5
    vocal_carve_db: float = 4.0       # max dip of guitar/keys 1.5-4 kHz while the vocal sings
    master_lowmid_db: float = -2.0     # 250-500 Hz measured ~2 dB heavy vs typical mixes
    master_lowmid_hz: float = 350.0
    master_air_db: float = 1.5
    max_move_db: float = 12.0           # cap on a stem's balance boost above its natural level
    stabilize_strength: float = 0.5    # tonal_stabilizer on the bus (0 = off); keeps the tone consistent over the song
    stem_stabilize_strength: float = 0.7   # same, per raw stem (vocals/guitar/bass/drums + B stems) before processing
    stem_stabilize_max_db: float = 4.0
    stem_stabilize_list: str = "vocals_dry,guitar,bass,drums"
    vocal_stabilize_lo_hz: float = 300.0   # vocal tone matching only from here up (its low-mids are partly bleed;
                                           # lifting them cost up to -0.5 PQ where a take was thin)
    stabilize_max_db: float = 3.0
    stabilize_ref_start: float = -1.0  # reference section (master s); < 0 = song median shape
    stabilize_ref_end: float = -1.0
    tone_ref: str = ""                  # optional studio recording of the song: match the master's tonal balance
    tone_ref_strength: float = 0.7
    tone_ref_max_db: float = 6.0
    extras: dict = field(default_factory=dict)


def _load(p: Path) -> np.ndarray:
    x, sr = sf.read(p, dtype="float32")
    assert sr == SR, (p, sr)
    return x if x.ndim == 1 else x.mean(axis=1)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def process_vocals(dry: np.ndarray, room: np.ndarray, p: MixParams, report: dict) -> np.ndarray:
    v = dry + p.vocal_room_keep * room
    v = highpass(v, SR, 85.0, order=2)
    v, report["vocals_eq"] = M.tonal_correct(v, "vocals", max_boost=6.0, max_cut=8.0, ref_band=(250, 1600),
                                             top_hz=14000, no_boost_below=160.0,
                                             boost_caps=[(5000, 20000, 4.0)])
    v = M.match_loudness(v, -20.0)
    v = M.level_rider(v, target_db=float(db(rms(v[np.abs(v) > 1e-4]))) if np.any(np.abs(v) > 1e-4) else -26.0,
                      max_boost=4.0, max_cut=4.0, window_s=1.5, smooth_s=3.0)
    # leveling compressor: threshold at the loud-phrase level
    e = M.envelope(v, 10.0, 150.0, mode="rms")
    thr = float(db(np.percentile(e[e > 1e-4], 70))) if np.any(e > 1e-4) else -30.0
    v, gr = M.compress(v, thr, 2.5, 8.0, 140.0, knee_db=8.0)
    report["vocal_comp_gr_p95_db"] = round(float(np.percentile(gr, 95)), 2)
    v = M.deess(v, 5200.0, 10500.0, thresh_rel_db=4.0, ratio=4.0)
    # exciter: harmonics generated from 3-8 kHz content only, then high-passed at 5 kHz
    band = M.highpass(v, SR, 3000.0, order=2)
    band = M.lowpass(band, SR, 8000.0, order=2)
    exc = M.saturate(band * 3.0, 9.0) / 3.0
    exc = M.highpass(exc, SR, 5000.0, order=4)
    v = v + p.exciter_mix * exc / (rms(exc) / max(rms(band), 1e-9) + 1e-9) * 0.5
    return v.astype(np.float32)


def process_guitar(g: np.ndarray, p: MixParams, report: dict) -> np.ndarray:
    g = highpass(g, SR, 75.0, order=2)
    g, report["guitar_eq"] = M.tonal_correct(g, "guitar", max_boost=5.0, max_cut=9.0, top_hz=5500,
                                             no_boost_below=120.0)
    g = M.lowpass(g, SR, 12000.0, order=2)
    e = M.envelope(g, 15.0, 200.0, mode="rms")
    thr = float(db(np.percentile(e[e > 1e-5], 80))) if np.any(e > 1e-5) else -30
    g, _ = M.compress(g, thr, 1.8, 15.0, 200.0, knee_db=8.0)
    return g


def process_bass(b: np.ndarray, p: MixParams, report: dict) -> np.ndarray:
    b = highpass(b, SR, 32.0, order=4)
    b, report["bass_eq"] = M.tonal_correct(b, "bass", max_boost=5.0, max_cut=10.0, ref_band=(45, 120),
                                           top_hz=2500, no_boost_below=35.0, boost_caps=[(160, 2500, 0.0)])
    e = M.envelope(b, 20.0, 160.0, mode="rms")
    thr = float(db(np.percentile(e[e > 1e-5], 60))) if np.any(e > 1e-5) else -30
    b, gr = M.compress(b, thr, 3.0, 20.0, 160.0, knee_db=6.0)
    report["bass_comp_gr_p95_db"] = round(float(np.percentile(gr, 95)), 2)
    # parallel harmonics so the bass line reads on small speakers (oversampled: no aliasing buzz)
    h = M.saturate(M.lowpass(b, SR, 300.0, order=2) * 4.0, 6.0) / 4.0
    h = M.highpass(h, SR, 250.0, order=2)
    h = M.lowpass(h, SR, 1200.0, order=4)
    b = b + p.bass_harmonics_mix * h * (rms(b) / max(rms(h), 1e-9)) * 0.5
    b = M.lowpass(b, SR, 5000.0, order=2)
    return b.astype(np.float32)


def process_drums(d: np.ndarray, p: MixParams, report: dict) -> np.ndarray:
    d = highpass(d, SR, 30.0, order=2)
    d, report["drums_eq"] = M.tonal_correct(d, "drums", max_boost=5.0, max_cut=8.0, ref_band=(63, 4000),
                                            top_hz=15000)
    # parallel (NY) compression for body without killing transients
    e = M.envelope(d, 1.0, 80.0, mode="peak")
    thr = float(db(np.percentile(e[e > 1e-5], 50))) if np.any(e > 1e-5) else -30
    crushed, _ = M.compress(d, thr, 6.0, 3.0, 90.0, knee_db=4.0)
    d = d + p.drum_parallel * crushed * (rms(d) / max(rms(crushed), 1e-9))
    # transient shaper: lift attacks where the fast envelope leads the slow one
    fast = M.envelope(d, 0.5, 40.0, mode="peak")
    slow = M.envelope(d, 15.0, 120.0, mode="peak")
    lift = np.clip(p.drum_attack * (db(fast + 1e-9) - db(slow + 1e-9)), 0.0, 6.0)
    from scipy import ndimage as _nd
    d = d * 10 ** (_nd.uniform_filter1d(lift, 64) / 20.0)
    report["drum_attack_lift_p95_db"] = round(float(np.percentile(lift, 95)), 2)
    return d.astype(np.float32)


def process_keys(k: np.ndarray, kind: str, p: MixParams, report: dict) -> np.ndarray:
    k = highpass(k, SR, 90.0, order=2)
    k, report[f"{kind}_eq"] = M.tonal_correct(k, "keys", max_boost=4.0, max_cut=8.0, top_hz=12000,
                                              no_boost_below=250.0 if kind == "other" else 120.0)
    return k


def vocal_carve(x: np.ndarray, vocal: np.ndarray, max_db: float) -> np.ndarray:
    """Sidechain dynamic EQ: dip x's 1.5-4 kHz band only while the vocal is active (frees the voice)."""
    if max_db <= 0:
        return x
    h = signal.firwin(1023, [1500.0, 4000.0], pass_zero=False, fs=SR).astype(np.float32)
    band = M.apply_fir(x, h, linear_phase=True)
    ve = db(M.envelope(vocal, 15.0, 180.0, mode="rms") + 1e-9)
    act = ve > np.max(ve) - 40
    ref = np.median(ve[act]) if act.any() else -40.0
    gr = np.clip(0.5 * (ve - (ref - 8.0)), 0.0, max_db)
    from scipy import ndimage as _nd
    gr = _nd.uniform_filter1d(gr, 2048)
    return (x - band * (1.0 - 10 ** (-gr / 20.0))).astype(np.float32)


def real_stereo(a: np.ndarray, b: np.ndarray | None, width: float, hp_hz: float, seed: int,
                fallback_width: float) -> np.ndarray:
    """L = A + S, R = A - S with S = width * (B' - A) / 2, B' = B slowly level-matched to A.
    Mono sum is exactly A; the side is the real difference between two microphone positions."""
    if b is None or width <= 0:
        return M.stereoize(a, fallback_width, seed=seed, hp_hz=hp_hz)
    b = _fit(b, len(a))
    b = b * (rms(a) / max(rms(b), 1e-9))           # global match first; B is not balance-placed
    from scipy import ndimage as _nd
    hop = 2048
    n = len(a) // hop
    ea = np.sqrt(np.mean(a[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    eb = np.sqrt(np.mean(b[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    g = _nd.uniform_filter1d(np.clip(20 * np.log10(ea / eb), -6.0, 6.0), 43)
    gs = 10 ** (np.interp(np.arange(len(a)), np.arange(n) * hop + hop // 2, g) / 20.0)
    side = 0.5 * width * (b * gs - a)
    side = M.highpass(side, SR, hp_hz, order=2)
    side = M.lowpass(side, SR, 14000.0, order=2)
    return np.stack([a + side, a - side], 1).astype(np.float32)


def mix_and_master(stems: dict[str, np.ndarray], crowd: np.ndarray, p: MixParams, out_dir: Path,
                   tag: str = "v8") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(len(v) for v in stems.values())
    S = {k: _fit(v, n) for k, v in stems.items()}
    crowd = _fit(crowd, n)
    rep: dict = {"params": asdict(p)}

    # natural balance reference: the unprocessed stem sum
    raw_sum = sum(S[k] for k in ("vocals_dry", "guitar", "bass", "drums", "piano", "other"))
    ref_lufs = M.lufs(raw_sum)
    # stems far below the band are separation residue: fold them into "other" untouched
    if M.lufs(S["piano"]) < ref_lufs - 18.0:
        S["other"] = S["other"] + S["piano"]
        S["piano"] = np.zeros(n, np.float32)
        rep["piano_folded_into_other"] = True
    natural = {k: M.lufs(S[k]) for k in ("vocals_dry", "guitar", "bass", "drums", "piano", "other")}
    # tone consistency: the comp hops between phones with different voicing (one dark take for minutes, then bright
    # thin ones); pull each main stem's slow tonal shape toward its own song-median shape (level preserved)
    if p.stem_stabilize_strength > 0:
        bands = {"vocals_dry": (p.vocal_stabilize_lo_hz, 16000.0), "guitar": (60.0, 16000.0), "bass": (30.0, 5000.0),
                 "drums": (30.0, 16000.0)}
        on = {k.strip() for k in p.stem_stabilize_list.replace(":", ",").split(",") if k.strip()}
        for k_, (lo_, hi_) in ((k, v) for k, v in bands.items() if k in on):
            for kk in (k_, "B_" + k_):
                if S.get(kk) is not None and np.any(S[kk]):
                    S[kk] = M.tonal_stabilizer(_fit(S[kk], n), SR, p.stem_stabilize_strength, p.stem_stabilize_max_db,
                                               f_lo=lo_, f_hi=hi_, report=rep, label=kk)

    voc = process_vocals(S["vocals_dry"], S.get("vocals_room", np.zeros(n, np.float32)), p, rep)
    gtr = process_guitar(S["guitar"], p, rep)
    bas = process_bass(S["bass"], p, rep)
    drm = process_drums(S["drums"], p, rep)
    key = process_keys(S["piano"], "piano", p, rep) if rms(S["piano"]) > 0 else S["piano"]
    oth = process_keys(S["other"], "other", p, rep)

    common = -17.0 - ref_lufs            # brings the raw band to ~-17 LUFS
    moves = {}
    # Genre guard: how present each stem really is (LU vs. the whole band). A stem far below the band is
    # separation residue (the "vocals" of an instrumental, the "guitar" of an electronic track): never boost
    # it toward a target, and do not carve the band around a voice that is not there.
    presence = {k: (natural[k] - ref_lufs) if np.isfinite(natural[k]) else -99.0 for k in natural}
    allow = {k: float(np.clip((v + 24.0) / 10.0, 0.0, 1.0)) for k, v in presence.items()}   # -24 LU: 0, -14 LU: 1
    # Arrangement: the vocals-first targets assume a singer. With no clear voice (instrumental, electronic,
    # the "vocals" stem is a synth lead or residue), melodic stems keep the band's own balance (+-2 dB).
    sung = float(np.clip((presence["vocals_dry"] + 16.0) / 6.0, 0.0, 1.0))      # -16 LU: 0, -10 LU: 1
    melodic = ("vocals_dry", "guitar", "piano", "other")
    rep["ref_lufs"] = round(float(ref_lufs), 1)
    rep["presence_lu"] = {k: round(v, 1) for k, v in presence.items()}
    rep["vocal_song"] = round(sung, 2)

    def place(x, key_name, target):
        if not np.any(x) or not np.isfinite(natural[key_name]):
            return x
        natural_after = natural[key_name] + common
        lo, hi = -8.0, p.max_move_db * allow[key_name]
        if key_name in melodic:                       # instrumental: stay near the natural balance
            lo, hi = lo * sung + -2.0 * (1 - sung), min(hi, hi * sung + 2.0 * (1 - sung))
        if presence[key_name] >= -4.0:                # the arrangement's dominant element: never bury it
            lo = max(lo, -3.0)
        move = float(np.clip(target - natural_after, lo, hi))
        moves[key_name] = round(move, 1)
        return M.match_loudness(x, natural_after + move)

    # second POV (B) versions of the instruments, processed identically, for real stereo sides
    B = {}
    for k_, fn in (("guitar", lambda x: process_guitar(x, p, {})), ("piano", lambda x: process_keys(x, "piano", p, {})),
                   ("other", lambda x: process_keys(x, "other", p, {})), ("drums", lambda x: process_drums(x, p, {}))):
        if S.get(f"B_{k_}") is not None and np.any(S[f"B_{k_}"]):
            B[k_] = fn(_fit(S[f"B_{k_}"], n))
    rep["real_stereo_stems"] = sorted(B)

    voc = place(voc, "vocals_dry", p.vocals_lufs)
    gtr = place(gtr, "guitar", p.guitar_lufs)
    bas = place(bas, "bass", p.bass_lufs)
    drm = place(drm, "drums", p.drums_lufs)
    key = place(key, "piano", p.keys_lufs)
    oth = place(oth, "other", p.other_lufs)
    rep["balance_moves_db"] = moves
    # B stems follow A's placement gain (level match happens inside real_stereo)
    carve_db = p.vocal_carve_db * allow["vocals_dry"] * sung
    rep["vocal_carve_db_used"] = round(carve_db, 2)
    gtr = vocal_carve(gtr, voc, carve_db)
    key = vocal_carve(key, voc, carve_db)
    oth = vocal_carve(oth, voc, carve_db)
    for k_, a_ in (("guitar", gtr), ("piano", key), ("other", oth), ("drums", drm)):
        if k_ in B:
            B[k_] = vocal_carve(B[k_], voc, carve_db) if k_ != "drums" else B[k_]
    rep["natural_lufs"] = {k: round(v, 1) for k, v in natural.items()}

    # crowd: band-limited room bed
    crd = M.highpass(crowd, SR, 150.0, order=2)
    crd = M.lowpass(crd, SR, 9000.0, order=2)
    crd = M.match_loudness(crd, p.crowd_lufs) if rms(crd) > 1e-6 else crd

    # stereo
    st = {
        "vocals": np.stack([voc, voc], 1),
        "guitar": real_stereo(gtr, B.get("guitar"), p.real_width_guitar, 250.0, 11, p.guitar_width),
        "bass": np.stack([bas, bas], 1),
        "drums": real_stereo(drm, B.get("drums"), p.real_width_drums, 2500.0, 23, p.drums_width),
        "piano": real_stereo(key, B.get("piano"), p.real_width_keys, 250.0, 31, p.keys_width),
        "other": real_stereo(oth, B.get("other"), p.real_width_keys, 250.0, 41, p.other_width),
        "crowd": M.stereoize(crd, p.crowd_width, seed=53, hp_hz=200.0),
    }
    # reverbs (true stereo synthetic IRs)
    plate = M.synth_reverb_ir(1.25, 35.0, seed=3, hf_damp=0.55, lf_rt_mult=0.8, er=False)
    hall = M.synth_reverb_ir(1.9, 22.0, seed=5, hf_damp=0.4, lf_rt_mult=1.1, er=True)
    send_v = M.highpass(voc, SR, 180.0, order=2)
    send_v = M.lowpass(send_v, SR, 9000.0, order=2)
    plate_ret = M.convolve_stereo(send_v, plate) * p.vocal_plate_send
    hall_in = (p.hall_send_vocal * send_v + p.hall_send_guitar * M.highpass(gtr, SR, 200.0)
               + p.hall_send_drums * M.highpass(drm, SR, 300.0))
    hall_ret = M.convolve_stereo(hall_in, hall)
    st["fx_plate"] = plate_ret.astype(np.float32)
    st["fx_hall"] = hall_ret.astype(np.float32)

    bus = sum(st.values())
    if p.stabilize_strength > 0:
        ref = (p.stabilize_ref_start, p.stabilize_ref_end) if p.stabilize_ref_start >= 0 else None
        bus = M.tonal_stabilizer(bus, SR, p.stabilize_strength, p.stabilize_max_db, ref_range=ref, report=rep,
                                 label="bus")
    # glue: slow RMS compression, low ratio
    mono_sc = bus.mean(axis=1)
    e = M.envelope(mono_sc, 30.0, 250.0, mode="rms")
    thr = float(db(np.percentile(e[e > 1e-5], 75)))
    bus, gr = M.compress(bus, thr, p.glue_ratio, 30.0, 250.0, knee_db=10.0, sidechain=mono_sc)
    rep["glue_gr_p95_db"] = round(float(np.percentile(gr, 95)), 2)
    # bass mono below 120 Hz (stereoize already keeps it there; guard anyway)
    mid, side = (bus[:, 0] + bus[:, 1]) / 2, (bus[:, 0] - bus[:, 1]) / 2
    side = M.highpass(side, SR, 120.0, order=2)
    bus = np.stack([mid + side, mid - side], 1).astype(np.float32)

    # mastering EQ: de-box the low mids, a touch of air (minimum-phase biquads, both channels)
    bus = M.eq(bus, [("peak", p.master_lowmid_hz, p.master_lowmid_db, 0.9),
                     ("highshelf", 11000.0, p.master_air_db, 0.7)])

    if p.tone_ref:
        ref_audio, ref_sr = sf.read(p.tone_ref, dtype="float32", always_2d=True)
        if ref_sr != SR:
            ref_audio = signal.resample_poly(ref_audio, SR, ref_sr, axis=0).astype(np.float32)
        bus = M.reference_tone_match(bus, ref_audio, p.tone_ref_strength, p.tone_ref_max_db, report=rep)

    # natural top and tail: short fade-in, 4 s fade-out into the room
    fi, fo = int(0.25 * SR), int(4.0 * SR)
    bus[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
    bus[-fo:] *= (np.cos(np.linspace(0, np.pi / 2, fo, dtype=np.float32)) ** 2)[:, None]

    # pre-limiter bus for external mastering (Ozone 12): float, normalised to -18 LUFS, no limiting
    L = M.lufs(bus)
    sf.write(out_dir / f"{tag}_premaster_f32.wav", (bus * 10 ** ((-18.0 - L) / 20.0)).astype(np.float32), SR,
             subtype="FLOAT")

    # loudness + true-peak limit
    bus = bus * 10 ** ((p.target_lufs - L) / 20.0)
    pre_peak = M.true_peak_db(bus)
    master = M.limiter(bus, p.ceiling_dbtp)
    rep["limiter_input_tp_db"] = round(pre_peak, 2)
    a, b = np.abs(bus).max(axis=1), np.abs(master).max(axis=1)
    sel = a > 10 ** (-20 / 20)
    grs = -db(np.maximum(b[sel], 1e-9) / np.maximum(a[sel], 1e-9))
    rep["limiter_gr_db"] = {"p50": round(float(np.percentile(grs, 50)), 2), "p95": round(float(np.percentile(grs, 95)), 2),
                            "p99.9": round(float(np.percentile(grs, 99.9)), 2),
                            "time_over_3db_pct": round(float(100 * np.mean(grs > 3.0) * sel.mean()), 3)}
    rep["master_lufs"] = round(M.lufs(master), 2)
    rep["master_tp_db"] = round(M.true_peak_db(master), 2)

    # export: 48 kHz / 24-bit master (+ 44.1 kHz), and processed stereo stems for the UI
    m48 = signal.resample_poly(master, 160, 147, axis=0).astype(np.float32)
    peak48 = M.true_peak_db(m48)
    if peak48 > p.ceiling_dbtp:
        m48 *= 10 ** ((p.ceiling_dbtp - peak48 - 0.05) / 20.0)
    sf.write(out_dir / f"{tag}_master_48k24.wav", m48, 48000, subtype="PCM_24")
    sf.write(out_dir / f"{tag}_master_44k24.wav", master, SR, subtype="PCM_24")
    gain = 10 ** ((p.target_lufs - L) / 20.0)
    for k, v in st.items():
        if k.startswith("fx_"):
            continue
        sf.write(out_dir / f"{tag}_stem_{k}.wav", (v * gain).astype(np.float32), SR, subtype="PCM_24")
    (out_dir / f"{tag}_mix_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def load_stems(stem_dir: Path) -> dict[str, np.ndarray]:
    s = {
        "vocals_dry": _load(stem_dir / "vocals_noreverb.wav"),
        "vocals_room": _load(stem_dir / "vocals_reverb.wav"),
        "guitar": _load(stem_dir / "sw_guitar.wav"),
        "bass": _load(stem_dir / "sw_bass.wav"),
        "drums": _load(stem_dir / "sw_drums.wav"),
        "piano": _load(stem_dir / "sw_piano.wav"),
        "other": _load(stem_dir / "sw_other.wav"),
    }
    for k in ("guitar", "piano", "other", "drums"):
        f = stem_dir / f"swB_{k}.wav"
        s[f"B_{k}"] = _load(f) if f.exists() else None
    # vocal residue the SW model found in the instrumental is mostly backing vox / bleed: fold into other
    resid = stem_dir / "sw_vocals.wav"
    if resid.exists():
        r = _load(resid)
        s["other"] = _fit(s["other"], len(r)) + 0.5 * r
    return s
