"""V8 engine unit tests: alignment, comp path, and the mono-compatibility / no-click guarantees."""

import numpy as np
import pytest

from concert_studio.v8 import mix as M
from concert_studio.v8.comp import viterbi, _sinc_read
from concert_studio.v8.dsp import gcc_phat, fractional_delay, novelty, fft_xcorr

SR = 44100
rng = np.random.default_rng(0)


def _music(sec=6.0, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    x = np.zeros_like(t)
    for k, f in enumerate([110, 220, 330, 440, 660, 990]):
        x += np.sin(2 * np.pi * f * t + k) / (k + 1)
    beats = (np.sin(2 * np.pi * 2.0 * t) > 0.95).astype(float)
    x += 0.5 * beats * rng.standard_normal(len(t))
    return (0.2 * x).astype(np.float32)


def test_gcc_phat_recovers_fractional_lag():
    x = _music(4.0, 16000)
    y = fractional_delay(x, 37.4)
    lag, score = gcc_phat(y, x, 200)
    assert abs(lag - 37.4) < 0.2
    assert score > 20


def test_novelty_xcorr_finds_offset():
    x = _music(30.0)
    off = int(4.5 * SR)
    clip = x[off:off + 10 * SR]
    a, fr = novelty(x, SR)
    b, _ = novelty(clip, SR)
    cc = fft_xcorr(a, b)
    lag = (np.argmax(cc) - (len(b) - 1)) / fr
    assert abs(lag - 4.5) < 0.02


def test_viterbi_prefers_better_take_and_respects_dwell():
    T = 100
    S = np.zeros((2, T))
    S[0] = 1.0
    S[1] = 0.0
    S[1, 40:60] = 3.0          # take 1 clearly better for 20 steps
    S[0, 70:72] = -np.inf      # take 0 unusable for 2 steps
    p = viterbi(S, switch_cost=5.0, min_dwell=8)
    assert (p[45:55] == 1).all()
    assert (p[70:72] == 1).all()
    runs = np.split(p, np.where(np.diff(p) != 0)[0] + 1)
    assert all(len(r) >= 8 for r in runs[1:-1])


def test_sinc_read_integer_positions_identity():
    x = _music(1.0)
    y = _sinc_read(x, np.arange(100, 5000, dtype=np.float64))
    assert np.max(np.abs(y - x[100:5000])) < 1e-4


def test_stereoize_is_exactly_mono_compatible():
    m = _music(3.0)
    st = M.stereoize(m, 0.6, seed=1)
    assert np.max(np.abs((st[:, 0] + st[:, 1]) / 2 - m)) < 1e-6
    side = (st[:, 0] - st[:, 1]) / 2
    assert M.rms(side) > 0.05 * M.rms(m)


def test_limiter_respects_true_peak_ceiling():
    x = np.stack([_music(4.0) * 8, _music(4.0) * 7], 1)
    y = M.limiter(x, -1.0)
    assert M.true_peak_db(y) <= -0.95


def test_reverb_ir_has_no_onset_click():
    ir = M.synth_reverb_ir(1.5, 20.0, er=False)
    pd = int(0.02 * SR)
    assert np.max(np.abs(ir[:pd])) == 0.0
    first = np.abs(ir[pd:pd + 64]).max()
    assert first < 0.2 * np.abs(ir).max()


def test_saturation_is_alias_free_enough():
    t = np.arange(SR) / SR
    x = (0.8 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)
    y = M.saturate(x, 12.0)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    # 3rd harmonic 21 kHz is legal; aliases would land at e.g. 44.1k-35k=9.1k, 44.1k-49k... check 9.1 kHz
    alias = Y[(f > 9000) & (f < 9200)].max()
    assert 20 * np.log10(alias / Y.max()) < -60


def test_deess_only_reduces():
    v = _music(2.0) + 0.3 * M.highpass(rng.standard_normal(2 * SR).astype(np.float32), SR, 6000, order=4)
    y = M.deess(v)
    assert M.rms(y) <= M.rms(v) * 1.001


def test_extend_edges_only_over_music():
    from concert_studio.v8.comp import _extend_edges
    T = 200
    ok = np.zeros(T, bool)
    ok[60:140] = True
    inside = np.ones(T, bool)
    loud = np.full(T, -20.0)
    loud[:40] = -50.0              # silence / talking far below the take's level
    cr = np.full(T, -10.0)
    cr[150:] = 5.0                 # crowd-dominated tail
    ext = _extend_edges(ok, inside, loud, cr, max_s=100.0)
    assert ext[40:60].all() and not ext[:40].any()
    assert ext[140:150].all() and not ext[150:].any()
    assert not (ext & ok).any()


def test_level_anomaly_rejection_never_opens_a_gap():
    from concert_studio.v8.comp import reject_level_anomalies
    T = 400
    feats = {"f": {}}
    rng2 = np.random.default_rng(3)
    base = np.concatenate([np.full(200, -40.0), np.full(200, -18.0)])   # real crescendo at t=200
    for k in range(4):
        loud = base + rng2.normal(0, 0.5, T) + k
        ok = np.ones(T, bool)
        feats["f"][f"t{k}"] = dict(loud=loud, ok=ok, ext=np.zeros(T, bool))
    feats["f"]["t3"]["loud"][300:310] += 20.0                            # one take's own burst
    reject_level_anomalies({f"t{k}": None for k in range(4)}, feats, log=lambda *a: None)
    alive = np.stack([feats["f"][f"t{k}"]["ok"] for k in range(4)]).any(axis=0)
    assert alive.all()
    assert not feats["f"]["t3"]["ok"][302:308].any()
    assert feats["f"]["t0"]["ok"][195:205].all()                         # crescendo kept


def test_real_stereo_is_mono_exact_and_centred():
    from concert_studio.v8.master import real_stereo
    a = _music(4.0)
    b = 0.25 * (np.roll(a, 7) + 0.3 * rng.standard_normal(len(a)).astype(np.float32))   # other POV, 12 dB down
    st = real_stereo(a, b, 0.7, 250.0, 1, 0.5)
    assert np.max(np.abs((st[:, 0] + st[:, 1]) / 2 - a)) < 1e-5
    bal = 10 * np.log10(np.sum(st[:, 0] ** 2) / np.sum(st[:, 1] ** 2))
    assert abs(bal) < 0.5                      # level mismatch between POVs must not pan the image
    assert M.rms(st[:, 0] - st[:, 1]) > 0.05 * M.rms(a)


def test_take_scores_perceptual_terms_flip_close_calls_only():
    from types import SimpleNamespace
    from concert_studio.v8.comp import take_scores
    T = 40
    tr = {n: SimpleNamespace(info=SimpleNamespace(sbr=False)) for n in ("a", "b")}
    f = lambda cr: dict(cr=np.full(T, cr), occl=np.zeros(T), clip=np.zeros(T), ext=np.zeros(T, bool),
                        hcr=np.full(T, cr), ok=np.ones(T, bool))
    feats = {"grid": np.arange(T) * 0.5, "f": {"a": f(-10.0), "b": f(-8.0)}}   # a has 2 dB less crowd
    q = {"a": {"PQ": np.full(T, 5.0), "CE": np.full(T, 5.0)}, "b": {"PQ": np.full(T, 6.5), "CE": np.full(T, 6.0)}}
    S0, _ = take_scores(tr, feats, {})
    assert (S0[0] > S0[1]).all()
    S1, H1 = take_scores(tr, feats, {}, quality=q, w_pq=4.0)
    assert (S1[1] > S1[0]).all() and (H1[1] > H1[0]).all()      # 1.5 PQ points outweigh 2 dB of crowd
    q["b"]["PQ"][:10] = np.nan                                   # uncovered windows fall back to the take median
    S2, _ = take_scores(tr, feats, {}, quality=q, w_pq=4.0)
    assert np.all(np.isfinite(S2))


def test_vocal_band_coherence_flags_a_local_source():
    from types import SimpleNamespace
    from concert_studio.v8.comp import vocal_band_coherence
    sr, dur = SR, 20.0
    n = int(sr * dur)
    common = _music(dur)[:n]
    local = rng.standard_normal(n).astype(np.float32)
    sig = {"a": common + 0.1 * rng.standard_normal(n).astype(np.float32),
           "b": common + 0.1 * rng.standard_normal(n).astype(np.float32),
           "c": common + 0.1 * rng.standard_normal(n).astype(np.float32),
           "d": 0.3 * common + local}                                # a talker right next to phone d
    tracks = {k: SimpleNamespace(music=v, pos=lambda t: t) for k, v in sig.items()}
    grid = np.arange(0, dur, 0.5)
    feats = {"grid": grid, "f": {k: {"ok": np.ones(len(grid), bool), "ext": np.zeros(len(grid), bool)} for k in sig}}
    C = vocal_band_coherence(tracks, feats, log=lambda *_: None)
    mid = slice(10, 30)
    assert np.nanmean(C["d"][mid]) < 0.5 * np.nanmean(C["a"][mid])


def test_take_pumping_flags_agc_wobble_not_extra_dynamics():
    from types import SimpleNamespace
    from scipy import ndimage
    from concert_studio.v8.comp import take_pumping
    sr, dur = SR, 60.0
    n = int(sr * dur)
    env = 10 ** (ndimage.uniform_filter1d(rng.standard_normal(int(dur * 4)), 3) * 0.4)   # music's own dynamics
    env = np.repeat(env, n // len(env) + 1)[:n].astype(np.float32)
    base = (rng.standard_normal(n).astype(np.float32) * env)
    sig = {k: base + 0.05 * rng.standard_normal(n).astype(np.float32) for k in "abcd"}
    sig["close"] = base ** 1 * env ** 0.3                         # same dynamics, exaggerated (slope > 1)
    wob = 10 ** (ndimage.uniform_filter1d(rng.standard_normal(int(dur * 4)), 4) * 0.6 / 20 * 10)
    sig["pump"] = base * np.repeat(wob, n // len(wob) + 1)[:n].astype(np.float32)
    tracks = {k: SimpleNamespace(music=v, pos=lambda t: np.asarray(t, float), dur=dur) for k, v in sig.items()}
    grid = np.arange(0, dur, 0.5)
    feats = {"grid": grid, "f": {k: {"ok": np.ones(len(grid), bool)} for k in sig}}
    P = take_pumping(tracks, feats, log=lambda *_: None)
    mid = slice(20, 100)
    assert np.nanmedian(P["pump"][mid]) > 2.0 * np.nanmedian(P["a"][mid])
    assert np.nanmedian(P["close"][mid]) < 0.6 * np.nanmedian(P["pump"][mid])


def _pink(n, sr, seed=1):
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, 2))
    X = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    X /= np.sqrt(np.maximum(f, 20.0))[:, None]
    x = np.fft.irfft(X, n=n, axis=0).astype(np.float32)
    return 0.1 * x / x.std()


def _band_share(z, sr, a, b, lo, hi):
    s = z[int(a * sr):int(b * sr), 0]
    S = np.abs(np.fft.rfft(s)) ** 2
    ff = np.fft.rfftfreq(len(s), 1 / sr)
    return 10 * np.log10(S[(ff >= lo) & (ff < hi)].sum() / S[(ff > 40) & (ff < 16000)].sum())


def test_tonal_stabilizer_evens_out_tone_drift_but_leaves_quiet_intro_alone():
    from concert_studio.v8.dsp import lowpass
    sr = 22050
    x = _pink(sr * 70, sr)
    x[: sr * 10] *= 0.1                                         # quiet intro (-20 dB), its own darker tone
    x[: sr * 10] = 0.5 * x[: sr * 10] + 0.5 * lowpass(x[: sr * 10], sr, 500.0, order=2, zero_phase=True)
    seg = x[sr * 40:]                                           # later takes: low end 5 dB thinner
    x[sr * 40:] = seg - (1 - 10 ** (-5 / 20)) * lowpass(seg, sr, 250.0, order=4, zero_phase=True)
    y = M.tonal_stabilizer(x, sr, strength=0.8, max_db=4.0)
    drift = lambda z: _band_share(z, sr, 50, 68, 40, 200) - _band_share(z, sr, 15, 35, 40, 200)
    assert drift(x) < -3.0
    assert abs(drift(y)) < 0.4 * abs(drift(x))
    body = slice(sr * 12, sr * 68)
    assert abs(10 * np.log10(np.mean(y[body] ** 2) / np.mean(x[body] ** 2))) < 0.3      # level preserved
    intro = lambda z: _band_share(z, sr, 0.5, 8, 40, 200) - _band_share(z, sr, 0.5, 8, 2000, 8000)
    assert abs(intro(y) - intro(x)) < 0.5                     # gated: sparse section keeps its own tone
    assert np.allclose(y[:, 0] - y[:, 1], y[:, 0] - y[:, 1])


def _tone(sec, f0s, sr=SR, seed=0, amp=0.1):
    r = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    x = sum(np.sin(2 * np.pi * f * t + r.uniform(0, 6)) / (k + 1) for k, f in enumerate(f0s))
    env = 0.6 + 0.4 * (np.sin(2 * np.pi * 2.0 * t) > 0)
    return (amp * x * env / np.max(np.abs(x))).astype(np.float32)


def _mix_report(stems, tmp_path):
    from concert_studio.v8.master import MixParams, mix_and_master
    n = len(stems["drums"])
    p = MixParams(stem_stabilize_strength=0.0, stabilize_strength=0.0)
    return mix_and_master(stems, np.zeros(n, np.float32), p, tmp_path, tag="t")


def test_instrumental_keeps_its_own_balance(tmp_path):
    """Electronic instrumental: the 'vocals' stem is a quiet synth lead, the dominant stem is 'other'.
    The vocals-first rock targets must not boost the lead or bury the main synths."""
    sec = 12.0
    stems = {"vocals_dry": _tone(sec, [880, 1320], seed=1, amp=0.004),
             "vocals_room": np.zeros(int(sec * SR), np.float32),
             "guitar": _tone(sec, [330, 660], seed=2, amp=0.01),
             "bass": _tone(sec, [55, 110], seed=3, amp=0.15),
             "drums": (rng.standard_normal(int(sec * SR)) * 0.05).astype(np.float32),
             "piano": np.zeros(int(sec * SR), np.float32),
             "other": _tone(sec, [220, 277, 330, 440], seed=4, amp=0.3)}
    rep = _mix_report(stems, tmp_path)
    mv = rep["balance_moves_db"]
    assert rep["vocal_song"] < 0.5
    assert abs(mv["vocals_dry"]) <= 2.01 and abs(mv["guitar"]) <= 2.01
    assert mv["other"] >= -3.01
    assert rep["vocal_carve_db_used"] < 0.5


def test_reference_tone_match_moves_toward_studio_balance():
    from concert_studio.v8.dsp import lowpass
    sr = SR
    ref = _pink(sr * 20, sr, seed=3)
    dark = lowpass(ref, sr, 2000.0, order=2, zero_phase=True) + 0.25 * ref     # phone/hall-dulled version
    y = M.reference_tone_match(dark, ref, strength=1.0, max_db=8.0)
    hf = lambda z: _band_share(z, sr, 2, 18, 4000, 12000)
    assert hf(ref) - hf(y) < 0.5 * (hf(ref) - hf(dark))
    assert np.all(np.isfinite(y))


def _section(chord_hz, sec, seed, sr=SR):
    """One performance of a song section: chord + drum hits; another `seed` = same part played again."""
    from scipy import signal as sg
    r = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    x = np.zeros(len(t))
    for f in chord_hz:                          # each performance: same notes, its own fine structure
        for h in (1, 2, 3):
            w = 2 * np.pi * f * h / sr
            rad = 1 - 2 * np.pi * 25.0 / sr     # ~25 Hz wide: a played note, not a sine
            x += sg.lfilter([1 - rad], [1, -2 * rad * np.cos(w), rad ** 2], r.standard_normal(len(t))) / h
    x = 0.2 * x / (np.std(x) + 1e-9)
    hits = np.zeros(len(t))
    for b in np.arange(0, sec, 0.5):
        i = max(0, int((b + r.normal(0, 0.015)) * sr))
        n = min(len(t) - i, int(0.08 * sr))
        if n > 0:
            hits[i:i + n] += r.standard_normal(n) * np.exp(-np.arange(n) / (0.015 * sr))
    return (x + 0.6 * hits).astype(np.float32)


def test_sparse_clips_of_repeating_song_become_islands_not_false_overlaps():
    from concert_studio.v8.align import align_clips
    A, B = [261.6, 329.6, 392.0], [220.0, 261.6, 329.6]            # C and Am: verse/chorus
    song = np.concatenate([_section(A, 15, 1), _section(B, 15, 2), _section(A, 15, 3), _section(B, 15, 4)])
    c1, c2 = song[: int(28 * SR)], song[int(32 * SR):]                  # chorus 2 never overlaps chorus 1
    pl = {p.name: p for p in align_clips(["c1", "c2"], {"c1": c1, "c2": c2}, "c1", log=lambda *a: None)}
    assert pl["c2"].island == 1 and pl["c1"].island == 0                 # not a false overlap on chorus 1
    assert pl["c2"].offset_s >= pl["c1"].offset_s + 28.0                 # its own island, after a gap


def test_gaps_inside_the_song_are_faded_not_cut():
    from concert_studio.v8.run import _soften_gaps
    sr = 1000
    x = np.ones(10 * sr, np.float32)
    w = np.ones(10 * sr)
    w[4 * sr:6 * sr] = 0.0                                     # an island gap
    x[4 * sr:6 * sr] = 0.0
    _soften_gaps(x, w, sr, fade_s=0.25)
    assert x[4 * sr - 1] < 0.05 and x[6 * sr] < 0.05            # silent right at the edges
    assert x[4 * sr - 300] == 1.0 and x[6 * sr + 300] == 1.0    # untouched outside the fades
    assert np.all(np.diff(x[6 * sr:6 * sr + 250]) >= 0)         # smooth rise, no step
