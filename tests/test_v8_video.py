"""V8 multi-cam director: planning respects coverage, trim and cut rules; timing math is exact."""

import numpy as np
import pytest

from concert_studio.v8 import video as V


def _cam(label, usable, dur, offset=0.0, closeness="mid", sharp=1.0, w=1080, h=1920):
    kt = list(np.arange(0.0, dur, 2.0))
    return {"label": label, "video": f"/x/{label}.mp4", "usable": usable, "dur": dur, "offset_s": offset,
            "drift_ppm": 0.0, "av_offset_s": 0.0, "trim_start_s": 0.0, "closeness": closeness, "w": w, "h": h,
            "kf": {"t": kt, "sharp": [sharp] * len(kt), "luma": [110.0] * len(kt)}, "heard": []}


def _roster():
    downs = list(np.arange(0.0, 120.0, 2.0))
    cams = {
        "a": _cam("a", [[1.0, 60.0]], 70.0, offset=0.0, closeness="close"),
        "b": _cam("b", [[0.5, 119.0]], 125.0, offset=0.0, closeness="wide"),
        "c": _cam("c", [[40.0, 118.0]], 90.0, offset=35.0, closeness="mid", sharp=1.5),
        "d": _cam("d", [[0.5, 30.0], [70.0, 119.0]], 125.0, offset=0.0, closeness="close"),
    }
    return {"cams": cams, "music": {"duration": 120.0, "downbeats": downs, "vocal_entries": [21.0, 61.3],
                                    "energy_10hz": list(np.r_[np.zeros(600), np.ones(600)])}}


def test_plan_is_contiguous_covered_and_alternates():
    r = _roster()
    shots = V.plan_shots(r, seed=3)
    assert shots[0]["start_sec"] == 0.0 and shots[-1]["end_sec"] == 120.0   # 0-0.5 s uncovered: lead-in hold
    assert shots[0]["lead_in_sec"] == pytest.approx(0.5)
    for a, b in zip(shots, shots[1:]):
        assert a["end_sec"] == pytest.approx(b["start_sec"], abs=1e-3)     # no holes
        assert a["camera"] != b["camera"]                                   # merged: never the same camera twice
    for s in shots:
        cam = r["cams"][s["camera"]]
        if s["switch_reason"] != "unvalidated_hold":
            s0 = s["start_sec"] + s.get("lead_in_sec", 0.0)      # validated footage (a merged hold may follow)
            assert V._covers(cam, s0, min(s["end_sec"], s0 + V.MIN_SHOT))
        assert s["src_in_sec"] == pytest.approx(float(V.master_to_container(cam, s["start_sec"])), abs=1e-3)


def test_plan_trim_and_seed_variation():
    r = _roster()
    shots = V.plan_shots(r, start=30.0, end=75.0, seed=0)
    assert shots[0]["start_sec"] == 30.0 and shots[-1]["end_sec"] == 75.0
    assert all(30.0 <= s["start_sec"] < s["end_sec"] <= 75.0 for s in shots)
    with pytest.raises(ValueError):
        V.plan_shots(r, start=50.0, end=50.5)
    plans = {tuple(s["camera"] for s in V.plan_shots(r, seed=k)) for k in range(6)}
    assert len(plans) > 1                                                   # re-direct gives other cuts


def test_master_to_container_maps_trim_offset_drift_and_av():
    cam = {"trim_start_s": 9.0, "offset_s": 4.0, "drift_ppm": 100.0, "av_offset_s": -0.025}
    t = np.array([0.0, 100.0])
    want = (t + 9.0 - 4.0) / (1 + 100e-6) - 0.025
    assert np.allclose(V.master_to_container(cam, t), want)


def test_edl_uses_real_source_timecode():
    r = _roster()
    shots = V.plan_shots(r, start=40.0, end=60.0, seed=1)
    edl = V.export_edl(shots, r)
    first = [l for l in edl.splitlines() if l.startswith("001")][0].split()
    src_in, rec_in = first[4], first[6]
    assert rec_in == "00:00:40:00"
    cam = r["cams"][shots[0]["camera"]]
    f = int(round(float(V.master_to_container(cam, 40.0)) * V.FPS))
    assert src_in == f"00:00:{f // V.FPS:02d}:{f % V.FPS:02d}"


def test_single_camera_plan_is_one_continuous_shot():
    r = _roster()
    r["cams"] = {"b": r["cams"]["b"]}
    shots = V.plan_shots(r, seed=0)
    assert len(shots) == 1 and shots[0]["start_sec"] == 0.0 and shots[0]["end_sec"] == 120.0
