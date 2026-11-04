"""V8 cut: layouts, framing geometry, band-aware planning, transitions and colour consensus."""

import numpy as np
import pytest

from concert_studio.v8 import cut as C


def _cam(label, usable, dur=130.0, w=720, h=1280, offset=0.0):
    kt = list(np.arange(0.0, dur, 2.0))
    return {"label": label, "video": f"/x/{label}.mp4", "usable": usable, "dur": dur, "offset_s": offset,
            "drift_ppm": 0.0, "av_offset_s": 0.0, "trim_start_s": 0.0, "closeness": "mid", "w": w, "h": h,
            "crop": None, "kf": {"t": kt, "sharp": [1.0] * len(kt), "luma": [110.0] * len(kt)}, "heard": []}


def _samples(dur, persons, sharp=1.0):
    return {"samples": [{"t": t, "rgb": [0.1, 0.1, 0.2], "luma": [0.02, 0.2, 0.7], "sharp": sharp,
                         "persons": persons, "n_small": 0} for t in np.arange(0.75, dur, 1.5)], "overlay": None}


SINGER = [0.3, 0.15, 0.7, 0.95, 0.99, "vocals", 0.95, 0.0]
GUITAR = [0.25, 0.2, 0.75, 0.95, 0.99, "guitar", 0.9, 0.0]
WIDE = [[x, 0.6, x + 0.06, 0.75, 0.9, r, 0.6, 0.0] for x, r in ((0.2, "guitar"), (0.45, "vocals"), (0.7, "bass"))]


def _band():
    cams = {"sing": _cam("@sing", [[0.5, 125.0]]), "gtr": _cam("@gtr", [[0.5, 125.0]]),
            "wide": _cam("@wide", [[0.5, 125.0]], w=1280, h=720), "wide2": _cam("@wide2", [[0.5, 125.0]])}
    vis = {"sing": _samples(130, [SINGER]), "gtr": _samples(130, [GUITAR]), "wide": _samples(130, WIDE),
           "wide2": _samples(130, WIDE, sharp=0.8)}
    downs = list(np.arange(0.0, 120.0, 2.4))
    music = {"duration": 120.0, "downbeats": downs, "vocal_entries": [12.0],
             "energy_10hz": list(np.r_[np.full(600, -30.0), np.full(600, -12.0)])}
    hop = 0.5
    feat = ["vocals"] * int(60 / hop) + ["guitar"] * int(30 / hop) + ["vocals"] * int(30 / hop)
    ft = {"hop_s": hop, "featured": feat,
          "activity": {"vocals": [1.0 if f == "vocals" else 0.0 for f in feat], "guitar": [0.8] * len(feat)}}
    return {"cams": cams, "music": music}, vis, ft


def test_layouts_tile_the_frame_without_overlap():
    for fmt, (W, H) in (("vertical", (1080, 1920)), ("landscape", (2560, 1440))):
        for n in (1, 2, 3, 4):
            rects = C.layout(fmt, n, W, H)
            assert len(rects) == n
            area = sum(w * h for _, _, w, h in rects)
            assert 0.97 * W * H <= area <= W * H                 # gutters only
            for i, a in enumerate(rects):
                assert a[0] >= 0 and a[1] >= 0 and a[0] + a[2] <= W and a[1] + a[3] <= H
                assert a[2] % 2 == 0 and a[3] % 2 == 0
                for b in rects[i + 1:]:
                    assert a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]
    x, y, w, h = C.layout("landscape", 3, 2560, 1440)[0]
    assert x > 0 and x + w < 2560 and abs(w / h - 9 / 16) < 0.05      # hero is the centre column, phone-shaped


def test_fit_caps_upscale_and_reports_fill():
    w, h, s, fill = C.fit(720, 1280, 1080, 1920, 2.7)
    assert (round(w), round(h), s, fill) == (720, 1280, 1.5, 1.0)
    w, h, s, fill = C.fit(720, 1280, 2560, 1440, 2.7)                   # a phone alone in a 16:9 frame
    assert s == 2.7 and round(w) == 720 and fill < 0.8              # crop to the middle, blurred sides
    w, h, s, fill = C.fit(1280, 720, 853, 1440, 2.7)
    assert fill == pytest.approx(1.0) and h == 720


def test_guitar_feature_puts_the_guitarist_in_the_hero_panel():
    roster, vis, ft = _band()
    for multi in (False, True):
        st = C.Style(fmt="landscape", multi=multi, min_cams=1, max_cams=3).norm()
        d = C.Director(roster, vis, ft, st, 2560, 1440)
        runs = d.plan(0.0, 120.0)
        assert runs[0]["t0"] == 0.0 and max(r["t1"] for r in runs) == 120.0
        hero_solo = [r for r in runs if r["slot"] == 0 and r["t0"] >= 61 and r["t1"] <= 89]
        assert hero_solo and (not multi or all(r["want"] == "strings" for r in hero_solo))
        on_gtr = sum(r["t1"] - r["t0"] for r in hero_solo if r["cam"] == "gtr")
        assert on_gtr >= (0.5 if multi else 0.25) * sum(r["t1"] - r["t0"] for r in hero_solo)
        assert multi or all(r["cam"] in ("gtr", "wide", "wide2") for r in hero_solo if r["want"] == "strings")
        sung = [r for r in runs if r["slot"] == 0 and r["t1"] <= 58 and r["want"] == "vocals"]
        assert any(r["cam"] == "sing" for r in sung)
        # every instant is covered by exactly the layout's panels, never the same phone twice
        for t in np.arange(0.25, 120.0, 0.5):
            on = [r for r in runs if r["t0"] <= t < r["t1"]]
            assert on and len({r["cam"] for r in on}) == len(on)
            assert len({r["slot"] for r in on}) == len(on)
        if not multi:
            assert all(r["slot"] == 0 for r in runs)


def test_multi_respects_min_and_max_cams():
    roster, vis, ft = _band()
    st = C.Style(fmt="vertical", multi=True, min_cams=2, max_cams=3).norm()
    runs = C.Director(roster, vis, ft, st, 1080, 1920).plan(0.0, 120.0)
    for t in np.arange(0.25, 120.0, 0.5):
        n = len([r for r in runs if r["t0"] <= t < r["t1"]])
        assert 2 <= n <= 3


def test_framing_zooms_toward_the_wanted_member():
    roster, vis, ft = _band()
    st = C.Style(fmt="vertical").norm()
    run = {"cam": "wide", "rect": [0, 0, 1080, 1920], "t0": 10.0, "t1": 20.0, "want": "vocals", "slot": 0}
    fr = C.frame_run(run, roster["cams"]["wide"], vis, st)
    w, h, x, y = fr["win"]
    assert x <= 0.45 * 1280 and x + w >= 0.51 * 1280                   # the singer's column is in frame
    assert fr["out"][0] <= 1080 and fr["out"][1] <= 1920


def test_cinematic_dissolves_only_at_quiet_layout_changes():
    runs = [{"id": 0, "slot": 0, "t0": 0.0, "t1": 10.0, "why": "phrase"},
            {"id": 1, "slot": 0, "t0": 10.0, "t1": 20.0, "why": "phrase"},
            {"id": 2, "slot": 1, "t0": 10.0, "t1": 20.0, "why": "phrase"}]
    st = C.Style(cinematic=True)
    p = C.pieces(runs, st, 0.0, 20.0)
    assert [len(x["layers"]) for x in p] == [1, 2, 1]
    assert p[1]["t0"] == pytest.approx(10.0 - C.DISSOLVE / 2) and p[1]["t1"] == pytest.approx(10.0 + C.DISSOLVE / 2)
    assert sum(x["t1"] - x["t0"] for x in p) == pytest.approx(20.0)
    assert len(C.pieces(runs, C.Style(), 0.0, 20.0)) == 2                # standard: hard cuts


def test_colour_consensus_pulls_an_odd_phone_toward_the_others():
    cams = {k: _cam(k, [[0, 100]]) for k in "abcd"}
    vis = {k: {"samples": [{"t": t, "rgb": [0.1, 0.1, 0.1], "luma": [0.0, 0.3, 0.8], "sharp": 1, "persons": []}
                           for t in np.arange(1, 90, 1.5)]} for k in "abc"}
    vis["d"] = {"samples": [{"t": t, "rgb": [0.2, 0.1, 0.05], "luma": [0.0, 0.15, 0.8], "sharp": 1, "persons": []}
                            for t in np.arange(1, 90, 1.5)]}
    corr = C.color_corrections({"cams": cams}, vis)
    r, g, b = corr["d"]["gains"]
    assert r < 1.0 < b                                               # too warm -> cooled
    assert corr["d"]["mid"][1] > corr["d"]["mid"][0]                  # too dark -> lifted
    assert abs(corr["a"]["gains"][0] - 1.0) < 0.02


def test_highlight_prefers_the_loud_section():
    _, _, _ = _band()
    roster, _, _ = _band()
    a, b = C.highlight(roster["music"], 30.0, roster)
    assert a >= 55.0 and b - a == pytest.approx(30.0)


def test_caption_bands_are_framed_out_when_the_zoom_allows():
    bands = [[0.1, 0.15, 0.9, 0.19], [0.2, 0.83, 0.8, 0.87]]
    free = C.caption_free(bands, 1024.0, 400.0)
    assert free == (pytest.approx(0.19 * 1024), pytest.approx(0.83 * 1024))
    assert C.caption_free(bands, 1024.0, 800.0) is None
    cam = {"label": "@x", "w": 576, "h": 1024, "crop": None, "trim_start_s": 0.0, "offset_s": 0.0, "drift_ppm": 0.0,
           "av_offset_s": 0.0}
    vis = {"x": {"samples": [{"t": t, "persons": [SINGER], "rgb": [0.1] * 3, "luma": [0, .2, .7], "sharp": 1}
                             for t in np.arange(0.75, 30, 1.5)], "captions": bands}}
    run = {"cam": "x", "rect": [0, 0, 540, 960], "t0": 2.0, "t1": 8.0, "want": "vocals", "slot": 0}
    fr = C.frame_run(run, cam, vis, C.Style(fmt="vertical").norm())
    assert not fr["caption_visible"]
