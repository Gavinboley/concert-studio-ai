"""Tests for Setlist.fm API Connection & Automated Song Timeline Segmentation."""

from __future__ import annotations

import json
import pytest

from concert_studio.setlist_fm import ConcertSetlist, SetlistFmConnector, SetlistSong


def test_setlist_parsing_and_alignment():
    # Mock payload from setlist.fm
    mock_payload = {
        "id": "abc1234",
        "eventDate": "26-07-2025",
        "artist": {"name": "Oasis"},
        "venue": {"name": "Wembley Stadium", "city": {"name": "London"}},
        "tour": {"name": "Live '25 Tour"},
        "sets": {
            "set": [
                {
                    "song": [
                        {"name": "Hello"},
                        {"name": "Acquiesce"},
                        {"name": "Supersonic"},
                    ]
                },
                {
                    "encore": 1,
                    "song": [
                        {"name": "Don't Look Back in Anger"},
                        {"name": "Wonderwall"},
                    ],
                },
            ]
        },
    }

    setlist = SetlistFmConnector._parse_setlist_payload(mock_payload)
    assert setlist.artist == "Oasis"
    assert setlist.venue == "Wembley Stadium"
    assert len(setlist.songs) == 5
    assert setlist.songs[3].is_encore is True
    assert setlist.songs[4].name == "Wonderwall"

    # Align to a 600-second timeline
    aligned = SetlistFmConnector.align_setlist_to_timeline(setlist, timeline_duration_sec=600.0)
    assert aligned.songs[0].start_sec == 0.0
    assert aligned.songs[0].end_sec == 120.0
    assert aligned.songs[-1].end_sec == 600.0

    # Generate CUE sheet
    cue = SetlistFmConnector.generate_cue_sheet(aligned, "studio_hd_master.wav")
    assert 'PERFORMER "Oasis"' in cue
    assert 'TRACK 01 AUDIO' in cue
    assert 'TITLE "Wonderwall"' in cue
    assert "INDEX 01" in cue
