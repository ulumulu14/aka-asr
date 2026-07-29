"""Scout CC-BY (yt_cc_scout): parser ISO8601 i ranking kanalow. Bez sieci/API."""

from utils.yt_cc_scout import aggregate_channels, parse_iso_duration


def test_parse_iso_duration():
    assert parse_iso_duration("PT1H2M3S") == 3723
    assert parse_iso_duration("PT45M") == 2700
    assert parse_iso_duration("PT30S") == 30
    assert parse_iso_duration("") == 0
    assert parse_iso_duration("bzdura") == 0


def test_aggregate_channels_ranking():
    vids = [
        {"channel_id": "A", "channel": "Kanal A", "title": "x1", "duration_s": 3600},
        {"channel_id": "B", "channel": "Kanal B", "title": "y1", "duration_s": 600},
        {"channel_id": "A", "channel": "Kanal A", "title": "x2", "duration_s": 1800},
    ]
    out = aggregate_channels(vids)
    assert [c["channel_id"] for c in out] == ["A", "B"]  # sort po godzinach malejaco
    assert out[0]["n"] == 2 and out[0]["sec"] == 5400
    assert out[0]["titles"] == ["x1", "x2"]
