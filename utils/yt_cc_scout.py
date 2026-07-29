#!/usr/bin/env python3
"""
FAZA 2b: scout kandydatow CC-BY z YouTube (Data API v3) -- automat robi discovery
i weryfikacje licencji, CZLOWIEK zatwierdza kanaly (flaga CC to deklaracja
uploadera, nie weryfikacja praw -- klasyczny falszywy pozytyw: reupload cudzego
programu oznaczony CC; ostatnie oko musi byc ludzkie).

    export YT_API_KEY=...   # darmowy klucz: console.cloud.google.com -> YouTube Data API v3
    python utils/yt_cc_scout.py --queries "podcast" "wywiad" "rozmowa" \
        --pages 2 --min-duration 600 --out data/yt_cc_candidates.jsonl

Wyjscie: ranking KANALOW (godziny CC, liczba filmow, przykladowe tytuly) na stdout
+ JSONL per film (channel_id, video_id, url, tytul, sekundy) do dalszej obrobki.
Po Twoim przegladzie: harvest zatwierdzonych kanalow przez yt-dlp z bramka CC-BY
(DATA_SCALING FAZA 2b) -- licencja jest wtedy sprawdzana DRUGI raz przy pobieraniu.

Kwoty API: search=100 jedn./strone (limit dzienny 10k => ~100 stron), videos=1 jedn./50.
Bez nowych zaleznosci (urllib).
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://www.googleapis.com/youtube/v3"
_DUR = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parse_iso_duration(s: str) -> int:
    """ISO8601 (PT1H2M3S) -> sekundy; nieparsowalne -> 0."""
    m = _DUR.fullmatch(s or "")
    if not m:
        return 0
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + se


def aggregate_channels(videos: list[dict]) -> list[dict]:
    """Filmy -> ranking kanalow (suma godzin CC malejaco). Kuracja idzie per KANAL."""
    per: dict[str, dict] = defaultdict(lambda: {"n": 0, "sec": 0, "titles": []})
    for v in videos:
        c = per[v["channel_id"]]
        c["channel"] = v["channel"]
        c["n"] += 1
        c["sec"] += v["duration_s"]
        if len(c["titles"]) < 3:
            c["titles"].append(v["title"][:60])
    out = [{"channel_id": k, **v} for k, v in per.items()]
    return sorted(out, key=lambda c: -c["sec"])


def _get(path: str, **params) -> dict:
    params["key"] = os.environ["YT_API_KEY"]
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def search_cc(query: str, pages: int) -> list[str]:
    """search.list z videoLicense=creativeCommon -> lista videoId."""
    ids, token = [], None
    for _ in range(pages):
        params = dict(part="id", type="video", videoLicense="creativeCommon",
                      relevanceLanguage="pl", maxResults=50, q=query)
        if token:
            params["pageToken"] = token
        d = _get("search", **params)
        ids += [it["id"]["videoId"] for it in d.get("items", []) if it["id"].get("videoId")]
        token = d.get("nextPageToken")
        if not token:
            break
    return ids


def verify_videos(ids: list[str]) -> list[dict]:
    """videos.list batchami po 50: twarda weryfikacja status.license=creativeCommon
    (search bywa niedokladny) + metadane do rankingu."""
    out = []
    for i in range(0, len(ids), 50):
        d = _get("videos", part="status,snippet,contentDetails", id=",".join(ids[i:i + 50]))
        for it in d.get("items", []):
            if it.get("status", {}).get("license") != "creativeCommon":
                continue
            sn, cd = it.get("snippet", {}), it.get("contentDetails", {})
            out.append({
                "video_id": it["id"],
                "url": f"https://youtu.be/{it['id']}",
                "title": sn.get("title", ""),
                "channel": sn.get("channelTitle", ""),
                "channel_id": sn.get("channelId", ""),
                "duration_s": parse_iso_duration(cd.get("duration", "")),
                "lang": sn.get("defaultAudioLanguage") or sn.get("defaultLanguage") or "",
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Scout kandydatow CC-BY z YT (Data API v3)")
    ap.add_argument("--queries", nargs="+",
                    default=["podcast", "wywiad", "rozmowa", "vlog", "audycja"])
    ap.add_argument("--pages", type=int, default=1, help="stron search na fraze (100 jedn./strona)")
    ap.add_argument("--min-duration", type=int, default=600,
                    help="min. sekund filmu (600 = dlugie formy, wiecej h na 1 decyzje kuracji)")
    ap.add_argument("--out", type=Path, default=Path("data/yt_cc_candidates.jsonl"))
    args = ap.parse_args()

    if not os.environ.get("YT_API_KEY"):
        sys.exit("Brak YT_API_KEY w env (darmowy klucz: console.cloud.google.com, "
                 "wlacz 'YouTube Data API v3')")

    seen: set[str] = set()
    ids: list[str] = []
    for q in args.queries:
        for vid in search_cc(q, args.pages):
            if vid not in seen:
                seen.add(vid)
                ids.append(vid)
    print(f"[scout] kandydatow z search: {len(ids)} (frazy: {args.queries})")

    videos = [v for v in verify_videos(ids) if v["duration_s"] >= args.min_duration]
    hours = sum(v["duration_s"] for v in videos) / 3600
    print(f"[scout] po weryfikacji licencji + min {args.min_duration}s: "
          f"{len(videos)} filmow / {hours:.1f} h")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for v in videos:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"\n=== RANKING KANALOW (kuracja: zatwierdzasz KANALY, nie filmy) ===")
    for c in aggregate_channels(videos)[:25]:
        print(f"  {c['sec'] / 3600:6.1f} h  {c['n']:>3} filmow  {c['channel'][:40]:40} "
              f"https://youtube.com/channel/{c['channel_id']}")
        for t in c["titles"]:
            print(f"           - {t}")
    print(f"\n[OK] pelna lista filmow -> {args.out}")
    print("nastepnie (zatwierdzone kanaly): yt-dlp z bramka CC-BY -- DATA_SCALING FAZA 2b")


if __name__ == "__main__":
    main()
