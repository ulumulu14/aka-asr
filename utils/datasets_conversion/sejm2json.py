#!/usr/bin/env python3
"""
FAZA 2 (Sejm, ground-truth): oficjalne stenogramy z api.sejm.gov.pl jako PRAWDZIWE
etykiety (nie pseudo Whispera). Sejm = Tier 1 (art. 4 pr.aut., materialy urzedowe).

Stenogram jest REDAGOWANY (nie verbatim: stenografowie czyszcza "yyy"/falstarty,
usuwaja didaskalia "(Oklaski)"), wiec nie da sie go slepo przypisac do audio.
Rozwiazanie = AGREEMENT: Whisper etykietuje segmenty (kotwica alignmentu), a
stenogram jest referencja ground-truth. Gdzie hyp Whispera zgadza sie ze
stenogramem (WER < prog) -> bierzemy tekst STENOGRAMU (czysty, z casingiem/
interpunkcja); gdzie nie -> drop (redakcja albo blad Whispera).

Dwa tryby:
  # 1. pobierz stenogram posiedzenia (JSON API -> czysty tekst):
  python utils/datasets_conversion/sejm2json.py fetch --num 1 --date 2023-11-13 \
      --out data/sejm/steno_1_2023-11-13.txt
  # 2. (miedzy: longform2json na audio -> pseudo_whisper -> manifest_whisper.jsonl)
  # 3. agreement stenogram <-> hipotezy Whispera -> manifest ground-truth:
  python utils/datasets_conversion/sejm2json.py align \
      --whisper data/pseudo_sejm/manifest_whisper.jsonl \
      --steno data/sejm/steno_1_2023-11-13.txt \
      --out data/sejm/manifest_sejm.jsonl --max-wer 0.3

Audio: pobierz pelny dzien posiedzenia (yt-dlp @SejmRP_PL albo sejm.gov.pl),
longform2json (VAD, BEZ --require-license -- Tier 1), pseudo_whisper (etykiety).
Align zaklada JEDEN plik audio na stenogram (segmenty w kolejnosci czasowej po
nazwie _0000.._NNNN); dla wielu plikow uruchom per plik.
"""
import argparse
import difflib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.sejm.gov.pl/sejm/term{term}/proceedings/{num}/{date}/transcripts"
API_KOM = "https://api.sejm.gov.pl/sejm/term{term}/committees"
# mowca w zapisie komisji: <p><b>Przewodniczaca posel X (KO):</b><br>tekst...
# (boldy BEZ dwukropka to lista obecnosci w naglowku -- nie ruszamy ich regexem,
# tylko ucinamy caly naglowek do pierwszego mowcy)
_KOM_SPEAKER = re.compile(r"<b>[^<]{3,160}:\s*</b>", re.DOTALL)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_PARENS = re.compile(r"\([^)]*\)")  # didaskalia stenogramu: (Oklaski), (Gwar na sali)
_HEAD = re.compile(r".*?<body[^>]*>", re.DOTALL)   # head+title (naglowki dokumentu)
_HEADERS = re.compile(r"<h[12][^>]*>.*?</h[12]>", re.DOTALL)  # h1 (kadencja) + h2 (mowca)
_P_TITLED = re.compile(r'<p class="[^"]*">.*?</p>', re.DOTALL)  # tytuly punktow (punkt-tytul)
_TAG = re.compile(r"<[^>]+>")


def strip_statement_html(raw: str) -> str:
    """HTML wypowiedzi -> czysty tekst WYPOWIEDZIANY. Wycina: head+title, wszystkie
    naglowki h1/h2 (kadencja/posiedzenie + naglowek mowcy), tytuly punktow (p.class),
    didaskalia w nawiasach, tagi. Zostaje tresc plain <p> = to, co padlo z mownicy."""
    raw = _HEAD.sub(" ", raw, count=1)   # usun wszystko do <body> (bez efektu gdy brak body)
    raw = _HEADERS.sub(" ", raw)
    raw = _P_TITLED.sub(" ", raw)
    raw = _TAG.sub(" ", raw)
    raw = html.unescape(raw)
    raw = _PARENS.sub(" ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def norm_words(text: str) -> list[str]:
    """lowercase + bez interpunkcji -> lista slow (forma do dopasowania, jak _acoustic)."""
    return _PUNCT.sub(" ", text.lower()).split()


def tokenize_steno(text: str) -> tuple[list[str], list[str]]:
    """Stenogram -> (tokeny ORYGINALNE z casingiem/interpunkcja, tokeny znormalizowane).
    Rownolegle listy; tokeny czysto-interpunkcyjne (np. '—') pomijane."""
    orig, norm = [], []
    for w in text.split():
        n = _PUNCT.sub("", w.lower())
        if n:
            orig.append(w)
            norm.append(n)
    return orig, norm


def word_wer(ref: list[str], hyp: list[str]) -> float:
    """Levenshtein na slowach / len(ref). Miara zgodnosci hyp Whispera ze stenogramem."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def align_segment(
    hyp_words: list[str],
    steno_norm: list[str],
    steno_orig: list[str],
    cursor: int,
    window: int = 600,
) -> tuple[dict | None, int]:
    """Dopasuj hipoteze Whispera do fragmentu stenogramu w oknie [cursor, cursor+window).
    Monotonicznie (oba ciagi w kolejnosci czasowej): difflib znajduje bloki wspolne,
    span = od pierwszego do ostatniego dopasowanego slowa. Zwraca (wynik|None, nowy_kursor).
    Kursor przesuwa sie tylko przy trafieniu (caller decyduje po progu WER)."""
    if not hyp_words:
        return None, cursor
    hi = min(len(steno_norm), cursor + window)
    win = steno_norm[cursor:hi]
    if not win:
        return None, cursor
    sm = difflib.SequenceMatcher(None, win, hyp_words, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None, cursor
    a0 = blocks[0].a
    a1 = blocks[-1].a + blocks[-1].size
    span_norm = win[a0:a1]
    span_orig = steno_orig[cursor + a0 : cursor + a1]
    return {"orig": span_orig, "norm": span_norm, "wer": word_wer(span_norm, hyp_words)}, cursor + a1


def _seg_index(path: str) -> tuple:
    """Klucz sortowania segmentow po nazwie (_0000.._NNNN) -> kolejnosc CZASOWA
    (pseudo_whisper sortuje po duracji -- tu przywracamy porzadek nagrania)."""
    m = re.search(r"_(\d+)\.wav$", path)
    return (re.sub(r"_\d+\.wav$", "", path), int(m.group(1)) if m else 0)


def strip_committee_html(raw: str) -> str:
    """Pelny zapis posiedzenia KOMISJI (jeden HTML na posiedzenie) -> czysty tekst
    wypowiedziany: ucina naglowek (porzadek+obecnosc) do PIERWSZEGO mowcy, usuwa
    znaczniki mowcow (<b>...:</b>), didaskalia w nawiasach i tagi."""
    m = _KOM_SPEAKER.search(raw)
    if m:
        raw = raw[m.start():]
    raw = _KOM_SPEAKER.sub(" ", raw)
    raw = _TAG.sub(" ", raw)
    raw = html.unescape(raw)
    raw = _PARENS.sub(" ", raw)
    return re.sub(r"\s+", " ", raw).strip()


# ------------------------------------------------------------------ fetch

def _get(url: str, retries: int = 3, accept: str | None = None) -> str:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as e:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return ""


def cmd_fetch(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    base = API.format(term=args.term, num=args.num, date=args.date)
    meta = json.loads(_get(base))
    statements = meta.get("statements", [])
    if args.max_statements:
        statements = statements[: args.max_statements]
    print(f"[fetch] posiedzenie {args.num}, dzien {args.date}: "
          f"{len(statements)} wypowiedzi ({args.workers} watkow)")

    def one(s: dict) -> tuple[int, str]:
        try:
            return s["num"], strip_statement_html(_get(f"{base}/{s['num']}"))
        except Exception:
            return s["num"], ""

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = sorted(ex.map(one, statements), key=lambda x: x[0])  # kolejnosc = num
    parts = [txt for _, txt in results if txt]
    full = "\n".join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(full, encoding="utf-8")
    print(f"[OK] {len(parts)} wypowiedzi, {len(full.split())} slow -> {args.out}")


def cmd_komisje(args: argparse.Namespace) -> None:
    """Listing: bez --code lista komisji; z --code posiedzenia (num/data/audio m3u8)."""
    base = API_KOM.format(term=args.term)
    if not args.code:
        for c in json.loads(_get(base)):
            print(f"{c.get('code'):6} {c.get('name', '')[:70]}")
        return
    sittings = json.loads(_get(f"{base}/{args.code}/sittings"))
    print(f"# {args.code}: {len(sittings)} posiedzen (audio: ffmpeg -i <m3u8> -vn -ac 1 -ar 16000 out.wav)")
    for s in sittings:
        vids = s.get("video") or []
        link = (vids[0].get("videoLink") or "") if vids else ""
        print(f"{s.get('num'):>4}  {s.get('date')}  {'zamkniete' if s.get('closed') else 'otwarte '}"
              f"  {link[:110]}")


def cmd_fetch_komisja(args: argparse.Namespace) -> None:
    """Stenogram POSIEDZENIA KOMISJI (ludzki zapis) -> czysty tekst do align."""
    url = f"{API_KOM.format(term=args.term)}/{args.code}/sittings/{args.num}/html"
    raw = _get(url, accept="text/html")
    txt = strip_committee_html(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(txt, encoding="utf-8")
    print(f"[OK] komisja {args.code} pos. {args.num}: {len(txt.split())} slow -> {args.out}")


# ------------------------------------------------------------------ align

def cmd_align(args: argparse.Namespace) -> None:
    to_spoken = None
    if args.spoken_numbers:  # cyfry stenogramu -> slowa (num2words, requirements-pseudo)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from utils.spoken_numbers import to_spoken_pl
        to_spoken = to_spoken_pl

    steno_orig, steno_norm = tokenize_steno(args.steno.read_text(encoding="utf-8"))
    print(f"[align] stenogram: {len(steno_norm)} slow"
          + (" | cyfry->slowa: ON" if to_spoken else " | cyfry->slowa: OFF"))

    with open(args.whisper, encoding="utf-8") as f:
        entries = [json.loads(ln) for ln in f if ln.strip()]
    entries.sort(key=lambda e: _seg_index(e["audio_path"]))  # kolejnosc czasowa
    print(f"[align] segmentow Whispera: {len(entries)}")

    cursor = 0
    kept = 0
    kept_h = 0.0
    wers = []
    n_nomatch = n_highwer = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for e in entries:
            hyp = norm_words(e.get("text", ""))
            res, new_cursor = align_segment(hyp, steno_norm, steno_orig, cursor, args.window)
            if res is None:
                n_nomatch += 1
                continue
            if res["wer"] > args.max_wer:
                n_highwer += 1
                continue
            cursor = new_cursor  # przesun kotwice tylko przy trafieniu
            text = " ".join(res["orig"])
            if to_spoken is not None:
                text = to_spoken(text)  # cyfry -> forma mowiona (zachowuje wolumen)
            entry = {"audio_path": e["audio_path"], "duration": e["duration"],
                     "text": text, "dataset_source": "sejm"}
            for k in ("source_id", "source_url"):
                if e.get(k):
                    entry[k] = e[k]
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1
            kept_h += e["duration"] / 3600
            wers.append(res["wer"])

    n = len(entries)
    avg_wer = sum(wers) / len(wers) if wers else 0.0
    print(f"[OK] zachowane {kept}/{n} ({kept/max(1,n):.0%}), {kept_h:.2f} h -> {args.out}")
    print(f"     odrzucone: {n_nomatch} bez dopasowania, {n_highwer} WER>{args.max_wer}")
    print(f"     srednie WER(hyp,stenogram) zachowanych: {avg_wer:.3f} "
          f"(nisko = Whisper i stenogram zgodni = pewny ground-truth)")
    if kept / max(1, n) < 0.4:
        print("[uwaga] <40% zachowanych -- zly dzien/audio-stenogram niesparowane? "
              "sprawdz czy audio odpowiada dacie posiedzenia")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sejm stenogramy (ground-truth) -> manifest przez agreement")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="pobierz stenogram posiedzenia z api.sejm.gov.pl")
    pf.add_argument("--term", type=int, default=10)
    pf.add_argument("--num", type=int, required=True, help="numer posiedzenia")
    pf.add_argument("--date", required=True, help="data dnia, YYYY-MM-DD")
    pf.add_argument("--out", type=Path, required=True)
    pf.add_argument("--workers", type=int, default=16, help="rownolegle pobieranie wypowiedzi")
    pf.add_argument("--max-statements", type=int, default=None, help="limit wypowiedzi (pilot)")
    pf.set_defaults(fn=cmd_fetch)

    pk = sub.add_parser("komisje", help="lista komisji / posiedzen (z linkami audio m3u8)")
    pk.add_argument("--term", type=int, default=10)
    pk.add_argument("--code", default=None, help="np. ASW; bez -> lista komisji")
    pk.set_defaults(fn=cmd_komisje)

    pfk = sub.add_parser("fetch-komisja", help="stenogram posiedzenia komisji -> tekst do align")
    pfk.add_argument("--term", type=int, default=10)
    pfk.add_argument("--code", required=True, help="kod komisji, np. ASW")
    pfk.add_argument("--num", type=int, required=True, help="numer posiedzenia")
    pfk.add_argument("--out", type=Path, required=True)
    pfk.set_defaults(fn=cmd_fetch_komisja)

    pa = sub.add_parser("align", help="agreement stenogram <-> hipotezy Whispera")
    pa.add_argument("--whisper", type=Path, required=True, help="manifest z pseudo_whisper.py")
    pa.add_argument("--steno", type=Path, required=True, help="stenogram z 'fetch'")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--max-wer", type=float, default=0.3,
                    help="prog zgodnosci: segment zachowany gdy WER(hyp,stenogram) <= tego")
    pa.add_argument("--window", type=int, default=600, help="okno szukania w slowach stenogramu")
    pa.add_argument("--no-spoken-numbers", dest="spoken_numbers", action="store_false",
                    help="NIE zamieniaj cyfr na slowa (domyslnie zamienia -- filtr treningowy tnie cyfry)")
    pa.set_defaults(fn=cmd_align, spoken_numbers=True)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
