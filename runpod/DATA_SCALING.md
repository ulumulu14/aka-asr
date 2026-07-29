# Skalowanie danych treningowych

Miks bazowy ~438 h / 213 k sampli (cv 24,8k · mls 25,0k · bigos 37,9k · voxpopuli
34,7k · **yodas 90,8k = 43%**). YODAS już dominuje i jest najbrudniejszy (napisy YT),
więc balans miksu ma znaczenie -- nie sam wolumen. Trzy dźwignie, w kolejności
wartość/wysiłek: **CV validated** (czyste, balansuje), **YODAS +shardy** (łatwe,
ale skręca w YT), **pseudo-labeling** (skalowalne, faza po dobrym modelu).

---

## 1. YODAS surowe shardy -- ZASTAPIONE (2026-07-28)

Surowe napisy YT (`hf2json.py yodas`, env `YODAS_SHARDS`) zastapione w bootstrapie
przez **yodas-granary** (FAZA 2a nizej): to samo audio, pseudo-etykiety Whisper-v3
po filtrach NVIDII, ~680 h vs 379 h. Stary konwerter zostaje na wypadek A/B
(`hf2json.py yodas --shards N` recznie); stare `data/yodas_pl/` do skasowania (~44 GB).

---

## 2. CV `validated` (czyste, bez przecieku dev/test) -- WPROST NA PODZIE

`train` split (24,8k) -> `validated` (~70k). Nie trzeba konwertowac lokalnie ani
wysylac ~35k wav -- pobierz surowy tar wprost na pod (cv2json pisze sciezki
/workspace/... przez p.resolve(), ffmpeg z bootstrapa):

```bash
# 1. Presigned URL przez API Mozilla Data Collective (API key z konta). Na podzie:
cd /workspace/aka-asr/data
RESP=$(curl -s -X POST \
  "https://mozilladatacollective.com/api/datasets/cmqinmu2a00winq07gyrtri0q/download" \
  -H "Authorization: Bearer <TWOJ_API_KEY>" -H "Content-Type: application/json")
# wyciagnij downloadUrl (jq; fallback bez jq: python3 -c "import sys,json;print(json.load(sys.stdin)['downloadUrl'])"):
URL=$(echo "$RESP" | jq -r '.downloadUrl')
curl -o cv-26-pl.tar.gz "$URL"                      # ~kilka GB

# 2. NAJPIERW sprawdz strukture tara (nowe "scripted-speech" moze roznic sie od
#    klasycznego cv-corpus; cv2json oczekuje validated.tsv + clips/*.mp3):
tar tzf cv-26-pl.tar.gz | grep -E "validated.tsv|clips/" | head

# 3. Konwersja validated (osobny wav-dir, zeby nie mieszac z CV 25.0 z zipa):
cd /workspace/aka-asr
python utils/datasets_conversion/cv2json.py data/cv-26-pl.tar.gz \
    --split validated --jsonl --out data/cv_validated.jsonl \
    --wav-dir data/cv-26-pl/wav --workers 12

# 3. FILTR PRZECIEKU: wywal klipy z naszego val setu (validated je zawiera)
python - <<'PY'
import json, os
val = {os.path.basename(json.loads(l)["audio_path"])
       for l in open("data/val_manifest_cv.jsonl", encoding="utf-8")}
kept = dropped = 0
with open("data/train_manifest_cv.jsonl", "w", encoding="utf-8") as out:
    for l in open("data/cv_validated.jsonl", encoding="utf-8"):
        if os.path.basename(json.loads(l)["audio_path"]) in val:
            dropped += 1
        else:
            out.write(l); kept += 1
print(f"train_manifest_cv: {kept} (usunieto {dropped} z val)")
PY

# 4. przesklej *_all z powiekszonym CV
bash runpod/bootstrap.sh
```
Caveat: filtr per-klip; nakladanie mowcow (client_id) train/val nie jest usuwane
-- lekko zawyza val, do dopracowania pozniej. Alternatywa (najczystsza): dodac
subkomende `cv` do hf2json.py i ciagnac validated z HF (mozilla-foundation/
common_voice_25_0, gated) jak bigos/voxpopuli -- wtedy zero URL-i i tarow.

---

## 3. Pseudo-labeling / czyszczenie agreement -- ZASTAPIONE przez FAZA 2a

(Historyczne: czyszczenie SUROWYCH napisow yodas naszym modelem. yodas-granary
robi to lepiej u zrodla -- Whisper-v3 + QE NVIDII. Zostaje jako wzorzec dla
przyszlych zrodel z watpliwymi etykietami.)

**Fakt (sprawdzone na HF)**: `pl000` to JEDYNY polski subset YODAS (78 shardów,
napisy MANUALNE); `pl100` (auto) nie istnieje. Wniosek: klasyczny pseudo-labeling
na dalszych shardach yodas NIE ma sensu -- napisy sa lepsze niz nasz model,
nadpisywanie ich pseudo-etykietami to degradacja. Docelowe zrodla pseudo to
naprawde nieopisane audio (podcasty, Sejm) -- osobny zaciag.

**Co ma sens JUZ TERAZ: agreement-cleaning napisow YODAS** -- dekodujemy audio
najlepszym checkpointem (pseudo_label.py z wylaczonymi filtrami zbiera hyp per
sample) i wyrzucamy sample, gdzie WER(hyp, napis) > progu (zle zsynchronizowany /
smieciowy napis). Etykiety zostaja z napisow, model tylko filtruje:

```bash
python utils/pseudo_label.py --checkpoint <best_lub_avg.pt> \
    --manifest data/train_manifest_yodas.jsonl --out data/yodas_hyps.jsonl \
    --min-score -99 --min-chars 0 --max-batch-duration 400 --num-workers 12
# potem join hyp<->napis po audio_path, drop przy WER > ~0.8 (snippet: patrz plan
# w pamieci sesji); typowo wypada kilka-kilkanascie %. UWAGA: ponowny bootstrap
# z wiekszym YODAS_SHARDS przebuduje manifest z napisow -> filtr powtorzyc.
```

Dla przyszlych naprawde nieopisanych zrodel: `--min-score -0.4` (ostry) ...
`-0.7` (luzny), `--max-tokens-per-sec 25`, pseudo w OSOBNYM manifescie (latwy A/B).

---

## FAZA 2a: yodas-granary -- gotowa destylacja NVIDII (PRIORYTET, zero GPU)

`espnet/yodas-granary` = audio YODAS2 z pseudo-etykietami **Whisper large-v3**
przepuszczonymi przez pipeline filtrujacy NVIDII (Granary, arXiv 2505.13404;
CC-BY-3.0, niegated). Polski: ~293k sampli / **~680 h** (vs nasze yodas-40
z surowych napisow: 379 h) -- wiecej godzin I czystsze etykiety (casing,
interpunkcja, odfiltrowane halucynacje). To robi za nas odlozony
agreement-cleaning i zastepuje pomysl "yodas 41-78 surowe".

```bash
# pilot (1 parquet na split), potem calosc (~78 GB parquet, transientnie kasowane):
python utils/datasets_conversion/hf2json.py yodas_granary --shards 1
python utils/datasets_conversion/hf2json.py yodas_granary   # df -h /workspace najpierw!
```

**SWAP w bootstrapie (decyzja 2026-07-28)**: bootstrap buduje `train_manifest_all.jsonl`
z `train_manifest_yodas_granary.jsonl` zamiast surowego yodas -- glowny config
(`config_runpod_big_bpe.yaml`) trenuje na nowym miksie bez zmian w yamlu:

```bash
cd /workspace/aka-asr && git pull
df -h /workspace                        # ~78 GB wav + parquety transientnie!
bash runpod/bootstrap.sh                # pilot: GRANARY_SHARDS=1 bash runpod/bootstrap.sh
rm -rf data/yodas_pl data/train_manifest_yodas.jsonl   # stare surowe (~44 GB)
CONFIG=configs/config_runpod_big_bpe.yaml bash runpod/train.sh
```

Val bez zmian (nie mial yodas) -> val_WER porownywalny miedzy miksami. NIE laczyc
obu yodasow w jednym manifescie (to samo zrodlowe audio YT -> duplikaty z
konfliktujacymi etykietami). ~4% sampli granary ma cyfry (odpadna na filtrze
treningowym); segmenty do 40 s: czesc >20 s utnie `max_audio_len` (konwerter
raportuje ile; ew. podniesc limit). Powrot do surowych napisow (A/B): recznie
`hf2json.py yodas --shards 40` + podmiana w sklejce.

---

## FAZA 2b: destylacja z Whisper large-v3 (wlasne nieopisane audio)

Cel z 2a+2b: 2-3k h. Zrodla ktorych Granary nie ma (swieze podcasty, Sejm,
wlasna kuracja YT) etykietujemy sami -- ten sam mechanizm co NVIDIA, nasz
pipeline (nowe pliki, `requirements-pseudo.txt` + ffmpeg).

**LEGALNOSC KOMERCYJNA (wazne)**: bierzemy tylko Tier 1 (Sejm/instytucje publiczne
= poza prawem autorskim, art. 4 pr.aut.) i Tier 2 (YouTube CC-BY). Bramka na dwoch
poziomach: yt-dlp `--match-filter` pobiera TYLKO CC-BY, a longform2json
`--require-license` odrzuca zrodla bez dowodu licencji w `.info.json` (fail-safe).
Licencja+zrodlo sa nieslione do manifestu (slad atrybucji). NIE redystrybuujemy
audio -- wysylamy tylko wagi. Przed launchem: opinia prawnika IP.

```bash
# 1a. YouTube CC-BY: pobierz TYLKO 'reuse allowed' + zapis metadanych (.info.json):
yt-dlp --match-filter "license *= 'Creative Commons'" --write-info-json \
    -f bestaudio -x --audio-format m4a -o 'data/raw_yt/%(id)s.%(ext)s' <URL...>
# 1b. Sejm/instytucje (Tier 1, bez bramki -- to nie YT): wrzuc audio do data/raw_sejm/

# 2. Segmentacja: dlugie pliki -> mowa 3-18 s (silero VAD) + manifest bez tekstu.
#    --require-license = druga bramka (odrzuca gdy .info.json != CC-BY); dla Sejm pomin.
python utils/datasets_conversion/longform2json.py data/raw_yt --out-root data/pseudo_yt \
    --require-license "Creative Commons"
python utils/datasets_conversion/longform2json.py data/raw_sejm --out-root data/pseudo_sejm

# 3. Etykiety Whisperem (batch 32+ na H100, 8-16 na RTX 5080; wznawialne; niesie licencje):
python utils/pseudo_whisper.py --manifest data/pseudo_yt/manifest_nolabel.jsonl \
    --out data/pseudo_yt/manifest_whisper.jsonl --batch-size 32

# 4. Do miksu (OSOBNY manifest = latwy A/B ablacji):
cat data/pseudo_yt/manifest_whisper.jsonl >> data/train_manifest_all.jsonl
```

Filtry jakosci wbudowane w pseudo_whisper.py (cps, powtorzenia, watermarki
napisow, cyfry); raport odrzutow na koncu -- >30% = obejrzec zrodlo. Val/test
NIE dostaja pseudo (zostaja recznie opisane). Pilot: `--limit-sources 3` w
kroku 2 i `--limit 200` w kroku 3, obejrzec pary przed masowka.

Do sprawdzenia (wieksze zrodla tej samej rodziny): `nvidia/Granary` ma dla pl
takze YouTube-Commons i MOSEL (nieopisane VoxPopuli) -- potencjalnie tysiace
godzin; zweryfikowac strukture/rozmiar zanim wejdzie do planu.

---

## FAZA 2c: Sejm ground-truth (stenogramy z api.sejm.gov.pl)

Sejm = Tier 1 (art. 4 pr.aut., materialy urzedowe). Stenogramy to LUDZKIE
etykiety (nie pseudo Whispera) -> najwyzsza jakosc. API daje per-wypowiedz:
mowca + czasy + tekst. Stenogram jest REDAGOWANY, wiec parujemy przez AGREEMENT:
Whisper = kotwica alignmentu, stenogram = ground-truth. Pilot (12 min posiedzenia
62): 26/28 segmentow zachowane, WER(whisper,stenogram)=0.064, stenogram POPRAWIL
bledy Whispera (61 vs 67 posiedzenie, "nad tym wnioskiem", casing).

```bash
# 1. stenogram (rownolegle; --max-statements na pilot):
python utils/datasets_conversion/sejm2json.py fetch --num 62 --date 2026-07-15 \
    --out data/sejm/steno_62.txt
# 2. audio: pobierz format audio-only (139=m4a 49k) i utnij (yt-dlp --download-sections
#    na dlugich sesjach dostaje 403 od ffmpeg -> pobierz caly maly format natywnie, tnij lokalnie):
python -m yt_dlp -f 139 -o data/sejm/raw/%(id)s.%(ext)s "https://youtube.com/watch?v=<ID>"
ffmpeg -i data/sejm/raw/<ID>.m4a -ar 16000 -ac 1 data/sejm/raw/<ID>.wav   # caly dzien
# 3. VAD (Tier 1 -> bez --require-license) + Whisper KOTWICA (--keep-digits! mowa
#    sejmowa pelna liczb, tu Whisper nie jest etykieta wiec filtr cyfr off):
python utils/datasets_conversion/longform2json.py data/sejm/raw --out-root data/sejm/seg
python utils/pseudo_whisper.py --manifest data/sejm/seg/manifest_nolabel.jsonl \
    --out data/sejm/seg/manifest_whisper.jsonl --batch-size 32 --keep-digits
# 4. agreement -> ground-truth:
python utils/datasets_conversion/sejm2json.py align \
    --whisper data/sejm/seg/manifest_whisper.jsonl --steno data/sejm/steno_62.txt \
    --out data/sejm/manifest_sejm.jsonl --max-wer 0.3
```

UWAGA liczby: ~73% etykiet Sejmu ma cyfry (artykuly/punkty/daty), a filtr treningowy
tnie cyfry. Do rozstrzygniecia: num2words pl (cyfry->slowa, ale mianownik zamiast
odmiany) vs drop segmentow z cyframi (strata wolumenu) vs keep. Domena formalna
(jak VoxPopuli) -> limitowac udzial w miksie.

---

## Held-out SPONTANICZNY: PELCRA (benchmark-only)

`pelcra/pl-asr-pelcra-for-bigos` (gated auto; konto+token musza miec dostep do
gated repos). **CC BY-NC-ND -> WYLACZNIE ewaluacja, NIE trening komercyjny.**
Test split ma UKRYTE refy (benchmark) -> held-out = validation (~50 h, 12
subsetow spontanicznej: rozmowy/wywiady/podcasty/VC/telefon):

```bash
python utils/datasets_conversion/hf2json.py pelcra --splits validation
# -> data/pelcra/... + data/val_manifest_pelcra.jsonl (~28,5k sampli / ~49,6 h)
python test.py --checkpoint <best_lub_avg.pt> --manifest data/val_manifest_pelcra.jsonl
# test.py rozbija per-subset (pelcra:spokes_biz_luz itd.)
```

---

## FAZA 2d: KOMISJE sejmowe -- semi-spontaniczna z ground-truth (najlepszy stosunek)

40 komisji x ~100+ posiedzen x 1-3 h = tysiace godzin DYSKUSJI (przerywanie,
pytania, odpowiedzi z glowy -- nie deklamacja) z ludzkim stenogramem. Tier 1.
Pilot (ASW/100, 8 min): 33/35 zachowane, WER(whisper,stenogram)=0.062 -- jakosc
identyczna z plenarnymi.

```bash
# 1. wybor: lista komisji / posiedzen z linkami audio (m3u8):
python utils/datasets_conversion/sejm2json.py komisje                # lista komisji
python utils/datasets_conversion/sejm2json.py komisje --code ASW     # posiedzenia + m3u8

# 2. stenogram + audio jednego posiedzenia:
python utils/datasets_conversion/sejm2json.py fetch-komisja --code ASW --num 100 \
    --out data/sejm/steno_ASW_100.txt
ffmpeg -i "<m3u8 z listingu>" -vn -ac 1 -ar 16000 data/sejm/raw/ASW_100.wav

# 3-4. jak FAZA 2c: longform2json (VAD) -> pseudo_whisper --keep-digits (kotwica)
#      -> sejm2json align (ground-truth ze stenogramu + cyfry->slowa)
```

Masowka: petla po (code, num, m3u8) z listingu; wznawialnosc per plik (done_sources
longform2json + resume pseudo_whisper). Limitowac udzial domeny w miksie tak samo
jak plenarne (formalno-urzedowa, choc zywsza).
