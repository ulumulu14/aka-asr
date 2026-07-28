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
pipeline (nowe pliki, `requirements-pseudo.txt` + ffmpeg):

```bash
# 1. Audio (przyklad YT; playlisty/kanaly wg wlasnej kuracji):
yt-dlp -f bestaudio -x --audio-format m4a -o 'data/raw_yt/%(id)s.%(ext)s' <URL...>

# 2. Segmentacja: dlugie pliki -> mowa 3-18 s (silero VAD) + manifest bez tekstu:
python utils/datasets_conversion/longform2json.py data/raw_yt --out-root data/pseudo_yt

# 3. Etykiety Whisperem (batch 32+ na H100, 8-16 na RTX 5080; wznawialne):
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
