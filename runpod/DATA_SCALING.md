# Skalowanie danych treningowych

Miks bazowy ~438 h / 213 k sampli (cv 24,8k · mls 25,0k · bigos 37,9k · voxpopuli
34,7k · **yodas 90,8k = 43%**). YODAS już dominuje i jest najbrudniejszy (napisy YT),
więc balans miksu ma znaczenie -- nie sam wolumen. Trzy dźwignie, w kolejności
wartość/wysiłek: **CV validated** (czyste, balansuje), **YODAS +shardy** (łatwe,
ale skręca w YT), **pseudo-labeling** (skalowalne, faza po dobrym modelu).

---

## 1. YODAS -> 20 shardów (łatwe, na podzie)

`YODAS_SHARDS` domyślnie 20 (pl000 ma 78). Bootstrap dociąga brakujące shardy i
przebudowuje manifest (pomija już pobrane), gdy `mam < YODAS_SHARDS`:

```bash
cd /workspace/aka-asr && git pull
bash runpod/bootstrap.sh                 # dociagnie shardy 10-19, przebuduje *_all
# wiecej: YODAS_SHARDS=30 bash runpod/bootstrap.sh
```
Sanity: `grep -c . data/train_manifest_yodas.jsonl` (~2x po 20 shardach) i
`wc -l data/train_manifest_all.jsonl`. Uwaga: 20 shardów robi z YODAS ~60% miksu.

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

## 3. Pseudo-labeling (faza 3 -- PO dobrym modelu)

Whisper-owa dźwignia: dekoduj NIEOPISANE audio najlepszym modelem, filtruj po
pewności, dorzuć do miksu. Rób z modelu docelowego (retry BPE albo mid 12,71%),
nie ze słabego -- inaczej wstrzykujesz błędy.

```bash
# audio bez etykiet, np. dalsze shardy YODAS (dekodujemy AUDIO, ignorujac napisy)
# albo polskie podcasty/YT sciagniete osobno -> manifest {audio_path, duration}
python utils/pseudo_label.py \
    --checkpoint checkpoints/conformer_big/<exp>/avg_last5.pt \
    --manifest data/unlabeled_pl.jsonl --out data/pseudo_pl.jsonl \
    --min-score -0.4 --max-tokens-per-sec 25

# scal z miksem i przetrenuj (pseudo osobno, zeby latwo wylaczyc/przewazyc)
cat data/train_manifest_all.jsonl data/pseudo_pl.jsonl > data/train_manifest_all_ps.jsonl
```
`--min-score` (średni log-prob tokenów): -0.3 ostry ... -0.7 luźny. Iteruj:
lepszy model -> lepsze pseudo-etykiety -> lepszy model. Ewaluuj, czy pseudo realnie
zbija WER (osobny manifest = łatwy A/B), nie dokładaj na ślepo.
