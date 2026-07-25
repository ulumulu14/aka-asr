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

## 2. CV `validated` (czyste, bez przecieku dev/test)

`train` split (24,8k) -> `validated` (~70k, wszystkie zweryfikowane klipy). Wymaga
**surowego korpusu Mozilli** (mp3 + tsv). **Lokalnie**:

```bash
# 1. Pobierz CV pl 25.0 z commonvoice.mozilla.org (konto, jezyk polski) -> tar.gz
# 2. Konwersja validated -> wav + manifest (absolutne sciezki przez cv2json)
python utils/datasets_conversion/cv2json.py cv-corpus-25.0-2026-03-09-pl.tar.gz \
    --split validated --jsonl --out data/cv_validated.jsonl \
    --wav-dir data/cv-corpus-25.0-2026-03-09-pl/wav --workers 8

# 3. FILTR PRZECIEKU: wywal klipy z naszego val setu (inaczej validated je zawiera)
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
```
Zastępuje `train_manifest_cv.jsonl` (był 24,8k) większym, leakage-safe. **Wyślij na
pod** NOWE wavy (validated minus już wysłane) + nowy `train_manifest_cv.jsonl`
(zip jak w README, ścieżki remapowane na `/workspace/...`), potem `bootstrap.sh`
przeskleja `*_all`. Caveat: filtr jest per-klip; nakładanie mówców (client_id)
train/val nie jest usuwane -- lekko zawyża val, do dopracowania później.

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
