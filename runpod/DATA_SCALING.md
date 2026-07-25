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

## 3. Pseudo-labeling / czyszczenie agreement

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
