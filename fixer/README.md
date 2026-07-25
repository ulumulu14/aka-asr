# Fixer -- post-korektor bledow ASR

Model jezykowy seq2seq (fine-tune **plT5**) trenowany na parach
`{bledna hipoteza ASR -> poprawna transkrypcja}`. Po dekodowaniu ASR poprawia
systematyczne bledy (diakrytyki, koncowki fleksyjne, homofony).

**Ta sama korekta w tescie i w inferencji** -- obie sciezki uzywaja jednej klasy
`fixer.fixer.AutoFixer`, wiec WER mierzony w tescie odpowiada temu, co dostaje
uzytkownik. To NIE jest czesc ASR-a: dziala na tekscie, niezaleznie od tego, czy
ASR jest char czy BPE.

```
audio --[ASR (train.py)]--> tekst --[AutoFixer (plT5)]--> tekst poprawiony
```

## Zaleznosci

```bash
pip install -r requirements-fixer.txt      # transformers, sentencepiece, protobuf
```
Nie sa potrzebne do treningu ASR (na podzie ich nie instaluj).

## Pipeline

### 1. Harvest par -- `harvest.py`
Uruchamia checkpoint ASR na manifescie i zapisuje `{hyp, ref}` (greedy decode 1:1
jak walidacja). Zachowuje tez pary `hyp==ref` (ucza fixer, zeby nie psul dobrych wyjsc).

```bash
# pary TRENINGOWE: duzo danych, --augment (SpecAugment -> wiecej bledow = gestsze pary)
python -m fixer.harvest --checkpoint checkpoints/conformer_big/<exp>/best.pt \
    --manifest data/train_manifest_all.jsonl --out fixer/pairs_train.jsonl \
    --augment --limit 60000

# pary TESTOWE: osobny manifest, BEZ augmentacji (uczciwy WER), NIE trenuj na nich
python -m fixer.harvest --checkpoint checkpoints/conformer_big/<exp>/best.pt \
    --manifest data/val_manifest_all.jsonl --out fixer/pairs_test.jsonl
```

### 2. Trening fixera -- `train_fixer.py`
```bash
python -m fixer.train_fixer --pairs fixer/pairs_train.jsonl \
    --val-pairs fixer/pairs_test.jsonl \
    --base allegro/plt5-base --out checkpoints/fixer/plt5-base_v1 --epochs 3
```
Zapisuje `save_pretrained` + `fixer_meta.json` (prefix, max_len).

### 3. Test (testowanie) -- `evaluate.py`
```bash
python -m fixer.evaluate --pairs fixer/pairs_test.jsonl \
    --fixer checkpoints/fixer/plt5-base_v1
```
Wypisuje **WER ASR-samego vs ASR+fixer** oraz ile zdan poprawionych / pogorszonych /
bez zmian. Fixer trzymamy tylko jesli netto poprawia i nie psuje za duzo.

### 4. Inferencja -- demo apka
`asr_demo_app/transcriber.py` odpala fixer, gdy ustawisz zmienna srodowiskowa:
```bash
export ASR_DEMO_FIXER=checkpoints/fixer/plt5-base_v1
```
Bez niej demo dziala jak dotychczas (sam ASR).

## Uwagi

- **Rob to na FINALNYM ASR** (big). Fixer uczy sie bledow konkretnego modelu; jak
  podmienisz ASR, rozklad bledow sie przesuwa -> przetrenuj fixer. Na mid (12,71%)
  mozna rozpykac pipeline, ale wagi produkcyjne z modelu docelowego.
- **Over-correction** to glowne ryzyko. `evaluate.py` pokazuje `pogorszone` -- jak
  rosnie, uzyj `--guard-max-len-ratio` (odrzuca korekty drastycznie zmieniajace
  dlugosc), przytnij epoki albo dodaj wiecej par `hyp==ref`.
- **Bez przecieku**: nie ewaluuj na parach, na ktorych trenowano fixer.
