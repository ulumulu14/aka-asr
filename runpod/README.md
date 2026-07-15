# Trening aka-asr na RunPodzie

Strategia: **kod przez git, dane regenerowane na podzie z zewnetrznych zrodel**
(tempo datacenter zamiast uploadu 61 GB z domu). Wyjatek: CV (~5.7 GB) dosylany
p2p, bo `cv2json.py` wymaga surowego korpusu Mozilli.

## 0. Pod i network volume

- Utworz **Network Volume** (~120 GB) w wybranym regionie -- dane i checkpointy na
  nim przezyja ubicie poda i mozna go podpiac do kolejnych podow.
- Wynajmij pod z GPU (**A100 40/80 GB** rekomendowana) na templacie, ktory MA juz
  `torch`+CUDA (np. oficjalny *RunPod PyTorch 2.x*) -- unikniesz kompilacji torcha.
  Sprzet A100/4090 to Ampere/Ada, wiec hack cu128/sm_120 z RTX 5080 NIE jest tu
  potrzebny; `bf16` z configu dziala natywnie.
- Zamontuj volume w `/workspace`.

## 1. Kod (git)

Repo `ulumulu14/aka-asr` jest **prywatne** -> clone wymaga tokenu z dostepem do
odczytu (fine-grained z `Contents: Read` na tym repo, albo classic ze scope `repo`).

```bash
cd /workspace
export GH_TOKEN=ghp_...              # token TYLKO do odczytu tego repo
git clone "https://ulumulu14:${GH_TOKEN}@github.com/ulumulu14/aka-asr.git" aka-asr
cd aka-asr
git remote set-url origin https://github.com/ulumulu14/aka-asr.git   # zdejmij token z .git/config
git checkout main
```
> Alternatywa: ustaw repo na **public** w Settings -> wtedy clone bez tokenu
> (`git clone https://github.com/ulumulu14/aka-asr.git aka-asr`).
> Na podzie i tak tylko czytasz kod (checkpointy ida na network volume), wiec
> token do odczytu wystarczy -- nie dawaj tam tokenu z zapisem.

## 2. Dane (regeneracja)

```bash
export HF_TOKEN=hf_...              # token z zaakceptowanymi warunkami BIGOS
bash runpod/bootstrap.sh
```
Regeneruje yodas+voxpopuli+bigos (HF), mls (OpenSLR), RIRS (OpenSLR) prosto do
`/workspace/aka-asr/data` z absolutnymi sciezkami `/workspace/...` -- zadnego
remapowania nie trzeba. Idempotentny: ponowne uruchomienie pomija gotowe zrodla.

### CV (dosylany z lokala)

Na **lokalu** wytnij sample CV z miksu i przemapuj sciezki root na `/workspace`:
```bash
sed 's|/home/arekk/workspace/aka-asr/data|/workspace/aka-asr/data|g' \
  <(grep cv-corpus data/train_manifest_all.jsonl) > /tmp/train_manifest_cv.jsonl
sed 's|/home/arekk/workspace/aka-asr/data|/workspace/aka-asr/data|g' \
  <(grep cv-corpus data/val_manifest_all.jsonl)   > /tmp/val_manifest_cv.jsonl

runpodctl send data/cv-corpus-25.0-2026-03-09-pl          # ~5.7 GB, wypisze kod
runpodctl send /tmp/train_manifest_cv.jsonl /tmp/val_manifest_cv.jsonl
```
Na **podzie** odbierz do `data/` (sciezki musza sie zgadzac z manifestem):
```bash
cd /workspace/aka-asr/data
runpodctl receive <kod_katalogu_cv>       # -> data/cv-corpus-25.0-2026-03-09-pl/
runpodctl receive <kod_manifestow>        # -> data/{train,val}_manifest_cv.jsonl
bash /workspace/aka-asr/runpod/bootstrap.sh   # ponownie: doklei CV do *_all.jsonl
```

## 3. Trening

```bash
bash runpod/train.sh                 # setsid+nohup, przezyje rozlaczenie SSH
tail -f /workspace/aka-asr/train_runpod.log
```
- TensorBoard: `train.sh` startuje go na porcie 6006 -> wystaw port w konfiguracji
  poda (HTTP proxy RunPod).
- Wznowienie po ubiciu poda (checkpointy sa na volume):
  ```bash
  RESUME=/workspace/aka-asr/checkpoints/conformer_mid/<exp>/last.pt bash runpod/train.sh
  ```
- Model `big` zamiast `mid`: zmien `model.size` w
  `configs/config_runpod_mid_all.yaml` (na A100 w zasiegu, powinien pobic WER 30%).
- A100-80GB: podnies `training.max_batch_duration` do ~700 (domyslnie 400 pod 40GB).

## 4. Po treningu

`best.pt` lezy na network volume (`checkpoints/conformer_mid/<exp>/best.pt`).
Sciagnij na lokal: `runpodctl send checkpoints/conformer_mid/<exp>/best.pt`
(albo przez RunPod file browser). Konsumuje go demo apka.
