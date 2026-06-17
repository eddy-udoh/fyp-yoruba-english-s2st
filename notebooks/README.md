# Colab training

Fine-tune both MarianMT directions (yo→en and en→yo) on a free Colab T4 GPU.

## Open in Colab
Open the notebook directly from GitHub:

https://colab.research.google.com/github/eddy-udoh/fyp-yoruba-english-s2st/blob/main/notebooks/train_marian_colab.ipynb

(or upload `train_marian_colab.ipynb` via *File → Upload notebook*).

## What it does
1. Checks the GPU (set `Runtime → Change runtime type → GPU (T4)` first).
2. Clones this repo and installs the minimal training deps.
3. Mounts Google Drive and writes **checkpoints + final models to Drive**, so a
   dropped session is recoverable — re-run the training cell to **auto-resume**.
4. Trains **yo→en**, then **en→yo** (each ~20–40 min on a T4).
5. Copies eval JSONs to Drive and verifies the saved model files.

## After training
Download the two model folders from Drive into the repo's `models/` folder
(top level), matching where `src/api/app.py` loads them:

```
models/marian-yoruba-medical/    # yo->en
models/marian-english-yoruba/    # en->yo
```

Each must contain `config.json`, `generation_config.json`, `source.spm`,
`target.spm`, `tokenizer_config.json`, `vocab.json`, and `model.safetensors`.
The weights are git-ignored (too large for GitHub) — keep the Drive copy as backup.

## Then evaluate
On the machine with the models + a GPU:

```bash
python src/eval/end_to_end_eval.py --n 100
python src/eval/results_report.py
```
