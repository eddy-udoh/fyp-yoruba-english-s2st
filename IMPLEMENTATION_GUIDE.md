# MedSpeak YO-EN: Yorùbá–English Medical Speech-to-Speech Translation
## Complete Implementation Guide — Setup, Training, Evaluation, Deployment

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Environment Setup](#3-environment-setup)
4. [Phase 1 — Data, Corpus & EDA](#4-phase-1--data-corpus--eda)
5. [Phase 2 — ASR (Whisper)](#5-phase-2--asr-whisper)
6. [Phase 3 — Neural Machine Translation (bidirectional)](#6-phase-3--neural-machine-translation-bidirectional)
7. [Phase 4 — Text-to-Speech](#7-phase-4--text-to-speech)
8. [Phase 5 — Flask API](#8-phase-5--flask-api)
9. [Phase 6 — React Frontend](#9-phase-6--react-frontend)
10. [Evaluation & Metrics](#10-evaluation--metrics) — **component vs end-to-end**
11. [Running the Full System](#11-running-the-full-system)
12. [Known Limitations & Future Work](#12-known-limitations--future-work)

---

## 1. Project Overview

**MedSpeak YO-EN** is a Final Year Project implementing a **bidirectional speech-to-speech translation (S2ST) system** between Yorùbá and English for **medical consultations**. A Yorùbá-speaking patient's speech is transcribed, translated to English for the clinician, and the clinician's English is translated and spoken back in Yorùbá. **Both translation directions are fine-tuned** on a custom medical corpus.

### Pipeline
```
[speech] → ASR (Whisper) → diacritic normalisation → NMT (MarianMT) → TTS → [speech]
```

### Technology stack
| Layer | Technology |
|---|---|
| ASR | OpenAI Whisper **small, fine-tuned on Yorùbá** |
| NMT yo→en | `Helsinki-NLP/opus-mt-yo-en`, fine-tuned (domain-balanced) |
| NMT en→yo | `Helsinki-NLP/opus-mt-en-nic` + `>>yor<<`, fine-tuned (domain-balanced) |
| TTS Yorùbá | `facebook/mms-tts-yor` (VITS) → WAV |
| TTS English | Microsoft Edge-TTS (neural) → gTTS fallback → MP3 |
| Backend | Python, Flask, flask-cors |
| Frontend | React 19, Axios |
| Eval | sacrebleu, jiwer, scikit-learn, matplotlib |
| Training | HuggingFace Transformers, PyTorch |

---

## 2. System Architecture

```
REACT FRONTEND (src/frontend/src/App.js)
  Mic recorder + direction toggle (YO↔EN)
        │  POST /translate (FormData or JSON)
        ▼
FLASK API (src/api/app.py)
  Whisper ASR → diacritic normalise → MarianMT NMT → MMS-TTS / Edge-TTS
  Response: transcript + translation + audio_base64 + latency + diacritic flags
        ▼
MODEL STORE (models/)
  whisper-small-yoruba-finetuned/     (ASR, local weights)
  marian-yoruba-medical/              (yo→en, fine-tuned)
  marian-english-yoruba/             (en→yo, fine-tuned)
  + facebook/mms-tts-yor downloaded from HuggingFace at runtime
```

> Model weights (`*.safetensors`) are git-ignored (too large for GitHub) and kept on Google Drive; the repo tracks config/tokenizer files only.

---

## 3. Environment Setup

### Prerequisites
- Python 3.10+, Node.js 18+, Git
- CUDA GPU recommended (training was done on a Colab T4; the app runs on a GTX 1650 Ti / 4 GB or CPU)

### Install
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
For an NVIDIA GPU, install the CUDA PyTorch wheel first (see pytorch.org for the `--index-url`).

`requirements.txt` now includes everything the code imports: `torch, torchaudio, transformers, datasets, accelerate, sentencepiece, openai-whisper, sacrebleu, jiwer, pandas, numpy, scikit-learn, matplotlib, flask, flask-cors, gtts, edge-tts, scipy, soundfile, librosa, python-dotenv, google-genai, requests` (Coqui `TTS` is optional/prototype only).

Secrets (e.g. `GEMINI_API_KEY`) go in a `.env` file at the repo root (git-ignored).

---

## 4. Phase 1 — Data, Corpus & EDA

### 4.1 Corpus
- **`data/raw/medical_dialogues_15k.csv`** — ~15,000 Yorùbá↔English medical dialogue pairs, generated with **Gemini 2.5 Flash** (`src/utils/generate_corpus.py`), fully diacritised.
- Columns: `Doctor_English`, `Patient_Yoruba`, `Clinical_Translation_English`, `Direct_Translation_English`.
- NMT pairs: `Patient_Yoruba` ↔ `Clinical_Translation_English`.

### 4.2 Splits
`src/utils/split_dataset.py` (seed 42) → `data/splits/`: **train 12,000 / val 1,500 / test 1,500** (80/10/10). The test split is **held out** and only used for final reporting.

### 4.3 OLDCARTS classification
`data/raw/classify_oldcarts_gemini.py` labels each `Doctor_English` question into one of the eight OLDCARTS clinical categories — **Onset, Location, Duration, Characteristics, Aggravating, Relieving, Timing, Severity** — via Gemini (batched, resumable, cost-capped; key read from `.env`). Output: `medical_dialogues_15k_classified.csv`.

### 4.4 EDA (`src/utils/eda_oldcarts.py`)
Generates figures into `evaluation/eda/` (OLDCARTS pie + bar, utterance lengths, diacritic density, split sizes) and `eda_summary.json`.

**Corpus class balance — the data is imbalanced, not balanced.** Across 15,000 rows:

| Category | Share |
|---|---:|
| Characteristics | **44.95%** |
| Relieving | 17.88% |
| Timing | 8.24% |
| Onset | 7.90% |
| Aggravating | 7.71% |
| Duration | 5.19% |
| Location | 4.65% |
| Severity | 3.49% |

> **Correction to earlier drafts:** the corpus is **long-tailed, dominated by `Characteristics` (~45%)**, with six of eight categories below the 12% line. It should be described as *imbalanced* (mitigated downstream by class weighting), **not** as a "reasonably balanced distribution."

Diacritic density: mean **33%**, 0 rows below the 4% threshold — confirming corpus generation enforced proper Yorùbá orthography.

### 4.5 Diacritic validation
`src/utils/02_preprocess.py` / `diacritic_validator.py` NFC-normalise and check for tonal marks (à á è é ì í ò ó ù ú — precomposed **and** combining) and underdots (ọ ẹ ṣ). Used for data-quality auditing and the live API warning layer.

---

## 5. Phase 2 — ASR (Whisper)

- **Model:** OpenAI Whisper **small**, fine-tuned on Yorùbá → `models/whisper-small-yoruba-finetuned/`. Processor loaded from base `openai/whisper-small`.
- **Baseline eval** (`src/asr/03_whisper_baseline.py`, FLEURS yo_ng, 20 samples): zero-shot WER **103.65%** (>100% because the zero-shot model hallucinated foreign-script tokens, inserting more words than the reference).
- **Fine-tuned:** WER **63.40%** (`evaluation/asr_finetuned_wer.json`) — a **−40.25 pp** improvement; hallucination eliminated. Residual WER is largely diacritic mismatch (partially-diacritised output vs fully-diacritised references).
- In the API, language + task tokens are forced (`get_decoder_prompt_ids`) so Whisper doesn't run unreliable language detection.

---

## 6. Phase 3 — Neural Machine Translation (bidirectional)

### 6.1 Domain-balanced fine-tuning — why
Fine-tuning **only** on medical text caused **domain collapse / catastrophic forgetting** — every input (even "Good morning") came back as a clinical sentence. The fix: each training run blends three sources so general ability is preserved:
- **Medical** (~52%) — `data/splits/train.csv`
- **General** (~45%) — `opus-100` en-yo pairs (`Helsinki-NLP/opus-100`)
- **Greetings/small-talk** (~4%) — hand-written, oversampled

### 6.2 Training (`src/nmt/marian_finetune.py`)
One script, both directions:
```powershell
python src/nmt/marian_finetune.py --direction yo-en
python src/nmt/marian_finetune.py --direction en-yo
```
| Parameter | Value |
|---|---|
| yo→en base | `Helsinki-NLP/opus-mt-yo-en` |
| en→yo base | `Helsinki-NLP/opus-mt-en-nic` (source prefixed `>>yor<< `) |
| Epochs | 5 |
| Learning rate | 3e-5 |
| Effective batch | 16 (8 × grad-accum 2) |
| Eval set | **validation split** (not test → no leakage) |
| Best model | highest **val** BLEU; `EarlyStoppingCallback` |
| Generation guards | `no_repeat_ngram_size=3`, `repetition_penalty=1.5` (kill degenerate repetition) |

Outputs save to `models/marian-yoruba-medical/` and `models/marian-english-yoruba/` (top-level — the API loads from there). Cross-session training is supported via Google Drive checkpointing + auto-resume (see `notebooks/train_marian_colab.ipynb`).

### 6.3 Results (clean-text, component-level)
| Direction | BLEU | chrF | Baseline BLEU |
|---|---:|---:|---:|
| Yorùbá → English | **54.32** | 66.28 | 0.45 |
| English → Yorùbá | **50.32** | 65.02 | ~0 |

Both went from near-zero to strong, and the greeting sanity check now passes both ways (`Ẹ kú àárọ̀ ↔ Good morning`). **These are clean-text scores** (NMT fed perfect reference text) — see §10 for the end-to-end picture.

---

## 7. Phase 4 — Text-to-Speech

| Output | Engine | Format |
|---|---|---|
| Yorùbá | `facebook/mms-tts-yor` (VITS, 16 kHz) | WAV |
| English | Microsoft Edge-TTS `en-US-AriaNeural` → gTTS fallback | MP3 |

MMS-TTS produces correctly-toned Yorùbá when given diacritised input. (`src/tts/coqui_tts_demo.py` is an earlier prototype, not in the live pipeline.)

---

## 8. Phase 5 — Flask API

`src/api/app.py`, host `0.0.0.0:5000`. All models load once at startup (`load_all_models`, thread-locked).

- `GET /health` → `{"status":"ok","models_loaded":true}`
- `POST /translate` (multipart `audio`+`direction`, or JSON `text`+`direction`) → `transcript, translation, audio_base64, latency_ms, diacritic_density, diacritic_flags, tts_note`.

The en→yo model loads from the local fine-tuned folder if present, otherwise falls back to off-the-shelf `opus-mt-en-nic`. Audio MIME is set per direction (WAV for Yorùbá, MP3 for English). Requests are logged to `evaluation/api_logs.json`.

---

## 9. Phase 6 — React Frontend

`src/frontend/src/App.js` (React 19 + Axios). Single page: direction toggle, mic recorder (`MediaRecorder`, webm), animated pipeline stages, results panel (transcript / translation / audio / metrics). The displayed latency uses the **server's** `latency_ms`; a low-diacritic-density warning banner alerts clinicians to tonally ambiguous transcripts. `npm test` runs smoke tests for the header + direction toggles.

---

## 10. Evaluation & Metrics

> **This section is deliberately split into Component evaluation and End-to-End evaluation.** Reporting only the component (clean-text) NMT score overstates the real system, because ASR errors propagate into translation. Keep the two tiers clearly separate.

### 10.1 Component evaluation (each stage in isolation)
| Component | Metric | Value | File |
|---|---|---|---|
| ASR (Whisper small) | WER before → after | 103.65% → **63.40%** | `asr_*_wer.json` |
| NMT yo→en | BLEU / chrF (clean text) | **54.32 / 66.28** | `nmt_finetuned_yo-en.json` |
| NMT en→yo | BLEU / chrF (clean text) | **50.32 / 65.02** | `nmt_finetuned_en-yo.json` |
| TTS | MOS (1–5, human) | *to collect* | `mos_ratings.csv` |

### 10.2 End-to-end evaluation (speech → translation)
`src/eval/end_to_end_eval.py` runs the **full chain** — Yorùbá audio → Whisper → MarianMT → English — and compares it against the clean-text path on the same 100 test sentences:

| Evaluation | BLEU | chrF |
|---|---:|---:|
| Clean-text NMT (component) | 54.32 | 66.28 |
| **End-to-end (speech→EN)** | **5.65** | 23.80 |
| **Error-propagation gap** | **48.67** | 42.48 |

ASR WER on this synthesized set: **70.79%**.

**Interpretation:** the component BLEU (54.32) collapses to **5.65** once translation runs on Whisper's actual output — a ~49-point gap that quantifies ASR error propagation. The ASR stage is the dominant bottleneck.

**Caveat (important for the write-up):** audio here is **synthesized** with MMS-TTS, which is *out-of-domain* for a Whisper fine-tuned on real FLEURS speech — this **inflates WER**, so the end-to-end 5.65 is a **pessimistic / worst-case** bound. A real-speech evaluation (`--mode recorded`, 50–100 utterances, 2–4 speakers) would give a fairer figure and is the recommended next step.

### 10.3 OLDCARTS classification — where ROC-AUC actually applies
`src/eval/oldcarts_classification_eval.py` trains a transparent **TF-IDF + Logistic Regression** classifier on the Gemini labels (80/20 stratified, `class_weight="balanced"` to counter the imbalance) and reports:

| Metric | Value |
|---|---:|
| Accuracy | 74.97% |
| Macro-F1 | 0.736 |
| **Macro ROC-AUC** | **0.964** |

Plus a confusion matrix and per-class one-vs-rest ROC curves (`evaluation/report/oldcarts_*`).

> **ROC-AUC scope.** ROC-AUC requires class labels and probability scores, so it is meaningful **only for the OLDCARTS classification task** — **not** for translation (BLEU/chrF) or ASR (WER), which are sequence-generation tasks with no threshold to sweep. (Caveat: Gemini's labels are treated as ground truth, so this measures the scheme's learnability/consistency rather than agreement with human gold labels.)

### 10.4 Metric definitions
- **WER** = (S+D+I)/N; lower better; can exceed 100% when insertions exceed reference length.
- **BLEU** — n-gram precision with brevity penalty; 0–100; higher better.
- **chrF** — character n-gram F-score; more robust for morphologically rich/low-resource Yorùbá.
- **MOS** — mean human rating (1–5) of TTS naturalness; subjective.
- **Diacritic density** — fraction of diacritised chars; data-quality/safety check.

### 10.5 Report builder
`src/eval/results_report.py` regenerates all figures (`wer_improvement`, `e2e_comparison`, per-direction `training_curves_*` / `bleu_vs_epoch_*`, `oldcarts_*`, `mos`) and `results_tables.md` into `evaluation/report/`.

---

## 11. Running the Full System

```powershell
# Terminal 1 — API
.\venv\Scripts\Activate.ps1
python src/api/app.py            # http://0.0.0.0:5000

# Terminal 2 — Frontend
cd src/frontend
npm start                        # http://localhost:3000
```

Re-running pipeline phases:
```powershell
python src/utils/02_preprocess.py            # preprocess + diacritic audit
python src/asr/03_whisper_baseline.py        # ASR WER
python src/nmt/marian_finetune.py --direction yo-en   # NMT (and en-yo)
python src/eval/end_to_end_eval.py --n 100   # end-to-end eval
python src/eval/results_report.py            # figures + tables
python src/eval/oldcarts_classification_eval.py
python src/utils/eda_oldcarts.py
```

Training on Colab (GPU): open `notebooks/train_marian_colab.ipynb` (clone → install → Drive checkpoints → train both directions → download into `models/`).

---

## 12. Known Limitations & Future Work

| Limitation | Status / impact |
|---|---|
| **ASR is the end-to-end bottleneck** | 63.4% WER (real) / 70.8% (synth) propagates into NMT, collapsing end-to-end BLEU to ~5.65 (worst case). Biggest lever for system quality. |
| Synth-speech eval is pessimistic | MMS-TTS audio is OOD for the FLEURS-tuned Whisper; real-speech eval needed for a fair number. |
| Corpus imbalance | `Characteristics` ~45%; mitigated with class weighting; more balanced generation would help. |
| TTS MOS not yet collected | Human listening test pending (see `mos_ratings_TEMPLATE.csv`). |
| MMS-TTS latency | Can spike on long Yorùbá sentences. |
| `_append_log` not concurrency-safe | Acceptable for a single-user demo. |

### Future work
1. **More/better ASR fine-tuning** (more Yorùbá speech; diacritic-consistent references) to cut WER below ~30% — the single biggest end-to-end win.
2. **Real-speech end-to-end evaluation** (50–100 recordings, multiple speakers) alongside the synth worst-case.
3. **Balanced corpus regeneration** across OLDCARTS categories.
4. **Clinical validation** by Yorùbá-speaking clinicians beyond automatic metrics.

---

*Every script referenced exists in the repository and runs in the order shown. Component vs end-to-end evaluation is kept separate by design — the end-to-end figure, not the clean-text BLEU, represents real system quality.*
