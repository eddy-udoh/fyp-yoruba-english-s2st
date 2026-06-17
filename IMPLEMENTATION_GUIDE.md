# MedSpeak YO-EN: Yoruba-English Medical Speech-to-Speech Translation System
## Complete Implementation Guide — From Setup to Deployment

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Environment Setup](#3-environment-setup)
4. [Phase 1 — Data Collection & Preprocessing](#4-phase-1--data-collection--preprocessing)
5. [Phase 2 — ASR Baseline (Whisper)](#5-phase-2--asr-baseline-whisper)
6. [Phase 3 — Neural Machine Translation](#6-phase-3--neural-machine-translation)
7. [Phase 4 — Text-to-Speech Synthesis](#7-phase-4--text-to-speech-synthesis)
8. [Phase 5 — Flask API Integration](#8-phase-5--flask-api-integration)
9. [Phase 6 — React Frontend](#9-phase-6--react-frontend)
10. [Evaluation & Metrics](#10-evaluation--metrics)
11. [Running the Full System](#11-running-the-full-system)
12. [Known Limitations & Future Work](#12-known-limitations--future-work)

---

## 1. Project Overview

**MedSpeak YO-EN** is a Final Year Project (FYP) that implements a **bidirectional speech-to-speech translation system** between Yoruba and English, specifically tailored for **medical consultations**. A patient speaking Yoruba can have their speech automatically transcribed, translated to English for a doctor, and the doctor's English reply synthesized back as audio.

### End-to-End Pipeline

```
[User speaks]
      ↓
[ASR] Whisper — Speech → Yoruba text
      ↓
[NMT] Fine-tuned MarianMT — Yoruba text → English text
      ↓
[TTS] gTTS / Coqui — English text → Speech audio
      ↓
[User hears translated response]
```

The reverse direction (English → Yoruba text → Yoruba audio) is also supported.

### Technology Stack

| Layer | Technology |
|---|---|
| ASR | OpenAI Whisper Medium |
| NMT | Helsinki-NLP/opus-mt-yo-en (fine-tuned) |
| TTS | gTTS, Coqui TTS (VITS) |
| Backend API | Python, Flask |
| Frontend | React 19 |
| Evaluation | sacrebleu, jiwer |
| Training | HuggingFace Transformers, PyTorch |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  REACT FRONTEND                     │
│  src/frontend/src/App.js                            │
│                                                     │
│  ┌──────────────┐   ┌──────────────┐               │
│  │  Microphone  │   │  Lang Toggle  │               │
│  │  Recorder    │   │  YO ↔ EN     │               │
│  └──────┬───────┘   └──────────────┘               │
│         │                                           │
│         │   POST /translate (FormData)              │
└─────────┼───────────────────────────────────────────┘
          │
          ↓ HTTP (localhost:5000)
┌─────────────────────────────────────────────────────┐
│                  FLASK API                          │
│  src/api/app.py                                     │
│                                                     │
│  ┌──────┐  ┌──────────────┐  ┌───────────────┐    │
│  │Whisper│→│ Fine-tuned   │→ │ gTTS / Coqui  │   │
│  │ ASR  │  │ MarianMT NMT │  │ TTS Engine    │   │
│  └──────┘  └──────────────┘  └───────────────┘    │
│                                                     │
│  Response: transcript + translation + audio_base64  │
└─────────────────────────────────────────────────────┘
          ↕
┌─────────────────────────────────────────────────────┐
│                  MODEL STORE                        │
│  models/whisper/           (ASR)                    │
│  models/marianmt/          (NMT base)               │
│  models/marian-yoruba-medical/  (Fine-tuned NMT)    │
│  models/coqui_tts/         (TTS)                    │
└─────────────────────────────────────────────────────┘
```

---

## 3. Environment Setup

### 3.1 Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- CUDA-capable GPU (GTX 1650 Ti or better) — optional but strongly recommended
- Git, pip

### 3.2 Clone / Initialise Project Directory

```
fyp_s2st/
├── src/
│   ├── api/
│   ├── asr/
│   ├── nmt/
│   ├── tts/
│   ├── utils/
│   └── frontend/
├── data/
│   ├── raw/text/
│   └── processed/{train,val,test}/
├── models/
├── evaluation/
└── requirements.txt
```

### 3.3 Python Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 requirements.txt (key packages)

```
torch
torchaudio
transformers
sentencepiece
datasets
accelerate
openai-whisper
gtts
TTS
flask
flask-cors
pandas
numpy
scikit-learn
sacrebleu
jiwer
soundfile
librosa
```

> **Windows CUDA note**: If you have an NVIDIA GPU, install the CUDA-enabled PyTorch wheel first.
> Visit pytorch.org for the correct `--index-url` command before running `pip install -r requirements.txt`.

### 3.5 Verify Environment

Run `setup_env.py` to confirm all libraries import correctly and GPU is detected:

```powershell
python setup_env.py
```

Expected output includes:
- All packages imported successfully
- `CUDA available: True` (if GPU present)
- GPU name shown (e.g., NVIDIA GeForce GTX 1650 Ti)

---

## 4. Phase 1 — Data Collection & Preprocessing

### 4.1 Medical Corpus

The project uses a custom **1000-row medical dialogue corpus** (`medical_dialogues.csv`). Each row contains three columns:

| Column | Description |
|---|---|
| `Doctor_English` | Doctor's question in English |
| `Patient_Yoruba` | Patient's response in Yoruba (with full diacritics) |
| `Clinical_Translation_English` | Reference English translation |

Example row:
```
Doctor_English:
  "How long has your child been coughing?"

Patient_Yoruba:
  "Ọmọ mi ti ń sọ̀rọ̀ ikọ́ fún ọjọ́ mẹ́fà."

Clinical_Translation_English:
  "The child has experienced a productive cough for six days."
```

A secondary file `patient_profiles.csv` contains patient demographic metadata.

### 4.2 JW300 Supplementary Corpus

The script `src/asr/01_load_datasets.py` downloads 50 preview samples from the
**JW300 Yoruba-English parallel corpus** via HuggingFace Datasets:

```python
from datasets import load_dataset
dataset = load_dataset("opus100", "en-yo", split="train", streaming=True)
```

This provides additional general-domain Yoruba-English sentence pairs.

### 4.3 Dataset Loading Script

Run `src/asr/01_load_datasets.py`:

```powershell
python src/asr/01_load_datasets.py
```

What it does:
1. Copies `medical_dialogues.csv` and `patient_profiles.csv` to `data/raw/text/`
2. Downloads and saves 50 JW300 pairs to `data/raw/text/jw300_yo_en_preview.csv`
3. Prints row counts and column names to confirm loading

### 4.4 Diacritic Validation — Why It Matters

Yoruba is a tonal language. The same sequence of letters with different tones means different
words. In a medical context, the difference between "ilé" (house) and "ìlé" (building), or
between "òògùn" (medicine) and "oogún" (witchcraft), could cause catastrophic misinterpretation.

The diacritic validator `src/utils/diacritic_validator.py` checks each Yoruba string for:

- **Tonal marks**: à á è é ì í ò ó (low and high tone acute/grave)
- **Underdot characters**: ọ ẹ ṣ (open vowels and retroflex sibilant)
- **Density ratio**: `(diacritic_count / char_count)` must exceed 4% threshold

```python
def compute_diacritic_density(text: str) -> float:
    diacritics = set("àáèéìíòóọẹṣÀÁÈÉÌÍÒÓỌẸṢ")
    count = sum(1 for c in text if c in diacritics)
    return count / max(len(text), 1)
```

### 4.5 Preprocessing Script

Run `src/utils/02_preprocess.py`:

```powershell
python src/utils/02_preprocess.py
```

Steps performed:
1. Load `data/raw/text/medical_dialogues.csv`
2. Run diacritic validation on the `Patient_Yoruba` column
3. Apply text cleaning (Unicode NFC normalization, strip control characters)
4. Stratified 80/10/10 train-val-test split using scikit-learn's `train_test_split`
5. Save splits to `data/processed/train/`, `val/`, `test/`

Results:
- Train: 800 rows → `data/processed/train/medical_dialogues_train.csv`
- Validation: 100 rows → `data/processed/val/medical_dialogues_val.csv`
- Test: 100 rows → `data/processed/test/medical_dialogues_test.csv`
- All 1000 rows passed diacritic validation (100% pass rate, avg density 34.59%)

A full audit JSON is also written to `evaluation/diacritic_audit.json`.

---

## 5. Phase 2 — ASR Baseline (Whisper)

### 5.1 What is Whisper?

OpenAI Whisper is a large multilingual ASR model trained on 680,000 hours of audio. It supports
Yoruba (yo) as a language code but was not heavily trained on it, leading to high WER on Yoruba
speech.

### 5.2 ASR Baseline Evaluation

Script: `src/asr/03_whisper_baseline.py`

```powershell
python src/asr/03_whisper_baseline.py
```

What it does:
1. Loads `google/fleurs` dataset, `yo_ng` (Yoruba Nigeria) test split
2. Takes the first 20 audio samples
3. Transcribes each using `whisper.load_model("medium")`
4. Computes Word Error Rate (WER) with `jiwer.wer(reference, hypothesis)`
5. Saves all per-sample results + aggregate WER to `evaluation/asr_baseline_wer.json`

```python
import whisper, jiwer
model = whisper.load_model("medium")
result = model.transcribe(audio_path, language="yo")
wer = jiwer.wer(reference_text, result["text"])
```

**Result**: Average WER = **103.65%** — indicating that Whisper Medium struggles significantly
with Yoruba, often producing more word errors than correct words (insertions + substitutions
combined exceed reference length).

### 5.3 Why Such High WER?

- Yoruba was under-represented in Whisper's training data
- Tonal information is lost in phonetic transcription
- The model sometimes outputs transliteration rather than proper Yoruba orthography
- This is documented and acknowledged as a limitation in the project

### 5.4 How ASR is Used in the API

Despite the high zero-shot WER, Whisper is still used in the Flask API because there are no
readily available fine-tuned Yoruba ASR models that can be deployed locally. The API uses
`whisper.load_model("small")` (lighter than Medium) for acceptable inference speed.

---

## 6. Phase 3 — Neural Machine Translation

### 6.1 Base Model: Helsinki-NLP/opus-mt-yo-en

The MarianMT family provides pre-trained translation models. For Yoruba→English, the model is:

```
Helsinki-NLP/opus-mt-yo-en
```

It was pre-trained on the OPUS corpus (primarily news/wiki/religious text), which differs
significantly from the medical domain.

### 6.2 NMT Baseline Evaluation

Script: `src/nmt/marian_baseline.py`

```powershell
python src/nmt/marian_baseline.py
```

Steps:
1. Load test set from `data/processed/test/medical_dialogues_test.csv`
2. Load `Helsinki-NLP/opus-mt-yo-en` from HuggingFace
3. Translate 50 Yoruba patient utterances
4. Compute BLEU and chrF scores against reference translations

```python
from transformers import MarianMTModel, MarianTokenizer
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-yo-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-yo-en")
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs, num_beams=4)
```

**Baseline Result**: BLEU = **0.45**, chrF = **12.73**

### 6.3 NMT Fine-Tuning on Medical Corpus

Script: `src/nmt/marian_finetune.py`

```powershell
python src/nmt/marian_finetune.py
```

#### Training Configuration

| Parameter | Value |
|---|---|
| Base model | Helsinki-NLP/opus-mt-yo-en |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Gradient accumulation | 2 (effective batch = 16) |
| Max source length | 128 tokens |
| Max target length | 128 tokens |
| Precision | fp16 (on CUDA), fp32 (CPU fallback) |
| Optimizer | AdamW (via HuggingFace Trainer default) |

#### Training Pipeline

```python
from transformers import (
    MarianMTModel, MarianTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# 1. Tokenize source (Yoruba) and target (English)
def preprocess(examples):
    inputs = tokenizer(examples["Patient_Yoruba"], max_length=128, truncation=True)
    targets = tokenizer(
        examples["Clinical_Translation_English"], max_length=128, truncation=True
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

# 2. Define training arguments
args = Seq2SeqTrainingArguments(
    output_dir="models/marian-yoruba-medical",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    logging_steps=50,
    save_strategy="epoch"
)

# 3. Train
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)
trainer.train()
trainer.save_model("models/marian-yoruba-medical")
```

#### Fine-Tuning Result

| Metric | Baseline | Fine-Tuned | Improvement |
|---|---|---|---|
| BLEU | 0.45 | 2.51 | +456% |
| chrF | 12.73 | 22.29 | +75% |

The fine-tuned checkpoint is saved to `models/marian-yoruba-medical/`. The API loads this
checkpoint directly.

---

## 7. Phase 4 — Text-to-Speech Synthesis

### 7.1 TTS Engine Selection

| Engine | Quality | Yoruba Support | Used For |
|---|---|---|---|
| Coqui TTS (VITS) | High | No | English synthesis demo |
| gTTS | Medium | No (Hausa fallback) | API production TTS |
| pyttsx3 SAPI | Low | No | Windows offline fallback |

### 7.2 Coqui TTS Demo

Script: `src/tts/coqui_tts_demo.py`

```powershell
python src/tts/coqui_tts_demo.py
```

This script:
1. Instantiates the Coqui TTS engine with model `tts_models/en/ljspeech/vits`
2. Synthesises 5 English medical phrases:
   - "The patient presents with fever and cough."
   - "Please open your mouth and say ahh."
   - "You need to take this medicine twice daily."
   - "How long have you had this pain?"
   - "We will run some blood tests."
3. Saves each to `evaluation/tts_samples/phrase_0{1-5}.wav`
4. Writes a manifest JSON to `evaluation/tts_manifest.json`

```python
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)
tts.tts_to_file(text=phrase, file_path=output_path)
```

If VITS fails (VRAM OOM), it falls back to `tts_models/en/ljspeech/tacotron2-DDC`,
and then to pyttsx3 on Windows.

### 7.3 gTTS in the Production API

The Flask API uses gTTS because it requires no local model downloads:

```python
from gtts import gTTS
import io

def synthesize_speech(text: str, lang: str = "en") -> bytes:
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()
```

**Yoruba TTS limitation**: gTTS does not support Yoruba (`yo`). The API falls back to
Hausa (`ha`) as the closest available West African language on gTTS. This is acknowledged
as a limitation.

---

## 8. Phase 5 — Flask API Integration

### 8.1 Starting the API

```powershell
.\venv\Scripts\Activate.ps1
python src/api/app.py
```

The server starts on `http://0.0.0.0:5000`.

### 8.2 API Endpoints

#### `GET /health`

Returns system status:
```json
{"status": "ok", "models_loaded": true}
```

#### `POST /translate`

Accepts multipart FormData or JSON:

| Field | Type | Description |
|---|---|---|
| `audio` | File (optional) | WebM audio from browser |
| `text` | String (optional) | Direct text input |
| `direction` | String | `"yo-en"` or `"en-yo"` |

Response:
```json
{
  "transcript": "Ọmọ mi ti ń sọ̀rọ̀ ikọ́...",
  "translation": "My child has been coughing...",
  "audio_base64": "<base64-encoded MP3>",
  "latency_ms": 87,
  "diacritic_flags": {
    "density": 0.34,
    "missing_tones": false,
    "missing_underdots": false,
    "warning": null
  }
}
```

### 8.3 Model Loading Strategy

The API uses thread-safe lazy loading — models are loaded on the first request, not at startup:

```python
_models = {}
_load_lock = threading.Lock()

def get_models():
    with _load_lock:
        if "whisper" not in _models:
            _models["whisper"] = whisper.load_model("small")
            _models["nmt_yo_en"] = MarianMTModel.from_pretrained(
                "models/marian-yoruba-medical"
            )
            _models["nmt_tokenizer"] = MarianTokenizer.from_pretrained(
                "models/marian-yoruba-medical"
            )
        return _models
```

### 8.4 Full Pipeline in `POST /translate`

```python
@app.route("/translate", methods=["POST"])
def translate():
    direction = request.form.get("direction", "yo-en")

    # 1. Get input text (from audio or direct text)
    if "audio" in request.files:
        audio_file = request.files["audio"]
        audio_path = save_temp_audio(audio_file)
        transcript = models["whisper"].transcribe(audio_path, language="yo")["text"]
    else:
        transcript = request.form.get("text", "")

    # 2. Validate diacritics (Yoruba input only)
    diacritic_info = validate_diacritics(transcript) if direction == "yo-en" else {}

    # 3. Translate
    translation = run_nmt(transcript, direction, models)

    # 4. Synthesise speech
    tts_lang = "en" if direction == "yo-en" else "ha"
    audio_bytes = synthesize_speech(translation, lang=tts_lang)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # 5. Log request
    log_request(transcript, translation, latency_ms)

    return jsonify({
        "transcript": transcript,
        "translation": translation,
        "audio_base64": audio_b64,
        "latency_ms": latency_ms,
        "diacritic_flags": diacritic_info
    })
```

### 8.5 Request Logging

Every API request is appended to `evaluation/api_logs.json`:

```json
[
  {
    "timestamp": "2025-05-31T14:23:01",
    "direction": "yo-en",
    "input_length": 54,
    "latency_ms": 87,
    "diacritic_density": 0.34
  }
]
```

---

## 9. Phase 6 — React Frontend

### 9.1 Setup

```powershell
cd src/frontend
npm install
npm start
```

Runs at `http://localhost:3000`. The dev server proxies API calls to `http://localhost:5000`.

### 9.2 Application Structure

The entire frontend is in `src/frontend/src/App.js` (743 lines), implementing a
single-page application.

### 9.3 Component Breakdown

#### Language Toggle

```jsx
const [direction, setDirection] = useState("yo-en");

<button onClick={() => setDirection(
  direction === "yo-en" ? "en-yo" : "yo-en"
)}>
  {direction === "yo-en" ? "Yoruba → English" : "English → Yoruba"}
</button>
```

#### Audio Recording

Uses the browser's `MediaRecorder` API:

```jsx
const startRecording = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  const chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: "audio/webm" });
    sendToAPI(blob);
  };
  recorder.start();
  setMediaRecorder(recorder);
  setIsRecording(true);
};
```

#### API Call

```jsx
const sendToAPI = async (audioBlob) => {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  formData.append("direction", direction);

  const response = await axios.post(
    "http://localhost:5000/translate",
    formData,
    { timeout: 120000 }
  );

  setTranscript(response.data.transcript);
  setTranslation(response.data.translation);
  setAudioBase64(response.data.audio_base64);
  setDiacriticFlags(response.data.diacritic_flags);
};
```

#### Diacritic Warning Alert

Shown when `diacritic_flags.warning` is non-null:

```jsx
{diacriticFlags?.warning && (
  <div style={{ background: "#FFF3CD", border: "1px solid #FFCC00", padding: 12 }}>
    ⚠️ Diacritic Warning: {diacriticFlags.warning}
    <br/>Density: {(diacriticFlags.density * 100).toFixed(1)}%
  </div>
)}
```

#### Audio Playback

```jsx
{audioBase64 && (
  <audio controls>
    <source src={`data:audio/mp3;base64,${audioBase64}`} type="audio/mp3" />
  </audio>
)}
```

#### Results Grid

```jsx
<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
  <div>
    <h3>Transcript</h3>
    <p>{transcript}</p>
  </div>
  <div>
    <h3>Translation</h3>
    <p>{translation}</p>
  </div>
</div>
```

---

## 10. Evaluation & Metrics

### 10.1 Summary of All Evaluation Results

| Component | Metric | Value | File |
|---|---|---|---|
| ASR (Whisper Medium, zero-shot) | WER | 103.65% | `evaluation/asr_baseline_wer.json` |
| NMT Baseline (opus-mt-yo-en) | BLEU | 0.45 | `evaluation/nmt_baseline.json` |
| NMT Baseline | chrF | 12.73 | `evaluation/nmt_baseline.json` |
| NMT Fine-tuned | BLEU | 2.51 (+456%) | `evaluation/nmt_finetuned.json` |
| NMT Fine-tuned | chrF | 22.29 (+75%) | `evaluation/nmt_finetuned.json` |
| Data Quality (diacritics) | Pass rate | 100% | `evaluation/diacritic_audit.json` |
| API Latency | Avg ms | ~87ms | `evaluation/api_logs.json` |
| TTS | Samples | 5 WAV files | `evaluation/tts_samples/` |

### 10.2 Running the End-to-End Integration Test

```powershell
python test_translate.py
```

This script:
1. Hits `GET /health` to confirm the API is live
2. Sends a Yoruba text string (`yo-en` direction) via POST
3. Sends an audio file via POST (`en-yo` direction)
4. Validates response structure
5. Saves result to `evaluation/translate_test_result.json`

### 10.3 Interpreting BLEU and chrF

- **BLEU** (Bilingual Evaluation Understudy): Measures n-gram overlap between hypothesis and
  reference. Range 0–100; a BLEU of 2.51 on a niche medical domain with limited training data
  is reasonable.
- **chrF** (Character n-gram F-score): More suitable for morphologically rich languages like
  Yoruba. A chrF of 22.29 indicates meaningful partial word-level overlap.
- Both metrics improved substantially after fine-tuning, validating the domain adaptation
  approach.

---

## 11. Running the Full System

### Step-by-Step from Cold Start

**Terminal 1 — Start Backend API**
```powershell
cd c:\Users\Admin\fyp_s2st
.\venv\Scripts\Activate.ps1
python src/api/app.py
```
Wait for: `Running on http://0.0.0.0:5000`

**Terminal 2 — Start Frontend**
```powershell
cd c:\Users\Admin\fyp_s2st\src\frontend
npm start
```
Wait for: `Compiled successfully! Local: http://localhost:3000`

**Open in browser**: `http://localhost:3000`

### Using the Application

1. Select translation direction (Yoruba → English or English → Yoruba)
2. Click the **Record** button — browser will request microphone permission
3. Speak your medical phrase
4. Click **Stop & Translate**
5. View:
   - **Transcript**: What Whisper heard
   - **Translation**: MarianMT output
   - **Audio player**: gTTS synthesized speech
   - **Diacritic warning** (if applicable for Yoruba input)
   - **Latency metric** at the bottom

Alternatively, type text directly into the text input field and click **Translate Text**.

### Re-running Individual Pipeline Phases

```powershell
# Phase 1 - Load datasets
python src/asr/01_load_datasets.py

# Phase 1 - Preprocess + validate
python src/utils/02_preprocess.py

# Phase 2 - ASR baseline evaluation
python src/asr/03_whisper_baseline.py

# Phase 3 - NMT baseline evaluation
python src/nmt/marian_baseline.py

# Phase 3 - NMT fine-tuning (requires GPU for speed)
python src/nmt/marian_finetune.py

# Phase 4 - TTS synthesis demo
python src/tts/coqui_tts_demo.py

# Integration test
python test_translate.py
```

---

## 12. Known Limitations & Future Work

### Current Limitations

| Issue | Root Cause | Impact |
|---|---|---|
| High ASR WER (103%) | Whisper not fine-tuned on Yoruba | Transcription errors cascade into NMT |
| Low absolute BLEU (2.51) | Small 800-sample training corpus | Translations may be imprecise for rare medical terms |
| No Yoruba TTS | gTTS/Coqui lack yo-NG voice | Yoruba audio output uses Hausa fallback — not clinically ideal |
| Real-time latency | Whisper + NMT inference chain | ~87ms average; spikes on first load due to cold model start |
| Single-speaker TTS | gTTS has only one English voice | No voice customization for patient/doctor distinction |

### Recommended Future Improvements

1. **ASR Fine-tuning**: Fine-tune Whisper on FLEURS yo_ng or BABEL Yoruba corpus to reduce WER
   from >100% to sub-30%
2. **Larger NMT Corpus**: Augment the 800-row training set with OPUS100, JW300, or generated
   synthetic pairs to push BLEU above 15
3. **Yoruba TTS**: Integrate Mozilla TTS or Meta's MMS-TTS which has a native Yoruba voice model
4. **Speaker Diarisation**: Separate doctor/patient audio streams using pyannote.audio
5. **Offline Deployment**: Package with ONNX Runtime to remove internet dependency for gTTS
6. **Clinical Validation**: Expert review by Yoruba-speaking clinicians to assess translation
   adequacy beyond automatic BLEU metrics

---

*This guide covers the complete implementation of MedSpeak YO-EN, from raw data ingestion
through model fine-tuning, API integration, and frontend deployment. Every script listed exists
in the codebase and can be executed in the order shown.*
