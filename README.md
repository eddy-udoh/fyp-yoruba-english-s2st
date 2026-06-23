# MedSpeak YO–EN

A bidirectional, diacritic-aware **Yorùbá ↔ English speech-to-speech translation system** for medical consultations, built as a cascaded **ASR → NMT → TTS** pipeline and anchored to the OLDCARTS clinical history-taking framework. Final Year Project.

## Pipeline
`speech → Whisper (ASR) → diacritic normalisation → MarianMT (NMT) → MMS-TTS / Edge-TTS → speech`

- **ASR:** Whisper Small, fine-tuned on Yorùbá
- **NMT:** fine-tuned `opus-mt-yo-en` (yo→en) and `opus-mt-en-nic` + `>>yor<<` (en→yo)
- **TTS:** `facebook/mms-tts-yor` (Yorùbá) · Microsoft Edge-TTS → gTTS fallback (English)

## Quick start
```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt          # CUDA users: install a CUDA torch wheel first

# Backend (loads all models, serves http://localhost:5000)
python src/api/app.py

# Frontend (separate terminal)
cd src/frontend && npm install && npm start   # http://localhost:3000
```
> Model weights (`*.safetensors`) are git-ignored due to size. Place the fine-tuned
> Whisper and MarianMT models under `models/`, or retrain via `notebooks/train_marian_colab.ipynb`.

## Repository layout
| Path | Contents |
|------|----------|
| `src/api/` | Flask REST API (full pipeline) |
| `src/asr/`, `src/nmt/`, `src/tts/`, `src/utils/` | Pipeline + data scripts |
| `src/eval/` | Evaluation: end-to-end, results report, OLDCARTS classification |
| `src/frontend/` | React 19 single-page app |
| `scripts/` | Figure/diagram generators (UML, MOS) |
| `evaluation/` | Metrics, figures, and result tables |
| `notebooks/` | Colab training notebook |

## Documentation
- **`IMPLEMENTATION_GUIDE.md`** — full setup-to-deployment guide
- **`TECHNICAL_OVERVIEW.txt`** — architecture, algorithms, and design reference
