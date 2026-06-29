"""
app.py — Speech-to-Speech Translation REST API (Yoruba ↔ English)

Endpoints:
  POST /translate   — full pipeline (audio or text → transcript + translation + TTS audio)
  GET  /health      — liveness probe

Usage:
  python src/api/app.py
"""
from dotenv import load_dotenv
import os
load_dotenv()
import sys
import os
import json
import time
import base64
import io
import re
import tempfile
import unicodedata
import threading
import logging

import numpy as np

# ── import soundfile BEFORE whisper — prevents Windows numba/LLVM crash ───────
import soundfile as sf  # noqa: F401

import torch
import whisper as _whisper_mod
from transformers import (
    MarianMTModel, MarianTokenizer,
    WhisperForConditionalGeneration, WhisperProcessor,
)
from gtts import gTTS
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths (resolve relative to repo root regardless of cwd) ──────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))          # src/api/
_REPO      = os.path.dirname(os.path.dirname(_HERE))             # repo root
MODELS_DIR = os.path.join(_REPO, "models")
EVAL_DIR   = os.path.join(_REPO, "evaluation")
LOG_FILE   = os.path.join(EVAL_DIR, "api_logs.json")

# yo→en: load from the top-level folder — this is where marian_finetune.py saves
# (matches the en→yo convention below). Drop a freshly trained model straight here.
YO_EN_MODEL_PATH    = os.path.join(MODELS_DIR, "marian-yoruba-medical")
WHISPER_MODEL_PATH  = os.path.join(MODELS_DIR, "whisper-medium-yoruba-finetuned")
# Processor (feature extractor + tokenizer) is unchanged by fine-tuning; load from base
WHISPER_BASE_ID     = "openai/whisper-medium"
# en→yo: prefer the locally fine-tuned model; fall back to off-the-shelf opus-mt-en-nic.
# Both are based on opus-mt-en-nic (English → Niger-Congo), so the >>yor<< prefix is
# still required at inference to select Yoruba output.
EN_YO_MODEL_PATH = os.path.join(MODELS_DIR, "marian-english-yoruba")
EN_YO_MODEL_ID   = "Helsinki-NLP/opus-mt-en-nic"
EN_YO_LANG_TOKEN = ">>yor<<"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(EVAL_DIR, exist_ok=True)

# ── global model state ────────────────────────────────────────────────────────
_m: dict = {}
_models_loaded = False
_load_lock = threading.Lock()

# ── diacritic helpers ─────────────────────────────────────────────────────────
_TONAL    = set("àáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ")
_UNDERDOT = set("ọẹṣỌẸṢ")
_ALL_DIA  = _TONAL | _UNDERDOT
_MIN_DIA  = 0.04


def _diacritic_density(text: str) -> float:
    chars = [c for c in unicodedata.normalize("NFC", text) if c != " "]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c in _ALL_DIA) / len(chars)


def _normalise_yoruba(text: str) -> dict:
    nfc = unicodedata.normalize("NFC", text)
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f]", "", nfc).strip()
    cleaned = re.sub(r" +", " ", cleaned)

    density     = _diacritic_density(cleaned)
    has_tonal   = any(c in _TONAL for c in cleaned)
    has_underdot = any(c in _UNDERDOT for c in cleaned)

    flags = []
    if not has_tonal:
        flags.append("missing_tonal_marks")
    if not has_underdot:
        flags.append("missing_underdots")
    if density < _MIN_DIA:
        flags.append(f"low_diacritic_density_{density * 100:.1f}pct")

    return {
        "text":              cleaned,
        "diacritic_density": round(density, 4),
        "flags":             flags,
    }


# ── model loading ─────────────────────────────────────────────────────────────
def load_all_models() -> None:
    global _models_loaded
    with _load_lock:
        if _models_loaded:
            return

        log.info("Loading fine-tuned Whisper from %s on %s …", WHISPER_MODEL_PATH, DEVICE)
        _m["whisper_proc"] = WhisperProcessor.from_pretrained(WHISPER_BASE_ID)
        _m["whisper"] = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH).to(DEVICE).eval()
        log.info("  Whisper loaded.")

        log.info("Loading yo→en fine-tuned model from %s …", YO_EN_MODEL_PATH)
        if not os.path.isdir(YO_EN_MODEL_PATH):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {YO_EN_MODEL_PATH}. "
                "Run src/nmt/marian_finetune.py first."
            )
        _m["yo_en_tok"] = MarianTokenizer.from_pretrained(YO_EN_MODEL_PATH)
        _m["yo_en_mdl"] = MarianMTModel.from_pretrained(YO_EN_MODEL_PATH).eval()
        log.info("  yo→en model loaded.")

        if os.path.isdir(EN_YO_MODEL_PATH):
            en_yo_src = EN_YO_MODEL_PATH
            log.info("Loading fine-tuned en→yo model from %s …", EN_YO_MODEL_PATH)
        else:
            en_yo_src = EN_YO_MODEL_ID
            log.warning("Fine-tuned en→yo model not found at %s — "
                        "falling back to off-the-shelf %s.", EN_YO_MODEL_PATH, EN_YO_MODEL_ID)
        _m["en_yo_tok"] = MarianTokenizer.from_pretrained(en_yo_src)
        _m["en_yo_mdl"] = MarianMTModel.from_pretrained(en_yo_src).eval()
        log.info("  en→yo model loaded.")

        log.info("Loading MMS-TTS Yoruba (facebook/mms-tts-yor) …")
        from transformers import VitsModel, AutoTokenizer as _AutoTok
        _m["mms_tok"] = _AutoTok.from_pretrained("facebook/mms-tts-yor")
        _m["mms_mdl"] = VitsModel.from_pretrained("facebook/mms-tts-yor")
        log.info("  MMS-TTS loaded.")

        _models_loaded = True
        log.info("All models ready.")


# ── translation ───────────────────────────────────────────────────────────────
def _translate(text: str, tok: MarianTokenizer, mdl: MarianMTModel) -> str:
    inputs = tok(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
        )
    return tok.decode(out_ids[0], skip_special_tokens=True)


# ── audio helpers ─────────────────────────────────────────────────────────────
def _audio_bytes_to_array(file_bytes: bytes, filename: str) -> np.ndarray:
    """Save uploaded bytes to a temp file, let whisper.load_audio resample to 16 kHz."""
    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        audio = _whisper_mod.load_audio(tmp_path)   # returns float32 @ 16 kHz
    finally:
        os.unlink(tmp_path)
    return audio


def synthesise_yoruba(text):
    try:
        import scipy.io.wavfile
        tokenizer = _m["mms_tok"]
        model     = _m["mms_mdl"]
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        wav_array = output.squeeze().numpy()
        wav_array = (wav_array * 32767).astype(np.int16)
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, rate=model.config.sampling_rate, data=wav_array)
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return audio_b64, "meta-mms-tts-yoruba", len(buf.getvalue())
    except Exception as e:
        print(f"MMS-TTS error: {e}")
        return None, "mms-error", 0


async def synthesise_speech_async(text, language="en"):
    import edge_tts, io, asyncio
    voice = "en-US-AriaNeural" if language == "en" else "en-US-AriaNeural"
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def synthesise_speech(text, language="en"):
    if language == "yo":
        return synthesise_yoruba(text)
    import asyncio, base64
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(synthesise_speech_async(text, language))
        loop.close()
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            return audio_b64, "edge-tts-aria-neural", len(audio_bytes)
    except Exception as e:
        print(f"Edge TTS error: {e}, falling back to gTTS")

    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return audio_b64, "gtts-fallback", len(buf.getvalue())
    except Exception as e:
        print(f"gTTS error: {e}")
        return None, "none", 0


# ── CORS helper ──────────────────────────────────────────────────────────────
def _cors(data, status=200):
    """Wrap a jsonify payload with explicit CORS headers."""
    resp = make_response(jsonify(data), status)
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# ── request logging ───────────────────────────────────────────────────────────
def _append_log(entry: dict) -> None:
    try:
        logs: list = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as fh:
                logs = json.load(fh)
        logs.append(entry)
        with open(LOG_FILE, "w", encoding="utf-8") as fh:
            json.dump(logs, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        log.warning("Log write failed: %s", exc)


# ── /health ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return _cors({})
    return _cors({"status": "ok", "models_loaded": _models_loaded})


# ── /translate ────────────────────────────────────────────────────────────────
@app.route("/translate", methods=["POST", "OPTIONS"])
def translate():
    if request.method == "OPTIONS":
        return _cors({})
    t0 = time.perf_counter()

    # ── parse direction ───────────────────────────────────────────────────────
    if request.is_json:
        data      = request.get_json(force=True)
        direction = data.get("direction")
        text_in   = data.get("text", "").strip()
    else:
        direction = request.form.get("direction", "")
        text_in   = request.form.get("text", "").strip()

    if direction not in ("yo-en", "en-yo"):
        return _cors({"error": "direction must be 'yo-en' or 'en-yo'"}, 400)

    # ── get transcript (from uploaded audio or text string) ───────────────────
    transcript = ""
    norm_info: dict = {}

    if "audio" in request.files:
        f = request.files["audio"]
        audio_arr = _audio_bytes_to_array(f.read(), f.filename or "upload.wav")
        lang_code = "yo" if direction == "yo-en" else "en"
        proc   = _m["whisper_proc"]
        feats  = proc(audio_arr, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        forced = proc.get_decoder_prompt_ids(language=lang_code, task="transcribe")
        with torch.no_grad():
            ids = _m["whisper"].generate(feats, forced_decoder_ids=forced)
        transcript = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    elif text_in:
        transcript = text_in
    else:
        return _cors({"error": "Provide 'audio' file or 'text' string"}, 400)

    # ── guard: no speech detected (empty/blank ASR output) ────────────────────
    if not transcript.strip():
        return _cors({"error": "No speech detected. Please record again and speak "
                               "clearly, close to the microphone."}, 422)

    # ── pipeline ──────────────────────────────────────────────────────────────
    if direction == "yo-en":
        norm_info   = _normalise_yoruba(transcript)
        src_text    = norm_info["text"]
        translation = _translate(src_text, _m["yo_en_tok"], _m["yo_en_mdl"])
        if not translation or not translation.strip():
            return _cors({"error": "NMT returned empty translation. Check model."}, 422)
        audio_b64, tts_engine, audio_size = synthesise_speech(translation, "en")
        tts_note = tts_engine
    else:  # en-yo
        norm_info   = {
            "text":              transcript,
            "diacritic_density": None,
            "flags":             [],
        }
        # opus-mt-en-nic requires a language prefix token to select Yoruba output
        src_nic = f"{EN_YO_LANG_TOKEN} {transcript}"
        translation = _translate(src_nic, _m["en_yo_tok"], _m["en_yo_mdl"])
        if not translation or not translation.strip():
            return _cors({"error": "NMT returned empty translation. Check model."}, 422)
        audio_b64, tts_engine, audio_size = synthesise_speech(translation, "yo")
        tts_note = tts_engine

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    response = {
        "direction":          direction,
        "transcript":         transcript,
        "translation":        translation,
        "direct_translation": translation,
        "audio_base64":       audio_b64,
        "latency_ms":        latency_ms,
        "diacritic_density": norm_info.get("diacritic_density"),
        "diacritic_flags":   norm_info.get("flags", []),
        "tts_note":          tts_note,
    }

    _append_log({
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "direction":         direction,
        "transcript":        transcript,
        "translation":       translation,
        "latency_ms":        latency_ms,
        "diacritic_density": norm_info.get("diacritic_density"),
        "diacritic_flags":   norm_info.get("flags", []),
        "tts_note":          tts_note,
    })

    return _cors(response)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_all_models()
    log.info("Starting Flask on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
