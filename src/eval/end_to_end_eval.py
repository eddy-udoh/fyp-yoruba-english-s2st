"""
end_to_end_eval.py — TRUE speech→translation evaluation (error propagation included).

WHY THIS EXISTS
───────────────
The reported BLEU (49.82) is a COMPONENT score: the NMT was fed clean reference
Yorùbá text from the CSV, never Whisper's output. So it does not reflect the
real system, where ASR errors (WER ~63%) corrupt the NMT input. This script
measures the full chain so component vs end-to-end can be compared honestly:

    Yorùbá audio ──► Whisper (ASR) ──► MarianMT (NMT) ──► English
                       │                                   │
                       ▼                                   ▼
                   ASR WER vs ref Yorùbá          END-TO-END BLEU/chrF vs ref English

It also re-runs the NMT on the CLEAN reference text, so you get both numbers and
their GAP (the cost of error propagation) in one table.

AUDIO SOURCE
────────────
  --mode synth   (default) : synthesize the test Yorùbá with MMS-TTS, then ASR it.
                             Fully automatic; TTS speech is cleaner than human
                             speech, so treat this as an optimistic upper bound.
  --mode recorded          : use real WAVs from --audio-dir named <row>.wav
                             (row index into the test CSV). Most rigorous.

Run:
  python src/eval/end_to_end_eval.py --n 100
  python src/eval/end_to_end_eval.py --mode recorded --audio-dir data/raw/e2e_audio --n 60
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd

# soundfile before whisper avoids a Windows numba/LLVM crash (same as app.py)
import soundfile as sf  # noqa: F401
import librosa
import torch
import jiwer
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    MarianMTModel, MarianTokenizer,
    WhisperForConditionalGeneration, WhisperProcessor,
    VitsModel, AutoTokenizer,
)

# ── paths (resolve relative to repo root regardless of cwd) ─────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))      # src/eval/
_REPO      = os.path.dirname(os.path.dirname(_HERE))         # repo root
MODELS_DIR = os.path.join(_REPO, "models")
TEST_CSV   = os.path.join(_REPO, "data", "splits", "test.csv")
OUT_JSON   = os.path.join(_REPO, "evaluation", "end_to_end_eval.json")

WHISPER_MODEL_PATH = os.path.join(MODELS_DIR, "whisper-small-yoruba-finetuned")
WHISPER_BASE_ID    = "openai/whisper-small"
YO_EN_MODEL_PATH   = os.path.join(MODELS_DIR, "marian-yoruba-medical")
MMS_TTS_ID         = "facebook/mms-tts-yor"

YO_COL = "Patient_Yoruba"
EN_COL = "Clinical_Translation_English"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

# WER normalisation — same recipe as the ASR baseline script
WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def wer(ref: str, hyp: str) -> float:
    out = jiwer.process_words(ref, hyp,
                              reference_transform=WER_TRANSFORM,
                              hypothesis_transform=WER_TRANSFORM)
    return out.wer


def load_models():
    print(f"  Device: {DEVICE}")
    print("  Loading fine-tuned Whisper …")
    proc = WhisperProcessor.from_pretrained(WHISPER_BASE_ID)
    asr  = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH).to(DEVICE).eval()

    print("  Loading fine-tuned yo→en MarianMT …")
    if not os.path.isdir(YO_EN_MODEL_PATH):
        sys.exit(f"[ERROR] yo→en model not found at {YO_EN_MODEL_PATH}")
    nmt_tok = MarianTokenizer.from_pretrained(YO_EN_MODEL_PATH)
    nmt_mdl = MarianMTModel.from_pretrained(YO_EN_MODEL_PATH).to(DEVICE).eval()

    return proc, asr, nmt_tok, nmt_mdl


def transcribe(audio_16k, proc, asr) -> str:
    feats = proc(audio_16k, sampling_rate=TARGET_SR,
                 return_tensors="pt").input_features.to(DEVICE)
    forced = proc.get_decoder_prompt_ids(language="yo", task="transcribe")
    with torch.no_grad():
        ids = asr.generate(feats, forced_decoder_ids=forced)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def translate(text: str, tok, mdl) -> str:
    inputs = tok([text], return_tensors="pt", padding=True,
                 truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=512, num_beams=4,
                           no_repeat_ngram_size=3, repetition_penalty=1.5)
    return tok.decode(out[0], skip_special_tokens=True)


def make_synthesizer():
    print("  Loading MMS-TTS (facebook/mms-tts-yor) for speech synthesis …")
    tok = AutoTokenizer.from_pretrained(MMS_TTS_ID)
    mdl = VitsModel.from_pretrained(MMS_TTS_ID).to(DEVICE).eval()
    sr  = mdl.config.sampling_rate

    def synth(text: str) -> np.ndarray:
        inputs = tok(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            wav = mdl(**inputs).waveform.squeeze().cpu().numpy().astype(np.float32)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        return wav

    return synth


def load_recorded(audio_dir: str, row: int) -> np.ndarray:
    path = os.path.join(audio_dir, f"{row}.wav")
    if not os.path.exists(path):
        return None
    wav, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return wav.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="End-to-end speech→translation evaluation")
    ap.add_argument("--n", type=int, default=100, help="number of test rows to evaluate")
    ap.add_argument("--mode", choices=["synth", "recorded"], default="synth")
    ap.add_argument("--audio-dir", default=None, help="dir of <row>.wav files (recorded mode)")
    args = ap.parse_args()

    if args.mode == "recorded" and not args.audio_dir:
        sys.exit("[ERROR] --mode recorded requires --audio-dir")

    print("=" * 64)
    print(f"  FYP S2ST — END-TO-END EVALUATION  ({args.mode} speech)")
    print("=" * 64)

    df = pd.read_csv(TEST_CSV, encoding="utf-8").head(args.n).reset_index(drop=True)
    print(f"  Test rows: {len(df)}  (from {TEST_CSV})")

    proc, asr, nmt_tok, nmt_mdl = load_models()
    synth = make_synthesizer() if args.mode == "synth" else None

    rows, wers = [], []
    e2e_hyps, clean_hyps, refs = [], [], []

    for i, r in df.iterrows():
        ref_yo = str(r[YO_COL])
        ref_en = str(r[EN_COL])

        # 1. get audio
        if args.mode == "synth":
            audio = synth(ref_yo)
        else:
            audio = load_recorded(args.audio_dir, i)
            if audio is None:
                print(f"  [skip] no recording for row {i}")
                continue

        # 2. ASR → 3. NMT (end-to-end path)
        asr_yo  = transcribe(audio, proc, asr)
        e2e_en  = translate(asr_yo, nmt_tok, nmt_mdl)
        # clean-text path (component) for comparison
        clean_en = translate(ref_yo, nmt_tok, nmt_mdl)

        w = wer(ref_yo, asr_yo)
        wers.append(w)
        e2e_hyps.append(e2e_en); clean_hyps.append(clean_en); refs.append(ref_en)

        rows.append({
            "row": int(i),
            "ref_yoruba":   ref_yo,
            "asr_yoruba":   asr_yo,
            "asr_wer":      round(w, 4),
            "e2e_english":  e2e_en,
            "clean_english": clean_en,
            "ref_english":  ref_en,
        })
        if (i + 1) % 10 == 0:
            print(f"    processed {i+1}/{len(df)} …")

    if not refs:
        sys.exit("[ERROR] No utterances evaluated.")

    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    e2e_bleu   = round(bleu.corpus_score(e2e_hyps,   [refs]).score, 2)
    e2e_chrf   = round(chrf.corpus_score(e2e_hyps,   [refs]).score, 2)
    clean_bleu = round(bleu.corpus_score(clean_hyps, [refs]).score, 2)
    clean_chrf = round(chrf.corpus_score(clean_hyps, [refs]).score, 2)
    avg_wer    = round(float(np.mean(wers)) * 100, 2)

    summary = {
        "mode":            args.mode,
        "n":               len(refs),
        "asr_wer_pct":     avg_wer,
        "end_to_end":      {"bleu": e2e_bleu,   "chrf": e2e_chrf},
        "clean_text_nmt":  {"bleu": clean_bleu, "chrf": clean_chrf},
        "error_propagation_gap": {
            "bleu": round(clean_bleu - e2e_bleu, 2),
            "chrf": round(clean_chrf - e2e_chrf, 2),
        },
        "per_row": rows,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"  ASR WER (on {args.mode} speech)     : {avg_wer:>6.2f}%")
    print("  ── NMT quality ──────────────────────────────────")
    print(f"  Clean-text  (component)  BLEU/chrF  : {clean_bleu:>6.2f} / {clean_chrf:.2f}")
    print(f"  END-TO-END  (speech→EN)  BLEU/chrF  : {e2e_bleu:>6.2f} / {e2e_chrf:.2f}")
    print(f"  Error-propagation gap    BLEU/chrF  : "
          f"{clean_bleu - e2e_bleu:>6.2f} / {clean_chrf - e2e_chrf:.2f}")
    print("=" * 64)
    print(f"\n  [SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
