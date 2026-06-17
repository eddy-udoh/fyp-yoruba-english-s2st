"""
03_whisper_baseline.py — Whisper Medium zero-shot WER evaluation on Yoruba
Phase 2, Step 1  |  Run: python src/asr/03_whisper_baseline.py

Dataset: google/fleurs yo_ng — fully public, no HF auth required.
NOTE: datasets must be loaded BEFORE importing whisper/numba to avoid
      a native STATUS_ACCESS_VIOLATION crash on Windows (numba/LLVM conflict
      with the multiprocessing used in dataset generation).
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

# ── load dataset FIRST (before whisper/numba import) ─────────────────────────
from datasets import load_dataset

EVAL      = "evaluation"
N_SAMPLES = 20
OUT_JSON  = os.path.join(EVAL, "asr_baseline_wer.json")

DATASET   = "google/fleurs"
CONFIG    = "yo_ng"
SPLIT     = "test"
REF_COL   = "transcription"

# ── now safe to import whisper / numba ───────────────────────────────────────
import jiwer
import torch
import whisper

os.makedirs(EVAL, exist_ok=True)

WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

def compute_wer(ref: str, hyp: str) -> float:
    out = jiwer.process_words(
        ref, hyp,
        reference_transform=WER_TRANSFORM,
        hypothesis_transform=WER_TRANSFORM,
    )
    return out.wer

def main():
    print("=" * 60)
    print("  FYP S2ST — Whisper Baseline | Phase 2 Step 1")
    print("=" * 60)

    print(f"\n  Loading {DATASET} ({CONFIG}) test split...")
    ds        = load_dataset(DATASET, CONFIG, split=SPLIT)
    available = len(ds)
    n         = min(N_SAMPLES, available)
    print(f"  [OK] {available} test samples. Using first {n}.\n")

    gpu    = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    label  = f"GPU — {torch.cuda.get_device_name(0)}" if gpu else "CPU"
    print(f"  Device: {label}")
    print("  Loading Whisper Medium...")
    model = whisper.load_model("medium", device=device)
    print("  [OK] Whisper Medium loaded.\n")

    per_sample = []
    wer_scores = []

    for i in range(n):
        sample    = ds[i]
        audio_arr = np.array(sample["audio"]["array"], dtype=np.float32)
        reference = sample[REF_COL].strip()

        print(f"  Sample {i+1:>2}/{n}")
        print(f"  Reference : {reference}")

        t0 = time.time()
        result = model.transcribe(
            audio_arr,
            language="yo",
            task="transcribe",
            fp16=False,
        )
        elapsed    = time.time() - t0
        hypothesis = result["text"].strip()
        wer        = compute_wer(reference, hypothesis)
        wer_scores.append(wer)

        print(f"  Hypothesis: {hypothesis}")
        print(f"  WER       : {wer * 100:.2f}%   ({elapsed:.1f}s)")
        print()

        per_sample.append({
            "sample_index": i,
            "reference":   reference,
            "hypothesis":  hypothesis,
            "wer":         round(wer, 4),
            "wer_pct":     round(wer * 100, 2),
            "latency_sec": round(elapsed, 2),
        })

    avg_wer = sum(wer_scores) / len(wer_scores)

    print("=" * 60)
    print(f"  AVERAGE WER across {n} samples: {avg_wer * 100:.2f}%")
    print("=" * 60)

    output = {
        "model":       "whisper-medium",
        "mode":        "zero-shot",
        "language":    "yo",
        "dataset":     f"{DATASET}/{CONFIG}",
        "n_samples":   n,
        "avg_wer":     round(avg_wer, 4),
        "avg_wer_pct": round(avg_wer * 100, 2),
        "per_sample":  per_sample,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] {OUT_JSON}")
    print("  Next: python src/nmt/marian_baseline.py")


if __name__ == "__main__":
    main()
