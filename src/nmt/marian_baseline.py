"""
marian_baseline.py — Helsinki-NLP/opus-mt-yo-en Baseline (BLEU + chrF)
Phase 3, Step 1  |  Run: python src/nmt/marian_baseline.py
"""
import json
import os
import sys

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from transformers import MarianMTModel, MarianTokenizer

EVAL     = "evaluation"
TEST_CSV = os.path.join("data", "processed", "test", "medical_dialogues_test.csv")
OUT_JSON = os.path.join(EVAL, "nmt_baseline.json")
MODEL_ID = "Helsinki-NLP/opus-mt-yo-en"
N_ROWS   = 50

os.makedirs(EVAL, exist_ok=True)

def translate_batch(texts: list[str], tokenizer, model, batch_size: int = 8) -> list[str]:
    results = []
    for start in range(0, len(texts), batch_size):
        batch  = texts[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids    = model.generate(**tokens, max_length=512)
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
        results.extend(decoded)
        print(f"  Translated {min(start + batch_size, len(texts))}/{len(texts)}...", flush=True)
    return results

def main():
    print("=" * 60)
    print("  FYP S2ST — MarianMT Baseline | Phase 3 Step 1")
    print("=" * 60)

    # ── load test data ────────────────────────────────────────────
    if not os.path.exists(TEST_CSV):
        print(f"[ERROR] Not found: {TEST_CSV}")
        sys.exit(1)

    df       = pd.read_csv(TEST_CSV, encoding="utf-8")
    df_sub   = df.head(N_ROWS).copy()
    sources  = df_sub["Patient_Yoruba"].tolist()
    refs     = df_sub["Clinical_Translation_English"].tolist()

    print(f"\n  Test CSV  : {TEST_CSV}")
    print(f"  Rows used : {len(df_sub)} (of {len(df)} available)")

    # ── load model ────────────────────────────────────────────────
    print(f"\n  Loading {MODEL_ID}...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
    model     = MarianMTModel.from_pretrained(MODEL_ID)
    model.eval()
    print("  [OK] Model loaded.\n")

    # ── translate ─────────────────────────────────────────────────
    print("  Translating...")
    hypotheses = translate_batch(sources, tokenizer, model)

    # ── print sample outputs ──────────────────────────────────────
    print("\n  Sample translations (first 5):")
    print("  " + "-" * 56)
    for i in range(min(5, len(sources))):
        print(f"  [{i+1}] Source : {sources[i][:80]}")
        print(f"       Hypothesis: {hypotheses[i][:80]}")
        print(f"       Reference : {refs[i][:80]}")
        print()

    # ── compute scores ────────────────────────────────────────────
    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()

    bleu_result = bleu_metric.corpus_score(hypotheses, [refs])
    chrf_result = chrf_metric.corpus_score(hypotheses, [refs])

    bleu_score = round(bleu_result.score, 2)
    chrf_score = round(chrf_result.score, 2)

    print("=" * 60)
    print(f"  BLEU  : {bleu_score}")
    print(f"  chrF  : {chrf_score}")
    print("=" * 60)

    # ── save results ──────────────────────────────────────────────
    per_row = [
        {
            "row":        i,
            "source":     sources[i],
            "hypothesis": hypotheses[i],
            "reference":  refs[i],
        }
        for i in range(len(sources))
    ]

    output = {
        "model":     MODEL_ID,
        "n_rows":    len(df_sub),
        "bleu":      bleu_score,
        "chrf":      chrf_score,
        "per_row":   per_row,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
