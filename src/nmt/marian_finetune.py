"""
marian_finetune.py — Fine-tune Helsinki-NLP/opus-mt-yo-en on medical corpus
Phase 3, Step 2  |  Run: python src/nmt/marian_finetune.py

Strategy:
  - per_device_train_batch_size=8  + gradient_accumulation_steps=2
    → effective batch size = 16 (user spec), safe on 4 GB VRAM
  - 3 epochs, lr=5e-5, fp16 on GPU
  - Best checkpoint saved to models/marian-yoruba-medical/
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    DataCollatorForSeq2Seq,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ── paths & config ─────────────────────────────────────────────────────────
BASE_MODEL  = "Helsinki-NLP/opus-mt-yo-en"
TRAIN_CSV   = os.path.join("data", "processed", "train", "medical_dialogues_train.csv")
TEST_CSV    = os.path.join("data", "processed", "test",  "medical_dialogues_test.csv")
OUTPUT_DIR  = os.path.join("models", "marian-yoruba-medical")
EVAL_DIR    = "evaluation"
OUT_JSON    = os.path.join(EVAL_DIR, "nmt_finetuned.json")

SRC_COL     = "Patient_Yoruba"
TGT_COL     = "Clinical_Translation_English"
MAX_SRC_LEN = 128
MAX_TGT_LEN = 256
EPOCHS      = 3
LR          = 5e-5
BATCH_SIZE  = 8       # per device; grad_accum=2 → effective = 16
GRAD_ACCUM  = 2
N_TEST_ROWS = 50

BASELINE_BLEU = 0.45
BASELINE_CHRF = 12.73

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,   exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────
def sep(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def load_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8")
    print(f"  {label}: {len(df)} rows")
    return df

# ── tokenize ───────────────────────────────────────────────────────────────
def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        model_inputs = tokenizer(
            batch[SRC_COL],
            max_length=MAX_SRC_LEN,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[TGT_COL],
            max_length=MAX_TGT_LEN,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenize

# ── compute BLEU during training (fast, token-level) ──────────────────────
def make_compute_metrics(tokenizer):
    bleu_metric = BLEU(effective_order=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # preds come as token ids when predict_with_generate=True
        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)
        # strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        result = bleu_metric.corpus_score(decoded_preds, [decoded_labels])
        return {"bleu": round(result.score, 4)}

    return compute_metrics

# ── translate batch (post-training eval) ──────────────────────────────────
def translate_batch(texts, tokenizer, model, device, batch_size=8):
    model.eval()
    results = []
    for start in range(0, len(texts), batch_size):
        batch  = texts[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_SRC_LEN).to(device)
        with torch.no_grad():
            ids = model.generate(**tokens, max_length=MAX_TGT_LEN, num_beams=4)
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
        results.extend(decoded)
    return results

# ── main ───────────────────────────────────────────────────────────────────
def main():
    sep("FYP S2ST — MarianMT Fine-Tune | Phase 3 Step 2")

    gpu    = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    print(f"  Device : {'GPU — ' + torch.cuda.get_device_name(0) if gpu else 'CPU'}")
    print(f"  Effective batch : {BATCH_SIZE * GRAD_ACCUM}  "
          f"(per_device={BATCH_SIZE} × grad_accum={GRAD_ACCUM})")

    # ── load data ────────────────────────────────────────────────
    sep("1 / 4  Loading data")
    train_df = load_csv(TRAIN_CSV, "Train")
    test_df  = load_csv(TEST_CSV,  "Test")

    train_dataset = Dataset.from_pandas(train_df[[SRC_COL, TGT_COL]])
    eval_dataset  = Dataset.from_pandas(test_df[[SRC_COL, TGT_COL]])

    # ── load tokenizer & model ───────────────────────────────────
    sep("2 / 4  Loading tokenizer & model")
    print(f"  Base model: {BASE_MODEL}")
    tokenizer = MarianTokenizer.from_pretrained(BASE_MODEL)
    model     = MarianMTModel.from_pretrained(BASE_MODEL)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenize_fn = make_tokenize_fn(tokenizer)
    train_tok   = train_dataset.map(tokenize_fn, batched=True,
                                    remove_columns=train_dataset.column_names)
    eval_tok    = eval_dataset.map(tokenize_fn, batched=True,
                                   remove_columns=eval_dataset.column_names)
    print(f"  Train tokenised: {len(train_tok)} rows")
    print(f"  Eval  tokenised: {len(eval_tok)}  rows")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    # ── training args ────────────────────────────────────────────
    sep("3 / 4  Fine-tuning")
    steps_per_epoch = max(1, len(train_tok) // (BATCH_SIZE * GRAD_ACCUM))
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Steps / epoch   : {steps_per_epoch}")
    print(f"  Total steps     : {steps_per_epoch * EPOCHS}")
    print()

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_steps                = max(1, steps_per_epoch // 5),
        weight_decay                = 0.01,
        fp16                        = gpu,
        predict_with_generate       = True,
        generation_max_length       = MAX_TGT_LEN,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "bleu",
        greater_is_better           = True,
        logging_steps               = max(1, steps_per_epoch // 5),
        report_to                   = "none",
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_tok,
        eval_dataset    = eval_tok,
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = make_compute_metrics(tokenizer),
    )

    train_result = trainer.train()
    print(f"\n  Training complete.")
    print(f"  Final train loss : {train_result.training_loss:.4f}")

    # ── save best model ─────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Best model saved -> {OUTPUT_DIR}/")

    # ── post-training evaluation ─────────────────────────────────
    sep("4 / 4  Post-training evaluation")
    print(f"  Loading best model from {OUTPUT_DIR}...")
    best_tokenizer = MarianTokenizer.from_pretrained(OUTPUT_DIR)
    best_model     = MarianMTModel.from_pretrained(OUTPUT_DIR).to(device)

    test_sub  = test_df.head(N_TEST_ROWS)
    sources   = test_sub[SRC_COL].tolist()
    refs      = test_sub[TGT_COL].tolist()

    print(f"  Translating {len(sources)} test rows with beam search (num_beams=4)...")
    hypotheses = translate_batch(sources, best_tokenizer, best_model, device)

    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()
    bleu_result = bleu_metric.corpus_score(hypotheses, [refs])
    chrf_result = chrf_metric.corpus_score(hypotheses, [refs])
    bleu_score  = round(bleu_result.score, 2)
    chrf_score  = round(chrf_result.score, 2)

    print("\n  Sample translations (first 3):")
    print("  " + "-" * 56)
    for i in range(min(3, len(sources))):
        print(f"  [{i+1}] Source     : {sources[i][:70]}")
        print(f"       Fine-tuned : {hypotheses[i][:70]}")
        print(f"       Reference  : {refs[i][:70]}")
        print()

    print("=" * 60)
    print(f"  Fine-tuned  BLEU : {bleu_score:>7.2f}  (baseline: {BASELINE_BLEU})")
    print(f"  Fine-tuned  chrF : {chrf_score:>7.2f}  (baseline: {BASELINE_CHRF})")
    bleu_delta = round(bleu_score - BASELINE_BLEU, 2)
    chrf_delta = round(chrf_score - BASELINE_CHRF, 2)
    print(f"  BLEU improvement : {bleu_delta:+.2f} points")
    print(f"  chrF improvement : {chrf_delta:+.2f} points")
    print("=" * 60)

    # ── save results ─────────────────────────────────────────────
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
        "model":          OUTPUT_DIR,
        "base_model":     BASE_MODEL,
        "epochs":         EPOCHS,
        "learning_rate":  LR,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "n_test_rows":    N_TEST_ROWS,
        "bleu":           bleu_score,
        "chrf":           chrf_score,
        "baseline_bleu":  BASELINE_BLEU,
        "baseline_chrf":  BASELINE_CHRF,
        "bleu_delta":     bleu_delta,
        "chrf_delta":     chrf_delta,
        "per_row":        per_row,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
