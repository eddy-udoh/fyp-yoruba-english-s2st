"""
marian_finetune.py — Fine-tune MarianMT on a domain-balanced corpus, EITHER
                     direction (yo→en or en→yo).
Phase 3, Step 2

Run:
  python src/nmt/marian_finetune.py --direction yo-en   # Yoruba → English (default)
  python src/nmt/marian_finetune.py --direction en-yo   # English → Yoruba

Why domain-balanced
───────────────────
Fine-tuning ONLY on medical dialogues caused domain collapse / catastrophic
forgetting (even "Good morning" came back as a medical sentence). Each training
run blends three sources so general translation ability is preserved:

  1. Medical corpus      — data/splits/train.csv (the 15k corpus, ~12k rows)
  2. General domain      — opus100 en-yo parallel pairs (downloaded)
  3. Greetings/small-talk — a small hand-written set, oversampled

Directions
──────────
  yo-en : base Helsinki-NLP/opus-mt-yo-en   | src=Patient_Yoruba  tgt=Clinical_EN
  en-yo : base Helsinki-NLP/opus-mt-en-nic  | src=Clinical_EN      tgt=Patient_Yoruba
          (en-nic is multilingual → every English source is prefixed ">>yor<< "
           to select Yoruba output, both in training and at inference)

Methodology
───────────
  - eval_dataset is the VAL split — NOT test — so there is no leakage.
  - test split is held out and only used for the final reported score.
  - generation guards (no_repeat_ngram_size, repetition_penalty) kill the
    "patient's patient's patient's" degeneration.
  - EarlyStoppingCallback stops once eval BLEU stops improving.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# ── shared paths & config ──────────────────────────────────────────────────
TRAIN_CSV   = os.path.join("data", "splits", "train.csv")   # 15k corpus, ~12k rows
VAL_CSV     = os.path.join("data", "splits", "val.csv")     # eval during training
TEST_CSV    = os.path.join("data", "splits", "test.csv")    # held-out, final score only
EVAL_DIR    = "evaluation"

MED_YO_COL  = "Patient_Yoruba"
MED_EN_COL  = "Clinical_Translation_English"

MAX_SRC_LEN = 128
MAX_TGT_LEN = 256
EPOCHS      = 5
LR          = 3e-5      # gentler than 5e-5 — slows catastrophic forgetting
BATCH_SIZE  = 8         # per device; grad_accum=2 → effective = 16
GRAD_ACCUM  = 2
N_TEST_ROWS = 100

# ── domain-balancing config ────────────────────────────────────────────────
GENERAL_MAX         = 12_000   # max general-domain (opus100) pairs to blend in
GREETING_OVERSAMPLE = 50       # repeat each greeting pair this many times

# ── generation guards (kill repetitive degeneration) ───────────────────────
NUM_BEAMS          = 4
NO_REPEAT_NGRAM    = 3
REPETITION_PENALTY = 1.5

# ── early stopping ─────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 2

# ── per-direction configuration ────────────────────────────────────────────
#   src_prefix is prepended to EVERY source string (training + inference).
#   opus_src/opus_tgt pick which side of the opus100 pair is source vs target.
DIRECTIONS = {
    "yo-en": {
        "base":       "Helsinki-NLP/opus-mt-yo-en",
        "med_src":    MED_YO_COL,
        "med_tgt":    MED_EN_COL,
        "src_prefix": "",
        "opus_src":   "yo",
        "opus_tgt":   "en",
        "output_dir": os.path.join("models", "marian-yoruba-medical"),
        "baseline_bleu": 0.45,
        "baseline_chrf": 12.73,
    },
    "en-yo": {
        "base":       "Helsinki-NLP/opus-mt-en-nic",
        "med_src":    MED_EN_COL,
        "med_tgt":    MED_YO_COL,
        "src_prefix": ">>yor<< ",
        "opus_src":   "en",
        "opus_tgt":   "yo",
        "output_dir": os.path.join("models", "marian-english-yoruba"),
        "baseline_bleu": 0.0,
        "baseline_chrf": 0.0,
    },
}

# ── hand-written greetings / small-talk (yo, en) ───────────────────────────
GREETINGS = [
    ("Ẹ kú àárọ̀",                  "Good morning"),
    ("Ẹ kú ọ̀sán",                  "Good afternoon"),
    ("Ẹ kú alẹ́",                   "Good evening"),
    ("Ẹ káàbọ̀",                    "Welcome"),
    ("Báwo ni?",                    "How are you?"),
    ("Ṣé àlàáfíà ni?",              "Are you well?"),
    ("Mo wà dáadáa, ẹ ṣé",          "I am fine, thank you"),
    ("Ẹ ṣé",                        "Thank you"),
    ("Mo dúpẹ́",                    "I am grateful"),
    ("Ẹ jọ̀ọ́",                     "Please"),
    ("Ó dàbọ̀",                     "Goodbye"),
    ("Kí ni orúkọ rẹ?",            "What is your name?"),
    ("Orúkọ mi ni Adé",            "My name is Ade"),
    ("Ṣé o gbọ́ mi?",              "Do you understand me?"),
    ("Ẹ jẹ́ kí n ràn ọ́ lọ́wọ́",     "Let me help you"),
    ("Kí ló dé?",                   "What is wrong?"),
    ("Jọ̀wọ́ jókòó",               "Please sit down"),
    ("Ẹ kú iṣẹ́",                   "Well done with your work"),
]

# greeting source phrases used for the post-training sanity check, per direction
SANITY = {
    "yo-en": ["Ẹ kú àárọ̀", "Báwo ni?", "Ẹ ṣé"],
    "en-yo": ["Good morning", "How are you?", "Thank you"],
}

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

def _frame(src_list, tgt_list, prefix: str) -> pd.DataFrame:
    """Build a standardized src/tgt frame, prefixing sources and dropping blanks."""
    df = pd.DataFrame({"src": src_list, "tgt": tgt_list}).dropna()
    df = df[(df["src"].astype(str).str.strip() != "") &
            (df["tgt"].astype(str).str.strip() != "")].reset_index(drop=True)
    df["src"] = prefix + df["src"].astype(str)
    return df

def medical_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return _frame(df[cfg["med_src"]].tolist(), df[cfg["med_tgt"]].tolist(), cfg["src_prefix"])

def general_frame(cfg: dict, max_rows: int) -> pd.DataFrame:
    """Download opus100 en-yo and return src/tgt pairs in the chosen direction."""
    try:
        ds = load_dataset("opus100", "en-yo", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  [WARN] Could not load opus100 general data ({e}).")
        print("  [WARN] Proceeding with medical + greetings only.")
        return pd.DataFrame(columns=["src", "tgt"])

    n = min(max_rows, len(ds))
    ds = ds.select(range(n))
    src_list = [r["translation"][cfg["opus_src"]] for r in ds]
    tgt_list = [r["translation"][cfg["opus_tgt"]] for r in ds]
    return _frame(src_list, tgt_list, cfg["src_prefix"])

def greetings_frame(cfg: dict, oversample: int) -> pd.DataFrame:
    if cfg["opus_src"] == "yo":          # yo→en : source=Yoruba, target=English
        src_list = [yo for yo, _ in GREETINGS]
        tgt_list = [en for _, en in GREETINGS]
    else:                                # en→yo : source=English, target=Yoruba
        src_list = [en for _, en in GREETINGS]
        tgt_list = [yo for yo, _ in GREETINGS]
    df = _frame(src_list, tgt_list, cfg["src_prefix"])
    return pd.concat([df] * oversample, ignore_index=True)

# ── tokenize ───────────────────────────────────────────────────────────────
def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        model_inputs = tokenizer(batch["src"], max_length=MAX_SRC_LEN, truncation=True)
        labels = tokenizer(text_target=batch["tgt"], max_length=MAX_TGT_LEN, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenize

# ── compute BLEU during training (fast, token-level) ──────────────────────
def make_compute_metrics(tokenizer):
    bleu_metric = BLEU(effective_order=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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
            ids = model.generate(
                **tokens,
                max_length=MAX_TGT_LEN,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                repetition_penalty=REPETITION_PENALTY,
            )
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
        results.extend(decoded)
    return results

# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune MarianMT (yo-en or en-yo)")
    parser.add_argument("--direction", choices=list(DIRECTIONS), default="yo-en",
                        help="Translation direction to fine-tune (default: yo-en)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save checkpoints/model. On Colab point this at "
                             "Drive, e.g. /content/drive/MyDrive/marian-yoruba-medical, "
                             "so checkpoints survive a disconnect. Defaults to models/<dir>/.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing checkpoint and train from scratch.")
    args = parser.parse_args()
    direction = args.direction
    cfg       = DIRECTIONS[direction]

    output_dir = args.output_dir or cfg["output_dir"]
    out_json   = os.path.join(EVAL_DIR, f"nmt_finetuned_{direction}.json")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(EVAL_DIR,   exist_ok=True)

    sep(f"FYP S2ST — MarianMT Fine-Tune [{direction}] | Phase 3 Step 2")

    gpu    = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    print(f"  Direction : {direction}   (base: {cfg['base']})")
    print(f"  Device    : {'GPU — ' + torch.cuda.get_device_name(0) if gpu else 'CPU'}")
    print(f"  Effective batch : {BATCH_SIZE * GRAD_ACCUM}  "
          f"(per_device={BATCH_SIZE} × grad_accum={GRAD_ACCUM})")

    # ── load + blend data ────────────────────────────────────────
    sep("1 / 4  Loading & blending data")
    med_train = medical_frame(load_csv(TRAIN_CSV, "Medical train"), cfg)
    med_val   = medical_frame(load_csv(VAL_CSV,   "Medical val"),   cfg)
    test_df   = load_csv(TEST_CSV, "Medical test (held out)")

    print("\n  Downloading general-domain pairs (opus100 en-yo) …")
    gen_train = general_frame(cfg, GENERAL_MAX)
    print(f"  General train  : {len(gen_train)} rows")

    greet_train = greetings_frame(cfg, GREETING_OVERSAMPLE)
    print(f"  Greetings train: {len(greet_train)} rows "
          f"({len(GREETINGS)} unique × {GREETING_OVERSAMPLE})")

    train_blend = pd.concat([med_train, gen_train, greet_train], ignore_index=True)
    train_blend = train_blend.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Blended TRAIN  : {len(train_blend)} rows total")
    print(f"    medical   : {len(med_train)}  ({len(med_train)/len(train_blend)*100:.1f}%)")
    print(f"    general   : {len(gen_train)}  ({len(gen_train)/len(train_blend)*100:.1f}%)")
    print(f"    greetings : {len(greet_train)}  ({len(greet_train)/len(train_blend)*100:.1f}%)")

    train_dataset = Dataset.from_pandas(train_blend)
    eval_dataset  = Dataset.from_pandas(med_val)   # VAL, not test — no leakage

    # ── load tokenizer & model ───────────────────────────────────
    sep("2 / 4  Loading tokenizer & model")
    print(f"  Base model: {cfg['base']}")
    tokenizer = MarianTokenizer.from_pretrained(cfg["base"])
    model     = MarianMTModel.from_pretrained(cfg["base"])
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # bake generation guards into the model so the trainer's eval uses them too
    model.generation_config.num_beams            = NUM_BEAMS
    model.generation_config.no_repeat_ngram_size = NO_REPEAT_NGRAM
    model.generation_config.repetition_penalty   = REPETITION_PENALTY

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
        output_dir                  = output_dir,
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
        generation_num_beams        = NUM_BEAMS,
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
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    # ── resume from last checkpoint if the previous run was interrupted ──
    resume_ckpt = None
    if not args.no_resume and os.path.isdir(output_dir):
        resume_ckpt = get_last_checkpoint(output_dir)
        if resume_ckpt:
            print(f"  [RESUME] Found checkpoint — continuing from {resume_ckpt}")
        else:
            print("  [RESUME] No checkpoint found — training from scratch.")

    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    print(f"\n  Training complete.")
    print(f"  Final train loss : {train_result.training_loss:.4f}")

    # ── save best model ─────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Best model saved -> {output_dir}/")

    # ── post-training evaluation (on held-out TEST) ──────────────
    sep("4 / 4  Post-training evaluation (held-out test split)")
    print(f"  Loading best model from {output_dir}...")
    best_tokenizer = MarianTokenizer.from_pretrained(output_dir)
    best_model     = MarianMTModel.from_pretrained(output_dir).to(device)

    test_sub  = test_df.head(N_TEST_ROWS)
    sources   = [cfg["src_prefix"] + str(s) for s in test_sub[cfg["med_src"]].tolist()]
    refs      = [str(t) for t in test_sub[cfg["med_tgt"]].tolist()]

    print(f"  Translating {len(sources)} test rows with beam search (num_beams={NUM_BEAMS})...")
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

    # sanity check: greetings should NOT come back medical
    print("  Greeting sanity check:")
    print("  " + "-" * 56)
    sanity_src = [cfg["src_prefix"] + s for s in SANITY[direction]]
    sanity_out = translate_batch(sanity_src, best_tokenizer, best_model, device)
    for s, o in zip(SANITY[direction], sanity_out):
        print(f"  {s!r:>20}  ->  {o}")
    print()

    bleu_delta = round(bleu_score - cfg["baseline_bleu"], 2)
    chrf_delta = round(chrf_score - cfg["baseline_chrf"], 2)
    print("=" * 60)
    print(f"  Fine-tuned  BLEU : {bleu_score:>7.2f}  (baseline: {cfg['baseline_bleu']})")
    print(f"  Fine-tuned  chrF : {chrf_score:>7.2f}  (baseline: {cfg['baseline_chrf']})")
    print(f"  BLEU improvement : {bleu_delta:+.2f} points")
    print(f"  chrF improvement : {chrf_delta:+.2f} points")
    print("=" * 60)

    # ── save results ─────────────────────────────────────────────
    per_row = [
        {"row": i, "source": sources[i], "hypothesis": hypotheses[i], "reference": refs[i]}
        for i in range(len(sources))
    ]
    output = {
        "direction":      direction,
        "model":          output_dir,
        "base_model":     cfg["base"],
        "epochs":         EPOCHS,
        "learning_rate":  LR,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "train_blend": {
            "medical":   len(med_train),
            "general":   len(gen_train),
            "greetings": len(greet_train),
            "total":     len(train_blend),
        },
        "eval_split":     "val (data/splits/val.csv)",
        "n_test_rows":    N_TEST_ROWS,
        "generation": {
            "num_beams":            NUM_BEAMS,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM,
            "repetition_penalty":   REPETITION_PENALTY,
        },
        "bleu":          bleu_score,
        "chrf":          chrf_score,
        "baseline_bleu": cfg["baseline_bleu"],
        "baseline_chrf": cfg["baseline_chrf"],
        "bleu_delta":    bleu_delta,
        "chrf_delta":    chrf_delta,
        "per_row":       per_row,
    }
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] {out_json}")


if __name__ == "__main__":
    main()
