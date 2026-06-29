"""
whisper_finetune.py — Fine-tune Whisper (Medium by default) for Yorùbá ASR on a
                      POOLED, multi-source corpus, with optional audio augmentation.
Phase 2, Step 2  (the yo→en speech direction — the real fix for end-to-end BLEU)

Run (Colab T4, recommended)
───────────────────────────
  # pool FLEURS yo + OpenSLR SLR86, augment, checkpoint to Drive:
  python src/asr/whisper_finetune.py \
      --model-size medium \
      --augment \
      --output-dir /content/drive/MyDrive/whisper-medium-yoruba-finetuned

  # quick smoke test (tiny model, few rows, no augment) to prove the pipeline:
  python src/asr/whisper_finetune.py --smoke

Why a POOLED corpus
───────────────────
The deployed Whisper-small was fine-tuned on FLEURS Yorùbá alone (read, single
domain), which is why end-to-end speech quality lags the clean-text NMT BLEU.
Pooling more real Yorùbá speech adds speakers + acoustic variety:

  1. google/fleurs (yo_ng)      — read speech.  TEST split is HELD OUT for
                                  comparability with the existing WER numbers.
  2. openslr/openslr (SLR86)    — crowdsourced multi-speaker read speech, 16 kHz,
                                  CC BY-SA 4.0.  ~3.5k clips, column `sentence`.
  3. (optional) --extra-hf      — any other verified HF speech dataset, given as
                                  "repo:config:split:audio_col:text_col".
                                  e.g. ÌròyìnSpeech once you confirm its repo id.
  4. (optional) --clinical-manifest CSV(audio_path,sentence) — drop IN-DOMAIN
                                  clinical/antenatal audio here if you obtain it.
                                  This is the in-domain lever; it pools straight in.

NOTE on the "Antenatal-Speech" corpus: it is NOT openly downloadable (the only
real artifact is a gated Springer dataset, no public transcripts). Use the
--clinical-manifest hook above if/when you get access to clinical recordings.

Methodology (mirrors marian_finetune.py)
─────────────────────────────────────────
  - eval during training = a held-out VALIDATION pool (FLEURS validation +
    a carved slice of the extra sources) — NOT the FLEURS test split.
  - FLEURS test stays held out; final WER is reported on it for comparability,
    plus an optional clinical test split (--clinical-test-manifest) for in-domain.
  - audio augmentation (Gaussian noise / gain / speed) is applied ON THE FLY at
    collate time, TRAIN ONLY, so the model generalises to real-mic conditions.
  - Drive checkpointing + auto-resume (get_last_checkpoint), like the NMT script.

Colab deps (pin datasets<3.0 as in notebooks/train_marian_colab.ipynb):
    pip install -q "datasets<3.0" "transformers>=4.46" accelerate jiwer soundfile librosa
"""
import argparse
import inspect
import json
import os
import random
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Yorùbá transcripts contain chars (ẹ ọ ṣ + tone marks) the Windows cp1252 console
# can't encode → print() would crash. Force UTF-8 stdout/stderr (no-op on Colab).
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset

import jiwer
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint


class WhisperTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that uses a SEPARATE, non-augmenting collator for evaluation.
    The base Trainer reuses one data_collator for train + eval; without this, audio
    augmentation would leak into the eval batches and corrupt the WER signal that
    drives best-model selection / early stopping."""

    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        train_collator = self.data_collator
        self.data_collator = self._eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = train_collator

# ── shared paths & config ───────────────────────────────────────────────────
EVAL_DIR     = "evaluation"
MODELS_DIR   = "models"
SR           = 16_000          # Whisper always works at 16 kHz mono
LANGUAGE     = "yo"
TASK         = "transcribe"

# unified schema every source is normalised to: {"audio": <16k array>, "sentence": str}
AUDIO_COL    = "audio"
TEXT_COL     = "sentence"

# ── model-size → HF base id ─────────────────────────────────────────────────
SIZE_TO_BASE = {
    "tiny":   "openai/whisper-tiny",
    "base":   "openai/whisper-base",
    "small":  "openai/whisper-small",
    "medium": "openai/whisper-medium",     # ← recommended sweet spot for T4
    "large":  "openai/whisper-large-v3",
}

# ── training hyper-parameters (T4-friendly defaults for MEDIUM) ─────────────
EPOCHS      = 3
LR          = 1e-5            # gentle — Whisper fine-tunes well at 1e-5..1e-4
BATCH_SIZE  = 8              # per device; medium on a 16 GB T4 usually needs 4
GRAD_ACCUM  = 2              # effective batch = BATCH_SIZE × GRAD_ACCUM
WARMUP      = 500
MAX_LABEL_LEN = 225          # Whisper's decoding cap is 448; transcripts are short

# step-based checkpointing so a Colab disconnect costs minutes, not a whole epoch
SAVE_EVAL_STEPS = 200        # checkpoint + eval every N steps (capped to steps/epoch)
EVAL_SUBSET     = 300        # clips used for the in-training eval metric (keeps it fast)

EARLY_STOP_PATIENCE = 4      # in EVAL events; eval now fires ~2x/epoch, so allow more grace

# ── audio augmentation knobs (TRAIN ONLY, applied on the fly) ───────────────
AUG_PROB    = 0.5            # probability a given clip is augmented at all
NOISE_SNR_DB = (10.0, 30.0)  # uniform SNR range for additive Gaussian noise
GAIN_DB     = (-6.0, 6.0)    # uniform gain range
SPEED_RANGE = (0.9, 1.1)     # uniform speed/tempo factor (pitch shifts too — ok)

# WER text normalisation — same spirit as src/asr/03_whisper_baseline.py
WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


# ── helpers ─────────────────────────────────────────────────────────────────
def sep(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def _supports_trc() -> bool:
    return "trust_remote_code" in inspect.signature(load_dataset).parameters


def _load(name, config=None, split=None):
    """load_dataset wrapper that passes trust_remote_code only if supported."""
    kwargs = {}
    if config is not None:
        kwargs["name"] = config
    if split is not None:
        kwargs["split"] = split
    if _supports_trc():
        kwargs["trust_remote_code"] = True
    return load_dataset(name, **kwargs)


def _normalise(ds, audio_col: str, text_col: str) -> Dataset:
    """Cast audio to 16 kHz mono and reduce columns to {audio, sentence}."""
    keep_audio = ds.column_names  # remember to drop everything else
    ds = ds.cast_column(audio_col, Audio(sampling_rate=SR))
    if audio_col != AUDIO_COL:
        ds = ds.rename_column(audio_col, AUDIO_COL)
    if text_col != TEXT_COL:
        ds = ds.rename_column(text_col, TEXT_COL)
    drop = [c for c in ds.column_names if c not in (AUDIO_COL, TEXT_COL)]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def _cap(ds, n: Optional[int]) -> Dataset:
    if n is not None and n < len(ds):
        return ds.select(range(n))
    return ds


# ── source loaders ──────────────────────────────────────────────────────────
def load_fleurs(max_rows: Optional[int]):
    """Returns (train_pool, val_pool, test_heldout) for google/fleurs yo_ng."""
    sep("Loading google/fleurs (yo_ng)")
    train = _normalise(_load("google/fleurs", "yo_ng", "train"),
                       "audio", "transcription")
    val   = _normalise(_load("google/fleurs", "yo_ng", "validation"),
                       "audio", "transcription")
    test  = _normalise(_load("google/fleurs", "yo_ng", "test"),
                       "audio", "transcription")
    train = _cap(train, max_rows)
    print(f"  FLEURS train/val/test: {len(train)} / {len(val)} / {len(test)} "
          f"(test HELD OUT)")
    return train, val, test


def load_slr86(max_rows: Optional[int], val_frac: float = 0.05):
    """OpenSLR SLR86 — single 'train' split; carve a small val slice off it."""
    sep("Loading openslr/openslr (SLR86)")
    try:
        ds = _normalise(_load("openslr/openslr", "SLR86", "train"),
                        "audio", "sentence")
    except Exception as e:
        print(f"  [WARN] SLR86 unavailable: {str(e)[:160]}")
        return None, None
    ds = _cap(ds, max_rows)
    ds = ds.train_test_split(test_size=val_frac, seed=42)
    print(f"  SLR86 train/val: {len(ds['train'])} / {len(ds['test'])}")
    return ds["train"], ds["test"]


def load_extra_hf(spec: str, max_rows: Optional[int], val_frac: float = 0.05):
    """Parse 'repo:config:split:audio_col:text_col' and load it.
    config may be empty (repo::split:audio:text). Carves a small val slice."""
    parts = spec.split(":")
    if len(parts) != 5:
        print(f"  [WARN] --extra-hf must be 'repo:config:split:audio_col:text_col', "
              f"got {spec!r} — skipping.")
        return None, None
    repo, config, split, a_col, t_col = parts
    sep(f"Loading extra HF source: {repo} ({config or 'default'})")
    try:
        ds = _normalise(_load(repo, config or None, split), a_col, t_col)
    except Exception as e:
        print(f"  [WARN] {repo} unavailable: {str(e)[:160]}")
        return None, None
    ds = _cap(ds, max_rows)
    ds = ds.train_test_split(test_size=val_frac, seed=42)
    print(f"  {repo} train/val: {len(ds['train'])} / {len(ds['test'])}")
    return ds["train"], ds["test"]


def load_manifest(csv_path: str, val_frac: float = 0.0):
    """Local CSV manifest with columns (audio_path, sentence).
    This is the IN-DOMAIN clinical hook — point it at antenatal/clinical audio."""
    import pandas as pd
    if not os.path.exists(csv_path):
        print(f"  [WARN] manifest not found: {csv_path} — skipping.")
        return None, None
    sep(f"Loading local manifest: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = {c.lower(): c for c in df.columns}
    a = cols.get("audio_path") or cols.get("path") or cols.get("audio")
    t = cols.get("sentence") or cols.get("transcription") or cols.get("text")
    if not a or not t:
        print(f"  [WARN] manifest needs audio_path + sentence columns "
              f"(found {list(df.columns)}) — skipping.")
        return None, None
    df = df[[a, t]].dropna()
    ds = Dataset.from_dict({AUDIO_COL: df[a].astype(str).tolist(),
                            TEXT_COL:  df[t].astype(str).tolist()})
    ds = ds.cast_column(AUDIO_COL, Audio(sampling_rate=SR))
    print(f"  manifest rows: {len(ds)}")
    if val_frac > 0:
        ds = ds.train_test_split(test_size=val_frac, seed=42)
        return ds["train"], ds["test"]
    return ds, None


# ── audio augmentation (numpy-only; TRAIN ONLY) ─────────────────────────────
def _augment_waveform(wav: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light, dependency-free augmentation: speed → gain → additive noise.
    Each transform is applied independently with its own coin-flip so clips see
    varied combinations. Returns float32 in roughly [-1, 1]."""
    wav = wav.astype(np.float32)

    # speed / tempo perturbation via linear resampling (also shifts pitch — fine)
    if rng.random() < 0.5:
        factor = rng.uniform(*SPEED_RANGE)
        n = max(1, int(round(len(wav) / factor)))
        idx = np.linspace(0, len(wav) - 1, num=n)
        wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)

    # random gain
    if rng.random() < 0.5:
        gain = 10.0 ** (rng.uniform(*GAIN_DB) / 20.0)
        wav = wav * gain

    # additive Gaussian noise at a random SNR
    if rng.random() < 0.5:
        sig_power = float(np.mean(wav ** 2)) + 1e-12
        snr_db = rng.uniform(*NOISE_SNR_DB)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        wav = wav + np.random.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)

    # guard against clipping blow-ups from stacked gain+noise
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav


# ── collator: feature-extract (+ optional augment) at batch time ────────────
@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any
    augment: bool = False
    seed: int = 42

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        # the tokenizer prepends <|startoftranscript|> (NOT bos); the model re-adds
        # it via decoder_start_token_id, so we strip this leading copy from labels.
        self._sot_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        wavs, texts = [], []
        for f in features:
            wav = np.asarray(f[AUDIO_COL]["array"], dtype=np.float32)
            if self.augment:
                wav = _augment_waveform(wav, self._rng)
            wavs.append(wav)
            texts.append(str(f[TEXT_COL]))

        # log-mel features (padded to 30 s by the extractor)
        feats = self.processor.feature_extractor(
            wavs, sampling_rate=SR, return_tensors="pt"
        )
        # tokenise transcripts → labels, pad, and mask pad with -100
        labels = self.processor.tokenizer(
            text=texts, padding=True, truncation=True,
            max_length=MAX_LABEL_LEN, return_tensors="pt",
        )
        label_ids = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100
        )
        # drop the leading <|startoftranscript|> the model prepends itself
        if label_ids.shape[1] > 0 and (label_ids[:, 0] == self._sot_id).all().cpu().item():
            label_ids = label_ids[:, 1:]

        return {"input_features": feats["input_features"], "labels": label_ids}


# ── metrics ─────────────────────────────────────────────────────────────────
def make_compute_metrics(processor):
    tok = processor.tokenizer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, tok.pad_token_id)
        pred_str = tok.batch_decode(pred_ids, skip_special_tokens=True)
        ref_str  = tok.batch_decode(label_ids, skip_special_tokens=True)
        out = jiwer.process_words(
            ref_str, pred_str,
            reference_transform=WER_TRANSFORM,
            hypothesis_transform=WER_TRANSFORM,
        )
        return {"wer": round(out.wer * 100, 2)}

    return compute_metrics


# ── held-out evaluation (greedy decode over a dataset) ──────────────────────
@torch.no_grad()
def evaluate_wer(model, processor, ds, device, label, batch_size=8):
    model.eval()
    refs, hyps = [], []
    for start in range(0, len(ds), batch_size):
        chunk = ds.select(range(start, min(start + batch_size, len(ds))))
        wavs = [np.asarray(r[AUDIO_COL]["array"], dtype=np.float32) for r in chunk]
        refs.extend(str(r[TEXT_COL]) for r in chunk)
        feats = processor.feature_extractor(
            wavs, sampling_rate=SR, return_tensors="pt"
        ).input_features.to(device)
        if device == "cuda":
            feats = feats.half()
        # pass language/task directly (not forced_decoder_ids) to avoid the
        # "forced_decoder_ids conflicts with task" warning and keep Yorùbá output
        ids = model.generate(feats, language=LANGUAGE, task=TASK,
                             max_new_tokens=MAX_LABEL_LEN)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
    out = jiwer.process_words(refs, hyps,
                              reference_transform=WER_TRANSFORM,
                              hypothesis_transform=WER_TRANSFORM)
    wer = out.wer
    print(f"  {label}: WER {wer*100:.2f}%  (n={len(ds)})")
    return round(wer, 4), refs, hyps


# ── main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Fine-tune Whisper for Yorùbá ASR on a pooled corpus")
    p.add_argument("--model-size", choices=list(SIZE_TO_BASE), default="medium",
                   help="Whisper size to fine-tune (default: medium).")
    p.add_argument("--output-dir", default=None,
                   help="Where to save checkpoints/model. On Colab point this at Drive, e.g. "
                        "/content/drive/MyDrive/whisper-medium-yoruba-finetuned, so it survives "
                        "a disconnect. Defaults to models/whisper-<size>-yoruba-finetuned/.")
    p.add_argument("--augment", action="store_true",
                   help="Enable on-the-fly audio augmentation (noise/gain/speed) on TRAIN.")
    p.add_argument("--no-slr86", action="store_true", help="Skip the OpenSLR SLR86 source.")
    p.add_argument("--extra-hf", action="append", default=[],
                   help="Extra HF source 'repo:config:split:audio_col:text_col' (repeatable).")
    p.add_argument("--clinical-manifest", default=None,
                   help="CSV(audio_path,sentence) of in-domain clinical audio to POOL into train.")
    p.add_argument("--clinical-test-manifest", default=None,
                   help="CSV(audio_path,sentence) held out as an in-domain TEST split.")
    p.add_argument("--max-per-source", type=int, default=None,
                   help="Cap rows per source (debugging / quota control).")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"Per-device train batch (default {BATCH_SIZE}; drop to 4 if a T4 OOMs).")
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--save-steps", type=int, default=SAVE_EVAL_STEPS,
                   help=f"Checkpoint + eval every N steps (default {SAVE_EVAL_STEPS}). Step-based "
                        "so a Colab disconnect costs minutes, not a whole epoch. Capped to "
                        "steps-per-epoch automatically.")
    p.add_argument("--eval-subset", type=int, default=EVAL_SUBSET,
                   help=f"Cap clips used for the in-training eval metric (default {EVAL_SUBSET}) so "
                        "frequent step-based eval stays fast. The final held-out FLEURS-test eval "
                        "always uses ALL test clips.")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint.")
    p.add_argument("--smoke", action="store_true",
                   help="Fast pipeline check: tiny model, 32 rows/source, 1 epoch, no augment.")
    args = p.parse_args()

    if args.smoke:
        args.model_size = "tiny"
        args.max_per_source = 32
        args.epochs = 1
        args.augment = False
        args.eval_subset = 16        # keep the smoke's in-loop eval fast

    base_id    = SIZE_TO_BASE[args.model_size]
    output_dir = args.output_dir or os.path.join(
        MODELS_DIR, f"whisper-{args.model_size}-yoruba-finetuned")
    out_json   = os.path.join(EVAL_DIR, f"asr_finetuned_whisper-{args.model_size}.json")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    sep(f"FYP S2ST — Whisper Fine-Tune [{args.model_size}] | Phase 2 Step 2")
    gpu    = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    print(f"  Base model : {base_id}")
    print(f"  Device     : {'GPU — ' + torch.cuda.get_device_name(0) if gpu else 'CPU'}")
    print(f"  Augment    : {args.augment}")
    print(f"  Effective batch : {args.batch_size * args.grad_accum} "
          f"(per_device={args.batch_size} × grad_accum={args.grad_accum})")

    # ── 1/4  load & pool data ────────────────────────────────────
    sep("1 / 4  Loading & pooling speech sources")
    train_parts, val_parts = [], []

    fl_train, fl_val, fl_test = load_fleurs(args.max_per_source)
    train_parts.append(fl_train)
    val_parts.append(fl_val)

    if not args.no_slr86:
        s_train, s_val = load_slr86(args.max_per_source)
        if s_train is not None:
            train_parts.append(s_train)
            val_parts.append(s_val)

    for spec in args.extra_hf:
        e_train, e_val = load_extra_hf(spec, args.max_per_source)
        if e_train is not None:
            train_parts.append(e_train)
            val_parts.append(e_val)

    if args.clinical_manifest:
        c_train, _ = load_manifest(args.clinical_manifest, val_frac=0.0)
        if c_train is not None:
            train_parts.append(c_train)

    clinical_test = None
    if args.clinical_test_manifest:
        clinical_test, _ = load_manifest(args.clinical_test_manifest, val_frac=0.0)

    train_ds = concatenate_datasets(train_parts).shuffle(seed=42)
    val_ds   = concatenate_datasets(val_parts).shuffle(seed=42)
    print(f"\n  POOLED train : {len(train_ds)} clips")
    print(f"  POOLED val   : {len(val_ds)} clips  (FLEURS test held out: {len(fl_test)})")

    # ── 2/4  processor & model ───────────────────────────────────
    sep("2 / 4  Loading processor & model")
    processor = WhisperProcessor.from_pretrained(base_id, language=LANGUAGE, task=TASK)
    model     = WhisperForConditionalGeneration.from_pretrained(base_id)
    # let the model decode Yorùbá; clear any forced ids so language is set via generation_config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False           # required with gradient checkpointing
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # train collator may augment; eval collator NEVER does (clean WER signal)
    train_collator = DataCollatorSpeechSeq2Seq(processor=processor, augment=args.augment)
    eval_collator  = DataCollatorSpeechSeq2Seq(processor=processor, augment=False)

    # ── 3/4  train ───────────────────────────────────────────────
    sep("3 / 4  Fine-tuning")
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    # checkpoint + eval every N steps, but at least once per epoch (and never 0)
    eval_save_steps = max(1, min(args.save_steps, steps_per_epoch))
    # cap the eval set so frequent step-based eval stays fast (val_ds is shuffled,
    # so a head slice is a mix of FLEURS + SLR86); final test eval uses ALL clips
    eval_ds = val_ds.select(range(min(args.eval_subset, len(val_ds))))
    print(f"  Epochs        : {args.epochs}")
    print(f"  Steps / epoch : {steps_per_epoch}")
    print(f"  Total steps   : {steps_per_epoch * args.epochs}")
    print(f"  Save/eval every: {eval_save_steps} steps  (eval on {len(eval_ds)} clips)\n")

    targs = Seq2SeqTrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        gradient_checkpointing      = True,
        learning_rate               = args.lr,
        warmup_steps                = min(WARMUP, max(1, steps_per_epoch)),
        weight_decay                = 0.0,
        fp16                        = gpu,
        predict_with_generate       = True,
        generation_max_length       = MAX_LABEL_LEN,
        eval_strategy               = "steps",
        eval_steps                  = eval_save_steps,
        save_strategy               = "steps",
        save_steps                  = eval_save_steps,
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,      # lower WER is better
        logging_steps               = max(1, eval_save_steps // 2),
        report_to                   = "none",
        remove_unused_columns       = False,      # collator needs the raw audio column
    )

    trainer_kwargs = dict(
        model           = model,
        args            = targs,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        data_collator   = train_collator,
        eval_collator   = eval_collator,
        compute_metrics = make_compute_metrics(processor),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )
    # transformers ≥4.46 renamed the Trainer `tokenizer` arg to `processing_class`.
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
    trainer = WhisperTrainer(**trainer_kwargs)

    resume_ckpt = None
    if not args.no_resume and os.path.isdir(output_dir):
        resume_ckpt = get_last_checkpoint(output_dir)
        if resume_ckpt:
            print(f"  [RESUME] continuing from {resume_ckpt}")

    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    print(f"\n  Training complete. Final train loss: {train_result.training_loss:.4f}")

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"  Best model + processor saved -> {output_dir}/")

    # ── 4/4  held-out evaluation (FLEURS test, + optional clinical) ──
    sep("4 / 4  Held-out evaluation")
    best = WhisperForConditionalGeneration.from_pretrained(output_dir).to(device)
    if device == "cuda":
        best = best.half()

    fleurs_wer, fl_refs, fl_hyps = evaluate_wer(best, processor, fl_test, device,
                                                "FLEURS test", batch_size=args.batch_size)
    clinical_wer = None
    if clinical_test is not None and len(clinical_test) > 0:
        clinical_wer, _, _ = evaluate_wer(best, processor, clinical_test, device,
                                          "Clinical test", batch_size=args.batch_size)

    print("\n  Sample transcriptions (FLEURS test, first 3):")
    print("  " + "-" * 56)
    for i in range(min(3, len(fl_refs))):
        print(f"  [{i+1}] ref : {fl_refs[i][:70]}")
        print(f"       hyp : {fl_hyps[i][:70]}")

    print("\n" + "=" * 60)
    print(f"  Fine-tuned {args.model_size}  |  FLEURS-test WER : {fleurs_wer*100:.2f}%")
    if clinical_wer is not None:
        print(f"                              Clinical-test WER : {clinical_wer*100:.2f}%")
    print("=" * 60)

    output = {
        "model":        output_dir,
        "base_model":   base_id,
        "model_size":   args.model_size,
        "language":     LANGUAGE,
        "mode":         "fine-tuned",
        "augment":      args.augment,
        "epochs":       args.epochs,
        "learning_rate": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "train_clips":  len(train_ds),
        "val_clips":    len(val_ds),
        "sources": {
            "fleurs": True,
            "slr86": not args.no_slr86,
            "extra_hf": args.extra_hf,
            "clinical_manifest": args.clinical_manifest,
        },
        "fleurs_test_n":  len(fl_test),
        "avg_wer":        fleurs_wer,
        "avg_wer_pct":    round(fleurs_wer * 100, 2),
        "clinical_test_wer_pct": round(clinical_wer * 100, 2) if clinical_wer is not None else None,
        "per_sample": [
            {"reference": r, "hypothesis": h} for r, h in zip(fl_refs, fl_hyps)
        ],
    }
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] {out_json}")
    print(f"  Next: copy {output_dir} into models/ and re-run src/eval/end_to_end_eval.py")


if __name__ == "__main__":
    main()
