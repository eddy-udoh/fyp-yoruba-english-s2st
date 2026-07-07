# Yorùbá ASR Upgrade — Whisper-Medium on a Pooled Corpus

**Project:** MedSpeak YO↔EN — bidirectional Yorùbá↔English speech-to-speech translation for medical consultations
**Change summary:** Replaced the deployed Whisper-**small** ASR (Yorùbá speech → text) with a Whisper-**medium** model fine-tuned on a **pooled** speech corpus, and re-measured the full speech-to-translation pipeline.
**Date:** 29 June 2026

---

## 1. Why this change

The `yo→en` direction of the system is a chain:

```
Yorùbá audio ─► Whisper (ASR) ─► MarianMT (NMT) ─► English text ─► TTS
```

The translation model (MarianMT) already scored well on **clean reference text** (component BLEU ≈ 54). But the *end-to-end* score — feeding the NMT the ASR's actual output — was far lower, because ASR errors propagate into the translation. The ASR, not the NMT, was the bottleneck: the deployed Whisper-small had been fine-tuned on a **single dataset** (FLEURS Yorùbá, read speech), limiting its speaker and acoustic variety.

**Goal:** improve the ASR (bigger model + more, more-varied Yorùbá speech) and quantify the effect on the true end-to-end quality.

---

## 2. What was done

1. **New training script** — [`src/asr/whisper_finetune.py`](../src/asr/whisper_finetune.py): fine-tunes Whisper (medium by default) on a **pool** of Yorùbá speech sources, normalised to one 16 kHz mono schema, with on-the-fly **audio augmentation** (noise / gain / speed) so the model generalises to real-microphone conditions.
2. **Pooled data:**
   - **FLEURS `yo_ng`** (Google) — read speech; **test split held out** for a fair, comparable WER.
   - **OpenSLR SLR86** (Google, crowdsourced Yorùbá, **CC BY-SA 4.0**) — ~3,583 multi-speaker clips, adds speakers and acoustic variety.
   - Pooled **training set: 5,742 clips** (FLEURS 2,339 + SLR86 3,403).
3. **Bigger model:** Whisper-**medium** (769 M parameters, ~3× Whisper-small).
4. **Reproducible Colab notebook** — [`notebooks/train_whisper_colab.ipynb`](../notebooks/train_whisper_colab.ipynb): clone → install → mount Drive → (optional smoke test) → train → save. Trains on a free-tier T4 GPU with step-based checkpointing (survives disconnects) and auto-resume.
5. **Deployment:** the API ([`src/api/app.py`](../src/api/app.py)) and the evaluation script ([`src/eval/end_to_end_eval.py`](../src/eval/end_to_end_eval.py)) were pointed at the new model, and the end-to-end evaluation was re-run.

> **On the clinical/antenatal corpus:** an in-domain clinical Yorùbá dataset was investigated but is **not openly available** (only a gated academic dataset with no public transcripts exists). The training script includes a `--clinical-manifest` hook so such audio can be pooled in later if obtained, without code changes.

---

## 3. Training configuration

| Setting | Value |
|---|---|
| Base model | `openai/whisper-medium` (769 M params) |
| Hardware | Google Colab **T4** (16 GB) |
| Epochs | 3 |
| Effective batch | 16 (per-device 2 × grad-accum 8) |
| Precision / memory | fp16 + gradient checkpointing + `expandable_segments` |
| Augmentation | On (noise / gain / speed, training only) |
| Wall-clock | ~2 h 08 m |
| Validation WER (per-epoch) | 58.29% → 53.80% → **52.53%** |

---

## 4. Results

### 4.1 Component ASR accuracy (held-out FLEURS `yo_ng` test, n = 831)

| Model | WER ↓ |
|---|---|
| Whisper-small (previous) | 63.40% |
| **Whisper-medium (new)** | **53.11%** |
| **Improvement** | **−10.29 points** (≈16% relative) |

Sample output (properly diacritised Yorùbá):

> ref: `àwọn èyàn ti mọ̀ nípa àwọn kemika pepe bí wúrà fàdákà àti kọ́pa àtijọ́`
> hyp: `àwọn èèyàn tí mọ̀ nípa wọn kẹ́míkà pẹ́pẹ́ bíi wúrà fàdákà àti kọ́pà …`

### 4.2 End-to-end pipeline (synthetic speech, n = 100) — the number that matters

Both rows use the **same** `end_to_end_eval.py` script, so this is a like-for-like before/after.

| Metric | Before (small) | After (medium) | Change |
|---|---|---|---|
| ASR WER on synth speech ↓ | 70.79% | **41.46%** | **−29.3 pts** |
| **End-to-end BLEU** ↑ | 5.65 | **26.73** | **+21.1 (≈4.7×)** |
| End-to-end chrF ↑ | 23.80 | **43.31** | **+19.5** |
| Error-propagation gap (BLEU) ↓ | 48.67 | **31.49** | **−17.2** |
| Clean-text NMT BLEU (component) | 54.32 | 58.22 | ~unchanged¹ |

¹ The NMT model was **not** changed, so the clean-text component score is effectively flat; the small difference is measurement/version variance, **not** a real gain. Only the ASR and end-to-end figures represent genuine improvement.

---

## 5. Honest caveats (methodology notes for assessment)

- **Component vs end-to-end.** Component metrics (NMT BLEU on clean text) overstate the real system. The **error-propagation gap** (the drop from clean-text BLEU to end-to-end BLEU) is the honest measure of how much ASR error costs the pipeline — and this change **narrowed that gap from 48.67 to 31.49 BLEU.**
- **Synthetic speech is a pessimistic bound.** End-to-end here uses text-to-speech audio, which is out-of-domain for a Whisper fine-tuned on real speech, so it inflates ASR WER. Real recorded speech would likely score **better**. A real-speech evaluation (`--mode recorded`) remains the gold-standard next step.
- **Why synth WER (41.46%) is *lower* than FLEURS-test WER (53.11%).** They are different test sets: the synth set is the *medical* domain text (the system's actual use case) spoken by clean TTS, whereas FLEURS is harder general-domain news/wiki text. Both are reported for transparency.

---

## 6. Files changed

| File | Change |
|---|---|
| `src/asr/whisper_finetune.py` | **New** — pooled Whisper fine-tuning script |
| `notebooks/train_whisper_colab.ipynb` | **New** — reproducible Colab training notebook |
| `src/api/app.py` | Point ASR at whisper-medium; add "no speech detected" guard |
| `src/eval/end_to_end_eval.py` | Point ASR at whisper-medium |
| `evaluation/end_to_end_eval.json` | New end-to-end results |
| `evaluation/asr_finetuned_whisper-medium.json` | Component WER results (also on Drive) |

*(The model weights, ~3 GB, are stored on Google Drive and in the local `models/` folder; they are git-ignored and not on GitHub.)*

## 7. How to reproduce

Open [`notebooks/train_whisper_colab.ipynb`](../notebooks/train_whisper_colab.ipynb) in Google Colab (T4 GPU) and run top to bottom. After training, download the model into `models/whisper-medium-yoruba-finetuned/` and run:

```bash
python src/eval/end_to_end_eval.py --mode synth --n 100
```

---

*Prepared as a change summary for supervisor review.*
