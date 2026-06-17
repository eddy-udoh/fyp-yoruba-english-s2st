"""
results_report.py — Build report-ready figures + tables from the evaluation files.

Generates into evaluation/report/:
  Figures (PNG)
    training_curves.png   — training loss vs validation loss per epoch
    bleu_vs_epoch.png     — eval BLEU per epoch
    wer_improvement.png   — Whisper WER before vs after fine-tuning (bar)
    e2e_comparison.png    — clean-text NMT vs END-TO-END BLEU/chrF (the key chart)
    mos.png               — MOS rating distribution (if evaluation/mos_ratings.csv exists)
  Tables
    results_tables.md     — dataset stats, ASR, NMT, TTS, end-to-end, per-epoch tables

Every section is independent and skips cleanly if its source file is missing, so
you can run this at any stage. Run:
  python src/eval/results_report.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── paths ───────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_REPO   = os.path.dirname(os.path.dirname(_HERE))
EVAL    = os.path.join(_REPO, "evaluation")
SPLITS  = os.path.join(_REPO, "data", "splits")
RAW_CSV = os.path.join(_REPO, "data", "raw", "medical_dialogues_15k.csv")
MODEL_YOEN = os.path.join(_REPO, "models", "marian-yoruba-medical")
OUT     = os.path.join(EVAL, "report")
MOS_CSV = os.path.join(EVAL, "mos_ratings.csv")

# ── palette (matches the app / EDA figures) ─────────────────────────────────
GREEN, GREEN2, MINT, INK, AMBER = "#0F6E56", "#1D9E75", "#5DCAA5", "#0A2E1F", "#D97706"


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for item in (ax.title, ax.xaxis.label, ax.yaxis.label):
        item.set_color(INK)
    ax.tick_params(colors=INK)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [FIG]   {path}")


def find_trainer_state():
    """Prefer top-level trainer_state.json, else the highest checkpoint-*/ one."""
    top = os.path.join(MODEL_YOEN, "trainer_state.json")
    if os.path.exists(top):
        return top
    ckpts = glob.glob(os.path.join(MODEL_YOEN, "checkpoint-*", "trainer_state.json"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    return ckpts[-1]


# ── training curves + BLEU vs epoch ─────────────────────────────────────────
def training_figures(md: list):
    ts_path = find_trainer_state()
    state = _load(ts_path) if ts_path else None
    if not state or "log_history" not in state:
        print("  [skip] no trainer_state.json — training curves skipped.")
        return

    hist = state["log_history"]
    train = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    evals = [(h["epoch"], h.get("eval_loss"), h.get("eval_bleu")) for h in hist if "eval_loss" in h]

    # loss curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if train:
        ax.plot([e for e, _ in train], [l for _, l in train], color=GREEN, marker="o",
                label="Training loss", linewidth=2)
    if evals:
        ax.plot([e for e, _, _ in evals], [v for _, v, _ in evals], color=AMBER, marker="s",
                label="Validation loss", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=False); _style(ax)
    _save(fig, "training_curves.png")

    # bleu vs epoch
    bleus = [(e, b) for e, _, b in evals if b is not None]
    if bleus:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot([e for e, _ in bleus], [b for _, b in bleus], color=GREEN2, marker="o", linewidth=2)
        for e, b in bleus:
            ax.text(e, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation BLEU")
        ax.set_title("BLEU Score vs Epoch", fontsize=14, fontweight="bold", pad=12)
        _style(ax)
        _save(fig, "bleu_vs_epoch.png")

    # tables
    md.append("## Training loss per epoch\n")
    md.append("| Epoch | Training loss | Validation loss | Validation BLEU |")
    md.append("|------:|--------------:|----------------:|----------------:|")
    eval_by_epoch = {round(e): (vl, vb) for e, vl, vb in evals}
    seen = set()
    for e, l in train:
        re_ = round(e)
        if re_ in seen:
            continue
        seen.add(re_)
        vl, vb = eval_by_epoch.get(re_, (None, None))
        md.append(f"| {re_} | {l:.4f} | "
                  f"{'%.4f' % vl if vl is not None else '—'} | "
                  f"{'%.4f' % vb if vb is not None else '—'} |")
    md.append("")


# ── WER improvement bar ─────────────────────────────────────────────────────
def wer_figure(md: list):
    base = _load(os.path.join(EVAL, "asr_baseline_wer.json"))
    fine = _load(os.path.join(EVAL, "asr_finetuned_wer.json"))
    if not fine:
        print("  [skip] no asr_finetuned_wer.json — WER chart skipped.")
        return
    before = (base or {}).get("avg_wer_pct") or fine.get("baseline_wer_pct")
    after  = fine.get("avg_wer_pct")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(["Baseline\nWhisper", "Fine-tuned\nWhisper"], [before, after],
                  color=[MINT, GREEN], edgecolor=GREEN)
    for b, v in zip(bars, [before, after]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", fontsize=11, color=INK)
    ax.set_ylabel("Word Error Rate (%)")
    ax.set_title("ASR WER Improvement", fontsize=14, fontweight="bold", pad=12)
    _style(ax)
    _save(fig, "wer_improvement.png")

    md.append("## ASR results (WER)\n")
    md.append("| Model | WER before | WER after | Improvement |")
    md.append("|-------|-----------:|----------:|------------:|")
    md.append(f"| Whisper Small | {before:.2f}% | {after:.2f}% | {before - after:.2f} pts |")
    md.append("\n> Note: WER can exceed 100% when the model inserts more words than the "
              "reference (deletions+insertions+substitutions > reference length).\n")


# ── end-to-end vs clean-text comparison (the key chart) ─────────────────────
def e2e_figure(md: list):
    e2e = _load(os.path.join(EVAL, "end_to_end_eval.json"))
    if not e2e:
        print("  [skip] no end_to_end_eval.json — run end_to_end_eval.py first.")
        return
    c, e = e2e["clean_text_nmt"], e2e["end_to_end"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(2)
    w = 0.35
    ax.bar([i - w/2 for i in x], [c["bleu"], c["chrf"]], width=w, color=GREEN,
           label="Clean text (component)")
    ax.bar([i + w/2 for i in x], [e["bleu"], e["chrf"]], width=w, color=AMBER,
           label="End-to-end (speech)")
    ax.set_xticks(list(x)); ax.set_xticklabels(["BLEU", "chrF"])
    ax.set_ylabel("Score")
    ax.set_title(f"Component vs End-to-End ({e2e['mode']} speech, WER {e2e['asr_wer_pct']:.1f}%)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False); _style(ax)
    _save(fig, "e2e_comparison.png")

    md.append("## End-to-end vs component evaluation\n")
    md.append(f"Audio source: **{e2e['mode']}** speech · ASR WER on this set: "
              f"**{e2e['asr_wer_pct']:.2f}%** · n = {e2e['n']}\n")
    md.append("| Evaluation | BLEU | chrF |")
    md.append("|-----------|-----:|-----:|")
    md.append(f"| Clean-text NMT (component) | {c['bleu']:.2f} | {c['chrf']:.2f} |")
    md.append(f"| **End-to-end (speech→EN)** | {e['bleu']:.2f} | {e['chrf']:.2f} |")
    g = e2e["error_propagation_gap"]
    md.append(f"| Error-propagation gap | {g['bleu']:.2f} | {g['chrf']:.2f} |")
    md.append("")


# ── MOS ──────────────────────────────────────────────────────────────────────
def mos_figure(md: list):
    if not os.path.exists(MOS_CSV):
        print(f"  [skip] no {MOS_CSV} — copy mos_ratings_TEMPLATE.csv and fill it.")
        return
    df = pd.read_csv(MOS_CSV)
    if df["count"].sum() == 0:
        print("  [skip] mos_ratings.csv is empty (all zero) — fill it in.")
        return
    df = df.sort_values("rating")
    mos = (df["rating"] * df["count"]).sum() / df["count"].sum()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(df["rating"].astype(str), df["count"], color=GREEN, edgecolor=GREEN2)
    for b, v in zip(bars, df["count"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.2, str(int(v)), ha="center", color=INK)
    ax.set_xlabel("Rating (1–5)"); ax.set_ylabel("Number of listeners")
    ax.set_title(f"TTS Mean Opinion Score — MOS = {mos:.2f}", fontsize=13, fontweight="bold", pad=12)
    _style(ax)
    _save(fig, "mos.png")

    md.append("## TTS results (MOS)\n")
    md.append("| Metric | Value |\n|--------|------:|")
    md.append(f"| Mean Opinion Score | {mos:.2f} |")
    md.append(f"| Respondents | {int(df['count'].sum())} |\n")


# ── tables that don't need a figure ──────────────────────────────────────────
def dataset_table(md: list):
    def n_rows(p):
        return (sum(1 for _ in open(p, encoding="utf-8")) - 1) if os.path.exists(p) else None
    corpus = n_rows(RAW_CSV)
    tr = n_rows(os.path.join(SPLITS, "train.csv"))
    vl = n_rows(os.path.join(SPLITS, "val.csv"))
    te = n_rows(os.path.join(SPLITS, "test.csv"))
    md.append("## Dataset statistics\n")
    md.append("| Item | Value |\n|------|------:|")
    for label, v in [("Corpus size", corpus), ("Train", tr), ("Validation", vl), ("Test", te)]:
        if v is not None:
            md.append(f"| {label} | {v:,} |")
    md.append("")


def nmt_table(md: list):
    yoen = (_load(os.path.join(EVAL, "nmt_finetuned_yo-en.json"))
            or _load(os.path.join(EVAL, "nmt_finetuned_15k.json"))
            or _load(os.path.join(EVAL, "nmt_finetuned.json")))
    enyo = _load(os.path.join(EVAL, "nmt_finetuned_en-yo.json"))
    if not (yoen or enyo):
        return
    md.append("## NMT results (clean-text, component)\n")
    md.append("| Direction | BLEU | chrF |\n|-----------|-----:|-----:|")
    if yoen:
        md.append(f"| Yorùbá → English | {yoen.get('bleu','—')} | {yoen.get('chrf','—')} |")
    if enyo:
        md.append(f"| English → Yorùbá | {enyo.get('bleu','—')} | {enyo.get('chrf','—')} |")
    md.append("")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 60)
    print("  FYP S2ST — Results Report Builder")
    print("=" * 60)

    md = ["# Experimental Results\n",
          "_Auto-generated by src/eval/results_report.py_\n"]

    dataset_table(md)
    wer_figure(md)
    nmt_table(md)
    e2e_figure(md)
    mos_figure(md)
    training_figures(md)

    md_path = os.path.join(OUT, "results_tables.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"  [TABLE] {md_path}")
    print(f"\n  Done. Figures + tables in: {OUT}")


if __name__ == "__main__":
    main()
