"""
eda_oldcarts.py — Exploratory Data Analysis for the medical dialogue corpus.

Produces report-ready figures (PNG) + a summary JSON:
  1. oldcarts_pie.png      — OLDCARTS category distribution of the 15k corpus
  2. oldcarts_bar.png      — same, as a sorted bar chart with the 12% balance line
  3. length_distribution.png — word-count distributions (Yoruba vs English)
  4. diacritic_density.png — Yoruba diacritic-density histogram
  5. split_sizes.png       — train / val / test split sizes
  6. eda_summary.json      — all the numbers behind the charts

Run:
  python src/utils/eda_oldcarts.py

Notes:
  - The OLDCARTS charts need the classified CSV produced by
    data/raw/classify_oldcarts_gemini.py. If it's missing, those two charts are
    skipped and the rest still run off the raw 15k corpus.
"""
import json
import os
import unicodedata

import matplotlib
matplotlib.use("Agg")               # headless-safe (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths (resolve relative to repo root regardless of cwd) ─────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))      # src/utils/
_REPO      = os.path.dirname(os.path.dirname(_HERE))         # repo root
RAW_CSV        = os.path.join(_REPO, "data", "raw", "medical_dialogues_15k.csv")
CLASSIFIED_CSV = os.path.join(_REPO, "data", "raw", "medical_dialogues_15k_classified.csv")
SPLITS_DIR     = os.path.join(_REPO, "data", "splits")
OUT_DIR        = os.path.join(_REPO, "evaluation", "eda")

YO_COL  = "Patient_Yoruba"
EN_COL  = "Clinical_Translation_English"
CAT_COL = "OLDCARTS_Category"

OLDCARTS_ORDER = [
    "Onset", "Location", "Duration", "Characteristics",
    "Aggravating", "Relieving", "Timing", "Severity",
]
BALANCE_CUTOFF = 0.12   # 12% reference line (thesis "cut-off point")

# ── medical green palette (matches the frontend) ────────────────────────────
GREEN   = "#0F6E56"
GREEN2  = "#1D9E75"
GREENS  = ["#0F6E56", "#1D9E75", "#5DCAA5", "#9FE1CB", "#0A4D3C",
           "#37B488", "#7FD6B6", "#C8E6D8"]
INK     = "#0A2E1F"

# ── diacritic helpers ───────────────────────────────────────────────────────
_TONAL    = set("àáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ")
_UNDERDOT = set("ọẹṣỌẸṢ")
_ALL_DIA  = _TONAL | _UNDERDOT


def _diacritic_density(text: str) -> float:
    chars = [c for c in unicodedata.normalize("NFC", str(text)) if c != " "]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c in _ALL_DIA) / len(chars)


def _word_count(text) -> int:
    return len(str(text).split())


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


# ── 1 & 2: OLDCARTS distribution ────────────────────────────────────────────
def plot_oldcarts(df: pd.DataFrame, summary: dict):
    counts = df[CAT_COL].value_counts()
    # keep canonical order, include any zero categories
    counts = counts.reindex(OLDCARTS_ORDER).fillna(0).astype(int)
    total  = int(counts.sum())
    pct    = (counts / total * 100).round(2)

    summary["oldcarts"] = {
        "total": total,
        "counts": counts.to_dict(),
        "percent": pct.to_dict(),
    }

    # ── pie ──
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct=lambda p: f"{p:.1f}%",
        colors=GREENS,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(color=INK, fontsize=11),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.set_title(f"OLDCARTS Category Distribution — {total:,} dialogues",
                 fontsize=14, fontweight="bold", pad=18)
    _save(fig, "oldcarts_pie.png")

    # ── sorted bar with 12% balance line ──
    sorted_pct = pct.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_pct.index, sorted_pct.values, color=GREEN2, edgecolor=GREEN)
    ax.axhline(BALANCE_CUTOFF * 100, color="#D97706", linestyle="--", linewidth=1.5,
               label=f"{BALANCE_CUTOFF*100:.0f}% balance cut-off")
    for b, v in zip(bars, sorted_pct.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("Share of corpus (%)")
    ax.set_title("OLDCARTS Category Balance", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(0, max(sorted_pct.values) * 1.2)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _style_ax(ax)
    _save(fig, "oldcarts_bar.png")


# ── 3: length distributions ─────────────────────────────────────────────────
def plot_lengths(df: pd.DataFrame, summary: dict):
    yo_len = df[YO_COL].map(_word_count)
    en_len = df[EN_COL].map(_word_count)

    summary["word_counts"] = {
        "yoruba_source":  {"mean": round(yo_len.mean(), 2), "median": int(yo_len.median()),
                           "min": int(yo_len.min()), "max": int(yo_len.max())},
        "english_target": {"mean": round(en_len.mean(), 2), "median": int(en_len.median()),
                           "min": int(en_len.min()), "max": int(en_len.max())},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = range(0, int(max(yo_len.max(), en_len.max())) + 3, 2)
    ax.hist(yo_len, bins=bins, alpha=0.7, label=f"Yorùbá source (μ={yo_len.mean():.1f})", color=GREEN)
    ax.hist(en_len, bins=bins, alpha=0.6, label=f"English target (μ={en_len.mean():.1f})", color=GREEN2)
    ax.set_xlabel("Words per utterance")
    ax.set_ylabel("Number of dialogues")
    ax.set_title("Utterance Length Distribution", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False)
    _style_ax(ax)
    _save(fig, "length_distribution.png")


# ── 4: diacritic density ────────────────────────────────────────────────────
def plot_diacritics(df: pd.DataFrame, summary: dict):
    dens = df[YO_COL].map(_diacritic_density)
    summary["diacritic_density"] = {
        "mean_pct": round(dens.mean() * 100, 2),
        "median_pct": round(dens.median() * 100, 2),
        "below_4pct": int((dens < 0.04).sum()),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dens * 100, bins=40, color=GREEN, edgecolor="white")
    ax.axvline(4, color="#D97706", linestyle="--", linewidth=1.5, label="4% minimum threshold")
    ax.axvline(dens.mean() * 100, color=INK, linestyle="-", linewidth=1.5,
               label=f"mean = {dens.mean()*100:.1f}%")
    ax.set_xlabel("Diacritic density (%)")
    ax.set_ylabel("Number of dialogues")
    ax.set_title("Yorùbá Diacritic Density", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False)
    _style_ax(ax)
    _save(fig, "diacritic_density.png")


# ── 5: split sizes ──────────────────────────────────────────────────────────
def plot_splits(summary: dict):
    sizes = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(SPLITS_DIR, f"{split}.csv")
        if os.path.exists(path):
            sizes[split] = sum(1 for _ in open(path, encoding="utf-8")) - 1
    if not sizes:
        print("  [SKIP] No data/splits/*.csv found — skipping split chart.")
        return
    summary["splits"] = sizes

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{k}\n({v:,})" for k, v in sizes.items()]
    bars = ax.bar(labels, list(sizes.values()), color=[GREEN, GREEN2, "#5DCAA5"],
                  edgecolor=GREEN)
    total = sum(sizes.values())
    for b, v in zip(bars, sizes.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01,
                f"{v/total*100:.0f}%", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"Train / Val / Test Split — {total:,} pairs (80/10/10)",
                 fontsize=13, fontweight="bold", pad=14)
    _style_ax(ax)
    _save(fig, "split_sizes.png")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  FYP S2ST — Corpus EDA")
    print("=" * 60)

    summary: dict = {}

    # OLDCARTS charts need the classified CSV; everything else uses the raw 15k.
    if os.path.exists(CLASSIFIED_CSV):
        print(f"  Loading classified corpus: {CLASSIFIED_CSV}")
        df = pd.read_csv(CLASSIFIED_CSV, encoding="utf-8")
        print(f"  Rows: {len(df):,}\n")
        if CAT_COL in df.columns:
            plot_oldcarts(df, summary)
        else:
            print(f"  [WARN] '{CAT_COL}' column missing — skipping OLDCARTS charts.")
    elif os.path.exists(RAW_CSV):
        print(f"  [WARN] Classified CSV not found — OLDCARTS charts skipped.")
        print(f"         Run: python data/raw/classify_oldcarts_gemini.py")
        print(f"  Loading raw corpus: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, encoding="utf-8")
        print(f"  Rows: {len(df):,}\n")
    else:
        print(f"  [ERROR] No corpus found at {RAW_CSV}")
        return

    plot_lengths(df, summary)
    plot_diacritics(df, summary)
    plot_splits(summary)

    out_json = os.path.join(OUT_DIR, "eda_summary.json")
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {out_json}")
    print("\n  Done. Figures + summary in:", OUT_DIR)


if __name__ == "__main__":
    main()
