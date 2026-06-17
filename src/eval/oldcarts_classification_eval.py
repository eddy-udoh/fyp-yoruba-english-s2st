"""
oldcarts_classification_eval.py — Evaluate OLDCARTS classification with the
metrics that actually apply to a classifier: accuracy, per-class precision/
recall/F1, a confusion matrix, and one-vs-rest ROC-AUC.

WHY A SURROGATE MODEL
─────────────────────
The Gemini classifier ([data/raw/classify_oldcarts_gemini.py]) outputs a single
category string per question — no probability scores — so ROC-AUC can't be
computed from it directly. Here we treat Gemini's labels as the dataset, train a
transparent TF-IDF + Logistic Regression classifier, and evaluate it on a
held-out split. This yields a genuine, reportable ROC-AUC + confusion matrix.

  ⚠ Caveat for the write-up: Gemini's labels are used as ground truth. For a
    fully rigorous study you would validate against a human-labelled gold subset;
    this measures how learnable/consistent the category scheme is.

Outputs (evaluation/report/):
  oldcarts_confusion_matrix.png
  oldcarts_roc_auc.png
  oldcarts_classification_eval.json   (all numbers)

Run (after the classified CSV exists):
  python src/eval/oldcarts_classification_eval.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize

# ── paths ───────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
CLASSIFIED_CSV = os.path.join(_REPO, "data", "raw", "medical_dialogues_15k_classified.csv")
OUT_DIR  = os.path.join(_REPO, "evaluation", "report")
OUT_JSON = os.path.join(_REPO, "evaluation", "oldcarts_classification_eval.json")

TEXT_COL = "Doctor_English"
CAT_COL  = "OLDCARTS_Category"
TEST_SIZE = 0.2
SEED = 42

GREEN, INK = "#0F6E56", "#0A2E1F"


def main():
    if not os.path.exists(CLASSIFIED_CSV):
        sys.exit(f"[ERROR] {CLASSIFIED_CSV} not found.\n"
                 f"        Run data/raw/classify_oldcarts_gemini.py first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  FYP S2ST — OLDCARTS Classification Evaluation")
    print("=" * 60)

    df = pd.read_csv(CLASSIFIED_CSV, encoding="utf-8").dropna(subset=[TEXT_COL, CAT_COL])
    X, y = df[TEXT_COL].astype(str), df[CAT_COL].astype(str)
    classes = sorted(y.unique())
    print(f"  Rows: {len(df):,} | Classes: {len(classes)} -> {classes}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"  Train: {len(X_tr):,} | Test: {len(X_te):,}\n")

    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(X_tr, y_tr)

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    proba_classes = list(clf.classes_)

    # ── scalar metrics ──
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)

    # one-vs-rest macro ROC-AUC
    y_te_bin = label_binarize(y_te, classes=proba_classes)
    try:
        macro_auc = roc_auc_score(y_te_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError as e:
        macro_auc = None
        print(f"  [WARN] ROC-AUC unavailable: {e}")

    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Macro F1        : {report['macro avg']['f1-score']:.4f}")
    if macro_auc is not None:
        print(f"  Macro ROC-AUC   : {macro_auc:.4f}")

    # ── confusion matrix figure ──
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else INK, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("OLDCARTS Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(OUT_DIR, "oldcarts_confusion_matrix.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [FIG]   {os.path.join(OUT_DIR, 'oldcarts_confusion_matrix.png')}")

    # ── ROC curves (one-vs-rest) ──
    per_class_auc = {}
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(proba_classes):
        if y_te_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_proba[:, i])
        a = roc_auc_score(y_te_bin[:, i], y_proba[:, i])
        per_class_auc[cls] = round(float(a), 4)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{cls} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    title = "OLDCARTS ROC Curves (one-vs-rest)"
    if macro_auc is not None:
        title += f"  —  macro AUC = {macro_auc:.3f}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(OUT_DIR, "oldcarts_roc_auc.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [FIG]   {os.path.join(OUT_DIR, 'oldcarts_roc_auc.png')}")

    # ── save metrics ──
    out = {
        "model": "TF-IDF (1-2gram) + LogisticRegression",
        "n_rows": int(len(df)),
        "n_test": int(len(X_te)),
        "classes": classes,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "macro_roc_auc": round(float(macro_auc), 4) if macro_auc is not None else None,
        "per_class_roc_auc": per_class_auc,
        "per_class_report": {c: report[c] for c in classes if c in report},
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"  [JSON]  {OUT_JSON}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
