"""
diacritic_validator.py — Standalone Yoruba Diacritic Audit
Run: python src/utils/diacritic_validator.py
"""
import pandas as pd
import unicodedata
import json
import os
import sys

# ── constants ────────────────────────────────────────────────────────────────
# Tonal vowels (precomposed NFC forms) — grave, acute, macron variants
TONAL_CHARS = set("àáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ")

# Yoruba underdot characters
UNDERDOT_CHARS = set("ọẹṣỌẸṢ")

# Union used for density counting
ALL_DIACRITICS = TONAL_CHARS | UNDERDOT_CHARS

MIN_DENSITY = 0.04  # 4%

SRC_CSV  = os.path.join("data", "raw", "text", "medical_dialogues.csv")
OUT_JSON = os.path.join("evaluation", "diacritic_audit.json")

# ── helpers ──────────────────────────────────────────────────────────────────
def has_tonal(text: str) -> bool:
    """True if text contains at least one tonal vowel (handles both NFC & NFD)."""
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    return (
        any(c in TONAL_CHARS for c in nfc)
        or "́" in nfd  # combining acute
        or "̀" in nfd  # combining grave
        or "̂" in nfd  # combining circumflex
    )

def has_underdot(text: str) -> bool:
    nfc = unicodedata.normalize("NFC", text)
    return any(c in UNDERDOT_CHARS for c in nfc)

def diacritic_density(text: str) -> float:
    """Fraction of non-space characters that are diacritic characters."""
    chars = [c for c in unicodedata.normalize("NFC", text) if c != " "]
    if not chars:
        return 0.0
    count = sum(1 for c in chars if c in ALL_DIACRITICS)
    return count / len(chars)

def audit_row(text: str) -> dict:
    """Return audit result for a single Yoruba string."""
    tone   = has_tonal(text)
    udot   = has_underdot(text)
    dens   = diacritic_density(text)
    dens_ok = dens >= MIN_DENSITY

    reasons = []
    if not tone:
        reasons.append("missing tonal marks (à á è é ì í ò ó ù ú)")
    if not udot:
        reasons.append("missing underdots (ọ ẹ ṣ)")
    if not dens_ok:
        reasons.append(f"diacritic density {dens*100:.2f}% < 4% minimum")

    return {
        "has_tonal":   tone,
        "has_underdot": udot,
        "density":     round(dens, 4),
        "density_pct": round(dens * 100, 2),
        "passed":      len(reasons) == 0,
        "reasons":     reasons,
    }

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("evaluation", exist_ok=True)

    if not os.path.exists(SRC_CSV):
        print(f"[ERROR] CSV not found: {SRC_CSV}")
        sys.exit(1)

    df = pd.read_csv(SRC_CSV, encoding="utf-8")
    print("=" * 60)
    print("  FYP S2ST — Diacritic Audit")
    print("=" * 60)
    print(f"  Source : {SRC_CSV}")
    print(f"  Rows   : {len(df)}")
    print()

    results   = []
    failed    = []
    densities = []

    for idx, row in df.iterrows():
        text   = str(row.get("Patient_Yoruba", ""))
        result = audit_row(text)
        result["row"]  = int(idx)
        result["text"] = text
        results.append(result)
        densities.append(result["density"])
        if not result["passed"]:
            failed.append(result)

    total      = len(results)
    n_passed   = total - len(failed)
    avg_dens   = sum(densities) / total if total else 0

    # ── print summary ─────────────────────────────────────────────
    print(f"  Total rows   : {total}")
    print(f"  Passed       : {n_passed}  ({n_passed/total*100:.1f}%)")
    print(f"  Failed       : {len(failed)}  ({len(failed)/total*100:.1f}%)")
    print(f"  Avg density  : {avg_dens*100:.2f}%")
    print()

    if failed:
        print("  FAILED ROWS (row index | density | reasons | text preview)")
        print("  " + "-" * 56)
        for f in failed:
            preview = f["text"][:60].replace("\n", " ")
            reasons = "; ".join(f["reasons"])
            print(f"  Row {f['row']:>4} | {f['density_pct']:>5.2f}% | {reasons}")
            print(f"         text: {preview}")
            print()
    else:
        print("  All rows passed diacritic validation.")

    # ── save JSON ─────────────────────────────────────────────────
    audit = {
        "source_csv":   SRC_CSV,
        "total_rows":   total,
        "passed":       n_passed,
        "failed":       len(failed),
        "avg_density_pct": round(avg_dens * 100, 4),
        "min_density_threshold_pct": MIN_DENSITY * 100,
        "failed_rows":  [
            {
                "row":       f["row"],
                "text":      f["text"],
                "density_pct": f["density_pct"],
                "reasons":   f["reasons"],
            }
            for f in failed
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(audit, fp, ensure_ascii=False, indent=2)

    print(f"  Audit saved -> {OUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
