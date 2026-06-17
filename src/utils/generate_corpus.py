"""
generate_corpus.py — Generate Yoruba↔English medical dialogue corpus via Google Gemini API.

Appends rows to data/raw/medical_dialogues_15k.csv until 15,000 total rows.
Each row: Doctor_English, Patient_Yoruba, Clinical_Translation_English, Direct_Translation_English

Usage:
  python src/utils/generate_corpus.py
"""
from dotenv import load_dotenv
import os
load_dotenv()
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

from google import genai

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL   = "gemini-2.5-flash"

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO    = Path(__file__).resolve().parent.parent.parent
CSV_PATH = _REPO / "data" / "raw" / "medical_dialogues_15k.csv"
CSV_FIELDNAMES = [
    "Doctor_English",
    "Patient_Yoruba",
    "Clinical_Translation_English",
    "Direct_Translation_English",
]

# ── run config ────────────────────────────────────────────────────────────────
TARGET_ROWS   = 15_000
BATCH_SIZE    = 50
SLEEP_BETWEEN = 2

# ── diacritic validation ──────────────────────────────────────────────────────
_TONAL    = set("àáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ")
_UNDERDOT = set("ọẹṣỌẸṢ")
_ALL_DIA  = _TONAL | _UNDERDOT
_MIN_DENSITY = 0.04


def _diacritic_density(text: str) -> float:
    chars = [c for c in unicodedata.normalize("NFC", text) if c != " "]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c in _ALL_DIA) / len(chars)


def _valid_yoruba(text: str) -> bool:
    nfc = unicodedata.normalize("NFC", text)
    return (
        any(c in _TONAL    for c in nfc)
        and any(c in _UNDERDOT for c in nfc)
        and _diacritic_density(nfc) >= _MIN_DENSITY
    )


# ── prompt ────────────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = (
    "You are a medical linguistics expert fluent in English and Yoruba. "
    "Generate exactly {n} unique medical dialogue exchanges between a doctor (English) "
    "and a patient (Yoruba) for a Nigerian hospital setting.\n\n"
    "CRITICAL: Every Yoruba utterance MUST contain full tonal marks (àáèéìíòóùú) "
    "AND underdot characters (ọ, ẹ, ṣ). Never omit diacritics.\n\n"
    "Return ONLY a JSON array of {n} objects — no explanation, no markdown fences.\n"
    "Each object must have exactly these four keys:\n"
    '  "Doctor_English"               — doctor\'s utterance in English\n'
    '  "Patient_Yoruba"               — patient\'s reply in standard Yoruba with ALL tonal marks\n'
    '  "Clinical_Translation_English" — formal/clinical English translation of the Yoruba\n'
    '  "Direct_Translation_English"   — word-for-word literal English translation of the Yoruba\n\n'
    "Topics: symptoms (fever, pain, headache, cough, fatigue, vomiting, diarrhoea), "
    "chronic conditions (diabetes, hypertension, malaria, typhoid), medication, "
    "surgical consent, maternal health, paediatric care, emergency triage. "
    "Vary complexity. Each exchange must be distinct."
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _existing_row_count() -> int:
    if not CSV_PATH.exists():
        return 0
    with CSV_PATH.open("r", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def _append_rows(rows: list[dict]) -> int:
    write_header = not CSV_PATH.exists()
    written = 0
    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})
            written += 1
    return written


def _parse_rows(raw: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [r for r in data if isinstance(r, dict)]


def _validate_rows(rows: list[dict]) -> tuple[list[dict], int]:
    good, bad = [], 0
    for row in rows:
        yo = row.get("Patient_Yoruba", "")
        if not yo or not _valid_yoruba(yo):
            bad += 1
            continue
        if all(row.get(k, "").strip() for k in CSV_FIELDNAMES):
            good.append(row)
        else:
            bad += 1
    return good, bad


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    client = genai.Client(api_key=API_KEY)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = _existing_row_count()
    print(f"[start] CSV: {CSV_PATH}")
    print(f"[start] existing rows: {total:,} / {TARGET_ROWS:,}")

    batch_num = 0
    while total < TARGET_ROWS:
        need       = min(BATCH_SIZE, TARGET_ROWS - total)
        batch_num += 1
        print(f"\n-- batch {batch_num} (need {need}, total so far {total:,}) --")

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=_PROMPT_TEMPLATE.format(n=need),
            )
            raw_text = response.text if hasattr(response, "text") else ""

            rows = _parse_rows(raw_text)
            print(f"  parsed  : {len(rows)} objects")

            valid, rejected = _validate_rows(rows)
            print(f"  valid   : {len(valid)} | rejected: {rejected}")

            if valid:
                written = _append_rows(valid)
                total  += written
                print(f"  written : {written}  ->  cumulative: {total:,}")
            else:
                print("  WARNING : no valid rows — skipping batch")

        except Exception as exc:
            err_str = str(exc)
            print(f"  ERROR (batch {batch_num}): {exc}")
            # Daily quota exhausted — no point retrying until tomorrow
            if "429" in err_str and "PerDay" in err_str:
                print("\n  DAILY QUOTA EXHAUSTED. Restart the script tomorrow.")
                break
            # Per-minute rate limit — honour the suggested retry delay
            if "429" in err_str:
                import re as _re
                m = _re.search(r"(\d+)\.?\d*s'", err_str)
                wait = int(m.group(1)) + 5 if m else 60
                print(f"  Rate-limited. Sleeping {wait}s before next attempt...")
                time.sleep(wait)
                continue

        if total < TARGET_ROWS:
            time.sleep(SLEEP_BETWEEN)

    print(f"\n[done] {total:,} rows written to {CSV_PATH}")


if __name__ == "__main__":
    main()
