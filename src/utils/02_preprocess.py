"""
02_preprocess.py — Preprocessing & Diacritic Validator
Phase 1, Step 3  |  Run: python src/utils/02_preprocess.py
"""
import pandas as pd, unicodedata, re, os

TRAIN = "data/processed/train"
VAL   = "data/processed/val"
TEST  = "data/processed/test"
RAW   = "data/raw/text"
for p in [TRAIN,VAL,TEST]: os.makedirs(p,exist_ok=True)

TONE_MARKS    = ["\u0301","\u0300","\u0302"]
UNDERDOT_CHARS= ["\u1ECD","\u1EB9","\u1E63","\u1ECC","\u1EB8","\u1E62"]

def validate_yoruba(text):
    n = unicodedata.normalize("NFC", str(text))
    has_tones    = any(m in n for m in TONE_MARKS)
    has_underdot = any(c in n for c in UNDERDOT_CHARS)
    if not has_tones and not has_underdot:
        return {"risk":"HIGH",   "msg":"No diacritics — patient safety risk"}
    elif not has_tones:
        return {"risk":"MEDIUM", "msg":"Tone marks missing"}
    elif not has_underdot:
        return {"risk":"MEDIUM", "msg":"Underdots missing (check ọ/ẹ/ṣ)"}
    return     {"risk":"LOW",    "msg":"Diacritics validated OK"}

def clean(text):
    if not isinstance(text,str): return ""
    text = unicodedata.normalize("NFC",text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]","",text)
    return re.sub(r" +"," ",text).strip()

def split_df(df,seed=42):
    df = df.sample(frac=1,random_state=seed).reset_index(drop=True)
    n  = len(df)
    a,b = int(n*.8), int(n*.9)
    return df[:a], df[a:b], df[b:]

def demo():
    print("\n--- Diacritic Validator Demo ---")
    examples = [
        ("Orí mi ń dùn",  "My head hurts — correct"),
        ("Ori mi n dun",  "No diacritics — HIGH RISK"),
        ("ọmọ",           "child — ambiguous without tone"),
        ("ọmọ́",           "his child — unambiguous"),
    ]
    icons = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}
    for text,label in examples:
        r = validate_yoruba(text)
        print(f"  {icons[r['risk']]} \"{text}\"")
        print(f"     {label}")
        print(f"     Risk: {r['risk']} — {r['msg']}\n")

def process_dialogues():
    print("\n--- Processing medical_dialogues.csv ---")
    path = os.path.join(RAW,"medical_dialogues.csv")
    if not os.path.exists(path):
        print("  [SKIP] Not found. Place medical_dialogues.csv in project root first."); return
    df = pd.read_csv(path,encoding="utf-8")
    print(f"  Loaded {len(df)} rows.")
    df["Doctor_English"]               = df["Doctor_English"].apply(clean)
    df["Clinical_Translation_English"] = df["Clinical_Translation_English"].apply(clean)
    df["Patient_Yoruba"]               = df["Patient_Yoruba"].apply(clean)
    df["diacritic_risk"] = df["Patient_Yoruba"].apply(lambda x: validate_yoruba(x)["risk"])
    high = df[df["diacritic_risk"]=="HIGH"]
    if len(high): print(f"  ⚠  {len(high)} rows flagged HIGH RISK — check diacritics!")
    else:         print("  ✓  All rows passed diacritic validation.")
    tr,vl,te = split_df(df)
    tr.to_csv(f"{TRAIN}/medical_dialogues_train.csv",index=False,encoding="utf-8")
    vl.to_csv(f"{VAL}/medical_dialogues_val.csv",    index=False,encoding="utf-8")
    te.to_csv(f"{TEST}/medical_dialogues_test.csv",  index=False,encoding="utf-8")
    print(f"  Split: train={len(tr)}, val={len(vl)}, test={len(te)} — SAVED ✓")

def process_profiles():
    print("\n--- Processing patient_profiles.csv ---")
    path = os.path.join(RAW,"patient_profiles.csv")
    if not os.path.exists(path):
        print("  [SKIP] Not found."); return
    df = pd.read_csv(path,encoding="utf-8")
    print(f"  Loaded {len(df)} rows.")
    if "Primary_Complaint" in df.columns:
        df["Primary_Complaint"] = df["Primary_Complaint"].apply(clean)
    tr,vl,te = split_df(df)
    tr.to_csv(f"{TRAIN}/patient_profiles_train.csv",index=False)
    vl.to_csv(f"{VAL}/patient_profiles_val.csv",    index=False)
    te.to_csv(f"{TEST}/patient_profiles_test.csv",  index=False)
    print(f"  Split: train={len(tr)}, val={len(vl)}, test={len(te)} — SAVED ✓")

if __name__ == "__main__":
    print("="*55)
    print("  FYP S2ST — Preprocessing | Phase 1 Step 3")
    print("="*55)
    demo()
    process_dialogues()
    process_profiles()
    print("\n  DONE. Next: python src/asr/03_whisper_baseline.py")
