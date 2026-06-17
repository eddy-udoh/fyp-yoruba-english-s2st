"""
01_load_datasets.py — Dataset Loading Script
Phase 1, Step 2  |  Run: python src/asr/01_load_datasets.py
"""
from datasets import load_dataset
import pandas as pd, os, json, shutil

RAW_TEXT  = os.path.join("data","raw","text")
RAW_AUDIO = os.path.join("data","raw","audio")
os.makedirs(RAW_TEXT, exist_ok=True)

def sep(t): print(f"\n{'='*55}\n  {t}\n{'='*55}")

def load_jw300():
    sep("LOADING: JW300 Yoruba-English (NMT)")
    try:
        dataset = load_dataset("opus100","en-yo",trust_remote_code=True)
        train   = dataset["train"]
        print(f"  [OK] {len(train)} parallel sentence pairs loaded.")
        rows = [{"yoruba":r["translation"]["yo"],"english":r["translation"]["en"]}
                for r in train.select(range(min(50,len(train))))]
        pd.DataFrame(rows).to_csv(os.path.join(RAW_TEXT,"jw300_preview.csv"),
                                   index=False,encoding="utf-8")
        print(f"  [SAVED] Preview -> data/raw/text/jw300_preview.csv")
        print(f"  Sample Yoruba : {rows[0]['yoruba']}")
        print(f"  Sample English: {rows[0]['english']}")
        return dataset
    except Exception as e:
        print(f"  [ERROR] {e}"); return None

def load_medical_corpus():
    sep("LOADING: Custom Medical Corpus")
    for fname in ["medical_dialogues.csv","patient_profiles.csv"]:
        src = fname
        dst = os.path.join(RAW_TEXT, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  [COPIED] {fname} -> data/raw/text/")
        elif os.path.exists(dst):
            print(f"  [OK] {fname} already in data/raw/text/")
        else:
            print(f"  [ACTION NEEDED] Place {fname} in project root and re-run.")
    path = os.path.join(RAW_TEXT,"medical_dialogues.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n  medical_dialogues.csv: {len(df)} rows")
        print(f"  Doctor (EN) : {df['Doctor_English'].iloc[0]}")
        print(f"  Patient (YO): {df['Patient_Yoruba'].iloc[0]}")
        return df
    return None

if __name__ == "__main__":
    sep("FYP S2ST — Dataset Loading | Phase 1 Step 2")
    r1 = load_jw300()
    r2 = load_medical_corpus()
    sep("SUMMARY")
    print(f"  JW300          : {'[OK]' if r1 else '[FAILED]'}")
    print(f"  Medical Corpus : {'[OK]' if r2 is not None else '[ACTION NEEDED]'}")
    print("\n  Next: python src/utils/02_preprocess.py")
