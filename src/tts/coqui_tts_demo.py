"""
coqui_tts_demo.py — TTS synthesis demo for S2ST pipeline
Phase 4, Step 1  |  Run: python src/tts/coqui_tts_demo.py

Engine priority
───────────────
1. Coqui TTS (VITS model)  — preferred; works on Linux or Windows + MSVC
2. pyttsx3 SAPI            — automatic fallback on Windows without MSVC

Why the fallback exists
───────────────────────
Coqui TTS requires a Cython build step (monotonic_align.core) that
needs Microsoft Visual C++ ≥ 14.  It also requires numpy<2.0 via gruut,
which conflicts with the numpy 2.4.5 already used by PyTorch in this env.
On Linux (or Windows with Build Tools installed) use the commented-out
Coqui section directly.

Coqui TTS model that WOULD be used (for your dissertation):
    tts_models/en/ljspeech/vits          ← best quality, fast
    tts_models/en/vctk/vits              ← multi-speaker
    tts_models/en/ljspeech/tacotron2-DDC ← fallback if VITS OOM
"""

import importlib.util
import json
import os
import subprocess
import sys
import time

PHRASES = [
    "The patient reports severe headache for three days.",
    "Please take this medication twice daily after meals.",
    "Your blood pressure is slightly elevated today.",
    "The child has had a fever for two days.",
    "We need to run some tests before we can diagnose you.",
]
OUT_DIR  = os.path.join("evaluation", "tts_samples")
EVAL_DIR = "evaluation"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

SEP = "=" * 60


# ─────────────────────────────────────────────────────────────
# ENGINE 1: Coqui TTS  (active on Linux / Windows+MSVC)
# ─────────────────────────────────────────────────────────────
def try_coqui_tts():
    """Return a synthesise(text, path) callable if Coqui TTS is available."""
    spec = importlib.util.find_spec("TTS")
    if spec is None:
        return None, "TTS package not installed"
    try:
        from TTS.api import TTS as CoquiTTS
    except Exception as e:
        return None, f"TTS import failed: {e}"

    print("\n  Coqui TTS is available. Listing English models …")
    try:
        api      = CoquiTTS()
        all_mdls = api.list_models()
        en_mdls  = [m for m in all_mdls if "/en/" in m]
        vits_mdls = [m for m in en_mdls if "vits" in m.lower()]
        print(f"\n  All English models ({len(en_mdls)} total):")
        for m in en_mdls:
            print(f"    {'[VITS]' if 'vits' in m.lower() else '      '} {m}")
    except Exception as e:
        return None, f"model listing failed: {e}"

    target = "tts_models/en/ljspeech/vits"
    if not any(target in m for m in en_mdls):
        target = vits_mdls[0] if vits_mdls else en_mdls[0]
    print(f"\n  Selected model: {target}")
    tts = CoquiTTS(model_name=target, progress_bar=True)

    def synthesise(text: str, out_path: str):
        tts.tts_to_file(text=text, file_path=out_path)

    return synthesise, f"Coqui TTS — {target}"


# ─────────────────────────────────────────────────────────────
# ENGINE 2: pyttsx3 SAPI  (fallback on Windows without MSVC)
# ─────────────────────────────────────────────────────────────
def ensure_pyttsx3():
    if importlib.util.find_spec("pyttsx3") is None:
        print("  Installing pyttsx3 …")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "pyttsx3", "-q"])


def try_pyttsx3():
    """Return a synthesise(text, path) callable using Windows SAPI."""
    ensure_pyttsx3()
    try:
        import pyttsx3
    except ImportError as e:
        return None, f"pyttsx3 unavailable: {e}"

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    print("\n  Available SAPI voices:")
    for i, v in enumerate(voices):
        print(f"    [{i}] {v.name}  ({', '.join(v.languages)})")

    # prefer female voice for clinical clarity; fall back to first
    chosen = next(
        (v for v in voices if "zira" in v.name.lower()
         or "female" in v.name.lower()
         or "aria"   in v.name.lower()
         or "jenny"  in v.name.lower()),
        voices[0],
    )
    print(f"\n  Selected voice: {chosen.name}")
    engine.setProperty("voice", chosen.id)
    engine.setProperty("rate",  165)   # words per minute
    engine.setProperty("volume", 1.0)
    engine.stop()                       # release to free the init engine

    def synthesise(text: str, out_path: str):
        eng = pyttsx3.init()
        eng.setProperty("voice",  chosen.id)
        eng.setProperty("rate",   165)
        eng.setProperty("volume", 1.0)
        eng.save_to_file(text, out_path)
        eng.runAndWait()
        eng.stop()

    return synthesise, f"pyttsx3 SAPI — {chosen.name}"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("  FYP S2ST — TTS Synthesis Demo | Phase 4 Step 1")
    print(SEP)

    # ── select engine ─────────────────────────────────────────
    print("\n  Checking engine availability …")
    synthesise, engine_label = try_coqui_tts()
    if synthesise is None:
        print(f"  Coqui TTS unavailable — reason: {engine_label}")
        print("  → Falling back to pyttsx3 (Windows SAPI)\n")
        synthesise, engine_label = try_pyttsx3()

    if synthesise is None:
        print(f"\n  [ERROR] No TTS engine available: {engine_label}")
        sys.exit(1)

    print(f"\n  Engine : {engine_label}")
    print(f"  Output : {OUT_DIR}/")
    print()

    # ── synthesise each phrase ────────────────────────────────
    results = []
    total_start = time.time()

    for i, phrase in enumerate(PHRASES, 1):
        fname    = f"phrase_{i:02d}.wav"
        out_path = os.path.join(OUT_DIR, fname)

        print(f"  [{i}/{len(PHRASES)}] \"{phrase}\"")
        t0 = time.time()
        synthesise(phrase, out_path)
        elapsed = time.time() - t0

        size_kb = os.path.getsize(out_path) / 1024 if os.path.exists(out_path) else 0
        print(f"          → {fname}  ({size_kb:.1f} KB)  in {elapsed:.2f}s")
        results.append({
            "phrase_no":   i,
            "text":        phrase,
            "file":        fname,
            "size_kb":     round(size_kb, 1),
            "latency_sec": round(elapsed, 2),
        })

    total = time.time() - total_start
    print()
    print(SEP)
    print(f"  Synthesised {len(PHRASES)} phrases in {total:.2f}s total")
    print(f"  Avg latency per phrase : {total/len(PHRASES):.2f}s")
    print(SEP)

    # ── save manifest ─────────────────────────────────────────
    manifest = {
        "engine":          engine_label,
        "n_phrases":       len(PHRASES),
        "output_dir":      OUT_DIR,
        "total_sec":       round(total, 2),
        "avg_latency_sec": round(total / len(PHRASES), 2),
        "samples":         results,
    }
    manifest_path = os.path.join(EVAL_DIR, "tts_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\n  [SAVED] Manifest → {manifest_path}")
    print(f"  [SAVED] WAV files → {OUT_DIR}/")

    # ── dissertation note ─────────────────────────────────────
    print("""
  ── Dissertation note ──────────────────────────────────────
  On Linux (or Windows with Visual C++ Build Tools), replace
  the pyttsx3 engine with Coqui TTS VITS:

      pip install TTS
      # In coqui_tts_demo.py, engine selection auto-switches.

  Recommended model for this pipeline:
      tts_models/en/ljspeech/vits  (single speaker, 22 kHz)

  VITS advantages over SAPI for medical TTS:
    • End-to-end neural synthesis (no HMM/formant artifacts)
    • Controllable speaking rate and prosody
    • 22 kHz output (vs 16 kHz SAPI)
    • Can be fine-tuned on clinical English corpora
  ──────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
