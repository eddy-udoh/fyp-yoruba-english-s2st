"""Quick test: POST phrase_01.wav to /translate and save results."""
import requests
import json
import time
import os
import sys

BASE = "http://localhost:5000"

# ── /health ────────────────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/health", timeout=10)
print("=== /health ===")
print(json.dumps(r.json(), indent=2))
print()

# ── /translate  en-yo  (phrase_01.wav is English speech) ──────────────────────
wav = os.path.join("evaluation", "tts_samples", "phrase_01.wav")
print(f"=== /translate  direction=en-yo  file={wav} ===")
t0 = time.time()
with open(wav, "rb") as f:
    resp = requests.post(
        f"{BASE}/translate",
        files={"audio": ("phrase_01.wav", f, "audio/wav")},
        data={"direction": "en-yo"},
        timeout=180,
    )
wall_ms = round((time.time() - t0) * 1000)

data = resp.json()
b64_len = len(data.get("audio_base64", ""))

# ── pretty-print (exclude the long base64 blob) ───────────────────────────────
display = {k: v for k, v in data.items() if k != "audio_base64"}
display["audio_base64"] = f"<base64 mp3, {b64_len} chars>"
print(f"HTTP {resp.status_code}  (wall-clock {wall_ms} ms)")
print(json.dumps(display, indent=2, ensure_ascii=False))

# ── save full response ─────────────────────────────────────────────────────────
out_path = os.path.join("evaluation", "translate_test_result.json")
full = dict(data)
full["_http_status"] = resp.status_code
full["_wall_clock_ms"] = wall_ms
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(full, fh, ensure_ascii=False, indent=2)
print(f"\nFull JSON (including audio_base64) saved → {out_path}")

# ── also test yo-en with a text string ────────────────────────────────────────
print("\n=== /translate  direction=yo-en  text input ===")
sample_yo = "Orí mi ń dùn, mo ti ń gbà oogun"
t0 = time.time()
resp2 = requests.post(
    f"{BASE}/translate",
    json={"direction": "yo-en", "text": sample_yo},
    timeout=60,
)
wall_ms2 = round((time.time() - t0) * 1000)
data2 = resp2.json()
b64_len2 = len(data2.get("audio_base64", ""))
display2 = {k: v for k, v in data2.items() if k != "audio_base64"}
display2["audio_base64"] = f"<base64 mp3, {b64_len2} chars>"
print(f"HTTP {resp2.status_code}  (wall-clock {wall_ms2} ms)")
print(json.dumps(display2, indent=2, ensure_ascii=False))
