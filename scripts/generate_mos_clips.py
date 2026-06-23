"""Generate Yoruba clinical TTS clips (MMS-TTS) for the MOS survey."""
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import numpy as np, torch
from scipy.io.wavfile import write as wavwrite
from transformers import VitsModel, AutoTokenizer

OUT = r"C:\Users\Admin\OneDrive\Desktop\FYP OMEGA\mos_clips"
os.makedirs(OUT, exist_ok=True)

PHRASES = [
    "Ẹ kú àárọ̀, báwo ni ara yín?",
    "Orí mi ń dùn mí gan-an.",
    "Ara mi ń gbóná, ó sì ń mì mí.",
    "Ikọ́ ti ń dà mí láàmú fún ọjọ́ mẹ́ta.",
    "Inú mi ń rún mi láti àná.",
    "Mo ní ìrora ní àyà mi.",
    "Ẹsẹ̀ mi ọ̀tún ti wú.",
    "Ṣé ẹ ti mu oògùn yín lónìí?",
    "Ẹ jọ̀wọ́, ẹ fẹnu yín sí.",
    "A óò ṣe àyẹ̀wò ẹ̀jẹ̀ fún yín báyìí.",
]

print("Loading facebook/mms-tts-yor ...")
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-yor")
mdl = VitsModel.from_pretrained("facebook/mms-tts-yor").eval()
sr = mdl.config.sampling_rate

manifest = []
for i, text in enumerate(PHRASES, 1):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        wav = mdl(**inp).waveform.squeeze().cpu().numpy()
    pcm = np.clip(wav, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    fname = f"mms_{i:02d}.wav"
    wavwrite(os.path.join(OUT, fname), sr, pcm)
    manifest.append(f"{fname}\t{text}")
    print(f"  {fname}  ({len(pcm)/sr:.1f}s)  {text}")

with open(os.path.join(OUT, "MANIFEST.txt"), "w", encoding="utf-8") as f:
    f.write("file\ttext\n" + "\n".join(manifest))
print("\nDONE ->", OUT)
