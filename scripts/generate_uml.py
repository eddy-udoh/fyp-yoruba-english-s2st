"""Generate corrected UML PNGs (Activity, State, Component) matching app.py."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = r"C:\Users\Admin\OneDrive\Desktop\FYP OMEGA\uml"
os.makedirs(OUT, exist_ok=True)

GREEN, GREEN2, MINT, LIGHT, INK = "#0F6E56", "#1D9E75", "#C8E6D8", "#E1F5EE", "#0A2E1F"
AMBER, RED = "#FDE68A", "#FECACA"
FONT = 9


def box(ax, cx, cy, w, h, text, fc=LIGHT, ec=GREEN, fs=FONT, bold=False, round=True):
    style = "round,pad=0.02,rounding_size=0.08" if round else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style,
                                fc=fc, ec=ec, lw=1.3, mutation_scale=10))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=INK, weight="bold" if bold else "normal", wrap=True)


def diamond(ax, cx, cy, w, h, text, fc=AMBER, ec="#D97706"):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc=fc, ec=ec, lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FONT - 1, color=INK)


def oval(ax, cx, cy, w, h, text, fc=GREEN, tc="white"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc=fc, ec=GREEN, lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FONT, color=tc, weight="bold")


def arrow(ax, x1, y1, x2, y2, text="", color=INK, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=1.2, color=color, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, text, fontsize=FONT - 2, color=color, ha="left")


# ── 1. ACTIVITY DIAGRAM ─────────────────────────────────────────────────────
def activity():
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 24); ax.axis("off")
    # swimlane bands
    ax.axhspan(20.2, 24, color="#F4FAF7"); ax.axhspan(3.0, 20.2, color="#F0F8F4"); ax.axhspan(0, 3.0, color="#F4FAF7")
    ax.text(0.15, 22.1, "CLIENT", rotation=90, va="center", fontsize=9, color=GREEN, weight="bold")
    ax.text(0.15, 11.5, "FLASK API (server)", rotation=90, va="center", fontsize=9, color=GREEN, weight="bold")
    ax.text(0.15, 1.5, "CLIENT", rotation=90, va="center", fontsize=9, color=GREEN, weight="bold")

    cx = 4.2
    oval(ax, cx, 23.2, 1.6, 0.7, "Start")
    box(ax, cx, 21.8, 4.2, 0.9, "Select direction; record audio\nor enter text")
    box(ax, cx, 20.5, 4.2, 0.8, "POST /translate  (audio or text)", fc=MINT)
    diamond(ax, cx, 18.9, 2.6, 1.4, "Input\npresent?")
    box(ax, 8.2, 18.9, 2.6, 0.8, "Return 400\n(no input)", fc=RED, ec="#DC2626")
    box(ax, cx, 17.2, 4.4, 0.9, "Whisper ASR\ntranscribe (forced language token)", fc=LIGHT)
    box(ax, cx, 15.5, 4.8, 1.1, "Diacritic normalisation (NFC) +\ncompute density; set WARNING flag if < 8%\n(non-blocking — processing always continues)", fc="#FEF9E7", ec="#D97706")
    box(ax, cx, 13.7, 4.8, 0.95, "MarianMT translate\nyo→en: opus-mt-yo-en  |  en→yo: opus-mt-en-nic (>>yor<<)", fc=LIGHT)
    diamond(ax, cx, 12.0, 2.6, 1.4, "Translation\nempty?")
    box(ax, 8.2, 12.0, 2.6, 0.8, "Return 422\n(empty NMT)", fc=RED, ec="#DC2626")
    diamond(ax, cx, 10.1, 2.8, 1.5, "Direction?")
    box(ax, 1.7, 10.1, 2.7, 1.0, "yo→en: Edge TTS\n→ gTTS on failure", fc=MINT)
    box(ax, 7.2, 10.1, 2.7, 1.0, "en→yo: MMS-TTS\n(VITS)", fc=MINT)
    box(ax, cx, 8.2, 4.8, 0.95, "Assemble JSON: transcript, translation,\naudio_base64, latency, diacritic flag, tts_note", fc=LIGHT)
    box(ax, cx, 6.6, 4.4, 0.8, "Append request to api_logs.json", fc=LIGHT)
    box(ax, cx, 2.2, 5.0, 1.05, "Render transcript + translation + audio;\nshow amber WARNING banner if flag set", fc=LIGHT)
    oval(ax, cx, 0.7, 1.6, 0.7, "End")

    arrow(ax, cx, 22.85, cx, 22.25)
    arrow(ax, cx, 21.35, cx, 20.9)
    arrow(ax, cx, 20.1, cx, 19.6)
    arrow(ax, cx, 18.2, cx, 17.65)
    arrow(ax, cx + 1.3, 18.9, 6.9, 18.9, "no")
    ax.text(cx + 0.15, 18.0, "yes", fontsize=FONT - 2, color=INK)
    arrow(ax, cx, 16.75, cx, 16.05)
    arrow(ax, cx, 14.95, cx, 14.2)
    arrow(ax, cx, 13.25, cx, 12.7)
    arrow(ax, cx + 1.3, 12.0, 6.9, 12.0, "yes")
    ax.text(cx + 0.15, 11.1, "no", fontsize=FONT - 2, color=INK)
    arrow(ax, cx, 11.3, cx, 10.85)
    arrow(ax, cx - 1.4, 10.1, 3.05, 10.1)
    arrow(ax, cx + 1.4, 10.1, 5.85, 10.1)
    arrow(ax, 1.7, 9.6, 2.6, 8.7); arrow(ax, 7.2, 9.6, 5.8, 8.7)
    arrow(ax, cx, 7.7, cx, 7.0)
    arrow(ax, cx, 6.2, cx, 2.75)
    arrow(ax, cx, 1.65, cx, 1.05)

    ax.set_title("Figure 3.3: UML Activity Diagram — S2ST Translation Request (diacritic warning is non-blocking)",
                 fontsize=10, weight="bold", color=INK, pad=10)
    fig.savefig(os.path.join(OUT, "activity_diagram.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("activity_diagram.png")


# ── 2. STATE DIAGRAM ────────────────────────────────────────────────────────
def state():
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 22); ax.axis("off")
    seq = [("Idle", 19.5), ("Recording", 17.3), ("Processing", 15.1),
           ("ASR Complete", 12.9), ("NMT Complete", 10.7), ("TTS Complete", 8.5),
           ("Result Displayed", 6.3)]
    cx = 4.0
    oval(ax, cx, 21.2, 1.4, 0.7, "●", fc=INK)
    for name, y in seq:
        box(ax, cx, y, 3.4, 1.0, name, fc=LIGHT, bold=True)
    arrow(ax, cx, 20.85, cx, 20.05)  # start → Idle
    labels = ["tap record", "tap stop → POST", "ASR done", "NMT done", "TTS done", "JSON rendered"]
    for (n1, y1), (n2, y2), lab in zip(seq, seq[1:], labels):
        arrow(ax, cx, y1 - 0.5, cx, y2 + 0.5, lab)
    # return to Idle
    arrow(ax, cx + 1.7, 6.3, 8.6, 6.3)
    arrow(ax, 8.6, 6.3, 8.6, 19.5)
    arrow(ax, 8.6, 19.5, cx + 1.7, 19.5, "new request")
    # error state
    box(ax, 8.6, 15.1, 2.4, 0.9, "Error\n(400 / 422 / TTS fail)", fc=RED, ec="#DC2626")
    arrow(ax, cx + 1.7, 15.1, 7.4, 15.1, "invalid input /\nempty output")
    arrow(ax, 8.6, 14.65, 8.6, 6.75, "→ Idle", color="#DC2626")
    # warning note
    box(ax, 1.2, 6.3, 2.0, 1.5,
        "WARNING flag\n(density < 8%)\nshown WITH result —\nnon-terminal,\nNMT still runs", fc="#FEF9E7", ec="#D97706", fs=FONT - 1)
    arrow(ax, 2.2, 6.3, cx - 1.7, 6.3, color="#D97706")

    ax.set_title("Figure 3.7: UML State Diagram — Translation Request Lifecycle\n(Warning is a non-blocking flag on the result, not a terminal state)",
                 fontsize=10, weight="bold", color=INK, pad=10)
    fig.savefig(os.path.join(OUT, "state_diagram.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("state_diagram.png")


# ── 3. COMPONENT / MODULE DIAGRAM (replaces OOP class diagram) ───────────────
def component():
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11); ax.axis("off")

    box(ax, 2.4, 9.2, 4.0, 1.3, "React Frontend  (App.js)\nMediaRecorder · axios POST /translate\nrenders translation card, audio, metrics", fc=MINT, ec=GREEN, bold=False)
    box(ax, 8.8, 9.2, 5.4, 1.3, "Flask API  (app.py)\nroutes: /health, /translate\nload_all_models() · shared model registry  _m{}", fc=GREEN, ec=GREEN, bold=True)
    ax.text(8.8, 8.25, "(module-level functions + global dict — not OOP classes)", ha="center", fontsize=FONT - 2, color=INK, style="italic")
    arrow(ax, 4.4, 9.2, 6.1, 9.2, "HTTP")

    mods = [
        (2.4, 6.2, "ASR\nWhisperProcessor +\nWhisperForConditionalGeneration\n→ transcribe()"),
        (6.6, 6.2, "Diacritic Normalisation\n_normalise_yoruba()\n_diacritic_density()"),
        (10.9, 6.2, "NMT\n_translate()  ·  MarianMT\nopus-mt-yo-en / opus-mt-en-nic"),
        (4.5, 3.6, "TTS (English)\nsynthesise_speech()\nedge-tts → gTTS fallback"),
        (8.8, 3.6, "TTS (Yorùbá)\nsynthesise_yoruba()\nMMS-TTS (VITS)"),
        (12.4, 3.6, "Logging\n_append_log()\n→ api_logs.json"),
    ]
    for x, y, t in mods:
        box(ax, x, y, 3.5, 1.4, t, fc=LIGHT, ec=GREEN)
        arrow(ax, 8.8, 8.55, x, y + 0.75, color="#9EC8B5")

    box(ax, 7.0, 1.1, 9.0, 0.9, "Models (loaded once into  _m{}):  whisper-small-yoruba-finetuned · "
        "marian-yoruba-medical · marian-english-yoruba · facebook/mms-tts-yor", fc="#F4FAF7", ec=GREEN, fs=FONT - 1)
    for x in (2.4, 10.9, 4.5, 8.8):
        arrow(ax, x, 2.9, 6.0 if x < 7 else 8.0, 1.55, color="#C8E6D8")

    ax.set_title("Figure 3.6: Component / Module View of the Flask Pipeline\n(reflects the procedural implementation — functions and a shared model registry)",
                 fontsize=10, weight="bold", color=INK, pad=10)
    fig.savefig(os.path.join(OUT, "component_diagram.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("component_diagram.png")


activity(); state(); component()
print("DONE ->", OUT)
