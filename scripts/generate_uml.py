"""Generate all five UML diagrams (Activity, Use Case, Sequence, Class, State)
as clean, legible PNGs reflecting the implemented MedSpeak system."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle

OUT = r"C:\Users\Admin\OneDrive\Desktop\FYP OMEGA\uml"
os.makedirs(OUT, exist_ok=True)
GREEN, GREEN2, MINT, LIGHT, INK = "#0F6E56", "#1D9E75", "#C8E6D8", "#E8F5EE", "#0A2E1F"
AMBER, AMBERBG, RED = "#D97706", "#FEF3C7", "#FEE2E2"
FS = 10


def rbox(ax, cx, cy, w, h, text, fc=LIGHT, ec=GREEN, fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
            weight="bold" if bold else "normal")


def diamond(ax, cx, cy, w, h, text, fc=AMBERBG, ec=AMBER):
    ax.add_patch(Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                         closed=True, fc=fc, ec=ec, lw=1.4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS-1, color=INK)


def oval(ax, cx, cy, w, h, text, fc=GREEN, tc="white", fs=FS):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc=fc, ec=GREEN, lw=1.4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold")


def arr(ax, x1, y1, x2, y2, text="", color=INK, dashed=False, fs=FS-2, lbldx=0.12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.1,
                                 color=color, shrinkA=2, shrinkB=2,
                                 linestyle="--" if dashed else "-"))
    if text:
        ax.text((x1+x2)/2+lbldx, (y1+y2)/2+0.10, text, fontsize=fs, color=color, ha="center", va="bottom")


def actor(ax, x, y, label):
    ax.add_patch(Circle((x, y+0.55), 0.18, fc="white", ec=INK, lw=1.4))
    ax.plot([x, x], [y+0.37, y-0.25], color=INK, lw=1.4)
    ax.plot([x-0.32, x+0.32], [y+0.12, y+0.12], color=INK, lw=1.4)
    ax.plot([x, x-0.28], [y-0.25, y-0.75], color=INK, lw=1.4)
    ax.plot([x, x+0.28], [y-0.25, y-0.75], color=INK, lw=1.4)
    ax.text(x, y-1.05, label, ha="center", va="top", fontsize=FS, color=INK, weight="bold")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name)


# ── 1. ACTIVITY ─────────────────────────────────────────────────────────────
def activity():
    fig, ax = plt.subplots(figsize=(9.5, 12.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 25); ax.axis("off")
    ax.axhspan(21, 25, color="#F4FAF7"); ax.axhspan(3, 21, color="#EFF8F3"); ax.axhspan(0, 3, color="#F4FAF7")
    ax.text(0.2, 23, "CLIENT", rotation=90, va="center", fontsize=10, color=GREEN, weight="bold")
    ax.text(0.2, 12, "FLASK API (server)", rotation=90, va="center", fontsize=10, color=GREEN, weight="bold")
    ax.text(0.2, 1.5, "CLIENT", rotation=90, va="center", fontsize=10, color=GREEN, weight="bold")
    cx = 4.3
    oval(ax, cx, 24.1, 1.5, 0.7, "Start")
    rbox(ax, cx, 22.6, 4.4, 0.9, "Select direction; record audio\nor type text")
    rbox(ax, cx, 21.2, 4.4, 0.8, "POST /translate", fc=MINT)
    diamond(ax, cx, 19.5, 2.8, 1.5, "Input\npresent?")
    rbox(ax, 8.3, 19.5, 2.6, 0.85, "Return 400\n(no input)", fc=RED, ec="#DC2626")
    rbox(ax, cx, 17.7, 4.6, 0.9, "Whisper ASR — transcribe\n(language token forced)")
    rbox(ax, cx, 15.8, 5.0, 1.1, "Diacritic normalisation (NFC) +\ndensity check; set WARNING if < 8%\n(non-blocking — always continues)", fc=AMBERBG, ec=AMBER)
    rbox(ax, cx, 13.9, 5.0, 0.95, "MarianMT — translate\n(yo→en  or  en→yo with >>yor<<)")
    diamond(ax, cx, 12.1, 2.8, 1.5, "Output\nempty?")
    rbox(ax, 8.3, 12.1, 2.6, 0.85, "Return 422\n(empty NMT)", fc=RED, ec="#DC2626")
    diamond(ax, cx, 10.2, 3.0, 1.6, "Direction?")
    rbox(ax, 1.7, 10.2, 2.6, 1.0, "yo→en:\nEdge TTS → gTTS", fc=MINT)
    rbox(ax, 7.3, 10.2, 2.6, 1.0, "en→yo:\nMMS-TTS (VITS)", fc=MINT)
    rbox(ax, cx, 8.3, 5.0, 0.95, "Assemble JSON (transcript,\ntranslation, audio, flags)")
    rbox(ax, cx, 6.7, 4.4, 0.8, "Append to api_logs.json")
    rbox(ax, cx, 2.1, 5.2, 1.0, "Render result + audio;\nshow warning banner if flagged")
    oval(ax, cx, 0.7, 1.5, 0.7, "End")
    for y1, y2 in [(23.75, 23.05), (22.15, 21.6), (20.8, 20.25), (18.75, 18.15),
                   (17.25, 16.35), (15.25, 14.38), (13.42, 12.85)]:
        arr(ax, cx, y1, cx, y2)
    arr(ax, cx+1.4, 19.5, 7.0, 19.5, "no"); ax.text(cx+0.25, 18.7, "yes", fontsize=FS-2)
    arr(ax, cx+1.4, 12.1, 7.0, 12.1, "yes"); ax.text(cx+0.25, 11.3, "no", fontsize=FS-2)
    arr(ax, cx, 11.3, cx, 11.0)
    arr(ax, cx-1.5, 10.2, 3.05, 10.2); arr(ax, cx+1.5, 10.2, 6.0, 10.2)
    arr(ax, 1.7, 9.7, 2.7, 8.8); arr(ax, 7.3, 9.7, 5.9, 8.8)
    arr(ax, cx, 7.8, cx, 7.15); arr(ax, cx, 6.3, cx, 2.65); arr(ax, cx, 1.6, cx, 1.05)
    ax.set_title("Figure 3.3: Activity Diagram — Translation Request Flow", fontsize=12, weight="bold", color=INK, pad=12)
    save(fig, "activity_diagram.png")


# ── 2. USE CASE ─────────────────────────────────────────────────────────────
def use_case():
    fig, ax = plt.subplots(figsize=(12, 8)); ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis("off")
    ax.add_patch(Rectangle((4.3, 0.8), 5.4, 10.4, fill=False, ec=GREEN, lw=1.6))
    ax.text(7.0, 10.8, "MedSpeak YO–EN System", ha="center", fontsize=11, color=GREEN, weight="bold")
    ucs = [("Record Yorùbá speech", 9.6), ("Translate Yorùbá → English", 8.2),
           ("Record English speech", 6.8), ("Translate English → Yorùbá", 5.4),
           ("Play synthesised audio", 4.0), ("View pipeline metrics", 2.6)]
    pos = {}
    for name, y in ucs:
        oval(ax, 7.0, y, 4.4, 1.0, name, fs=FS-1); pos[name] = y
    actor(ax, 1.6, 7.0, "Patient\n(Yorùbá)")
    actor(ax, 12.4, 6.0, "Clinician\n(English)")
    L, R = 4.8, 9.2
    for name in ["Record Yorùbá speech", "Translate Yorùbá → English", "Play synthesised audio", "View pipeline metrics"]:
        ax.plot([1.95, L], [7.0, pos[name]], color="#7FB6A2", lw=1.0)
    for name in ["Record English speech", "Translate English → Yorùbá", "Play synthesised audio", "View pipeline metrics"]:
        ax.plot([12.05, R], [6.0, pos[name]], color="#7FB6A2", lw=1.0)
    ax.set_title("Figure 3.4: Use Case Diagram", fontsize=12, weight="bold", color=INK, pad=12)
    save(fig, "use_case_diagram.png")


# ── 3. SEQUENCE ─────────────────────────────────────────────────────────────
def sequence():
    fig, ax = plt.subplots(figsize=(13, 9)); ax.set_xlim(0, 14); ax.set_ylim(0, 13); ax.axis("off")
    lifelines = [("Browser /\nClient", 1.6), ("React App", 3.8), ("Flask API", 6.2),
                 ("Whisper\nASR", 8.6), ("MarianMT\nNMT", 10.8), ("TTS", 12.6)]
    xs = {}
    for name, x in lifelines:
        rbox(ax, x, 12.2, 1.9, 0.85, name, fc=GREEN, fs=FS-1)
        ax.text(x, 12.2, name, ha="center", va="center", fontsize=FS-1, color="white", weight="bold")
        ax.plot([x, x], [11.7, 0.8], color="#9EC8B5", lw=1.0, linestyle="--"); xs[name] = x
    C, R, F, W, M, T = (xs["Browser /\nClient"], xs["React App"], xs["Flask API"],
                        xs["Whisper\nASR"], xs["MarianMT\nNMT"], xs["TTS"])

    def msg(y, x1, x2, label, dashed=False):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color=INK, linestyle="--" if dashed else "-"))
        ax.text((x1+x2)/2, y+0.13, label, ha="center", va="bottom", fontsize=FS-2, color=INK)

    msg(11.0, C, R, "record / audio blob")
    msg(10.1, R, F, "POST /translate (direction)")
    msg(9.2, F, W, "transcribe(audio)")
    msg(8.3, W, F, "transcript", dashed=True)
    ax.annotate("", xy=(F, 7.2), xytext=(F+0.95, 7.2), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=INK))
    ax.plot([F, F+0.95, F+0.95], [7.55, 7.55, 7.2], color=INK, lw=1.1)
    ax.text(F+1.1, 7.5, "normalise + density (warn if <8%)", ha="left", va="center", fontsize=FS-2, color=AMBER)
    msg(6.3, F, M, "translate(text)")
    msg(5.4, M, F, "translated text", dashed=True)
    msg(4.5, F, T, "synthesise(text)")
    msg(3.6, T, F, "audio (base64)", dashed=True)
    msg(2.6, F, R, "JSON response", dashed=True)
    msg(1.6, R, C, "render + play audio")
    ax.set_title("Figure 3.5: Sequence Diagram — Yorùbá→English Request", fontsize=12, weight="bold", color=INK, pad=12)
    save(fig, "sequence_diagram.png")


# ── 4. CLASS ────────────────────────────────────────────────────────────────
def class_box(ax, cx, top, w, name, attrs, methods):
    lh = 0.42
    h = 0.6 + 0.12 + lh*len(attrs) + 0.10 + lh*len(methods) + 0.12
    y = top - h
    ax.add_patch(Rectangle((cx-w/2, y), w, h, fill=True, fc="white", ec=GREEN, lw=1.5))
    ax.add_patch(Rectangle((cx-w/2, top-0.6), w, 0.6, fill=True, fc=GREEN, ec=GREEN, lw=1.5))
    ax.text(cx, top-0.3, name, ha="center", va="center", fontsize=FS, color="white", weight="bold")
    yy = top-0.6-0.30
    for a in attrs:
        ax.text(cx-w/2+0.18, yy, a, ha="left", va="center", fontsize=FS-2, color=INK); yy -= lh
    div = yy + lh - 0.13
    ax.plot([cx-w/2, cx+w/2], [div, div], color=GREEN, lw=1.0)
    yy = div - 0.28
    for m in methods:
        ax.text(cx-w/2+0.18, yy, m, ha="left", va="center", fontsize=FS-2, color=INK); yy -= lh
    return y  # bottom y


def class_diagram():
    fig, ax = plt.subplots(figsize=(15, 9)); ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
    api_bot = class_box(ax, 8.0, 10.4, 5.6, "TranslationAPI",
        ["- models registry  _m{}"],
        ["+ health()", "+ translate(direction, audio | text)"])
    boxes = [
        (2.3, "WhisperASR", ["- model", "- processor"], ["+ transcribe(audio, lang)"]),
        (6.1, "DiacriticValidator", ["(stateless utility)"], ["+ normalise(text)", "+ diacritic_density(text)"]),
        (9.9, "MarianNMT", ["- yo_en, en_yo", "- tokenizers"], ["+ translate(text, dir)"]),
        (13.7, "TTSSynthesiser", ["- mms_model", "- edge_voice"], ["+ synthesise(text, lang)"]),
    ]
    for cx, name, attrs, methods in boxes:
        top = 5.8
        class_box(ax, cx, top, 3.4, name, attrs, methods)
        ax.add_patch(FancyArrowPatch((8.0, api_bot), (cx, top), arrowstyle="-|>",
                     mutation_scale=12, lw=1.1, color="#6A9E88", linestyle="--", shrinkA=4, shrinkB=4))
    ax.text(8.0, 6.15, "«use» dependencies", ha="center", fontsize=FS-2, color="#6A9E88", style="italic")
    ax.set_title("Figure 3.6: Class Diagram — Flask Pipeline (logical view)", fontsize=12, weight="bold", color=INK, pad=12)
    save(fig, "class_diagram.png")


# ── 5. STATE ────────────────────────────────────────────────────────────────
def state():
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.set_xlim(0, 10); ax.set_ylim(0, 22); ax.axis("off")
    seq = [("Idle", 19.3), ("Recording", 17.0), ("Processing", 14.7),
           ("ASR Complete", 12.4), ("NMT Complete", 10.1), ("TTS Complete", 7.8),
           ("Result Displayed", 5.5)]
    cx = 4.0
    ax.add_patch(Circle((cx, 21.1), 0.22, fc=INK, ec=INK))
    for name, y in seq: rbox(ax, cx, y, 3.4, 1.0, name, bold=True)
    arr(ax, cx, 20.85, cx, 19.85)
    labels = ["tap record", "tap stop → POST", "ASR done", "NMT done", "TTS done", "JSON rendered"]
    for (n1, y1), (n2, y2), lab in zip(seq, seq[1:], labels):
        arr(ax, cx, y1-0.5, cx, y2+0.5, lab, lbldx=0.0)
    # "new request" return — far-right corridor (does not touch the Error box)
    arr(ax, cx+1.7, 5.5, 9.5, 5.5); ax.plot([9.5, 9.5], [5.5, 19.6], color=INK, lw=1.1)
    arr(ax, 9.5, 19.6, cx+1.7, 19.6, "new request")
    # Error state + its own separate return path to Idle
    rbox(ax, 7.9, 14.7, 2.5, 0.95, "Error\n(400 / 422 / TTS fail)", fc=RED, ec="#DC2626")
    arr(ax, cx+1.7, 14.7, 6.65, 14.7, "invalid /\nempty", lbldx=0.0)
    ax.plot([7.9, 7.9], [15.18, 19.0], color="#DC2626", lw=1.1)
    arr(ax, 7.9, 19.0, cx+1.7, 19.0, "", color="#DC2626")
    ax.text(8.05, 17.2, "→ Idle", color="#DC2626", fontsize=FS-2, rotation=90, va="center")
    rbox(ax, 1.25, 5.5, 2.0, 1.7, "WARNING flag\n(density < 8%)\nshown WITH result\n— non-terminal,\nNMT still runs", fc=AMBERBG, ec=AMBER, fs=FS-2)
    arr(ax, 2.25, 5.5, cx-1.7, 5.5, color=AMBER)
    ax.set_title("Figure 3.7: State Diagram — Request Lifecycle", fontsize=12, weight="bold", color=INK, pad=12)
    save(fig, "state_diagram.png")


print("Generating UML diagrams ->", OUT)
activity(); use_case(); sequence(); class_diagram(); state()
print("DONE")
