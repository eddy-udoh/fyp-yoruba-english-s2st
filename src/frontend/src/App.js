import { useState, useRef, useEffect } from "react";
import axios from "axios";

const API = "http://localhost:5000";

const STAGES = [
  { id: "asr",   label: "Transcribing speech",  sub: "Whisper ASR",         color: "#0F6E56" },
  { id: "norm",  label: "Validating diacritics", sub: "Normalisation layer", color: "#1D9E75" },
  { id: "nmt",   label: "Translating",           sub: "MarianMT",            color: "#0F6E56" },
  { id: "tts",   label: "Synthesising audio",    sub: "Edge TTS Neural",     color: "#1D9E75" },
];

function PipelineStage({ stage, status }) {
  const colors = {
    idle:    { bg: "#F4FAF7", border: "#C8E6D8", icon: "#9EC8B5", text: "#6A9E88" },
    active:  { bg: "#E1F5EE", border: "#1D9E75", icon: "#0F6E56", text: "#085041" },
    done:    { bg: "#0F6E56", border: "#0F6E56", icon: "#ffffff", text: "#ffffff" },
  };
  const c = colors[status] || colors.idle;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12, padding: "10px 14px",
      borderRadius: 10, border: `1.5px solid ${c.border}`,
      background: c.bg, transition: "all 0.4s ease",
    }}>
      <div style={{
        width: 34, height: 34, borderRadius: "50%",
        background: status === "done" ? "#ffffff22" : c.bg,
        border: `1.5px solid ${c.border}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0,
      }}>
        {status === "done"
          ? <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 8l3.5 3.5L13 5" stroke={c.icon} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
          : status === "active"
          ? <div style={{ width: 10, height: 10, borderRadius: "50%", background: c.icon, animation: "ping 1s ease-in-out infinite" }} />
          : <div style={{ width: 8, height: 8, borderRadius: "50%", background: c.icon }} />
        }
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: c.text, letterSpacing: "-0.01em" }}>{stage.label}</div>
        <div style={{ fontSize: 11, color: status === "done" ? "#ffffff99" : "#6A9E88", marginTop: 1 }}>{stage.sub}</div>
      </div>
      {status === "active" && (
        <div style={{ marginLeft: "auto", display: "flex", gap: 3 }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width: 4, height: 4, borderRadius: "50%", background: "#0F6E56",
              animation: `bounce 0.9s ease-in-out ${i * 0.15}s infinite`,
            }} />
          ))}
        </div>
      )}
    </div>
  );
}

function WaveBar({ delay, active }) {
  return (
    <div style={{
      width: 3, borderRadius: 3,
      background: active ? "#0F6E56" : "#C8E6D8",
      minHeight: 4,
      animation: active ? `wave 0.7s ease-in-out ${delay}s infinite` : "none",
      height: active ? undefined : 4,
      transition: "background 0.3s",
    }} />
  );
}

export default function App() {
  const [direction, setDirection]     = useState("yo-en");
  const [recState, setRecState]       = useState("idle");
  const [stageState, setStageState]   = useState({});
  const [transcript, setTranscript]   = useState("");
  const [translation, setTranslation] = useState("");
  const [directTrans, setDirectTrans] = useState("");
  const [density, setDensity]         = useState(null);
  const [latency, setLatency]         = useState(null);
  const [ttsEngine, setTtsEngine]     = useState("");
  const [audioSrc, setAudioSrc]       = useState(null);
  const [warnDia, setWarnDia]         = useState(false);
  const [hasResult, setHasResult]     = useState(false);
  const mediaRef = useRef(null);
  const chunksRef = useRef([]);
  const audioRef  = useRef(null);

  const activateStage = (id) => setStageState(s => ({ ...s, [id]: "active" }));
  const completeStage = (id) => setStageState(s => ({ ...s, [id]: "done" }));
  const resetStages   = ()   => setStageState({});

  const handleRecord = async () => {
    if (recState === "idle") {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mr = new MediaRecorder(stream);
        chunksRef.current = [];
        mr.ondataavailable = e => chunksRef.current.push(e.data);
        mr.onstop = () => { stream.getTracks().forEach(t => t.stop()); sendAudio(); };
        mr.start();
        mediaRef.current = mr;
        setRecState("recording");
        resetStages();
        setHasResult(false);
        setTranscript(""); setTranslation(""); setDirectTrans("");
        setAudioSrc(null); setLatency(null); setDensity(null);
      } catch (e) {
        console.error(e);
      }
    } else if (recState === "recording") {
      mediaRef.current?.stop();
      setRecState("processing");
    }
  };

  const sendAudio = async () => {
    const blob = new Blob(chunksRef.current, { type: "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "recording.webm");
    form.append("direction", direction);

    try {
      activateStage("asr");
      const t0 = Date.now();
      const res = await axios.post(`${API}/translate`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: () => {},
      });
      const data = res.data;

      completeStage("asr");
      await delay(200);
      activateStage("norm");
      await delay(400);
      completeStage("norm");
      await delay(200);
      activateStage("nmt");
      await delay(500);
      completeStage("nmt");
      await delay(200);
      activateStage("tts");
      await delay(300);
      completeStage("tts");

      setTranscript(data.transcript || "");
      setTranslation(data.translation || "");
      setDirectTrans(data.direct_translation || "");
      setDensity(data.diacritic_density);
      // Prefer the server-measured pipeline latency; fall back to wall-clock only if absent.
      setLatency(data.latency_ms != null ? Math.round(data.latency_ms) : Math.round(Date.now() - t0));
      setTtsEngine(data.tts_note || "");
      setWarnDia(data.diacritic_density && data.diacritic_density < 0.08);
      if (data.audio_base64) {
        // Yoruba TTS (en→yo) returns WAV; English TTS (edge/gTTS) returns MP3.
        const mime = direction === "en-yo" ? "audio/wav" : "audio/mpeg";
        setAudioSrc(`data:${mime};base64,${data.audio_base64}`);
      }
      setHasResult(true);
    } catch (e) {
      console.error("API error:", e);
      resetStages();
    } finally {
      setRecState("idle");
    }
  };

  useEffect(() => {
    if (audioSrc && audioRef.current) {
      audioRef.current.load();
      audioRef.current.play().catch(() => {});
    }
  }, [audioSrc]);

  const delay = ms => new Promise(r => setTimeout(r, ms));
  const isRecording   = recState === "recording";
  const isProcessing  = recState === "processing";
  const isBusy        = isRecording || isProcessing;

  return (
    <div style={{
      minHeight: "100vh", background: "#F7FCF9",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      display: "flex", flexDirection: "column",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes wave { 0%,100%{height:6px} 50%{height:28px} }
        @keyframes ping  { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.3);opacity:0.6} }
        @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
        @keyframes spin  { to{transform:rotate(360deg)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
        @keyframes recPulse { 0%,100%{box-shadow:0 0 0 0 rgba(15,110,86,0.35)} 60%{box-shadow:0 0 0 18px rgba(15,110,86,0)} }
      `}</style>

      <header style={{
        padding: "16px 24px", borderBottom: "1px solid #DDF0E6",
        background: "#fff", display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, background: "#0F6E56",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="8" stroke="#fff" strokeWidth="1.5"/>
              <path d="M7 10h6M10 7v6" stroke="#fff" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 15, color: "#0A2E1F", letterSpacing: "-0.02em" }}>MedSpeak</div>
            <div style={{ fontSize: 11, color: "#6A9E88", fontWeight: 500 }}>Yorùbá · English · Healthcare</div>
          </div>
        </div>
        <div style={{
          fontSize: 11, fontWeight: 600, padding: "4px 10px", borderRadius: 99,
          background: "#E1F5EE", color: "#085041", letterSpacing: "0.04em", textTransform: "uppercase",
        }}>
          {isRecording ? "● Recording" : isProcessing ? "Processing" : "Ready"}
        </div>
      </header>

      <main style={{ flex: 1, padding: "24px 20px", maxWidth: 480, margin: "0 auto", width: "100%" }}>

        <div style={{
          display: "flex", background: "#E8F5EE", borderRadius: 99,
          padding: 3, marginBottom: 24, border: "1px solid #C8E6D8",
        }}>
          {["yo-en", "en-yo"].map(d => (
            <button key={d} onClick={() => !isBusy && setDirection(d)} style={{
              flex: 1, padding: "8px 0", borderRadius: 99, border: "none", cursor: isBusy ? "not-allowed" : "pointer",
              fontFamily: "inherit", fontSize: 13, fontWeight: 600, letterSpacing: "-0.01em",
              transition: "all 0.25s ease",
              background: direction === d ? "#0F6E56" : "transparent",
              color: direction === d ? "#fff" : "#6A9E88",
            }}>
              {d === "yo-en" ? "Yorùbá → English" : "English → Yorùbá"}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "center", gap: 3, height: 40, marginBottom: 16 }}>
            {[0,0.05,0.1,0.15,0.2,0.1,0.05,0.15,0.08,0.18,0.12,0.06,0.14,0.04,0.16].map((d, i) => (
              <WaveBar key={i} delay={d} active={isRecording} />
            ))}
          </div>

          <button onClick={handleRecord} disabled={isProcessing} style={{
            width: 76, height: 76, borderRadius: "50%", border: "none",
            background: isProcessing ? "#C8E6D8" : "#0F6E56",
            color: "#fff", cursor: isProcessing ? "not-allowed" : "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            animation: isRecording ? "recPulse 1.4s ease-out infinite" : "none",
            transition: "background 0.3s, transform 0.1s",
            transform: "scale(1)",
          }}>
            {isProcessing
              ? <div style={{ width: 22, height: 22, border: "2.5px solid #0F6E56", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
              : isRecording
              ? <svg width="22" height="22" viewBox="0 0 22 22" fill="none"><rect x="6" y="6" width="10" height="10" rx="2" fill="#fff"/></svg>
              : <svg width="22" height="22" viewBox="0 0 22 22" fill="none"><rect x="8" y="3" width="6" height="11" rx="3" fill="#fff"/><path d="M4 11a7 7 0 0014 0" stroke="#fff" strokeWidth="1.8" strokeLinecap="round"/><line x1="11" y1="18" x2="11" y2="21" stroke="#fff" strokeWidth="1.8" strokeLinecap="round"/></svg>
            }
          </button>
          <div style={{ marginTop: 10, fontSize: 12, fontWeight: 500, color: "#6A9E88" }}>
            {isRecording ? "Tap to stop" : isProcessing ? "Translating..." : "Tap to record"}
          </div>
        </div>

        {(isProcessing || hasResult) && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 24 }}>
            {STAGES.map(s => (
              <PipelineStage key={s.id} stage={s} status={stageState[s.id] || "idle"} />
            ))}
          </div>
        )}

        {hasResult && (
          <div style={{ animation: "fadeUp 0.5s ease forwards" }}>

            {warnDia && (
              <div style={{
                display: "flex", alignItems: "center", gap: 8, padding: "10px 14px",
                background: "#FFFBEB", border: "1px solid #FCD34D", borderRadius: 10,
                fontSize: 12, color: "#92400E", marginBottom: 16, fontWeight: 500,
              }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2L14 13H2L8 2Z" stroke="#D97706" strokeWidth="1.5" strokeLinejoin="round"/><line x1="8" y1="6" x2="8" y2="9" stroke="#D97706" strokeWidth="1.5" strokeLinecap="round"/><circle cx="8" cy="11" r="0.5" fill="#D97706" stroke="#D97706"/></svg>
                Low diacritic density — tonal accuracy may be reduced
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

              <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #DDF0E6", overflow: "hidden" }}>
                <div style={{ padding: "10px 14px", background: "#F4FAF7", borderBottom: "1px solid #DDF0E6", display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#0F6E56" }} />
                  <span style={{ fontSize: 11, fontWeight: 600, color: "#0F6E56", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    {direction === "yo-en" ? "Yorùbá speech recognised" : "English speech recognised"}
                  </span>
                </div>
                <div style={{ padding: "12px 14px", fontSize: 14, color: "#0A2E1F", lineHeight: 1.65, fontFamily: "'DM Mono', monospace" }}>
                  {transcript || "—"}
                </div>
              </div>

              {directTrans && (
                <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #DDF0E6", overflow: "hidden" }}>
                  <div style={{ padding: "10px 14px", background: "#F4FAF7", borderBottom: "1px solid #DDF0E6", display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#1D9E75" }} />
                    <span style={{ fontSize: 11, fontWeight: 600, color: "#1D9E75", textTransform: "uppercase", letterSpacing: "0.06em" }}>Direct translation</span>
                  </div>
                  <div style={{ padding: "12px 14px", fontSize: 14, color: "#0A2E1F", lineHeight: 1.65, fontStyle: "italic" }}>
                    {directTrans}
                  </div>
                </div>
              )}

              <div style={{ background: "#0F6E56", borderRadius: 12, overflow: "hidden" }}>
                <div style={{ padding: "10px 14px", background: "#085041", display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#5DCAA5" }} />
                  <span style={{ fontSize: 11, fontWeight: 600, color: "#9FE1CB", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    {direction === "yo-en" ? "English translation" : "Yorùbá translation"}
                  </span>
                </div>
                <div style={{ padding: "12px 14px", fontSize: 14, color: "#E1F5EE", lineHeight: 1.65, fontWeight: 500 }}>
                  {translation || "—"}
                </div>
              </div>

              {audioSrc && (
                <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #DDF0E6", padding: "12px 14px" }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "#0F6E56", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                    Synthesised audio · {ttsEngine}
                  </div>
                  <audio ref={audioRef} controls src={audioSrc} style={{ width: "100%", height: 36 }} />
                </div>
              )}

              <div style={{ display: "flex", gap: 8 }}>
                {[
                  { label: "Latency", val: latency ? `${latency}ms` : "—" },
                  { label: "Diacritic density", val: density ? `${(density * 100).toFixed(1)}%` : "—" },
                  { label: "Direction", val: direction === "yo-en" ? "YO→EN" : "EN→YO" },
                ].map(m => (
                  <div key={m.label} style={{
                    flex: 1, background: "#fff", borderRadius: 10, border: "1px solid #DDF0E6",
                    padding: "10px 12px", textAlign: "center",
                  }}>
                    <div style={{ fontSize: 16, fontWeight: 600, color: "#0A2E1F", letterSpacing: "-0.02em" }}>{m.val}</div>
                    <div style={{ fontSize: 10, color: "#6A9E88", fontWeight: 500, marginTop: 2, textTransform: "uppercase", letterSpacing: "0.04em" }}>{m.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {!hasResult && !isBusy && (
          <div style={{ textAlign: "center", padding: "20px 0", color: "#9EC8B5", fontSize: 13 }}>
            {direction === "yo-en"
              ? "Speak Yorùbá — get an English translation"
              : "Speak English — get a Yorùbá translation"}
          </div>
        )}
      </main>
    </div>
  );
}
