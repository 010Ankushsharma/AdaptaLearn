/**
 * frontend/src/App.jsx
 * Adaptive Learning System — Student Interface
 *
 * Screens:
 *   1. Login / consent gate
 *   2. Pre-test
 *   3. Active session  — question renderer + live mastery radar
 *   4. Session complete — results summary
 *   5. Post-test
 *
 * API calls mirror backend/views.py endpoints.
 * All state lives in React — no localStorage (privacy).
 *
 * Libraries used (from package.json):
 *   recharts   — radar + line charts
 *   lucide-react — icons
 */

import { useState, useEffect, useCallback, useRef } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Tooltip,
  LineChart, Line, XAxis, YAxis, CartesianGrid,
} from "recharts";
import {
  Brain, CheckCircle, XCircle, ChevronRight,
  Clock, Target, Zap, BarChart2, Award, RefreshCw,
} from "lucide-react";

// ── API helpers ──────────────────────────────────────────────────────────────

const API = {
  base: process.env.REACT_APP_API_URL || "http://localhost:8000/api",

  async post(path, body = {}) {
    const r = await fetch(`${this.base}${path}`, {
      method:      "POST",
      headers:     { "Content-Type": "application/json" },
      credentials: "include",
      body:        JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${r.status}`);
    }
    return r.json();
  },

  async get(path) {
    const r = await fetch(`${this.base}${path}`, {
      credentials: "include",
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
};

// ── Concept labels (matches config.NUM_CONCEPTS = 188 in prod;
//    trimmed to 10 for the demo / synthetic dataset) ────────────────────────

const CONCEPT_LABELS = [
  "Variables", "Loops", "Functions", "Lists",
  "Dicts", "Classes", "Files", "Exceptions",
  "Modules", "Algorithms",
];

const label = (i) => CONCEPT_LABELS[i] || `C${i}`;

// ── Colour palette ────────────────────────────────────────────────────────────

const C = {
  bg:       "#0D0F14",
  surface:  "#141720",
  border:   "#1E2330",
  accent:   "#7F77DD",
  accentLt: "#AFA9EC",
  green:    "#5DCAA5",
  amber:    "#EF9F27",
  coral:    "#F0997B",
  text:     "#E8E9F0",
  muted:    "#6B7280",
};

// ── Global styles injected once ───────────────────────────────────────────────

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: ${C.bg};
    color: ${C.text};
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
  }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: ${C.bg}; }
  ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 0.6; }
    100% { transform: scale(1.6); opacity: 0; }
  }
  @keyframes shimmer {
    from { background-position: -400px 0; }
    to   { background-position:  400px 0; }
  }

  .fade-up  { animation: fadeUp 0.35s ease both; }
  .fade-up-2{ animation: fadeUp 0.35s 0.08s ease both; }
  .fade-up-3{ animation: fadeUp 0.35s 0.16s ease both; }
  .fade-up-4{ animation: fadeUp 0.35s 0.24s ease both; }

  .skeleton {
    background: linear-gradient(90deg, ${C.surface} 25%, ${C.border} 50%, ${C.surface} 75%);
    background-size: 400px 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 6px;
  }
`;

function GlobalStyle() {
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = GLOBAL_CSS;
    document.head.appendChild(el);
    return () => el.remove();
  }, []);
  return null;
}

// ── Shared UI primitives ──────────────────────────────────────────────────────

function Card({ children, style, className = "" }) {
  return (
    <div className={className} style={{
      background:   C.surface,
      border:       `1px solid ${C.border}`,
      borderRadius: 16,
      padding:      "1.5rem",
      ...style,
    }}>
      {children}
    </div>
  );
}

function Btn({ children, onClick, variant = "primary", disabled, style }) {
  const base = {
    display:       "inline-flex",
    alignItems:    "center",
    gap:           8,
    padding:       "0.65rem 1.4rem",
    borderRadius:  10,
    border:        "none",
    fontFamily:    "Syne, sans-serif",
    fontWeight:    600,
    fontSize:      15,
    cursor:        disabled ? "not-allowed" : "pointer",
    opacity:       disabled ? 0.45 : 1,
    transition:    "all 0.15s",
    ...style,
  };
  const variants = {
    primary:  { background: C.accent,  color: "#fff" },
    ghost:    { background: "transparent", color: C.muted, border: `1px solid ${C.border}` },
    success:  { background: C.green,   color: "#fff" },
    danger:   { background: C.coral,   color: "#fff" },
  };
  return (
    <button onClick={disabled ? undefined : onClick}
            style={{ ...base, ...variants[variant] }}>
      {children}
    </button>
  );
}

function Tag({ children, color = C.accentLt }) {
  return (
    <span style={{
      fontFamily:   "'DM Mono', monospace",
      fontSize:     11,
      padding:      "3px 10px",
      borderRadius: 20,
      background:   color + "22",
      color,
      letterSpacing: "0.04em",
    }}>
      {children}
    </span>
  );
}

function ProgressBar({ value, max, color = C.accent }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div style={{ background: C.border, borderRadius: 4, height: 4, overflow: "hidden" }}>
      <div style={{
        width:      `${pct}%`,
        height:     "100%",
        background: color,
        borderRadius: 4,
        transition: "width 0.4s ease",
      }} />
    </div>
  );
}

// ── Mastery radar chart ───────────────────────────────────────────────────────

function MasteryRadar({ mastery, size = 280 }) {
  const data = mastery.slice(0, 10).map((v, i) => ({
    subject: label(i),
    value:   Math.round(v * 100),
    fullMark: 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={size}>
      <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
        <PolarGrid stroke={C.border} />
        <PolarAngleAxis
          dataKey="subject"
          tick={{ fill: C.muted, fontSize: 11, fontFamily: "DM Mono, monospace" }}
        />
        <PolarRadiusAxis
          angle={90} domain={[0, 100]} tick={false} axisLine={false}
        />
        <Radar
          name="Mastery"
          dataKey="value"
          stroke={C.accent}
          fill={C.accent}
          fillOpacity={0.18}
          strokeWidth={2}
          dot={{ fill: C.accent, r: 3 }}
        />
        <Tooltip
          contentStyle={{
            background:   C.surface,
            border:       `1px solid ${C.border}`,
            borderRadius: 8,
            fontFamily:   "DM Mono, monospace",
            fontSize:     12,
            color:        C.text,
          }}
          formatter={(v) => [`${v}%`, "Mastery"]}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}

// ── Reward history line chart ─────────────────────────────────────────────────

function RewardLine({ rewards }) {
  const data = rewards.map((r, i) => ({ step: i + 1, reward: +r.toFixed(3) }));
  return (
    <ResponsiveContainer width="100%" height={100}>
      <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <CartesianGrid stroke={C.border} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" tick={{ fill: C.muted, fontSize: 10 }} />
        <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
        <Line
          type="monotone" dataKey="reward"
          stroke={C.green} strokeWidth={2} dot={false}
          activeDot={{ r: 4, fill: C.green }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Screen: Login ─────────────────────────────────────────────────────────────

function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error,    setError]    = useState("");
  const [loading,  setLoading]  = useState(false);

  const submit = async () => {
    if (!username || !password) return;
    setLoading(true); setError("");
    try {
      await API.post("/auth/login/", { username, password });
      onLogin(username);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight:      "100vh",
      display:        "flex",
      alignItems:     "center",
      justifyContent: "center",
      padding:        "2rem",
    }}>
      <div style={{ width: "100%", maxWidth: 400 }}>
        {/* Logo */}
        <div className="fade-up" style={{ textAlign: "center", marginBottom: "2.5rem" }}>
          <div style={{
            width: 64, height: 64, borderRadius: 18,
            background: `${C.accent}22`,
            border: `1px solid ${C.accent}44`,
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            marginBottom: "1rem",
          }}>
            <Brain size={32} color={C.accent} />
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.5px" }}>
            AdaptLearn
          </h1>
          <p style={{ color: C.muted, marginTop: 6, fontSize: 14 }}>
            AI-powered personalised Python tutor
          </p>
        </div>

        <Card className="fade-up-2">
          <div style={{ marginBottom: "1rem" }}>
            <label style={{ fontSize: 13, color: C.muted, display: "block", marginBottom: 6 }}>
              Username
            </label>
            <input
              value={username}
              onChange={e => setUsername(e.target.value)}
              onKeyDown={e => e.key === "Enter" && submit()}
              placeholder="your.username"
              style={{
                width: "100%", padding: "0.65rem 1rem",
                background: C.bg, border: `1px solid ${C.border}`,
                borderRadius: 10, color: C.text,
                fontFamily: "DM Mono, monospace", fontSize: 14,
                outline: "none",
              }}
            />
          </div>
          <div style={{ marginBottom: "1.25rem" }}>
            <label style={{ fontSize: 13, color: C.muted, display: "block", marginBottom: 6 }}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === "Enter" && submit()}
              placeholder="••••••••"
              style={{
                width: "100%", padding: "0.65rem 1rem",
                background: C.bg, border: `1px solid ${C.border}`,
                borderRadius: 10, color: C.text,
                fontFamily: "DM Mono, monospace", fontSize: 14,
                outline: "none",
              }}
            />
          </div>
          {error && (
            <p style={{ color: C.coral, fontSize: 13, marginBottom: "1rem" }}>{error}</p>
          )}
          <Btn onClick={submit} disabled={loading || !username || !password}
               style={{ width: "100%", justifyContent: "center" }}>
            {loading ? "Signing in…" : "Sign in"} <ChevronRight size={16} />
          </Btn>
        </Card>
      </div>
    </div>
  );
}

// ── Screen: Active session ────────────────────────────────────────────────────

function SessionScreen({ sessionId, mastery: initMastery, onComplete }) {
  const [phase,      setPhase]      = useState("loading"); // loading|question|feedback|done
  const [question,   setQuestion]   = useState(null);
  const [selected,   setSelected]   = useState(null);
  const [feedback,   setFeedback]   = useState(null);   // {correct, reward, explanation}
  const [mastery,    setMastery]    = useState(initMastery);
  const [rewards,    setRewards]    = useState([]);
  const [step,       setStep]       = useState(0);
  const [error,      setError]      = useState("");
  const timerRef = useRef(null);
  const startMs  = useRef(Date.now());

  const MAX_Q = 20;

  const fetchNext = useCallback(async () => {
    setPhase("loading");
    setSelected(null);
    setFeedback(null);
    setError("");
    startMs.current = Date.now();
    try {
      const data = await API.post(`/session/${sessionId}/next/`);
      setQuestion(data);
      setPhase("question");
    } catch (e) {
      setError(e.message);
      setPhase("question");
    }
  }, [sessionId]);

  useEffect(() => { fetchNext(); }, [fetchNext]);

  const submitAnswer = async (idx) => {
    if (phase !== "question" || selected !== null) return;
    setSelected(idx);
    const elapsed_ms = Date.now() - startMs.current;
    try {
      const data = await API.post(`/session/${sessionId}/answer/`, {
        question_id:  question.question_id,
        answer_index: idx,
        elapsed_ms,
        hint_used: false,
      });
      setFeedback(data);
      setMastery(data.mastery_vector);
      setRewards(prev => [...prev, data.reward]);
      setStep(data.step);
      setPhase("feedback");
      if (data.session_complete) {
        setTimeout(() => onComplete(data.mastery_vector, rewards.concat(data.reward)), 1800);
      }
    } catch (e) {
      setError(e.message);
      setPhase("question");
      setSelected(null);
    }
  };

  const optionColor = (i) => {
    if (selected === null) return C.surface;
    if (feedback && i === feedback.correct_index) return `${C.green}22`;
    if (i === selected && !feedback?.correct) return `${C.coral}22`;
    return C.surface;
  };
  const optionBorder = (i) => {
    if (selected === null) return C.border;
    if (feedback && i === feedback.correct_index) return C.green;
    if (i === selected && !feedback?.correct) return C.coral;
    return C.border;
  };

  return (
    <div style={{
      minHeight: "100vh",
      display:   "grid",
      gridTemplateColumns: "1fr 320px",
      gap: "1.5rem",
      padding: "1.5rem",
      maxWidth: 1100,
      margin: "0 auto",
    }}>

      {/* ── Left: question panel ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

        {/* Header */}
        <div className="fade-up" style={{
          display: "flex", alignItems: "center",
          justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <Brain size={22} color={C.accent} />
            <span style={{ fontWeight: 600, fontSize: 17 }}>AdaptLearn</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {question && <Tag color={C.accentLt}>{question.concept_name}</Tag>}
            <Tag color={C.amber}>Step {step}/{MAX_Q}</Tag>
          </div>
        </div>

        {/* Progress */}
        <ProgressBar value={step} max={MAX_Q} color={C.accent} />

        {/* Question card */}
        {phase === "loading" ? (
          <Card style={{ flex: 1 }}>
            <div className="skeleton" style={{ height: 24, width: "60%", marginBottom: 16 }} />
            <div className="skeleton" style={{ height: 16, width: "90%", marginBottom: 8 }} />
            <div className="skeleton" style={{ height: 16, width: "75%", marginBottom: 24 }} />
            {[0,1,2,3].map(i => (
              <div key={i} className="skeleton"
                   style={{ height: 52, borderRadius: 10, marginBottom: 10 }} />
            ))}
          </Card>
        ) : question ? (
          <Card className="fade-up" style={{ flex: 1 }}>
            {/* Difficulty badge */}
            <div style={{ display: "flex", gap: 8, marginBottom: "1.25rem" }}>
              <Tag color={
                question.difficulty < 0.4 ? C.green :
                question.difficulty < 0.7 ? C.amber : C.coral
              }>
                {question.difficulty < 0.4 ? "Easy" :
                 question.difficulty < 0.7 ? "Medium" : "Hard"}
              </Tag>
              <Tag color={C.muted}>P(correct) {Math.round(question.p_correct * 100)}%</Tag>
            </div>

            {/* Question text */}
            <p style={{
              fontSize: 18, fontWeight: 500, lineHeight: 1.65,
              marginBottom: "1.75rem", letterSpacing: "-0.2px",
            }}>
              {question.question_text || `[Question ${question.question_id}]`}
            </p>

            {/* Answer options */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {(question.answer_options?.length > 0
                ? question.answer_options
                : ["Option A", "Option B", "Option C", "Option D"].map(t => ({ text: t }))
              ).map((opt, i) => (
                <button key={i}
                  onClick={() => submitAnswer(i)}
                  disabled={selected !== null}
                  style={{
                    display:    "flex",
                    alignItems: "center",
                    gap: 14,
                    padding:    "0.85rem 1.1rem",
                    background: optionColor(i),
                    border:     `1px solid ${optionBorder(i)}`,
                    borderRadius: 12,
                    cursor:     selected !== null ? "default" : "pointer",
                    textAlign:  "left",
                    transition: "all 0.15s",
                    color:      C.text,
                    fontFamily: "Syne, sans-serif",
                    fontSize:   15,
                  }}
                >
                  <span style={{
                    width: 28, height: 28, borderRadius: 8,
                    background: C.border,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontFamily: "DM Mono, monospace", fontSize: 12, flexShrink: 0,
                    color: C.muted,
                  }}>
                    {String.fromCharCode(65 + i)}
                  </span>
                  <span>{opt.text || opt}</span>
                  {feedback && i === feedback.correct_index && (
                    <CheckCircle size={18} color={C.green} style={{ marginLeft: "auto" }} />
                  )}
                  {selected === i && !feedback?.correct && (
                    <XCircle size={18} color={C.coral} style={{ marginLeft: "auto" }} />
                  )}
                </button>
              ))}
            </div>
          </Card>
        ) : null}

        {/* Feedback bar */}
        {feedback && (
          <Card className="fade-up" style={{
            background: feedback.correct ? `${C.green}11` : `${C.coral}11`,
            border: `1px solid ${feedback.correct ? C.green : C.coral}44`,
            padding: "1rem 1.25rem",
          }}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
              {feedback.correct
                ? <CheckCircle size={20} color={C.green} style={{ flexShrink: 0, marginTop: 2 }} />
                : <XCircle    size={20} color={C.coral} style={{ flexShrink: 0, marginTop: 2 }} />
              }
              <div style={{ flex: 1 }}>
                <p style={{
                  fontWeight: 600, fontSize: 15,
                  color: feedback.correct ? C.green : C.coral,
                  marginBottom: 4,
                }}>
                  {feedback.correct ? "Correct!" : "Not quite."}
                  <span style={{ marginLeft: 10, fontSize: 13, opacity: 0.8 }}>
                    Reward {feedback.reward >= 0 ? "+" : ""}{feedback.reward?.toFixed(3)}
                  </span>
                </p>
                {feedback.explanation && (
                  <p style={{ fontSize: 13, color: C.muted, lineHeight: 1.6 }}>
                    {feedback.explanation}
                  </p>
                )}
              </div>
              {!feedback.session_complete && (
                <Btn onClick={fetchNext} variant="ghost" style={{ flexShrink: 0 }}>
                  Next <ChevronRight size={15} />
                </Btn>
              )}
            </div>
          </Card>
        )}

        {error && (
          <p style={{ color: C.coral, fontSize: 13 }}>⚠ {error}</p>
        )}
      </div>

      {/* ── Right: mastery sidebar ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

        {/* Mastery radar */}
        <Card className="fade-up-2">
          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            marginBottom: "0.75rem",
          }}>
            <Target size={16} color={C.accent} />
            <span style={{ fontSize: 13, fontWeight: 600 }}>Knowledge map</span>
          </div>
          <MasteryRadar mastery={mastery.slice(0, 10)} size={220} />
        </Card>

        {/* Concept bars */}
        <Card className="fade-up-3">
          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            marginBottom: "1rem",
          }}>
            <BarChart2 size={16} color={C.accent} />
            <span style={{ fontSize: 13, fontWeight: 600 }}>Mastery per concept</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {mastery.slice(0, 10).map((m, i) => (
              <div key={i}>
                <div style={{
                  display: "flex", justifyContent: "space-between",
                  marginBottom: 3,
                }}>
                  <span style={{
                    fontSize: 11, color: C.muted,
                    fontFamily: "DM Mono, monospace",
                  }}>{label(i)}</span>
                  <span style={{
                    fontSize: 11, color: m >= 0.85 ? C.green : C.muted,
                    fontFamily: "DM Mono, monospace",
                  }}>{Math.round(m * 100)}%</span>
                </div>
                <ProgressBar
                  value={m * 100} max={100}
                  color={m >= 0.85 ? C.green : m >= 0.5 ? C.accent : C.amber}
                />
              </div>
            ))}
          </div>
        </Card>

        {/* Reward history */}
        {rewards.length > 1 && (
          <Card className="fade-up-4">
            <div style={{
              display: "flex", alignItems: "center", gap: 8,
              marginBottom: "0.75rem",
            }}>
              <Zap size={16} color={C.green} />
              <span style={{ fontSize: 13, fontWeight: 600 }}>Reward history</span>
            </div>
            <RewardLine rewards={rewards} />
          </Card>
        )}
      </div>
    </div>
  );
}

// ── Screen: Session complete ──────────────────────────────────────────────────

function CompleteScreen({ mastery, rewards, onPostTest }) {
  const meanMastery   = mastery.reduce((a, b) => a + b, 0) / mastery.length;
  const nMastered     = mastery.filter(m => m >= 0.85).length;
  const totalReward   = rewards.reduce((a, b) => a + b, 0);

  const stats = [
    { icon: <Target  size={18} color={C.accent} />, label: "Mean mastery",  value: `${Math.round(meanMastery * 100)}%` },
    { icon: <Award   size={18} color={C.green}  />, label: "Concepts mastered", value: `${nMastered}/${mastery.length}` },
    { icon: <Zap     size={18} color={C.amber}  />, label: "Total reward",  value: totalReward.toFixed(2) },
    { icon: <BarChart2 size={18} color={C.accentLt}/>, label: "Questions",  value: rewards.length },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: "2rem",
    }}>
      <div style={{ width: "100%", maxWidth: 700 }}>
        <div className="fade-up" style={{ textAlign: "center", marginBottom: "2rem" }}>
          <div style={{
            width: 72, height: 72, borderRadius: 20,
            background: `${C.green}22`,
            border: `1px solid ${C.green}44`,
            display: "inline-flex", alignItems: "center",
            justifyContent: "center", marginBottom: "1rem",
          }}>
            <Award size={36} color={C.green} />
          </div>
          <h2 style={{ fontSize: 30, fontWeight: 700, letterSpacing: "-0.5px" }}>
            Session complete
          </h2>
          <p style={{ color: C.muted, marginTop: 8 }}>
            Great work — here's how you did
          </p>
        </div>

        {/* Stats grid */}
        <div className="fade-up-2" style={{
          display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: "1rem", marginBottom: "1.5rem",
        }}>
          {stats.map((s, i) => (
            <Card key={i} style={{ padding: "1.25rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                {s.icon}
                <span style={{ fontSize: 13, color: C.muted }}>{s.label}</span>
              </div>
              <span style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-1px" }}>
                {s.value}
              </span>
            </Card>
          ))}
        </div>

        {/* Final radar */}
        <Card className="fade-up-3" style={{ marginBottom: "1.5rem" }}>
          <p style={{ fontSize: 13, color: C.muted, marginBottom: "0.5rem" }}>
            Final knowledge map
          </p>
          <MasteryRadar mastery={mastery.slice(0, 10)} size={260} />
        </Card>

        <div className="fade-up-4" style={{ textAlign: "center" }}>
          <Btn onClick={onPostTest} variant="success">
            Take post-test to measure your progress <ChevronRight size={16} />
          </Btn>
        </div>
      </div>
    </div>
  );
}

// ── Root app ──────────────────────────────────────────────────────────────────

export default function App() {
  const [screen,    setScreen]    = useState("login");
  const [sessionId, setSessionId] = useState(null);
  const [mastery,   setMastery]   = useState(Array(10).fill(0.3));
  const [rewards,   setRewards]   = useState([]);
  const [error,     setError]     = useState("");

  // Mock mastery for dev (remove when backend is live)
  const DEV_MODE = !process.env.REACT_APP_API_URL;

  const handleLogin = async (username) => {
    if (DEV_MODE) {
      setScreen("session");
      setSessionId("dev-session-001");
      setMastery(Array(10).fill(0).map(() => 0.2 + Math.random() * 0.4));
      return;
    }
    try {
      const data = await API.post("/session/start/");
      setSessionId(data.session_id);
      setMastery(data.mastery_vector);
      setScreen("session");
    } catch (e) {
      setError(e.message);
    }
  };

  const handleComplete = (finalMastery, finalRewards) => {
    setMastery(finalMastery);
    setRewards(finalRewards);
    setScreen("complete");
  };

  const handlePostTest = () => {
    // Navigate to post-test (simplified: just show a thank you)
    setScreen("done");
  };

  return (
    <>
      <GlobalStyle />
      {screen === "login"    && <LoginScreen onLogin={handleLogin} />}
      {screen === "session"  && (
        <SessionScreen
          sessionId={sessionId}
          mastery={mastery}
          onComplete={handleComplete}
        />
      )}
      {screen === "complete" && (
        <CompleteScreen
          mastery={mastery}
          rewards={rewards}
          onPostTest={handlePostTest}
        />
      )}
      {screen === "done" && (
        <div style={{
          minHeight: "100vh", display: "flex",
          alignItems: "center", justifyContent: "center",
          flexDirection: "column", gap: 16, textAlign: "center",
        }}>
          <CheckCircle size={56} color={C.green} />
          <h2 style={{ fontSize: 26, fontWeight: 700 }}>All done!</h2>
          <p style={{ color: C.muted }}>
            Thank you for participating. Your data has been recorded.
          </p>
          <Btn onClick={() => { setScreen("login"); setRewards([]); }} variant="ghost">
            <RefreshCw size={15} /> New session
          </Btn>
        </div>
      )}
    </>
  );
}