import { useMemo, useState } from "react";
import {
  explainPatient,
  getAnalyticsSummary,
  getCurrentUser,
  getPreferences,
  getTopFeatures,
  predictPatient,
  setPreferences,
  submitFeedback
} from "./api/client";

const INITIAL_PATIENT = {
  male: "",
  age: "",
  education: "",
  currentSmoker: "",
  cigsPerDay: "",
  BPMeds: "",
  prevalentStroke: "",
  prevalentHyp: "",
  diabetes: "",
  totChol: "",
  sysBP: "",
  diaBP: "",
  BMI: "",
  heartRate: "",
  glucose: ""
};

const FIELD_META = {
  male: { label: "Sex (Male=1, Female=0)", unit: "binary", step: "1", min: 0, max: 1 },
  age: { label: "Age", unit: "years", step: "1", min: 18, max: 110 },
  education: { label: "Education Level", unit: "1-4", step: "1", min: 1, max: 4, optional: true },
  currentSmoker: { label: "Current Smoker", unit: "binary", step: "1", min: 0, max: 1 },
  cigsPerDay: { label: "Cigarettes per Day", unit: "count", step: "1", min: 0, optional: true },
  BPMeds: { label: "On BP Medication", unit: "binary", step: "1", min: 0, max: 1, optional: true },
  prevalentStroke: { label: "History of Stroke", unit: "binary", step: "1", min: 0, max: 1 },
  prevalentHyp: { label: "History of Hypertension", unit: "binary", step: "1", min: 0, max: 1 },
  diabetes: { label: "Diabetes Diagnosis", unit: "binary", step: "1", min: 0, max: 1 },
  totChol: { label: "Total Cholesterol", unit: "mg/dL", step: "1", min: 50, optional: true },
  sysBP: { label: "Systolic BP", unit: "mmHg", step: "1", min: 60 },
  diaBP: { label: "Diastolic BP", unit: "mmHg", step: "1", min: 40 },
  BMI: { label: "BMI", unit: "kg/m^2", step: "0.1", min: 10, optional: true },
  heartRate: { label: "Heart Rate", unit: "bpm", step: "1", min: 30, optional: true },
  glucose: { label: "Glucose", unit: "mg/dL", step: "1", min: 30, optional: true }
};

const FIELD_ORDER = Object.keys(FIELD_META);
const FEATURE_OPTIONS = FIELD_ORDER;

const FEEDBACK_LABELS = {
  irrelevant: "This feature seems irrelevant",
  confusing: "This feature is confusing",
  prefer_short: "Prefer shorter explanations",
  prefer_long: "Prefer detailed explanations",
  relevant: "This feature is relevant"
};

function formatRisk(v) {
  if (typeof v !== "number") return "-";
  return `${(v * 100).toFixed(2)}%`;
}

function riskTone(risk, threshold) {
  if (typeof risk !== "number") return "neutral";
  if (typeof threshold !== "number") return "neutral";
  if (risk >= threshold * 2) return "high";
  if (risk >= threshold) return "medium";
  return "low";
}

function screeningStatusLabel(risk, threshold) {
  if (typeof risk !== "number" || typeof threshold !== "number") return "-";
  return risk >= threshold ? "Flagged for follow-up" : "Not flagged";
}

function feedbackTypeLabel(feedbackType) {
  return FEEDBACK_LABELS[feedbackType] || feedbackType;
}

function AlertModal({ open, title, message, onClose }) {
  if (!open) return null;
  return (
    <div className="alert-backdrop" role="alertdialog" aria-modal="true">
      <div className="alert-modal">
        <h3>{title}</h3>
        <p>{message}</p>
        <button onClick={onClose}>OK</button>
      </div>
    </div>
  );
}

function FeatureTable({ title, items }) {
  return (
    <div className="result-card">
      <h3>{title}</h3>
      {!items || items.length === 0 ? (
        <p className="muted">No items available.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Value</th>
              <th>SHAP</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item, idx) => (
              <tr key={`${item.feature}-${idx}`}>
                <td>{item.feature}</td>
                <td>{String(item.value)}</td>
                <td>{Number(item.shap).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function App() {
  const [authToken, setAuthToken] = useState("");
  const [sessionUser, setSessionUser] = useState("");
  const [patient, setPatient] = useState(INITIAL_PATIENT);
  const [feedbackType, setFeedbackType] = useState("irrelevant");
  const [feedbackFeature, setFeedbackFeature] = useState("sysBP");
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [predictOut, setPredictOut] = useState(null);
  const [explainOut, setExplainOut] = useState(null);
  const [prefsOut, setPrefsOut] = useState(null);
  const [analyticsOut, setAnalyticsOut] = useState(null);
  const [topFeaturesOut, setTopFeaturesOut] = useState([]);
  const [showRawJson, setShowRawJson] = useState(false);
  const [alertState, setAlertState] = useState({
    open: false,
    title: "",
    message: ""
  });

  const explainRisk = useMemo(() => formatRisk(explainOut?.risk), [explainOut]);
  const predictRisk = useMemo(() => formatRisk(predictOut?.risk), [predictOut]);
  const hasSession = authToken.trim().length > 0;
  const currentRisk = explainOut?.risk ?? predictOut?.risk;
  const currentThreshold = explainOut?.threshold ?? predictOut?.threshold;
  const tone = riskTone(currentRisk, currentThreshold);

  function updateField(name, value) {
    setPatient((prev) => ({ ...prev, [name]: value }));
  }

  function openAlert(title, message) {
    setAlertState({ open: true, title, message });
  }

  function closeAlert() {
    setAlertState({ open: false, title: "", message: "" });
  }

  function validatePatientInput() {
    const issues = [];
    for (const field of FIELD_ORDER) {
      const meta = FIELD_META[field];
      const raw = patient[field];
      if (raw === "" || raw === null || raw === undefined) {
        if (!meta.optional) {
          issues.push(`${meta.label}: this field is required.`);
        }
        continue;
      }
      const num = Number(raw);
      if (Number.isNaN(num)) {
        issues.push(`${meta.label}: must be a number.`);
        continue;
      }
      if (meta.min !== undefined && num < meta.min) {
        issues.push(`${meta.label}: cannot be less than ${meta.min}.`);
      }
      if (meta.max !== undefined && num > meta.max) {
        issues.push(`${meta.label}: cannot be greater than ${meta.max}.`);
      }
    }
    return issues;
  }

  async function runPredict() {
    if (!hasSession) {
      openAlert("Auth Token Required", "Enter an auth token before running prediction.");
      return;
    }
    const issues = validatePatientInput();
    if (issues.length > 0) {
      openAlert("Invalid Input", issues[0]);
      return;
    }
    setBusyLabel("Running prediction...");
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const data = await predictPatient(patient, authToken);
      setPredictOut(data);
      setNotice("Prediction completed.");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setBusyLabel("");
    }
  }

  async function runExplain() {
    if (!hasSession) {
      openAlert("Auth Token Required", "Enter an auth token before generating explanation.");
      return;
    }
    const issues = validatePatientInput();
    if (issues.length > 0) {
      openAlert("Invalid Input", issues[0]);
      return;
    }
    setBusyLabel("Generating explanation...");
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const data = await explainPatient(patient, authToken);
      setExplainOut(data);
      setNotice("Explanation generated.");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setBusyLabel("");
    }
  }

  async function saveFeedback() {
    if (!hasSession) {
      openAlert("Auth Token Required", "Enter an auth token before saving feedback.");
      return;
    }
    setBusyLabel("Saving feedback...");
    setLoading(true);
    setError("");
    setNotice("");
    try {
      await submitFeedback(
        {
          feedbackType,
          featureName: feedbackFeature,
          message: feedbackMessage
        },
        authToken
      );
      await loadCurrentUser();
      await loadPreferences();
      await loadAnalytics();
      setNotice("Feedback saved and profile refreshed.");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setBusyLabel("");
    }
  }

  async function loadCurrentUser() {
    const data = await getCurrentUser(authToken);
    setSessionUser(data.user_id || "");
  }

  async function loadPreferences() {
    const data = await getPreferences(authToken);
    setPrefsOut(data);
  }

  async function savePreferences() {
    if (!hasSession) {
      openAlert("Auth Token Required", "Enter an auth token before saving preferences.");
      return;
    }
    setBusyLabel("Saving preferences...");
    setLoading(true);
    setError("");
    setNotice("");
    try {
      await setPreferences({
        topK: Number(prefsOut?.top_k || 8),
        style: prefsOut?.style || "simple"
      }, authToken);
      await loadCurrentUser();
      await loadPreferences();
      setNotice("Preferences updated.");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setBusyLabel("");
    }
  }

  async function loadAnalytics() {
    if (!hasSession) {
      setAnalyticsOut(null);
      setTopFeaturesOut([]);
      return;
    }
    const [summary, top] = await Promise.all([
      getAnalyticsSummary(authToken),
      getTopFeatures({ feedbackType: "irrelevant", limit: 5 }, authToken)
    ]);
    setAnalyticsOut(summary);
    setTopFeaturesOut(top.top_features || []);
  }

  async function refreshAll() {
    if (!hasSession) {
      openAlert("Auth Token Required", "Enter an auth token before refreshing profile.");
      return;
    }
    setBusyLabel("Refreshing profile...");
    setLoading(true);
    setError("");
    setNotice("");
    try {
      await Promise.all([loadCurrentUser(), loadPreferences(), loadAnalytics()]);
      setNotice("Session, preferences, and analytics refreshed.");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setBusyLabel("");
    }
  }

  function resetPatient() {
    setPatient(INITIAL_PATIENT);
    setNotice("Patient form cleared.");
  }

  return (
    <div className="page">
      <AlertModal
        open={alertState.open}
        title={alertState.title}
        message={alertState.message}
        onClose={closeAlert}
      />

      <header className="hero">
        <h1>Interactive XAI Screening</h1>
        <p>Interactive Explainable AI for Healthcare Risk Prediction with adaptive user feedback.</p>
        <div className="hero-tags">
          <span>Live Reasoning</span>
          <span>Human Feedback Loop</span>
          <span>Screening Support</span>
        </div>
      </header>

      <section className="command-strip">
        <div className="command-group">
          <h3>Session Identity</h3>
          <label>
            Auth Token
            <input
              type="password"
              value={authToken}
              onChange={(e) => setAuthToken(e.target.value)}
              placeholder="Enter bearer token"
            />
          </label>
          <label>
            Authenticated User
            <input value={sessionUser} readOnly placeholder="No active session" />
          </label>
          <div className="actions">
            <button className="button-ghost" onClick={refreshAll} disabled={loading}>
              Refresh Profile
            </button>
          </div>
        </div>
      </section>

      <main className="layout">
        <section className="stack">
          <section className="card">
            <h2>Patient Input Matrix</h2>
            <p className="muted">Leave optional fields blank if unknown.</p>
            <div className="grid">
              {FIELD_ORDER.map((field) => {
                const meta = FIELD_META[field];
                return (
                  <label key={field}>
                    <span className="label-title">{meta.label}</span>
                    <span className="label-meta">
                      {meta.unit}
                      {meta.optional ? " | optional" : ""}
                    </span>
                    <input
                      type="number"
                      step={meta.step || "any"}
                      min={meta.min}
                      max={meta.max}
                      value={patient[field]}
                      onChange={(e) => updateField(field, e.target.value)}
                    />
                  </label>
                );
              })}
            </div>
            <div className="actions">
              <button onClick={runPredict} disabled={loading}>
                Predict Risk
              </button>
              <button onClick={runExplain} disabled={loading}>
                Explain Drivers
              </button>
              <button className="button-ghost" onClick={resetPatient} disabled={loading}>
                Reset Example
              </button>
            </div>
          </section>

          <section className="card">
            <h2>Feedback Capture</h2>
            <div className="grid-3">
              <label>
                feedback_type
                <select value={feedbackType} onChange={(e) => setFeedbackType(e.target.value)}>
                  <option value="irrelevant">{FEEDBACK_LABELS.irrelevant}</option>
                  <option value="confusing">{FEEDBACK_LABELS.confusing}</option>
                  <option value="prefer_short">{FEEDBACK_LABELS.prefer_short}</option>
                  <option value="prefer_long">{FEEDBACK_LABELS.prefer_long}</option>
                  <option value="relevant">{FEEDBACK_LABELS.relevant}</option>
                </select>
              </label>
              <label>
                feature_name
                <select value={feedbackFeature} onChange={(e) => setFeedbackFeature(e.target.value)}>
                  {FEATURE_OPTIONS.map((feature) => (
                    <option key={feature} value={feature}>
                      {feature}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                message
                <input value={feedbackMessage} onChange={(e) => setFeedbackMessage(e.target.value)} />
              </label>
            </div>
            <button onClick={saveFeedback} disabled={loading}>
              Save Feedback
            </button>
          </section>

          <section className="card">
            <h2>Preference Tuning</h2>
            <div className="grid-2">
              <label>
                top_k
                <input
                  type="number"
                  min={1}
                  max={10}
                  step={1}
                  value={prefsOut?.top_k ?? 8}
                  onChange={(e) =>
                    setPrefsOut((prev) => ({ ...(prev || {}), top_k: Number(e.target.value) }))
                  }
                />
              </label>
              <label>
                style
                <select
                  value={prefsOut?.style ?? "simple"}
                  onChange={(e) =>
                    setPrefsOut((prev) => ({ ...(prev || {}), style: e.target.value }))
                  }
                >
                  <option value="simple">simple</option>
                  <option value="detailed">detailed</option>
                </select>
              </label>
            </div>
            <button onClick={savePreferences} disabled={loading}>
              Save Preferences
            </button>
          </section>
        </section>

        <section className="stack">
          <section className={`insight-hero tone-${tone}`}>
            <div className="risk-ring" aria-hidden="true">
              <div className="risk-ring-inner">{formatRisk(currentRisk)}</div>
            </div>
            <div>
              <h2>Risk Command Center</h2>
              <p>{loading ? busyLabel || "Processing..." : notice || "Model ready for next case."}</p>
              <p className="muted">
                Screening threshold: {typeof currentThreshold === "number" ? currentThreshold.toFixed(4) : "-"} |
                Screening status: {screeningStatusLabel(currentRisk, currentThreshold)}
              </p>
              {error && <p className="error">{error}</p>}
            </div>
          </section>

          <section className="card">
            <h2>Results Snapshot</h2>
            <div className="result-grid">
              <div className="result-card">
                <h3>Prediction</h3>
                <p>
                  Risk: <strong>{predictRisk}</strong>
                </p>
                <p>
                  Screening status: <strong>{screeningStatusLabel(predictOut?.risk, predictOut?.threshold)}</strong>
                </p>
              </div>
              <div className="result-card">
                <h3>Explanation</h3>
                <p>
                  Risk: <strong>{explainRisk}</strong>
                </p>
                <p>
                  Screening status: <strong>{screeningStatusLabel(explainOut?.risk, explainOut?.threshold)}</strong>
                </p>
                <p>
                  Disputed: <strong>{explainOut?.disputed_features?.join(", ") || "-"}</strong>
                </p>
              </div>
              <div className="result-card">
                <h3>Analytics Summary</h3>
                {!hasSession ? (
                  <p className="muted">Enter a token and refresh profile.</p>
                ) : analyticsOut?.summary?.length ? (
                  <ul className="summary-list">
                    {analyticsOut.summary.map((s) => (
                      <li key={s.feedback_type}>
                        <span>{feedbackTypeLabel(s.feedback_type)}</span>
                        <strong>{s.count}</strong>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">No analytics yet.</p>
                )}
              </div>
              <div className="result-card">
                <h3>Top Irrelevant Features</h3>
                {!hasSession ? (
                  <p className="muted">Enter a token and refresh profile.</p>
                ) : topFeaturesOut.length ? (
                  <ul className="summary-list">
                    {topFeaturesOut.map((f) => (
                      <li key={f.feature}>
                        <span>{f.feature}</span>
                        <strong>{f.count}</strong>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">No feature feedback yet.</p>
                )}
              </div>
            </div>
          </section>

          <section className="card">
            <h2>Feature-Level Evidence</h2>
            <div className="result-grid">
              <FeatureTable title="Top Positive Contributors" items={explainOut?.top_positive || []} />
              <FeatureTable title="Top Negative Contributors" items={explainOut?.top_negative || []} />
              <FeatureTable title="Hidden Contributors (Disputed)" items={explainOut?.hidden_contributors || []} />
              <div className="result-card">
                <h3>Clarifications</h3>
                {explainOut?.meta?.clarifications?.length ? (
                  <ul className="clarify-list">
                    {explainOut.meta.clarifications.map((c) => (
                      <li key={c.feature}>
                        <strong>{c.feature}</strong>
                        <span>
                          {c.desc} ({c.unit})
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">No clarification items.</p>
                )}
              </div>
            </div>

            <button className="button-ghost" onClick={() => setShowRawJson((v) => !v)}>
              {showRawJson ? "Hide Raw JSON" : "Show Raw JSON"}
            </button>
            {showRawJson && (
              <div className="result-grid">
                <div className="result-card">
                  <h3>Predict JSON</h3>
                  <pre>{JSON.stringify(predictOut, null, 2)}</pre>
                </div>
                <div className="result-card">
                  <h3>Explain JSON</h3>
                  <pre>{JSON.stringify(explainOut, null, 2)}</pre>
                </div>
                <div className="result-card">
                  <h3>Preferences JSON</h3>
                  <pre>{JSON.stringify(prefsOut, null, 2)}</pre>
                </div>
                <div className="result-card">
                  <h3>Analytics JSON</h3>
                  <pre>{JSON.stringify(analyticsOut, null, 2)}</pre>
                </div>
              </div>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
