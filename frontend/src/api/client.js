export const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
export const API_TOKEN = import.meta.env.VITE_API_TOKEN || "";

const PATIENT_FIELDS = [
  "male",
  "age",
  "education",
  "currentSmoker",
  "cigsPerDay",
  "BPMeds",
  "prevalentStroke",
  "prevalentHyp",
  "diabetes",
  "totChol",
  "sysBP",
  "diaBP",
  "BMI",
  "heartRate",
  "glucose"
];

function normalizePatientPayload(form) {
  const payload = {};
  for (const key of PATIENT_FIELDS) {
    const raw = form[key];
    payload[key] = raw === "" ? null : Number(raw);
  }
  return payload;
}

async function apiFetch(path, options = {}) {
  const authHeaders = API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {};
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...authHeaders,
      ...(options.headers || {})
    },
    ...options
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export function predictPatient(patientForm, userId, caseId = null) {
  const caseParam = caseId ? `&case_id=${encodeURIComponent(caseId)}` : "";
  return apiFetch(`/predict?user_id=${encodeURIComponent(userId)}${caseParam}`, {
    method: "POST",
    body: JSON.stringify(normalizePatientPayload(patientForm))
  });
}

export function explainPatient(patientForm, userId, caseId = null) {
  const caseParam = caseId ? `&case_id=${encodeURIComponent(caseId)}` : "";
  return apiFetch(`/explain?user_id=${encodeURIComponent(userId)}${caseParam}`, {
    method: "POST",
    body: JSON.stringify(normalizePatientPayload(patientForm))
  });
}

export function submitFeedback({
  userId,
  feedbackType,
  featureName,
  caseId = null,
  message
}) {
  return apiFetch("/feedback", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      feedback_type: feedbackType,
      feature_name: featureName || null,
      case_id: caseId,
      message: message || null
    })
  });
}

export function getPreferences(userId) {
  return apiFetch(`/preferences?user_id=${encodeURIComponent(userId)}`);
}

export function setPreferences({ userId, topK, style }) {
  return apiFetch("/preferences", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      top_k: Number(topK || 8),
      style: style || "simple"
    })
  });
}

export function getAnalyticsSummary(userId) {
  return apiFetch(`/analytics/summary?user_id=${encodeURIComponent(userId)}`);
}

export function getTopFeatures({ feedbackType, limit = 10, userId }) {
  return apiFetch(
    `/analytics/top_features?feedback_type=${encodeURIComponent(feedbackType)}&limit=${Number(
      limit
    )}&user_id=${encodeURIComponent(userId)}`
  );
}
