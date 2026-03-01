export const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

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

function authHeaders(authToken) {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

async function apiFetch(path, options = {}, authToken = "") {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(authToken),
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

export function getCurrentUser(authToken) {
  return apiFetch("/auth/me", {}, authToken);
}

export function predictPatient(patientForm, authToken) {
  return apiFetch(
    "/predict",
    {
      method: "POST",
      body: JSON.stringify(normalizePatientPayload(patientForm))
    },
    authToken
  );
}

export function explainPatient(patientForm, authToken) {
  return apiFetch(
    "/explain",
    {
      method: "POST",
      body: JSON.stringify(normalizePatientPayload(patientForm))
    },
    authToken
  );
}

export function submitFeedback(
  {
    feedbackType,
    featureName,
    caseId = "frontend_case",
    message
  },
  authToken
) {
  return apiFetch(
    "/feedback",
    {
      method: "POST",
      body: JSON.stringify({
        feedback_type: feedbackType,
        feature_name: featureName || null,
        case_id: caseId,
        message: message || null
      })
    },
    authToken
  );
}

export function getPreferences(authToken) {
  return apiFetch("/preferences", {}, authToken);
}

export function setPreferences({ topK, style }, authToken) {
  return apiFetch(
    "/preferences",
    {
      method: "POST",
      body: JSON.stringify({
        top_k: Number(topK || 8),
        style: style || "simple"
      })
    },
    authToken
  );
}

export function getAnalyticsSummary(authToken) {
  return apiFetch("/analytics/summary", {}, authToken);
}

export function getTopFeatures({ feedbackType, limit = 10 }, authToken) {
  return apiFetch(
    `/analytics/top_features?feedback_type=${encodeURIComponent(feedbackType)}&limit=${Number(limit)}`,
    {},
    authToken
  );
}
