import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import app.db as db_module
import app.security as security_module
from app.main import app


def sample_patient():
    return {
        "male": 1,
        "age": 55,
        "education": 2,
        "currentSmoker": 0,
        "cigsPerDay": 0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 0,
        "totChol": 210,
        "sysBP": 140,
        "diaBP": 90,
        "BMI": 27.5,
        "heartRate": 72,
        "glucose": 95,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db = tmp_path / "test_feedback.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(test_db))
    monkeypatch.setattr(security_module, "AUTH_REQUIRED", False)
    monkeypatch.setattr(security_module, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(security_module, "RATE_LIMIT_PER_MIN", 60)
    monkeypatch.setattr(security_module, "TOKENS_BY_USER", {})
    monkeypatch.setattr(security_module, "USERS_BY_TOKEN", {})
    security_module.reset_rate_limiter()
    db_module.init_db()
    with TestClient(app) as c:
        yield c

@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    test_db = tmp_path / "test_feedback_auth.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(test_db))
    monkeypatch.setattr(security_module, "AUTH_REQUIRED", True)
    monkeypatch.setattr(security_module, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(security_module, "RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(security_module, "TOKENS_BY_USER", {"alice": "token_alice"})
    monkeypatch.setattr(security_module, "USERS_BY_TOKEN", {"token_alice": "alice"})
    security_module.reset_rate_limiter()
    db_module.init_db()
    with TestClient(app) as c:
        yield c


def test_home_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "/predict" in body["endpoints"]
    assert "/model/info" in body["endpoints"]


def test_predict_endpoint(client):
    resp = client.post("/predict?user_id=test_user", json=sample_patient())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert 0.0 <= body["threshold"] <= 1.0
    assert isinstance(body["case_id"], str)
    assert body["case_id"].startswith("case_")
    assert "model_version" in body

def test_predict_rejects_invalid_age(client):
    payload = sample_patient()
    payload["age"] = -3
    resp = client.post("/predict?user_id=test_user", json=payload)
    assert resp.status_code == 422


def test_explain_endpoint(client):
    resp = client.post("/explain?user_id=test_user", json=sample_patient())
    assert resp.status_code == 200
    body = resp.json()
    assert set(
        [
            "risk",
            "threshold",
            "flagged",
            "case_id",
            "top_positive",
            "top_negative",
            "disputed_features",
            "meta",
            "hidden_contributors",
        ]
    ).issubset(body.keys())
    assert isinstance(body["top_positive"], list)
    assert isinstance(body["top_negative"], list)
    assert isinstance(body["case_id"], str)


def test_feedback_prefer_short_updates_preferences(client):
    resp = client.post(
        "/feedback",
        json={
            "user_id": "pref_user",
            "feedback_type": "prefer_short",
            "feature_name": "sysBP",
            "case_id": "case_1",
            "message": "keep it short",
        },
    )
    assert resp.status_code == 200
    pref_resp = client.get("/preferences?user_id=pref_user")
    assert pref_resp.status_code == 200
    prefs = pref_resp.json()
    assert prefs["top_k"] == 3
    assert prefs["style"] == "simple"

def test_feedback_rejects_missing_feature_for_irrelevant(client):
    resp = client.post(
        "/feedback",
        json={
            "user_id": "pref_user",
            "feedback_type": "irrelevant",
        },
    )
    assert resp.status_code == 422

def test_feedback_rejects_missing_case_for_irrelevant(client):
    resp = client.post(
        "/feedback",
        json={
            "user_id": "pref_user",
            "feedback_type": "irrelevant",
            "feature_name": "sysBP",
        },
    )
    assert resp.status_code == 422


def test_preferences_set_and_get(client):
    post_resp = client.post(
        "/preferences",
        json={"user_id": "manual_pref_user", "top_k": 9, "style": "detailed"},
    )
    assert post_resp.status_code == 200
    get_resp = client.get("/preferences?user_id=manual_pref_user")
    assert get_resp.status_code == 200
    assert get_resp.json() == {"top_k": 9, "style": "detailed"}

def test_preferences_reject_invalid_style(client):
    post_resp = client.post(
        "/preferences",
        json={"user_id": "manual_pref_user", "top_k": 9, "style": "unknown"},
    )
    assert post_resp.status_code == 422


def test_analytics_endpoints(client):
    user_id = "analytics_user"
    events = [
        {
            "user_id": user_id,
            "feedback_type": "irrelevant",
            "feature_name": "sysBP",
            "case_id": "c1",
            "message": None,
        },
        {
            "user_id": user_id,
            "feedback_type": "irrelevant",
            "feature_name": "sysBP",
            "case_id": "c2",
            "message": None,
        },
        {
            "user_id": user_id,
            "feedback_type": "confusing",
            "feature_name": "totChol",
            "case_id": "c3",
            "message": None,
        },
    ]
    for event in events:
        resp = client.post("/feedback", json=event)
        assert resp.status_code == 200

    summary_resp = client.get(f"/analytics/summary?user_id={user_id}")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()["summary"]
    assert any(row["feedback_type"] == "irrelevant" for row in summary)
    assert any(row["feedback_type"] == "confusing" for row in summary)

    top_resp = client.get(
        f"/analytics/top_features?feedback_type=irrelevant&limit=5&user_id={user_id}"
    )
    assert top_resp.status_code == 200
    payload = top_resp.json()
    assert payload["feedback_type"] == "irrelevant"
    assert payload["top_features"][0]["feature"] == "sysBP"
    assert payload["top_features"][0]["count"] == 2

def test_top_features_reject_invalid_limit(client):
    resp = client.get("/analytics/top_features?feedback_type=irrelevant&limit=0&user_id=analytics_user")
    assert resp.status_code == 422


def test_model_info_endpoint(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_version" in body
    assert "screening_threshold" in body
    assert "features" in body
    assert isinstance(body["features"], list)


def test_explain_applies_case_scoped_disputed_features(client):
    user_id = "case_scope_user"
    case_a = "case_a"
    case_b = "case_b"

    # Register feature-level feedback for two different cases.
    r1 = client.post(
        "/feedback",
        json={
            "user_id": user_id,
            "feedback_type": "irrelevant",
            "feature_name": "sysBP",
            "case_id": case_a,
            "message": None,
        },
    )
    r2 = client.post(
        "/feedback",
        json={
            "user_id": user_id,
            "feedback_type": "irrelevant",
            "feature_name": "totChol",
            "case_id": case_b,
            "message": None,
        },
    )
    assert r1.status_code == 200
    assert r2.status_code == 200

    explain_resp = client.post(f"/explain?user_id={user_id}&case_id={case_a}", json=sample_patient())
    assert explain_resp.status_code == 200
    payload = explain_resp.json()
    assert payload["case_id"] == case_a
    assert "sysBP" in payload["disputed_features"]
    assert "totChol" not in payload["disputed_features"]
    assert payload["meta"]["adaptation_scope"] == "case"

    for item in payload["hidden_contributors"]:
        assert item["disputed"] is True


def test_auth_required_rejects_missing_header(auth_client):
    resp = auth_client.post("/predict?user_id=alice", json=sample_patient())
    assert resp.status_code == 401


def test_auth_rejects_user_id_mismatch(auth_client):
    headers = {"Authorization": "Bearer token_alice"}
    resp = auth_client.post("/predict?user_id=bob", json=sample_patient(), headers=headers)
    assert resp.status_code == 403


def test_auth_allows_valid_token(auth_client):
    headers = {"Authorization": "Bearer token_alice"}
    resp = auth_client.post("/predict?user_id=alice", json=sample_patient(), headers=headers)
    assert resp.status_code == 200


def test_rate_limit_enforced(auth_client):
    headers = {"Authorization": "Bearer token_alice"}
    r1 = auth_client.post("/predict?user_id=alice", json=sample_patient(), headers=headers)
    r2 = auth_client.post("/predict?user_id=alice", json=sample_patient(), headers=headers)
    r3 = auth_client.post("/predict?user_id=alice", json=sample_patient(), headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
