import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import app.db as db_module
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
    db_module.init_db()
    with TestClient(app) as c:
        yield c


def test_home_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "/predict" in body["endpoints"]


def test_predict_endpoint(client):
    resp = client.post("/predict?user_id=test_user", json=sample_patient())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert 0.0 <= body["threshold"] <= 1.0


def test_explain_endpoint(client):
    resp = client.post("/explain?user_id=test_user", json=sample_patient())
    assert resp.status_code == 200
    body = resp.json()
    assert set(
        [
            "risk",
            "threshold",
            "flagged",
            "top_positive",
            "top_negative",
            "disputed_features",
            "meta",
            "hidden_contributors",
        ]
    ).issubset(body.keys())
    assert isinstance(body["top_positive"], list)
    assert isinstance(body["top_negative"], list)


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


def test_preferences_set_and_get(client):
    post_resp = client.post(
        "/preferences",
        json={"user_id": "manual_pref_user", "top_k": 9, "style": "detailed"},
    )
    assert post_resp.status_code == 200
    get_resp = client.get("/preferences?user_id=manual_pref_user")
    assert get_resp.status_code == 200
    assert get_resp.json() == {"top_k": 9, "style": "detailed"}


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
