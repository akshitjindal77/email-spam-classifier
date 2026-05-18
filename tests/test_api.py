"""Integration tests for the FastAPI spam classifier app."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

_SPAM_TEXT = "Congratulations! You have won a free prize. Click here to claim your reward now."
_HAM_TEXT = "Hi team, please find the meeting agenda attached. See you Thursday."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Prediction correctness
# ---------------------------------------------------------------------------

class TestPredictCorrectness:
    def test_spam_email_classified_as_spam(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        assert r.json()["label"] == "spam"

    def test_ham_email_classified_as_ham(self, client):
        r = client.post("/predict", json={"text": _HAM_TEXT})
        assert r.json()["label"] == "ham"

    def test_logreg_classifies_spam(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT, "model": "logreg"})
        assert r.json()["label"] == "spam"

    def test_logreg_classifies_ham(self, client):
        r = client.post("/predict", json={"text": _HAM_TEXT, "model": "logreg"})
        assert r.json()["label"] == "ham"


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class TestResponseSchema:
    def test_status_200(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        assert r.status_code == 200

    def test_response_has_label(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        assert "label" in r.json()

    def test_response_has_confidence(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        assert "confidence" in r.json()

    def test_response_has_model(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        assert "model" in r.json()

    def test_label_is_ham_or_spam(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        assert r.json()["label"] in {"ham", "spam"}

    def test_confidence_between_0_and_1(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        confidence = r.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_model_field_echoes_request(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT, "model": "logreg"})
        assert r.json()["model"] == "logreg"

    def test_default_model_is_naive_bayes(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT})
        assert r.json()["model"] == "naive_bayes"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_text_returns_422(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_empty_text_returns_422(self, client):
        r = client.post("/predict", json={"text": ""})
        assert r.status_code == 422

    def test_invalid_model_name_returns_422(self, client):
        r = client.post("/predict", json={"text": _SPAM_TEXT, "model": "svm"})
        assert r.status_code == 422

    def test_non_string_text_returns_422(self, client):
        r = client.post("/predict", json={"text": 12345})
        assert r.status_code == 422
