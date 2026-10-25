from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_models_report_current_inventory_without_mocking():
    health = client.get("/api/health")
    models = client.get("/api/models")
    assert health.status_code == 200
    assert models.status_code == 200
    assert health.json()["available_models"] == sum(
        model["available"] for model in models.json()["models"]
    )


def test_short_article_is_rejected_before_model_loading():
    response = client.post("/api/predict", json={"headline": "A headline", "article": "Narendra modi did yesterday", "model": "lstm"})
    assert response.status_code == 422
    assert response.json()["code"] == "ARTICLE_TOO_SHORT"


def test_invalid_model_is_rejected():
    response = client.post("/api/predict", json={"headline": "A headline", "article": " ".join(["word"] * 60), "model": "unknown"})
    assert response.status_code == 400


def test_valid_length_input_uses_available_model_or_reports_missing_artifacts():
    response = client.post("/api/predict", json={"headline": "A headline", "article": " ".join(["news"] * 60), "model": "lstm"})
    if response.status_code == 200:
        assert response.json()["model"] == "lstm"
        assert response.json()["prediction"] in {"REAL", "FAKE", "UNCERTAIN"}
        assert 0 <= response.json()["real_probability"] <= 1
        assert 0 <= response.json()["fake_probability"] <= 1
    else:
        assert response.status_code == 503
        assert response.json()["code"] == "MODEL_UNAVAILABLE"


def test_compare_rejects_short_article_instead_of_returning_unavailable_models():
    response = client.post("/api/compare", json={"headline": "A headline", "article": "too short", "model": "lstm"})
    assert response.status_code == 422
    assert response.json()["code"] == "ARTICLE_TOO_SHORT"


def test_dataset_statistics_reflect_prepared_dataset_when_available():
    response = client.get("/api/dataset/statistics")
    if response.status_code == 200:
        assert response.json()["total"] > 0
        assert response.json()["real_count"] + response.json()["fake_count"] == response.json()["total"]
    else:
        assert response.status_code == 404
        assert response.json()["code"] == "DATASET_UNAVAILABLE"
