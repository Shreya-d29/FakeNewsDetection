import numpy as np
import pytest

from app.ml import predictor


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([0.05, 0.95], "REAL"),
        ([0.95, 0.05], "FAKE"),
        ([0.45, 0.55], "UNCERTAIN"),
    ],
)
def test_predict_news_selects_label_from_model_probabilities(monkeypatch, probabilities, expected):
    monkeypatch.setattr(predictor, "model_inventory", lambda: [{"name": "dense"}])
    monkeypatch.setattr(predictor, "predict_texts", lambda _model, _texts: np.array([probabilities]))
    monkeypatch.setattr(
        predictor,
        "read_json",
        lambda _path: {"class_to_index": {"FAKE": 0, "REAL": 1}},
    )

    result = predictor.predict_news("A news headline", " ".join(["article"] * 60), "dense")

    assert result["prediction"] == expected
