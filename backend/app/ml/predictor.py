import json
from functools import lru_cache

import numpy as np

from app.core.config import MAPPING_PATH, MINIMUM_ARTICLE_WORDS, MODELS_DIR, UNCERTAINTY_MARGIN, read_json
from app.preprocessing.text import combine_news
from app.services.artifacts import model_artifact_paths, model_inventory


class PredictionError(Exception):
    def __init__(self, message: str, code: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def validate_text_inputs(headline: str, article: str):
    if not headline.strip():
        raise PredictionError("Please provide a headline.", "EMPTY_HEADLINE", 422)
    if len(article.split()) < MINIMUM_ARTICLE_WORDS:
        raise PredictionError("Insufficient article text. Please provide a complete news article.", "ARTICLE_TOO_SHORT", 422)


def predict_texts(model_name: str, texts: list[str]) -> np.ndarray:
    model, tokenizer = load_artifacts(model_name)
    mapping = read_json(MAPPING_PATH)
    if not mapping or not isinstance(mapping.get("class_to_index"), dict):
        raise PredictionError("Label mapping is unavailable. Prepare the dataset and train a model first.", "MAPPING_UNAVAILABLE", 503)
    index_to_class = {int(index): label.upper() for label, index in mapping["class_to_index"].items()}
    if model_name == "bert":
        import torch
        encoded = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            output = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
    else:
        import tensorflow as tf
        sequences = tokenizer.texts_to_sequences(texts)
        padded = tf.keras.utils.pad_sequences(
            sequences, maxlen=int(mapping.get("sequence_length", 500)), padding="post", truncating="post"
        )
        positive_label = mapping.get("sigmoid_positive_label", index_to_class.get(1))
        positive = np.asarray(model.predict(padded, verbose=0)).reshape(-1)
        output = np.zeros((len(texts), 2), dtype=float)
        output[:, mapping["class_to_index"][positive_label]] = positive
        output[:, mapping["class_to_index"]["FAKE" if positive_label == "REAL" else "REAL"]] = 1 - positive
    if output.ndim != 2 or output.shape != (len(texts), 2) or not np.isfinite(output).all():
        raise PredictionError("The model returned invalid probabilities.", "INVALID_MODEL_OUTPUT", 503)
    return output


@lru_cache(maxsize=5)
def load_artifacts(name: str):
    if name == "bert":
        folder = MODELS_DIR / name
        if not (folder / "config.json").is_file() or not (folder / "tokenizer_config.json").is_file():
            raise PredictionError("This model has not been trained or its artifacts are missing.", "MODEL_UNAVAILABLE", 503)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(folder)
            model = AutoModelForSequenceClassification.from_pretrained(folder)
        except (OSError, ValueError, ImportError) as exc:
            raise PredictionError("The saved model could not be loaded.", "MODEL_LOAD_FAILED", 503) from exc
        return model, tokenizer
    model_path, tokenizer_path = model_artifact_paths(name)
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise PredictionError("This model has not been trained or its artifacts are missing.", "MODEL_UNAVAILABLE", 503)
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        with tokenizer_path.open(encoding="utf-8") as file:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(file.read())
    except (OSError, ValueError, ImportError) as exc:
        raise PredictionError("The saved model could not be loaded.", "MODEL_LOAD_FAILED", 503) from exc
    return model, tokenizer


def predict_news(headline: str, article: str, model_name: str) -> dict:
    if model_name not in {item["name"] for item in model_inventory()}:
        raise PredictionError("Choose a supported model.", "INVALID_MODEL", 400)
    validate_text_inputs(headline, article)
    text = combine_news(headline, article)
    output = predict_texts(model_name, [text])[0]
    mapping = read_json(MAPPING_PATH)
    if not mapping or not isinstance(mapping.get("class_to_index"), dict):
        raise PredictionError("Label mapping is unavailable. Prepare the dataset and train a model first.", "MAPPING_UNAVAILABLE", 503)
    class_to_index = mapping["class_to_index"]
    real_probability = float(np.clip(output[class_to_index["REAL"]], 0, 1))
    fake_probability = float(np.clip(output[class_to_index["FAKE"]], 0, 1))
    prediction = "UNCERTAIN" if abs(real_probability - fake_probability) < UNCERTAINTY_MARGIN else (
        "REAL" if real_probability > fake_probability else "FAKE"
    )
    confidence = max(real_probability, fake_probability)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "model": model_name,
        "explanation_note": "Model-based classification from learned text patterns; not independent fact verification.",
    }
