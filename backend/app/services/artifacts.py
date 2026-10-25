from pathlib import Path

from app.core.config import METRICS_DIR, MODELS_DIR, MODEL_NAMES, read_json


def model_artifact_paths(name: str) -> tuple[Path, Path]:
    folder = MODELS_DIR / name
    return folder / "model.keras", folder / "tokenizer.json"


def model_inventory() -> list[dict]:
    models = []
    for name in MODEL_NAMES:
        model_path, tokenizer_path = model_artifact_paths(name)
        if name == "bert":
            model_path = MODELS_DIR / name / "config.json"
            tokenizer_path = MODELS_DIR / name / "tokenizer_config.json"
        metrics = read_json(METRICS_DIR / f"{name}.json") or {}
        models.append({
            "name": name,
            "available": model_path.is_file() and tokenizer_path.is_file(),
            "parameter_count": metrics.get("parameter_count"),
            "description": metrics.get("description", f"{name.upper()} text classifier"),
        })
    return models
