import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
with (ROOT / "backend" / "config.yaml").open(encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "results" / "metrics"
MAPPING_PATH = ROOT / "data" / "processed" / "label_mapping.json"
DATASET_STATS_PATH = METRICS_DIR / "dataset_stats.json"
MINIMUM_ARTICLE_WORDS = int(os.getenv("TRUTHLENS_MINIMUM_ARTICLE_WORDS", str(CONFIG["prediction"]["minimum_article_words"])))
UNCERTAINTY_MARGIN = float(os.getenv("TRUTHLENS_UNCERTAINTY_MARGIN", str(CONFIG["prediction"]["uncertainty_margin"])))
ALLOWED_ORIGINS = CONFIG["api"]["allowed_origins"]
MODEL_NAMES = ("dense", "rnn", "lstm", "gru", "bert")


def read_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)
