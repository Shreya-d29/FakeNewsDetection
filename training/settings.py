from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "backend" / "config.yaml").open(encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

TRAINING = CONFIG["training"]
SEED = int(TRAINING["seed"])
MAX_VOCABULARY = int(TRAINING["max_vocabulary"])
SEQUENCE_LENGTH = int(TRAINING["sequence_length"])
EMBEDDING_DIMENSION = int(TRAINING["embedding_dimension"])
BATCH_SIZE = int(TRAINING["batch_size"])
EPOCHS = int(TRAINING["epochs"])
DENSE_UNITS = int(TRAINING["dense_units"])
RECURRENT_UNITS = int(TRAINING["recurrent_units"])
DROPOUT = float(TRAINING["dropout"])
BERT = TRAINING["bert"]
