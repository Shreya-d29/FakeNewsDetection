"""Fine-tune DistilBERT on the fixed train/validation split; evaluate once on test."""
import json
import argparse
import inspect
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

os.environ["USE_TF"] = "0"

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from data_pipeline import METRICS_DIR, PROCESSED_DIR, ROOT, prepare_dataset
from settings import BERT, SEED

BASE_MODEL = BERT["base_model"]


class NewsDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.inputs = tokenizer(
            (frame.headline + " " + frame.article).tolist(), truncation=True,
            max_length=max_length,
        )
        self.labels = frame.label_id.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: torch.tensor(values[index]) for key, values in self.inputs.items()},
                "labels": torch.tensor(self.labels[index])}


def main(fast=False):
    if not (PROCESSED_DIR / "news.csv").is_file():
        prepare_dataset()
    frame = pd.read_csv(PROCESSED_DIR / "news.csv")
    train = frame[frame.split == "train"]
    validation = frame[frame.split == "validation"]
    test = frame[frame.split == "test"]
    max_length = 96 if fast else int(BERT["max_sequence_length"])
    epochs = 1 if fast else int(BERT["epochs"])
    train_batch_size = 16 if fast else int(BERT["batch_size"])
    eval_batch_size = 32 if fast else 16
    if fast:
        train, _ = train_test_split(
            train, train_size=min(1024, len(train)), random_state=SEED,
            stratify=train.label_id,
        )
        validation, _ = train_test_split(
            validation, train_size=min(256, len(validation)), random_state=SEED,
            stratify=validation.label_id,
        )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    output_dir = ROOT / "models" / "bert"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_strategy_parameter = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    args = TrainingArguments(
        output_dir=str(ROOT / "results" / "bert_checkpoints"), learning_rate=float(BERT["learning_rate"]),
        per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size, num_train_epochs=epochs,
        weight_decay=0.01, save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", save_total_limit=1,
        logging_strategy="epoch", report_to="none", seed=SEED,
        **{eval_strategy_parameter: "epoch"},
    )
    def compute_metrics(eval_prediction):
        predicted_ids = np.asarray(eval_prediction.predictions).argmax(axis=-1)
        return {"accuracy": accuracy_score(eval_prediction.label_ids, predicted_ids)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=NewsDataset(train, tokenizer, max_length),
        eval_dataset=NewsDataset(validation, tokenizer, max_length),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    started = time.perf_counter()
    trainer.train()
    duration = time.perf_counter() - started
    test_dataset = NewsDataset(test, tokenizer, max_length)
    prediction = trainer.predict(test_dataset)
    logits = prediction.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = probabilities.argmax(axis=1)
    actual = test.label_id.to_numpy(dtype=int)
    if len(np.unique(actual)) < 2 or len(np.unique(predictions)) < 2 or min(np.bincount(predictions, minlength=2)) < max(10, math.ceil(len(predictions) * 0.05)):
        raise RuntimeError("BERT appears degenerate on the untouched test set; inspect data, labels, and preprocessing before continuing.")
    with (PROCESSED_DIR / "label_mapping.json").open(encoding="utf-8") as map_file:
        mapping = json.load(map_file)
    samples = test[["headline", "article", "label"]].copy()
    samples["predicted"] = [mapping["index_to_class"][str(int(index))] for index in predictions]
    samples["fake_probability"] = probabilities[:, mapping["class_to_index"]["FAKE"]]
    samples["real_probability"] = probabilities[:, mapping["class_to_index"]["REAL"]]
    validation_samples = pd.concat([samples[samples.label == label].head(10) for label in ("FAKE", "REAL")])
    if validation_samples.label.value_counts().min() < 10:
        raise RuntimeError("Test split needs at least 10 REAL and 10 FAKE samples for mandatory model validation.")
    print("Actual | Predicted | FAKE prob | REAL prob")
    for row in validation_samples.itertuples():
        print(f"{row.label} | {row.predicted} | {row.fake_probability:.4f} | {row.real_probability:.4f}")
    history = trainer.state.log_history
    train_loss = [item["loss"] for item in history if "loss" in item and "eval_loss" not in item]
    val_loss = [item["eval_loss"] for item in history if "eval_loss" in item]
    val_acc = [item.get("eval_accuracy", 0) for item in history if "eval_loss" in item]
    tokenizer.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    metrics = {
        "model": "bert", "description": "Fine-tuned DistilBERT sequence classifier",
        "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad),
        "training_seconds": duration, "training_date": datetime.now(timezone.utc).isoformat(),
        "history": {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
        "test_accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities[:, 1])),
        "confusion_matrix": confusion_matrix(actual, predictions).tolist(),
        "hyperparameters": {
            "base_model": BASE_MODEL,
            "learning_rate": float(BERT["learning_rate"]),
            "epochs": epochs,
            "max_length": max_length,
            "batch_size": train_batch_size,
            "fast_mode": fast,
            "training_samples": len(train),
            "validation_samples": len(validation),
            "test_samples": len(test),
            "seed": SEED,
        },
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "bert.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (METRICS_DIR / "bert_test_samples.csv").open("w", encoding="utf-8") as file:
        samples.to_csv(file, index=False)
    from train_text_model import refresh_comparison
    refresh_comparison()
    print(json.dumps({key: metrics[key] for key in ("test_accuracy", "precision", "recall", "f1", "roc_auc")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for news classification.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fine-tune on stratified subsets for one epoch, then evaluate on the full test split.",
    )
    main(fast=parser.parse_args().fast)
