import json
import re
import sys
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.preprocessing.text import clean_text
from sklearn.model_selection import train_test_split
from training.settings import ROOT, SEED, TRAINING

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "results" / "metrics"


def prepare_dataset():
    sources = []
    missing_values = 0
    for file_path in sorted(RAW_DIR.glob("*.csv")):
        stem = file_path.stem.lower()
        label = "FAKE" if "fake" in stem else "REAL" if ("real" in stem or "true" in stem) else None
        if label is None:
            continue
        frame = pd.read_csv(file_path)
        headline_column = next((c for c in frame.columns if c.lower() in ("title", "headline")), None)
        article_column = next((c for c in frame.columns if c.lower() in ("text", "article", "content")), None)
        if headline_column is None or article_column is None:
            raise ValueError(f"{file_path.name} must contain title/headline and text/article columns.")
        missing_values += int(frame[[headline_column, article_column]].isna().sum().sum())
        rows = pd.DataFrame({
            "headline": frame[headline_column].fillna("").map(clean_text),
            "article": frame[article_column].fillna("").map(clean_text),
            "label": label,
        })
        dateline = lambda s: re.sub(
            r"^\s*(?:[A-Z][A-Z\s,.]{1,40}\s+)?\(?\s*Reuters\s*\)?\s*[-:]\s*",
            "", s, flags=re.IGNORECASE,
        )
        rows["headline"] = rows["headline"].map(dateline)
        rows["article"] = rows["article"].map(dateline)
        sources.append(rows)
    if not sources or {row["label"].iloc[0] for row in sources} != {"FAKE", "REAL"}:
        raise FileNotFoundError("Place both Fake.csv and True.csv/Real.csv (with title/text columns) in data/raw.")

    combined = pd.concat(sources, ignore_index=True)
    before_cleanup = len(combined)
    combined = combined[(combined.headline.str.len() > 0) & (combined.article.str.len() > 0)]
    duplicate_count = int(combined.duplicated(subset=["headline", "article"]).sum())
    conflicting = combined.groupby(["headline", "article"]).label.transform("nunique") > 1
    combined = combined[~conflicting]
    combined = combined.drop_duplicates(subset=["headline", "article"]).reset_index(drop=True)
    labels = sorted(combined.label.unique().tolist())
    class_to_index = {label: index for index, label in enumerate(labels)}
    combined["label_id"] = combined.label.map(class_to_index)
    combined["text"] = combined.headline + " " + combined.article

    indices = combined.index.to_numpy()
    split = TRAINING["split"]
    temp_size = float(split["validation"]) + float(split["test"])
    test_fraction = float(split["test"]) / temp_size
    train_ids, temp_ids = train_test_split(indices, test_size=temp_size, random_state=SEED, stratify=combined.label_id)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_fraction, random_state=SEED, stratify=combined.loc[temp_ids, "label_id"])
    combined["split"] = "train"
    combined.loc[val_ids, "split"] = "validation"
    combined.loc[test_ids, "split"] = "test"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PROCESSED_DIR / "news.csv", index=False)
    with (PROCESSED_DIR / "split_indices.json").open("w", encoding="utf-8") as f:
        json.dump({"train": train_ids.tolist(), "validation": val_ids.tolist(), "test": test_ids.tolist()}, f)
    with (PROCESSED_DIR / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"class_to_index": class_to_index, "index_to_class": {str(v): k for k, v in class_to_index.items()},
                   "sigmoid_positive_label": next(label for label, value in class_to_index.items() if value == 1),
                   "sequence_length": int(TRAINING["sequence_length"]), "minimum_article_words": 50}, f, indent=2)

    stats = {
        "total": len(combined), "removed_empty_or_duplicate": before_cleanup - len(combined),
        "real_count": int((combined.label == "REAL").sum()), "fake_count": int((combined.label == "FAKE").sum()),
        "missing_values": missing_values,
        "duplicates": duplicate_count,
        "class_distribution": {label: int(count) for label, count in combined.label.value_counts().items()},
        "article_length_words": combined.article.map(lambda text: len(text.split())).describe().to_dict(),
        "headline_length_words": combined.headline.map(lambda text: len(text.split())).describe().to_dict(),
        "split_counts": combined.split.value_counts().to_dict(),
    }
    with (METRICS_DIR / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return combined, class_to_index


if __name__ == "__main__":
    try:
        data, mapping = prepare_dataset()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Prepared {len(data)} articles; class mapping: {mapping}")
