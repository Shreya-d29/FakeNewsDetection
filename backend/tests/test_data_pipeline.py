import json

import pandas as pd

from training import data_pipeline


def test_prepare_dataset_builds_mapping_stats_and_stratified_splits(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    metrics = tmp_path / "metrics"
    raw.mkdir()
    body = " ".join(["article"] * 60)
    pd.DataFrame({"title": [f"Fake headline {i}" for i in range(30)], "text": [body] * 30}).to_csv(raw / "Fake.csv", index=False)
    pd.DataFrame({"title": [f"Real headline {i}" for i in range(30)], "text": [f"(Reuters) - {body}"] * 30}).to_csv(raw / "True.csv", index=False)
    monkeypatch.setattr(data_pipeline, "RAW_DIR", raw)
    monkeypatch.setattr(data_pipeline, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data_pipeline, "METRICS_DIR", metrics)

    frame, mapping = data_pipeline.prepare_dataset()

    assert mapping == {"FAKE": 0, "REAL": 1}
    assert set(frame.split) == {"train", "validation", "test"}
    assert (frame[frame.label == "REAL"].article.str.contains("Reuters", case=False)).sum() == 0
    assert len(json.loads((processed / "split_indices.json").read_text())) == 3
    stats = json.loads((metrics / "dataset_stats.json").read_text())
    assert stats["total"] == 60
    assert stats["real_count"] == stats["fake_count"] == 30
