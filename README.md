# TruthLens AI

TruthLens AI is a training-ready deep-learning course project for binary fake/real **news text classification**. It is a probabilistic text classifier, not a fact-checker. It does not use news APIs, browse the web, validate sources, or store submitted articles.

> **Important:** This checkout does not contain a dataset or trained model artifacts. The API and interface run without them and explicitly show unavailable states. No predictions, evaluation metrics, or chart values are fabricated. Train models locally before expecting inference and performance metrics.

## Problem statement and objectives

The project explores whether language-modeling approaches can distinguish articles labeled REAL or FAKE in a public training corpus. Objectives are to prepare a reproducible stratified dataset, compare neural architectures on an untouched test split, expose saved predictions through a typed API, and communicate uncertainty and model limitations clearly.

## Features

- Dense, SimpleRNN, LSTM, GRU, and fine-tuned DistilBERT training workflows.
- Shared text cleanup and model-input construction (headline + article).
- Label mapping and dataset statistics produced by the data preparation script.
- Stratified 70/15/15 train/validation/test split; training-only vocabulary fit.
- Saved evaluation metrics, confusion matrices, epoch histories, and held-out test prediction samples.
- FastAPI inference, model inventory, evaluation, dataset statistics, and comparison endpoints.
- Responsive React/TypeScript application with Analyze, Dashboard, Models, Performance, Dataset, and About views.
- Missing models and results are reported as unavailable. The UI has no mock results.

## Architecture

```text
ISOT Fake.csv / Real.csv
        │
        ▼
training/data_pipeline.py ──► data/processed/news.csv + label_mapping.json
        │                                  │
        ▼                                  ▼
training/train_*.py ──► models/{name}/ + results/metrics/*.json
                                      │
                                      ▼
React + TypeScript ──HTTP──► FastAPI ──► cached model + shared preprocessing
```

## Dataset

This release supports the **ISOT Fake News Dataset** CSV format. Obtain it from its official academic distribution and place its two files in `data/raw/`:

```text
data/raw/Fake.csv
data/raw/True.csv
```

Each CSV must contain a headline column named `title` or `headline`, and an article column named `text`, `article`, or `content`. Filenames identify the class (`Fake.csv` and `True.csv`/`Real.csv`), and the preparation step validates that both classes exist before writing the single shared `data/processed/label_mapping.json`. Dataset files and trained artifacts are excluded from version control.

Cleaning removes HTML, replaces URLs with a token, normalizes whitespace, removes a leading Reuters dateline, empty examples, and duplicate headline/article pairs. Source/date/subject fields are not used as inputs. It preserves normal casing and punctuation. The prepared table has one combined text input (`headline + article`) and a deterministic stratified 70/15/15 split. The tokenizer is fitted only to training examples.

Statistics are saved to `results/metrics/dataset_stats.json`. Index assignments and class-to-id semantics are saved to `data/processed/label_mapping.json`; model output probabilities are interpreted using that mapping.

## Models and training

Install backend dependencies in a supported Python environment (Python 3.11 recommended). The Keras models use a 30,000-word vocabulary, 500-token sequences, an embedding, and the requested Dense/SimpleRNN/LSTM/GRU architecture. Training uses fixed seeds, binary cross-entropy, Adam, and validation-based early stopping/checkpointing for the recurrent baseline. BERT is a separately fine-tuned DistilBERT classifier and requires PyTorch/Transformers; a GPU is strongly recommended.

From the project root, create the environment and prepare the dataset:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
python training\data_pipeline.py
python training\train_dense.py
python training\train_rnn.py
python training\train_lstm.py
python training\train_gru.py
```

Fine-tune BERT (requires downloading the pretrained DistilBERT checkpoint):

```powershell
python -m pip install -r backend\requirements-bert.txt
python training\train_bert.py
```

Run the deliberately overfit / underfit demonstrations separately; these use only the train and validation partitions and save curves without overwriting model artifacts:

```powershell
python training\train_experiments.py
```

Each model training run evaluates exactly once on the held-out test partition, writes its metrics and confusion matrix, and prints at least 10 REAL and 10 FAKE test samples with predicted label and both probabilities. It stops before writing successful metrics if predictions collapse to one class or have less than 5% representation for a class. After evaluating a model, compare results in `results/metrics/comparison.json`.

### Colab workflow

1. Select a GPU runtime for BERT (Keras recurrent models can also benefit from GPU).
2. Install dependencies from the corresponding requirements file.
3. Upload the dataset CSV files into `data/raw/`.
4. Inspect the input columns, run `python training/data_pipeline.py`, and inspect the generated statistics and label mapping.
5. Train models one at a time; do not use test metrics to tune hyperparameters.
6. Check every printed class-balanced test sample validation table.
7. Download `models/`, `results/metrics/`, and `data/processed/label_mapping.json` into this project's corresponding paths.

## Backend

With the backend environment active, start the API:

```powershell
cd backend
uvicorn app.main:app --reload
```

Interactive API docs are at `http://127.0.0.1:8000/docs`.

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API and model availability |
| GET | `/api/models` | Model readiness and saved parameter counts |
| GET | `/api/metrics` | Existing model results and comparison |
| POST | `/api/predict` | Classify a headline and article with a selected model |
| POST | `/api/compare` | Run all available models on the same article |
| GET | `/api/model/{name}/metrics` | Saved per-model metrics and training history |
| GET | `/api/model/{name}/confusion-matrix` | Saved confusion matrix |
| GET | `/api/dataset/statistics` | Prepared dataset statistics |
| POST | `/api/explain` | LIME word-attribution approximation for a trained model |

Prediction input is `{ "headline": "...", "article": "...", "model": "lstm" }`. The minimum article length defaults to 50 words and can be changed with `TRUTHLENS_MINIMUM_ARTICLE_WORDS`. The uncertainty margin defaults to 0.2 and can be changed with `TRUTHLENS_UNCERTAINTY_MARGIN`. A result of `UNCERTAIN` is used when the REAL/FAKE probability difference is below the margin. No user article is persisted.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server runs at `http://127.0.0.1:5173`. Set `VITE_API_BASE` to override the API base URL. The frontend only renders backend results; missing data appears as an unavailable state.

## Evaluation

Evaluation JSON files are generated only by training. The Performance view derives comparison charts, training curves, and confusion matrices from saved metrics. Dataset charts are generated from saved dataset statistics. An absent file is shown as unavailable; no sample metric values or random predictions are used.

## Tests

```powershell
cd backend
python -m pytest
```

Tests cover preprocessing, API health and model availability, short/invalid input handling, and absent dataset statistics. Model inference tests require actual trained artifacts and should be run after training.

## Project structure

```text
backend/app/             FastAPI routes, schemas, preprocessing, model service
backend/tests/           pytest API and preprocessing tests
data/raw/                User-provided dataset (ignored by git)
data/processed/          Prepared CSV, split assignments, label mapping
frontend/src/             React app, API service, types, responsive styles
notebooks/                Seven Colab/Jupyter workflow notebooks (EDA through BERT)
models/{dense,rnn,lstm,gru,bert}/  Locally trained artifacts (ignored by git)
results/metrics/          Dataset and model evaluation JSON/CSV (ignored by git)
training/                 Dataset preparation and offline training scripts
```

## Syllabus mapping (indicative)

| Unit | Project connection |
|---|---|
| I — Python / data handling | Dataset inspection, cleaning, Pandas and NumPy |
| II — NLP preprocessing | Text normalization, vocabulary, token sequences |
| III — Neural networks | Dense network and embedding baseline |
| IV — Recurrent networks | SimpleRNN, LSTM, and GRU |
| V — Deep learning / Transformers | Fine-tuned DistilBERT |
| VI — Evaluation and deployment | Stratified evaluation, metrics, FastAPI, React UI |

## Limitations and future scope

Dataset biases can become shortcuts; removing Reuters datelines and metadata reduces but cannot eliminate leakage. Held-out test performance is not evidence of real-world truth or generalization. Model confidence is not factual certainty. The browser does not perform fact verification. LIME explanations are local surrogate approximations, not evidence that a claim is true or false. This initial release does not include pre-trained weights. Future work can add stronger dataset-shift testing, model calibration, and additional corpora.
