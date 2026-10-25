from fastapi import APIRouter, HTTPException

from app.core.config import DATASET_STATS_PATH, METRICS_DIR, MAPPING_PATH, MINIMUM_ARTICLE_WORDS, MODEL_NAMES, read_json
from app.ml.predictor import PredictionError, predict_news, predict_texts, validate_text_inputs
from app.core.config import UNCERTAINTY_MARGIN
from app.preprocessing.text import combine_news
from app.schemas import PredictRequest
from app.services.artifacts import model_inventory

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    inventory = model_inventory()
    return {"status": "ok", "available_models": sum(item["available"] for item in inventory), "models": inventory}


@router.get("/models")
def models():
    return {"models": model_inventory(), "minimum_article_words": MINIMUM_ARTICLE_WORDS}


@router.get("/metrics")
def metrics():
    all_metrics = {name: read_json(METRICS_DIR / f"{name}.json") for name in MODEL_NAMES}
    comparison = read_json(METRICS_DIR / "comparison.json")
    experiments = read_json(METRICS_DIR / "experiment_curves.json")
    return {"models": {key: value for key, value in all_metrics.items() if value is not None}, "comparison": comparison, "experiments": experiments}


@router.post("/predict")
def predict(request: PredictRequest):
    try:
        return predict_news(request.headline, request.article, request.model)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc), "code": exc.code}) from exc


@router.post("/compare")
def compare(request: PredictRequest):
    try:
        validate_text_inputs(request.headline, request.article)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc), "code": exc.code}) from exc
    outcomes = []
    for item in model_inventory():
        if item["available"]:
            try:
                outcomes.append(predict_news(request.headline, request.article, item["name"]))
            except PredictionError as exc:
                outcomes.append({"model": item["name"], "error": str(exc), "code": exc.code})
        else:
            outcomes.append({"model": item["name"], "error": "Model not trained yet.", "code": "MODEL_UNAVAILABLE"})
    if not outcomes:
        raise HTTPException(status_code=503, detail={"error": "No trained models are available.", "code": "NO_MODELS_AVAILABLE"})
    return {"results": outcomes}


@router.get("/model/{name}/metrics")
def model_metrics(name: str):
    if name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail={"error": "Choose a supported model.", "code": "INVALID_MODEL"})
    result = read_json(METRICS_DIR / f"{name}.json")
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "No saved evaluation results for this model.", "code": "METRICS_UNAVAILABLE"})
    return result


@router.get("/model/{name}/confusion-matrix")
def confusion_matrix(name: str):
    if name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail={"error": "Choose a supported model.", "code": "INVALID_MODEL"})
    result = read_json(METRICS_DIR / f"{name}.json")
    if result is None or "confusion_matrix" not in result:
        raise HTTPException(status_code=404, detail={"error": "No saved confusion matrix for this model.", "code": "METRICS_UNAVAILABLE"})
    return {"model": name, "confusion_matrix": result["confusion_matrix"]}


@router.get("/dataset/statistics")
def dataset_statistics():
    result = read_json(DATASET_STATS_PATH)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "Dataset statistics are not available until the dataset is prepared.", "code": "DATASET_UNAVAILABLE"})
    return result


@router.post("/explain")
def explain(request: PredictRequest):
    if request.model not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail={"error": "Choose a supported model.", "code": "INVALID_MODEL"})
    try:
        validate_text_inputs(request.headline, request.article)
        from lime.lime_text import LimeTextExplainer
        text = combine_news(request.headline, request.article)
        label_mapping = read_json(MAPPING_PATH)
        if not label_mapping:
            raise PredictionError("Label mapping is unavailable.", "MAPPING_UNAVAILABLE", 503)
        index_to_class = {int(index): label for label, index in label_mapping["class_to_index"].items()}
        class_names = [index_to_class[index] for index in range(len(index_to_class))]
        explainer = LimeTextExplainer(class_names=class_names, random_state=42)
        explanation = explainer.explain_instance(
            text, lambda samples: predict_texts(request.model, samples),
            labels=(0, 1), num_features=12, num_samples=300,
        )
        class_index = int(predict_texts(request.model, [text])[0].argmax())
        return {
            "model": request.model,
            "prediction": index_to_class[class_index],
            "words": [{"word": word, "weight": float(weight)} for word, weight in explanation.as_list(label=class_index)],
            "note": "Local surrogate explanation; influential words are approximate and do not establish factual accuracy.",
        }
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc), "code": exc.code}) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail={"error": "The LIME explanation dependency is not installed.", "code": "EXPLAINER_UNAVAILABLE"}) from exc
