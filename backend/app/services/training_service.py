import json
import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import __version__ as sklearn_version

from backend.app.config import LABELS, LABEL_IDS, MODEL_DIR
from backend.app.services.dataset_service import DatasetService, dataset_service
from backend.app.services.preprocessing import bigram_analyzer, preprocess_text

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "unigram": {
        "name": "Unigram Naive Bayes",
        "feature_extraction": "CountVectorizer, unigrams (1, 1)",
        "vectorizer": "count",
        "ngram_range": (1, 1),
        "alpha": 1.0,
    },
    "bigram": {
        "name": "Bigram Naive Bayes",
        "feature_extraction": "CountVectorizer, sentence-aware adjacent bigrams",
        "vectorizer": "count",
        "alpha": 1.0,
    },
    "tfidf": {
        "name": "TF-IDF + Naive Bayes",
        "feature_extraction": "TfidfVectorizer, unigrams (1, 1)",
        "vectorizer": "tfidf",
        "ngram_range": (1, 1),
        "alpha": 1.0,
    },
}
PREPROCESSING_VERSION = "social-text-v1"
BIGRAM_ANALYZER_VERSION = "sentence-boundary-bigrams-v1"
logger = logging.getLogger("sentiscope.models")


class TrainingInProgressError(RuntimeError):
    """Raised when a second training run is requested while one is active."""


class ModelUnavailableError(RuntimeError):
    """Raised when no persisted trained model is available."""


class TrainingService:
    def __init__(self, dataset_service: DatasetService) -> None:
        self.dataset_service = dataset_service
        self._lock = Lock()
        self._training = False
        self._runtime: dict[str, dict[str, Any]] = {
            model_id: {"training": False, "error": None}
            for model_id in MODEL_CONFIGS
        }
        self._loaded_models: dict[str, Pipeline] = {}
        self._invalid_models: set[str] = set()

    def status(self) -> list[dict[str, Any]]:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        statuses: list[dict[str, Any]] = []
        with self._lock:
            runtime = {model_id: dict(state) for model_id, state in self._runtime.items()}
            invalid_models = set(self._invalid_models)
        for model_id, config in MODEL_CONFIGS.items():
            stored_metadata = self._read_metadata(model_id)
            artifact_exists = self._artifact_path(model_id).is_file()
            is_trained = bool(
                stored_metadata
                and artifact_exists
                and model_id not in invalid_models
                and self._metadata_is_compatible(model_id, stored_metadata)
            )
            metadata = stored_metadata if is_trained else None
            state = runtime[model_id]
            compatibility_error = (
                "Saved model metadata is missing or incompatible; retraining is required."
                if artifact_exists and not is_trained
                    else state["error"] if model_id in invalid_models else None
            )
            statuses.append(
                {
                    "id": model_id,
                    "name": config["name"],
                    "feature_extraction": config["feature_extraction"],
                    "algorithm": "Multinomial Naive Bayes",
                    "smoothing": f"Laplace, alpha={config['alpha']}",
                    "trained": is_trained,
                    "training": state["training"],
                    "training_status": (
                        "training"
                        if state["training"]
                        else "trained"
                        if is_trained
                        else "failed"
                        if state["error"]
                        else "not_trained"
                    ),
                    "training_error": (
                        "Training failed. Check backend logs and retry."
                        if state["error"]
                        else compatibility_error
                    ),
                    "evaluation_status": "validation" if is_trained else "not_evaluated",
                    "evaluation": metadata.get("evaluation") if metadata else None,
                    "trained_at": metadata.get("trained_at") if metadata else None,
                }
            )
        return statuses

    def train_all(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._training:
                raise TrainingInProgressError("A model training run is already in progress.")
            self._training = True
            for state in self._runtime.values():
                state.update({"training": True, "error": None})

        messages: list[str] = []
        try:
            datasets = self.dataset_service.splits()
            dataset_fingerprint = self.dataset_service.fingerprint()
            if dataset_fingerprint is None:
                raise RuntimeError("The loaded dataset has no verified content fingerprint.")
            train = datasets["train"]
            validation = datasets["validation"]
            train_texts = np.asarray(train["text"], dtype=object)
            train_labels = np.asarray(train["label"], dtype=np.int64)
            validation_texts = np.asarray(validation["text"], dtype=object)
            validation_labels = np.asarray(validation["label"], dtype=np.int64)

            for model_id, config in MODEL_CONFIGS.items():
                if not force and self._is_trained(model_id):
                    try:
                        self._get_model(model_id)
                    except ModelUnavailableError as error:
                        logger.warning(
                            "Saved %s model could not be reused; retraining: %s",
                            model_id,
                            error,
                        )
                        messages.append(
                            f"{config['name']}: saved model could not be reused; "
                            "retraining."
                        )
                    else:
                        with self._lock:
                            self._runtime[model_id].update({"training": False, "error": None})
                        messages.append(f"{config['name']}: existing trained model reused.")
                        continue
                started = time.monotonic()
                try:
                    pipeline = self._build_pipeline(model_id, config)
                    pipeline.fit(train_texts, train_labels)
                    predicted = pipeline.predict(validation_texts)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        validation_labels,
                        predicted,
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    )
                    evaluation = {
                        "split": "validation",
                        "records": int(len(validation_labels)),
                        "accuracy": float(accuracy_score(validation_labels, predicted)),
                        "precision_macro": float(precision),
                        "recall_macro": float(recall),
                        "f1_macro": float(f1),
                        "confusion_matrix": confusion_matrix(
                            validation_labels,
                            predicted,
                            labels=list(LABELS),
                        ).tolist(),
                        "labels": [LABELS[label_id] for label_id in LABELS],
                    }
                    metadata = {
                        "model_id": model_id,
                        "name": config["name"],
                        "feature_extraction": config["feature_extraction"],
                        "algorithm": "Multinomial Naive Bayes",
                        "alpha": config["alpha"],
                        "pipeline_config": self._pipeline_config(model_id, config),
                        "preprocessing_version": PREPROCESSING_VERSION,
                        "dataset_fingerprint": dataset_fingerprint,
                        "scikit_learn_version": sklearn_version,
                        "training_split": "train",
                        "training_records": int(len(train_labels)),
                        "evaluation": evaluation,
                        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "duration_seconds": round(time.monotonic() - started, 3),
                    }
                    self._persist_model(model_id, pipeline, metadata)
                    with self._lock:
                        self._loaded_models[model_id] = pipeline
                        self._invalid_models.discard(model_id)
                        self._runtime[model_id].update({"training": False, "error": None})
                    messages.append(f"{config['name']}: trained and evaluated.")
                except Exception as error:
                    logger.exception("%s model training failed.", model_id)
                    message = "Training failed. Check backend logs and retry."
                    with self._lock:
                        self._runtime[model_id].update({"training": False, "error": message})
                    messages.append(
                        f"{config['name']}: training failed. Check backend logs and retry."
                    )
            return {"models": self.status(), "messages": messages}
        finally:
            with self._lock:
                self._training = False
                for state in self._runtime.values():
                    state["training"] = False

    def predict(self, model_id: str, text: str) -> dict[str, Any]:
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model identifier: {model_id}")
        if not text.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        pipeline = self._get_model(model_id)
        probabilities = pipeline.predict_proba([text])[0]
        classifier = pipeline.named_steps["classifier"]
        class_probabilities = {
            LABELS[int(label_id)]: float(probability)
            for label_id, probability in zip(classifier.classes_, probabilities, strict=True)
        }
        for sentiment in LABEL_IDS:
            class_probabilities.setdefault(sentiment, 0.0)
        if not np.isclose(sum(class_probabilities.values()), 1.0, atol=1e-6):
            raise RuntimeError("The trained classifier returned invalid class probabilities.")
        predicted_id = int(pipeline.predict([text])[0])
        vectorizer = pipeline.named_steps["vectorizer"]
        features = vectorizer.transform([text])
        feature_indices = features.indices
        feature_values = features.data
        predicted_index = int(np.flatnonzero(classifier.classes_ == predicted_id)[0])
        joint_scores = np.asarray(
            classifier.class_log_prior_ + features @ classifier.feature_log_prob_.T
        ).ravel()
        joint_scores[predicted_index] = -np.inf
        alternative_index = int(np.argmax(joint_scores))
        contributions = feature_values * (
            classifier.feature_log_prob_[predicted_index, feature_indices]
            - classifier.feature_log_prob_[alternative_index, feature_indices]
        )
        ranked_indices = np.argsort(np.abs(contributions))[::-1]
        feature_names = vectorizer.get_feature_names_out()
        explanations = [
            {
                "feature": str(feature_names[feature_indices[index]]),
                "score": float(contributions[index]),
                "effect": "supports" if contributions[index] > 0 else "opposes",
            }
            for index in ranked_indices[:8]
            if not np.isclose(contributions[index], 0.0)
        ]
        return {
            "predicted_sentiment": LABELS[predicted_id],
            "model_id": model_id,
            "model_name": MODEL_CONFIGS[model_id]["name"],
            "model_status": "trained",
            "probabilities": class_probabilities,
            "confidence": class_probabilities[LABELS[predicted_id]],
            "explanations": explanations,
            "metadata": {
                "source": "trained_multinomial_naive_bayes",
                "explanation_available": True,
            },
        }

    def _get_model(self, model_id: str) -> Pipeline:
        with self._lock:
            cached = self._loaded_models.get(model_id)
            if cached is not None:
                return cached
        if not self._is_trained(model_id):
            raise ModelUnavailableError(
                f"Model '{model_id}' is not trained. Train the models with POST /api/models/train."
            )
        try:
            artifact = joblib.load(self._artifact_path(model_id))
        except Exception as error:
            message = (
                f"Persisted model '{model_id}' could not be loaded and needs retraining: "
                f"{type(error).__name__}: {error}"
            )
            with self._lock:
                self._invalid_models.add(model_id)
                self._runtime[model_id]["error"] = message
            raise ModelUnavailableError(message) from error
        config = MODEL_CONFIGS[model_id]
        vectorizer_type = CountVectorizer if config["vectorizer"] == "count" else TfidfVectorizer
        vectorizer = artifact.named_steps.get("vectorizer") if isinstance(artifact, Pipeline) else None
        valid_artifact = (
            isinstance(artifact, Pipeline)
            and len(artifact.steps) == 2
            and artifact.steps[0][0] == "vectorizer"
            and artifact.steps[1][0] == "classifier"
            and isinstance(vectorizer, vectorizer_type)
            and isinstance(artifact.named_steps.get("classifier"), MultinomialNB)
            and (
                vectorizer.analyzer is bigram_analyzer
                if model_id == "bigram"
                else vectorizer.ngram_range == config["ngram_range"]
            )
            and artifact.named_steps["classifier"].alpha == config["alpha"]
            and hasattr(artifact.named_steps["classifier"], "classes_")
        )
        if not valid_artifact:
            message = (
                f"Persisted model '{model_id}' has an invalid or incompatible pipeline "
                "and needs retraining."
            )
            with self._lock:
                self._invalid_models.add(model_id)
                self._runtime[model_id]["error"] = message
            raise ModelUnavailableError(message)
        with self._lock:
            self._loaded_models[model_id] = artifact
        return artifact

    def _is_trained(self, model_id: str) -> bool:
        metadata = self._read_metadata(model_id)
        with self._lock:
            is_invalid = model_id in self._invalid_models
        return bool(
            not is_invalid
            and self._artifact_path(model_id).is_file()
            and metadata
            and self._metadata_is_compatible(model_id, metadata)
        )

    def _metadata_is_compatible(
        self,
        model_id: str,
        metadata: dict[str, Any],
    ) -> bool:
        config = MODEL_CONFIGS[model_id]
        expected_config = {
            **self._pipeline_config(model_id, config),
        }
        stored_fingerprint = metadata.get("dataset_fingerprint")
        current_fingerprint = self.dataset_service.fingerprint()
        return (
            metadata.get("model_id") == model_id
            and metadata.get("pipeline_config") == expected_config
            and metadata.get("preprocessing_version") == PREPROCESSING_VERSION
            and metadata.get("scikit_learn_version") == sklearn_version
            and isinstance(stored_fingerprint, str)
            and bool(stored_fingerprint)
            and current_fingerprint is not None
            and stored_fingerprint == current_fingerprint
        )

    def _read_metadata(self, model_id: str) -> dict[str, Any] | None:
        path = self._metadata_path(model_id)
        if not path.is_file():
            return None
        try:
            with path.open(encoding="utf-8") as file:
                result = json.load(file)
            return result if isinstance(result, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _persist_model(
        self,
        model_id: str,
        pipeline: Pipeline,
        metadata: dict[str, Any],
    ) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._artifact_path(model_id), lambda path: joblib.dump(pipeline, path))
        self._atomic_write(
            self._metadata_path(model_id),
            lambda path: path.write_text(json.dumps(metadata, indent=2), encoding="utf-8"),
        )

    @staticmethod
    def _atomic_write(destination: Path, writer: Any) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            writer(temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _artifact_path(model_id: str) -> Path:
        return MODEL_DIR / f"{model_id}.joblib"

    @staticmethod
    def _metadata_path(model_id: str) -> Path:
        return MODEL_DIR / f"{model_id}.json"

    @staticmethod
    def _build_pipeline(model_id: str, config: dict[str, Any]) -> Pipeline:
        vectorizer_type = CountVectorizer if config["vectorizer"] == "count" else TfidfVectorizer
        if model_id == "bigram":
            vectorizer = vectorizer_type(analyzer=bigram_analyzer)
        else:
            vectorizer = vectorizer_type(
                preprocessor=preprocess_text,
                lowercase=False,
                ngram_range=config["ngram_range"],
                token_pattern=r"(?u)\b\w+\b",
            )
        return Pipeline(
            [
                ("vectorizer", vectorizer),
                ("classifier", MultinomialNB(alpha=config["alpha"])),
            ]
        )

    @staticmethod
    def _pipeline_config(model_id: str, config: dict[str, Any]) -> dict[str, Any]:
        pipeline_config: dict[str, Any] = {
            "vectorizer": config["vectorizer"],
            "alpha": config["alpha"],
        }
        if model_id == "bigram":
            pipeline_config["analyzer"] = BIGRAM_ANALYZER_VERSION
        else:
            pipeline_config["ngram_range"] = list(config["ngram_range"])
        return pipeline_config


training_service = TrainingService(dataset_service=dataset_service)
