import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from backend.app.config import DATASET_CONFIG, DATASET_ID, LABELS, MODEL_DIR
from backend.app.services.dataset_service import DatasetService, dataset_service
from backend.app.services.training_service import (
    MODEL_CONFIGS,
    ModelUnavailableError,
    TrainingService,
    training_service,
)

EVALUATION_VERSION = "held-out-sentiment-v1"


class EvaluationUnavailableError(RuntimeError):
    """Raised when the frozen models or official splits cannot be evaluated."""


class ModelEvaluationService:
    def __init__(
        self,
        trainer: TrainingService,
        datasets: DatasetService,
    ) -> None:
        self.trainer = trainer
        self.datasets = datasets
        self._lock = Lock()

    def get_evaluation(self) -> dict[str, Any]:
        with self._lock:
            splits = self.datasets.splits()
            dataset_fingerprint = self.datasets.fingerprint()
            if dataset_fingerprint is None:
                raise EvaluationUnavailableError(
                    "The loaded dataset has no verified content fingerprint."
                )

            model_context: list[dict[str, Any]] = []
            pipelines = {}
            for model_id in MODEL_CONFIGS:
                metadata = self.trainer._read_metadata(model_id)
                if (
                    metadata is None
                    or metadata.get("training_split") != "train"
                    or metadata.get("dataset_fingerprint") != dataset_fingerprint
                    or not self.trainer._is_trained(model_id)
                ):
                    raise ModelUnavailableError(
                        f"Model '{model_id}' is unavailable or not trained on the "
                        "current training split."
                    )
                try:
                    pipeline = self.trainer._get_model(model_id)
                except ModelUnavailableError:
                    raise
                artifact_path = self.trainer._artifact_path(model_id)
                artifact_stat = artifact_path.stat()
                pipelines[model_id] = pipeline
                model_context.append(
                    {
                        "id": model_id,
                        "trained_at": metadata.get("trained_at"),
                        "training_records": metadata.get("training_records"),
                        "pipeline_config": metadata.get("pipeline_config"),
                        "preprocessing_version": metadata.get("preprocessing_version"),
                        "scikit_learn_version": metadata.get("scikit_learn_version"),
                        "artifact_size": artifact_stat.st_size,
                        "artifact_mtime_ns": artifact_stat.st_mtime_ns,
                    }
                )

            splits_count = {
                split: len(splits[split]["label"])
                for split in ("train", "validation", "test")
            }
            signature_source = json.dumps(
                {
                    "evaluation_version": EVALUATION_VERSION,
                    "dataset_fingerprint": dataset_fingerprint,
                    "split_counts": splits_count,
                    "models": model_context,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            evaluation_id = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
            evaluation_path = MODEL_DIR / "evaluations" / f"{evaluation_id}.json"
            if evaluation_path.is_file():
                return self._read_evaluation(evaluation_path, evaluation_id)
            cached = self._find_compatible_evaluation(
                evaluation_path.parent,
                dataset_split_records=splits_count,
                model_context=model_context,
            )
            if cached is not None:
                return cached

            result = self._evaluate(
                splits=splits,
                splits_count=splits_count,
                pipelines=pipelines,
                model_context=model_context,
                evaluation_id=evaluation_id,
            )
            evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_once(evaluation_path, result)
            return self._read_evaluation(evaluation_path, evaluation_id)

    @staticmethod
    def _find_compatible_evaluation(
        directory: Path,
        *,
        dataset_split_records: dict[str, int],
        model_context: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not directory.is_dir():
            return None
        expected_models = {
            model["id"]: model
            for model in model_context
        }
        for path in sorted(directory.glob("*.json")):
            try:
                with path.open(encoding="utf-8") as file:
                    candidate = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                not isinstance(candidate, dict)
                or candidate.get("evaluation_version") != EVALUATION_VERSION
                or candidate.get("status") != "complete"
                or candidate.get("dataset") != DATASET_ID
                or candidate.get("configuration") != DATASET_CONFIG
                or candidate.get("dataset_split_records") != dataset_split_records
                or candidate.get("test_split_usage") != "final_evaluation_only"
                or candidate.get("evaluation_id") != path.stem
                or not isinstance(candidate.get("models"), list)
                or len(candidate["models"]) != len(expected_models)
            ):
                continue
            candidate_models = {
                model.get("id"): model
                for model in candidate["models"]
                if isinstance(model, dict)
            }
            if set(candidate_models) != set(expected_models):
                continue
            if all(
                candidate_models[model_id].get("name") == MODEL_CONFIGS[model_id]["name"]
                and candidate_models[model_id].get("feature_extraction")
                == MODEL_CONFIGS[model_id]["feature_extraction"]
                and candidate_models[model_id].get("training_split") == "train"
                and candidate_models[model_id].get("training_records")
                == expected["training_records"]
                and candidate_models[model_id].get("trained_at") == expected["trained_at"]
                and isinstance(candidate_models[model_id].get("validation"), dict)
                and isinstance(candidate_models[model_id].get("test"), dict)
                for model_id, expected in expected_models.items()
            ):
                return candidate
        return None

    def _evaluate(
        self,
        *,
        splits: Any,
        splits_count: dict[str, int],
        pipelines: dict[str, Any],
        model_context: list[dict[str, Any]],
        evaluation_id: str,
    ) -> dict[str, Any]:
        predictions_by_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name in ("validation", "test"):
            split_data = splits[split_name]
            texts = np.asarray(split_data["text"], dtype=object)
            labels = np.asarray(split_data["label"], dtype=np.int64)
            if (
                not len(labels)
                or len(texts) != len(labels)
                or not np.isin(labels, list(LABELS)).all()
            ):
                raise EvaluationUnavailableError(
                    f"The {split_name} split is empty, has misaligned fields, or "
                    "contains an unsupported label."
                )
            predictions_by_split[split_name] = (texts, labels)

        model_results: list[dict[str, Any]] = []
        test_record_count: int | None = None
        for model_id, metadata in zip(MODEL_CONFIGS, model_context, strict=True):
            pipeline = pipelines[model_id]
            split_results: dict[str, dict[str, Any]] = {}
            for split_name in ("validation", "test"):
                texts, labels = predictions_by_split[split_name]
                predicted = np.asarray(pipeline.predict(texts), dtype=np.int64)
                if predicted.shape != labels.shape:
                    raise EvaluationUnavailableError(
                        f"Model '{model_id}' returned an invalid number of predictions "
                        f"for the {split_name} split."
                    )
                if not np.isin(predicted, list(LABELS)).all():
                    raise EvaluationUnavailableError(
                        f"Model '{model_id}' returned a label outside the three-class "
                        f"mapping for the {split_name} split."
                    )
                result = self._evaluate_split(labels, predicted, split_name)
                split_results[split_name] = result
                if split_name == "test":
                    if test_record_count is None:
                        test_record_count = result["records"]
                    elif result["records"] != test_record_count:
                        raise EvaluationUnavailableError(
                            "Models were not evaluated on identical test records."
                        )

            model_results.append(
                {
                    "id": model_id,
                    "name": MODEL_CONFIGS[model_id]["name"],
                    "feature_extraction": MODEL_CONFIGS[model_id]["feature_extraction"],
                    "training_split": "train",
                    "training_records": metadata["training_records"],
                    "trained_at": metadata["trained_at"],
                    "validation": split_results["validation"],
                    "test": split_results["test"],
                    "test_minus_validation": {
                        metric: (
                            split_results["test"][metric]
                            - split_results["validation"][metric]
                        )
                        for metric in (
                            "accuracy",
                            "precision_macro",
                            "recall_macro",
                            "f1_macro",
                        )
                    },
                }
            )

        highest_f1 = max(model["test"]["f1_macro"] for model in model_results)
        best_models = [
            model
            for model in model_results
            if np.isclose(model["test"]["f1_macro"], highest_f1, rtol=0.0, atol=1e-12)
        ]
        return {
            "evaluation_id": evaluation_id,
            "evaluation_version": EVALUATION_VERSION,
            "status": "complete",
            "dataset": DATASET_ID,
            "configuration": DATASET_CONFIG,
            "dataset_split_records": splits_count,
            "label_order": [LABELS[label_id] for label_id in LABELS],
            "test_split_usage": "final_evaluation_only",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": model_results,
            "best_model_by_macro_f1": {
                "score": float(highest_f1),
                "model_ids": [model["id"] for model in best_models],
                "model_names": [model["name"] for model in best_models],
                "tie": len(best_models) > 1,
            },
        }

    @staticmethod
    def _evaluate_split(
        labels: np.ndarray,
        predictions: np.ndarray,
        split_name: str,
    ) -> dict[str, Any]:
        ordered_ids = list(LABELS)
        ordered_names = [LABELS[label_id] for label_id in ordered_ids]
        report = classification_report(
            labels,
            predictions,
            labels=ordered_ids,
            target_names=ordered_names,
            zero_division=0,
            output_dict=True,
        )
        count = int(len(labels))
        correct = int(np.count_nonzero(labels == predictions))
        return {
            "split": split_name,
            "records": count,
            "correct": correct,
            "incorrect": count - correct,
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "per_class": {
                name: {
                    "precision": float(report[name]["precision"]),
                    "recall": float(report[name]["recall"]),
                    "f1": float(report[name]["f1-score"]),
                    "support": int(report[name]["support"]),
                }
                for name in ordered_names
            },
            "confusion_matrix": confusion_matrix(
                labels,
                predictions,
                labels=ordered_ids,
            ).tolist(),
            "classification_report": report,
        }

    @staticmethod
    def _write_once(path: Path, payload: dict[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                return
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _read_evaluation(path: Path, expected_id: str) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationUnavailableError(
                f"Stored final evaluation '{expected_id}' cannot be read: "
                f"{type(error).__name__}: {error}"
            ) from error
        if not isinstance(result, dict) or result.get("evaluation_id") != expected_id:
            raise EvaluationUnavailableError(
                f"Stored final evaluation '{expected_id}' is invalid."
            )
        return result


model_evaluation_service = ModelEvaluationService(
    trainer=training_service,
    datasets=dataset_service,
)
