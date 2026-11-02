import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score

from backend.app.routes import models as models_route
from backend.app.routes import dataset as dataset_route
from backend.app.schemas.api import ModelEvaluationResponse
from backend.app.services import evaluation_service as evaluation_module
from backend.app.services.dataset_service import DatasetService, DatasetUnavailableError
from backend.app.services.evaluation_service import (
    EvaluationUnavailableError,
    ModelEvaluationService,
)
from backend.app.services.training_service import (
    MODEL_CONFIGS,
    ModelUnavailableError,
    TrainingService,
)
import backend.app.services.training_service as training_module


class SyntheticDataset:
    def __init__(self) -> None:
        self.data = {
            "train": {
                "text": [
                    "awful bad service",
                    "terrible disappointing",
                    "hate this product",
                    "arrived today",
                    "ordinary parcel",
                    "meeting on monday",
                    "great helpful service",
                    "love this product",
                    "excellent experience",
                ],
                "label": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            },
            "validation": {
                "text": [
                    "terrible service",
                    "ordinary product",
                    "great experience",
                ],
                "label": [0, 1, 2],
            },
            "test": {
                "text": [
                    "bad disappointing product",
                    "arrived yesterday",
                    "excellent helpful staff",
                    "unseen vocabulary",
                ],
                "label": [0, 1, 2, 1],
            },
        }

    def splits(self) -> dict:
        return self.data

    def fingerprint(self) -> str:
        return "synthetic-frozen-split-fingerprint"


class ModelEvaluationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.model_directory = Path(self.temporary_directory.name) / "models"
        self.datasets = SyntheticDataset()
        self.trainer = TrainingService(self.datasets)
        self.training_directory_patch = patch.object(
            training_module,
            "MODEL_DIR",
            self.model_directory,
        )
        self.evaluation_directory_patch = patch.object(
            evaluation_module,
            "MODEL_DIR",
            self.model_directory,
        )
        self.training_directory_patch.start()
        self.evaluation_directory_patch.start()
        self.addCleanup(self.training_directory_patch.stop)
        self.addCleanup(self.evaluation_directory_patch.stop)
        trained = self.trainer.train_all()
        self.assertTrue(all(model["trained"] for model in trained["models"]))
        self.service = ModelEvaluationService(self.trainer, self.datasets)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_final_evaluation_reports_actual_metrics_and_class_support(self) -> None:
        pipelines = {
            model_id: self.trainer._get_model(model_id)
            for model_id in MODEL_CONFIGS
        }
        result = self.service.get_evaluation()

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evaluation_version"], "held-out-sentiment-v1")
        self.assertEqual(result["test_split_usage"], "final_evaluation_only")
        self.assertEqual(result["label_order"], ["negative", "neutral", "positive"])
        self.assertEqual(
            result["dataset_split_records"],
            {"train": 9, "validation": 3, "test": 4},
        )
        self.assertEqual(len(result["models"]), 3)

        test_labels = np.asarray(self.datasets.data["test"]["label"])
        test_texts = np.asarray(self.datasets.data["test"]["text"], dtype=object)
        for model_result in result["models"]:
            with self.subTest(model=model_result["id"]):
                model_id = model_result["id"]
                expected_predictions = pipelines[model_id].predict(test_texts)
                self.assertEqual(
                    model_result["test"]["accuracy"],
                    float(accuracy_score(test_labels, expected_predictions)),
                )
                self.assertEqual(
                    model_result["test"]["correct"]
                    + model_result["test"]["incorrect"],
                    4,
                )
                self.assertEqual(
                    sum(
                        item["support"]
                        for item in model_result["test"]["per_class"].values()
                    ),
                    4,
                )
                self.assertEqual(
                    sum(map(sum, model_result["test"]["confusion_matrix"])),
                    4,
                )
                self.assertEqual(
                    model_result["training_split"],
                    "train",
                )
                self.assertAlmostEqual(
                    model_result["test_minus_validation"]["f1_macro"],
                    model_result["test"]["f1_macro"]
                    - model_result["validation"]["f1_macro"],
                )
                self.assertEqual(
                    model_result["test"]["classification_report"]["accuracy"],
                    model_result["test"]["accuracy"],
                )

        ModelEvaluationResponse(**result)
        winner = result["best_model_by_macro_f1"]
        self.assertTrue(winner["model_ids"])
        self.assertEqual(winner["score"], max(item["test"]["f1_macro"] for item in result["models"]))

    def test_evaluation_is_cached_without_repeating_test_predictions(self) -> None:
        pipelines = {
            model_id: self.trainer._get_model(model_id)
            for model_id in MODEL_CONFIGS
        }
        metadata_before = {
            model_id: self.trainer._read_metadata(model_id)
            for model_id in MODEL_CONFIGS
        }
        spies = {}
        for model_id, pipeline in pipelines.items():
            spy = patch.object(pipeline, "predict", wraps=pipeline.predict)
            spies[model_id] = spy.start()
            self.addCleanup(spy.stop)

        first = self.service.get_evaluation()
        first_call_counts = {
            model_id: spy.call_count
            for model_id, spy in spies.items()
        }
        second = self.service.get_evaluation()

        self.assertEqual(first["evaluation_id"], second["evaluation_id"])
        self.assertEqual(first_call_counts, {model_id: 2 for model_id in MODEL_CONFIGS})
        self.assertEqual(
            {model_id: spy.call_count for model_id, spy in spies.items()},
            first_call_counts,
        )
        self.assertEqual(
            {
                model_id: self.trainer._read_metadata(model_id)
                for model_id in MODEL_CONFIGS
            },
            metadata_before,
        )
        saved_artifacts = list((self.model_directory / "evaluations").glob("*.json"))
        self.assertEqual(len(saved_artifacts), 1)

    def test_persisted_report_is_reused_when_deployment_changes_file_timestamps(self) -> None:
        original = self.service.get_evaluation()
        model = self.trainer._get_model("unigram")
        artifact = self.trainer._artifact_path("unigram")
        stat = artifact.stat()
        os.utime(
            artifact,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000),
        )
        with patch.object(model, "predict", side_effect=AssertionError("unexpected reevaluation")):
            restored = self.service.get_evaluation()
        self.assertEqual(restored["evaluation_id"], original["evaluation_id"])
        self.assertEqual(
            len(list((self.model_directory / "evaluations").glob("*.json"))),
            1,
        )

    def test_evaluation_refuses_untrained_or_mismatched_models(self) -> None:
        self.trainer._metadata_path("unigram").unlink()
        with self.assertRaises(ModelUnavailableError):
            self.service.get_evaluation()

    def test_invalid_split_labels_are_rejected(self) -> None:
        self.datasets.data["test"]["label"][0] = 9
        with self.assertRaises(EvaluationUnavailableError):
            self.service.get_evaluation()


class ModelEvaluationRouteTests(unittest.TestCase):
    def test_unavailable_dataset_returns_service_unavailable(self) -> None:
        with patch.object(
            models_route.model_evaluation_service,
            "get_evaluation",
            side_effect=DatasetUnavailableError("Official test split is unavailable."),
        ):
            with self.assertRaises(models_route.HTTPException) as raised:
                models_route.model_evaluation()
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("dataset and model readiness", raised.exception.detail)
        self.assertNotIn("Official test split", raised.exception.detail)

    def test_unexpected_error_is_returned_as_server_error(self) -> None:
        with patch.object(
            models_route.model_evaluation_service,
            "get_evaluation",
            side_effect=RuntimeError("classifier prediction failed"),
        ):
            with self.assertRaises(models_route.HTTPException) as raised:
                models_route.model_evaluation()
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("Check server logs", raised.exception.detail)
        self.assertNotIn("classifier prediction failed", raised.exception.detail)

    def test_training_failure_does_not_expose_exception_details(self) -> None:
        with patch.object(
            models_route.training_service,
            "train_all",
            side_effect=RuntimeError("C:\\private\\path\\secret"),
        ):
            with self.assertRaises(models_route.HTTPException) as raised:
                models_route.train_models()
        self.assertEqual(raised.exception.status_code, 500)
        self.assertNotIn("private", raised.exception.detail)

    def test_dataset_failure_does_not_expose_exception_details(self) -> None:
        with patch.object(
            dataset_route.dataset_service,
            "summary",
            side_effect=DatasetUnavailableError("C:\\private\\path\\secret"),
        ):
            with self.assertRaises(dataset_route.HTTPException) as raised:
                dataset_route.dataset_summary()
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("private", raised.exception.detail)


class DatasetHealthErrorTests(unittest.TestCase):
    def test_health_status_sanitizes_internal_dataset_error(self) -> None:
        datasets = DatasetService()
        datasets._datasets = None
        datasets._state = "unavailable"
        datasets._error = "C:\\private\\path\\secret"
        status = datasets.status()
        self.assertIn("backend logs", status["error"])
        self.assertNotIn("private", status["error"])


if __name__ == "__main__":
    unittest.main()
