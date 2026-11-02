import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.routes.analysis import analyze
from backend.app.schemas.api import AnalysisRequest
from backend.app.services.training_service import ModelUnavailableError


class AnalysisRouteTests(unittest.TestCase):
    def test_request_schema_rejects_empty_and_over_limit_text(self) -> None:
        for text in ("", "x" * 5001):
            with self.subTest(length=len(text)):
                with self.assertRaises(ValidationError):
                    AnalysisRequest(text=text, model="unigram")

    def test_whitespace_is_rejected_by_prediction_service_as_unprocessable(self) -> None:
        request = AnalysisRequest(text=" \t\n", model="unigram")
        with patch(
            "backend.app.routes.analysis.prediction_service.analyze",
            side_effect=ValueError(
                "Text must contain at least one non-whitespace character."
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                analyze(request)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("non-whitespace", raised.exception.detail)

    def test_unavailable_model_returns_actionable_service_unavailable_error(self) -> None:
        request = AnalysisRequest(text="A review", model="bigram")
        with patch(
            "backend.app.routes.analysis.prediction_service.analyze",
            side_effect=ModelUnavailableError("Model 'bigram' is not trained."),
        ):
            with self.assertRaises(HTTPException) as raised:
                analyze(request)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("Train the models", raised.exception.detail)

    def test_unexpected_prediction_error_is_logged_but_not_exposed(self) -> None:
        request = AnalysisRequest(text="A review", model="tfidf")
        with patch(
            "backend.app.routes.analysis.prediction_service.analyze",
            side_effect=RuntimeError("classifier state is invalid"),
        ):
            with self.assertRaises(HTTPException) as raised:
                analyze(request)
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("Check server logs", raised.exception.detail)
        self.assertNotIn("classifier state is invalid", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
