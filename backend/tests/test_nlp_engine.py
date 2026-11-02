import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

from backend.app.config import LABELS
from backend.app.services.dataset_service import DatasetService
from backend.app.services.preprocessing import bigram_analyzer, preprocess_text
from backend.app.services.training_service import (
    BIGRAM_ANALYZER_VERSION,
    MODEL_CONFIGS,
    ModelUnavailableError,
    PREPROCESSING_VERSION,
    TrainingService,
)
import backend.app.services.training_service as training_service_module


class PreprocessingTests(unittest.TestCase):
    def test_preserves_hashtag_words_and_negation(self) -> None:
        normalized = preprocess_text(
            "I CAN'T like #Helpful posts! Visit https://example.test @person"
        )
        self.assertEqual(
            normalized,
            "i can not like helpful posts visit urltoken usertoken",
        )

    def test_bigram_analyzer_does_not_cross_sentence_boundaries(self) -> None:
        features = bigram_analyzer(
            "I do not like it. The service is terrible! This quality? "
            "is surprisingly good."
        )
        self.assertIn("do not", features)
        self.assertIn("not like", features)
        self.assertIn("service is", features)
        self.assertIn("is terrible", features)
        self.assertIn("surprisingly good", features)
        self.assertNotIn("it the", features)
        self.assertNotIn("terrible this", features)
        self.assertNotIn("quality is", features)
        self.assertTrue(all(len(feature.split()) == 2 for feature in features))


class PipelineTests(unittest.TestCase):
    texts = [
        "awful bad disappointed",
        "terrible poor service",
        "not good at all",
        "arrived today",
        "parcel received today",
        "ordinary service",
        "great excellent lovely",
        "very good",
        "happy helpful service",
    ]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def setUp(self) -> None:
        self.service = TrainingService(DatasetService())

    def test_three_models_use_independent_feature_pipelines(self) -> None:
        pipelines = {}
        for model_id, config in MODEL_CONFIGS.items():
            pipeline = self.service._build_pipeline(model_id, config)
            pipeline.fit(self.texts, self.labels)
            pipelines[model_id] = pipeline

        self.assertIsInstance(pipelines["unigram"].named_steps["vectorizer"], CountVectorizer)
        self.assertEqual(
            pipelines["unigram"].named_steps["vectorizer"].ngram_range,
            (1, 1),
        )
        self.assertIsInstance(pipelines["bigram"].named_steps["vectorizer"], CountVectorizer)
        self.assertIs(pipelines["bigram"].named_steps["vectorizer"].analyzer, bigram_analyzer)
        self.assertIsInstance(pipelines["tfidf"].named_steps["vectorizer"], TfidfVectorizer)
        self.assertEqual(
            pipelines["tfidf"].named_steps["vectorizer"].ngram_range,
            (1, 1),
        )
        self.assertEqual(len({id(pipeline.named_steps["classifier"]) for pipeline in pipelines.values()}), 3)
        for pipeline in pipelines.values():
            self.assertEqual(set(pipeline.named_steps["classifier"].classes_), set(LABELS))
            self.assertEqual(pipeline.named_steps["classifier"].alpha, 1.0)

    def test_prediction_returns_probability_and_feature_contributions(self) -> None:
        text = "happy helpful service"
        for model_id, config in MODEL_CONFIGS.items():
            with self.subTest(model=model_id):
                pipeline = self.service._build_pipeline(model_id, config)
                pipeline.fit(self.texts, self.labels)
                with patch.object(self.service, "_get_model", return_value=pipeline):
                    prediction = self.service.predict(model_id, text)

                classifier = pipeline.named_steps["classifier"]
                vectorizer = pipeline.named_steps["vectorizer"]
                matrix = vectorizer.transform([text])
                actual_probabilities = pipeline.predict_proba([text])[0]
                expected_probabilities = {
                    LABELS[int(class_id)]: float(probability)
                    for class_id, probability in zip(
                        classifier.classes_,
                        actual_probabilities,
                        strict=True,
                    )
                }
                self.assertEqual(prediction["model_name"], config["name"])
                self.assertEqual(prediction["model_status"], "trained")
                self.assertEqual(prediction["probabilities"], expected_probabilities)
                self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0)
                self.assertTrue(prediction["metadata"]["explanation_available"])
                self.assertTrue(prediction["explanations"])

                predicted_id = int(pipeline.predict([text])[0])
                predicted_index = int(np.flatnonzero(classifier.classes_ == predicted_id)[0])
                class_scores = classifier.class_log_prior_ + np.asarray(
                    matrix @ classifier.feature_log_prob_.T
                ).ravel()
                predicted_score = float(class_scores[predicted_index])
                class_scores[predicted_index] = -np.inf
                alternative_index = int(np.argmax(class_scores))
                alternative_score = float(
                    classifier.class_log_prior_[alternative_index]
                    + (matrix @ classifier.feature_log_prob_.T).ravel()[alternative_index]
                )
                expected_scores = matrix.data * (
                    classifier.feature_log_prob_[predicted_index, matrix.indices]
                    - classifier.feature_log_prob_[alternative_index, matrix.indices]
                )
                names = vectorizer.get_feature_names_out()
                expected_contributions = {
                    str(names[index]): float(score)
                    for index, score in zip(matrix.indices, expected_scores, strict=True)
                    if not np.isclose(score, 0)
                }
                actual_contributions = {
                    item["feature"]: item["score"]
                    for item in prediction["explanations"]
                }
                self.assertEqual(set(actual_contributions), set(expected_contributions))
                for feature, expected_score in expected_contributions.items():
                    actual_score = actual_contributions[feature]
                    self.assertAlmostEqual(actual_score, expected_score)
                    self.assertEqual(
                        next(
                            item["effect"]
                            for item in prediction["explanations"]
                            if item["feature"] == feature
                        ),
                        "supports" if expected_score > 0 else "opposes",
                    )
                prior_difference = float(
                    classifier.class_log_prior_[predicted_index]
                    - classifier.class_log_prior_[alternative_index]
                )
                self.assertAlmostEqual(
                    sum(actual_contributions.values()) + prior_difference,
                    predicted_score - alternative_score,
                )

    def test_bigram_pipeline_uses_sentence_aware_vectorizer_for_fit_and_transform(self) -> None:
        pipeline = self.service._build_pipeline("bigram", MODEL_CONFIGS["bigram"])
        pipeline.fit(
            [*self.texts, "good the service", "the service arrived"],
            [*self.labels, 1, 1],
        )
        vectorizer = pipeline.named_steps["vectorizer"]
        self.assertIs(vectorizer.analyzer, bigram_analyzer)
        self.assertNotIn("ngram_range", MODEL_CONFIGS["bigram"])
        pipeline_config = self.service._pipeline_config("bigram", MODEL_CONFIGS["bigram"])
        self.assertNotIn("ngram_range", pipeline_config)
        self.assertEqual(pipeline_config["analyzer"], BIGRAM_ANALYZER_VERSION)
        matrix = vectorizer.transform(["good. The service arrived"])
        feature_names = vectorizer.get_feature_names_out()
        features = {feature_names[index] for index in matrix.indices}
        self.assertIn("the service", features)
        self.assertIn("service arrived", features)
        self.assertNotIn("good the", features)
        self.assertTrue(all(len(feature.split()) == 2 for feature in feature_names))

    def test_incompatible_or_legacy_metadata_is_not_reused(self) -> None:
        metadata = {
            "model_id": "unigram",
            "pipeline_config": {
                "vectorizer": "count",
                "ngram_range": [1, 1],
                "alpha": 1.0,
            },
            "preprocessing_version": PREPROCESSING_VERSION,
            "scikit_learn_version": __import__("sklearn").__version__,
            "dataset_fingerprint": "cached-dataset",
        }
        with patch.object(self.service.dataset_service, "fingerprint", return_value="cached-dataset"):
            self.assertTrue(self.service._metadata_is_compatible("unigram", metadata))
            self.assertFalse(self.service._metadata_is_compatible("unigram", {"model_id": "unigram"}))
            metadata["dataset_fingerprint"] = "different-dataset"
            self.assertFalse(self.service._metadata_is_compatible("unigram", metadata))

    def test_legacy_bigram_metadata_requires_retraining(self) -> None:
        metadata = {
            "model_id": "bigram",
            "pipeline_config": {
                "vectorizer": "count",
                "ngram_range": [2, 2],
                "alpha": 1.0,
            },
            "preprocessing_version": PREPROCESSING_VERSION,
            "scikit_learn_version": __import__("sklearn").__version__,
            "dataset_fingerprint": "cached-dataset",
        }
        with patch.object(self.service.dataset_service, "fingerprint", return_value="cached-dataset"):
            self.assertFalse(self.service._metadata_is_compatible("bigram", metadata))
            metadata["pipeline_config"] = self.service._pipeline_config(
                "bigram",
                MODEL_CONFIGS["bigram"],
            )
            self.assertTrue(self.service._metadata_is_compatible("bigram", metadata))

    def test_only_bigram_retrains_and_round_trips_with_sentence_aware_config(self) -> None:
        validation_texts = [
            "terrible service",
            "awful experience",
            "ordinary service",
            "parcel arrived",
            "excellent service",
            "happy experience",
        ]
        validation_labels = [0, 0, 1, 1, 2, 2]
        dataset_fingerprint = "small-regression-dataset"

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            with patch.object(training_service_module, "MODEL_DIR", model_dir):
                for model_id, config in MODEL_CONFIGS.items():
                    pipeline = self.service._build_pipeline(model_id, config)
                    pipeline.fit(self.texts, self.labels)
                    pipeline_config = self.service._pipeline_config(model_id, config)
                    if model_id == "bigram":
                        pipeline_config = {**pipeline_config, "ngram_range": [2, 2]}
                    self.service._persist_model(
                        model_id,
                        pipeline,
                        {
                            "model_id": model_id,
                            "pipeline_config": pipeline_config,
                            "preprocessing_version": PREPROCESSING_VERSION,
                            "scikit_learn_version": __import__("sklearn").__version__,
                            "dataset_fingerprint": dataset_fingerprint,
                            "evaluation": {"split": "validation", "records": 0},
                        },
                    )

                legacy_bigram_metadata = (model_dir / "bigram.json").read_bytes()
                untouched_model_files = {
                    model_id: (
                        (model_dir / f"{model_id}.joblib").read_bytes(),
                        (model_dir / f"{model_id}.json").read_bytes(),
                    )
                    for model_id in ("unigram", "tfidf")
                }

                with (
                    patch.object(
                        self.service.dataset_service,
                        "splits",
                        return_value={
                            "train": {"text": self.texts, "label": self.labels},
                            "validation": {
                                "text": validation_texts,
                                "label": validation_labels,
                            },
                        },
                    ),
                    patch.object(
                        self.service.dataset_service,
                        "fingerprint",
                        return_value=dataset_fingerprint,
                    ),
                ):
                    result = self.service.train_all()
                    reloaded_service = TrainingService(self.service.dataset_service)
                    loaded_bigram = reloaded_service._get_model("bigram")

                self.assertIn(
                    "Unigram Naive Bayes: existing trained model reused.",
                    result["messages"],
                )
                self.assertIn(
                    "Bigram Naive Bayes: trained and evaluated.",
                    result["messages"],
                )
                self.assertIn(
                    "TF-IDF + Naive Bayes: existing trained model reused.",
                    result["messages"],
                )
                self.assertNotEqual(
                    (model_dir / "bigram.json").read_bytes(),
                    legacy_bigram_metadata,
                )
                self.assertEqual(
                    {
                        model_id: (
                            (model_dir / f"{model_id}.joblib").read_bytes(),
                            (model_dir / f"{model_id}.json").read_bytes(),
                        )
                        for model_id in ("unigram", "tfidf")
                    },
                    untouched_model_files,
                )

                vectorizer = loaded_bigram.named_steps["vectorizer"]
                self.assertIsInstance(vectorizer, CountVectorizer)
                self.assertIs(vectorizer.analyzer, bigram_analyzer)
                self.assertEqual(
                    set(loaded_bigram.named_steps["classifier"].classes_),
                    set(LABELS),
                )
                self.assertEqual(loaded_bigram.named_steps["classifier"].alpha, 1.0)
                self.assertTrue(
                    all(
                        len(feature.split()) == 2
                        for feature in vectorizer.get_feature_names_out()
                    )
                )
                self.assertEqual(
                    self.service._read_metadata("bigram")["pipeline_config"],
                    self.service._pipeline_config(
                        "bigram",
                        MODEL_CONFIGS["bigram"],
                    ),
                )

    def test_corrupted_persisted_model_is_reported_for_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "corrupted.joblib"
            artifact_path.write_text("not a joblib artifact", encoding="utf-8")
            with (
                patch.object(self.service, "_is_trained", return_value=True),
                patch.object(self.service, "_artifact_path", return_value=artifact_path),
            ):
                with self.assertRaisesRegex(Exception, "needs retraining"):
                    self.service._get_model("unigram")

            self.assertIn("unigram", self.service._invalid_models)
            self.assertEqual(self.service.status()[0]["training_status"], "failed")


class ExplainabilityScenarioTests(unittest.TestCase):
    training_texts = [
        "love wonderful amazing excellent great happy beautiful best",
        "excellent product wonderful experience extremely helpful staff",
        "best experience ever had brilliant perfect quality",
        "i love product amazing service delivery",
        "hate terrible awful disappointing unreliable poor worst",
        "terrible product experience disappointing service",
        "regret purchasing complete waste money horrible",
        "service slow quality bad very poor",
        "package arrived yesterday product available three different colours",
        "meeting scheduled for monday morning",
        "order received today package delivered",
        "product is available in several sizes",
        "do not like this product not a bad experience",
        "never expected such terrible service product is not good at all",
        "design beautiful but quality disappointing love product although delivery terrible",
        "staff friendly but service extremely slow",
    ]
    training_labels = [
        2, 2, 2, 2,
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 0, 1, 0,
    ]
    scenarios = {
        "positive": [
            "I absolutely love this product. The experience was amazing!",
            "The service was excellent and the staff were extremely helpful.",
            "This is one of the best experiences I have ever had.",
        ],
        "negative": [
            "I hate this product. The experience was terrible.",
            "The service was disappointing and completely unreliable.",
            "I regret purchasing this. It was a complete waste of money.",
        ],
        "neutral": [
            "The package arrived yesterday.",
            "The product is available in three different colours.",
            "The meeting is scheduled for Monday morning.",
        ],
        "negation": [
            "I do not like this product.",
            "This is not a bad experience.",
            "I never expected such terrible service.",
            "The product is not good at all.",
        ],
        "mixed": [
            "The design is beautiful, but the quality is disappointing.",
            "I love the product, although the delivery was terrible.",
            "The staff were friendly, but the service was extremely slow.",
        ],
        "edge": [
            "Amazing!",
            "Terrible.",
            "Okay.",
            "unseen qzxv blorf",
            "Sooooo amazing!!! #Helpful",
            "!!!",
        ],
    }

    def setUp(self) -> None:
        self.service = TrainingService(DatasetService())
        self.pipelines = {}
        for model_id, config in MODEL_CONFIGS.items():
            pipeline = self.service._build_pipeline(model_id, config)
            pipeline.fit(self.training_texts, self.training_labels)
            self.pipelines[model_id] = pipeline

    def test_scenario_predictions_have_model_specific_probabilities_and_explanations(self) -> None:
        mixed_has_both_signs = False
        for category, texts in self.scenarios.items():
            for text in texts:
                for model_id, pipeline in self.pipelines.items():
                    with self.subTest(category=category, model=model_id, text=text):
                        with patch.object(self.service, "_get_model", return_value=pipeline):
                            result = self.service.predict(model_id, text)

                        classifier = pipeline.named_steps["classifier"]
                        vectorizer = pipeline.named_steps["vectorizer"]
                        matrix = vectorizer.transform([text])
                        predicted_id = int(pipeline.predict([text])[0])
                        expected_probabilities = pipeline.predict_proba([text])[0]
                        expected_probability_map = {
                            LABELS[int(class_id)]: float(probability)
                            for class_id, probability in zip(
                                classifier.classes_,
                                expected_probabilities,
                                strict=True,
                            )
                        }
                        self.assertEqual(result["model_id"], model_id)
                        self.assertEqual(result["model_name"], MODEL_CONFIGS[model_id]["name"])
                        self.assertEqual(result["predicted_sentiment"], LABELS[predicted_id])
                        self.assertEqual(result["probabilities"], expected_probability_map)
                        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
                        self.assertEqual(
                            result["confidence"],
                            expected_probability_map[LABELS[predicted_id]],
                        )

                        if model_id == "bigram":
                            self.assertTrue(
                                all(
                                    len(item["feature"].split()) == 2
                                    for item in result["explanations"]
                                )
                            )
                        else:
                            self.assertTrue(
                                all(
                                    len(item["feature"].split()) == 1
                                    for item in result["explanations"]
                                )
                            )

                        predicted_index = int(
                            np.flatnonzero(classifier.classes_ == predicted_id)[0]
                        )
                        evidence = np.asarray(
                            matrix @ classifier.feature_log_prob_.T
                        ).ravel()
                        joint_scores = classifier.class_log_prior_ + evidence
                        joint_scores[predicted_index] = -np.inf
                        alternative_index = int(np.argmax(joint_scores))
                        expected_scores = matrix.data * (
                            classifier.feature_log_prob_[
                                predicted_index,
                                matrix.indices,
                            ]
                            - classifier.feature_log_prob_[
                                alternative_index,
                                matrix.indices,
                            ]
                        )
                        feature_names = vectorizer.get_feature_names_out()
                        expected_by_feature = {
                            str(feature_names[index]): float(score)
                            for index, score in zip(
                                matrix.indices,
                                expected_scores,
                                strict=True,
                            )
                            if not np.isclose(score, 0.0)
                        }
                        actual_by_feature = {
                            item["feature"]: item["score"]
                            for item in result["explanations"]
                        }
                        self.assertLessEqual(len(actual_by_feature), 8)
                        self.assertTrue(set(actual_by_feature).issubset(expected_by_feature))
                        for item in result["explanations"]:
                            score = expected_by_feature[item["feature"]]
                            self.assertAlmostEqual(item["score"], score)
                            self.assertEqual(
                                item["effect"],
                                "supports" if score > 0 else "opposes",
                            )
                        if category == "mixed":
                            effects = {item["effect"] for item in result["explanations"]}
                            mixed_has_both_signs |= effects == {"supports", "opposes"}

        self.assertTrue(mixed_has_both_signs)

    def test_invalid_and_edge_inputs_are_handled_without_crashing(self) -> None:
        with patch.object(
            self.service,
            "_get_model",
            return_value=self.pipelines["unigram"],
        ):
            for text in ("", " \t\r\n"):
                with self.subTest(text=repr(text)):
                    with self.assertRaisesRegex(ValueError, "at least one non-whitespace"):
                        self.service.predict("unigram", text)

            for text in ("!!!", "unseen qzxv blorf", "Sooooo amazing!!! #Helpful", "Amazing!", "Terrible.", "Okay."):
                with self.subTest(text=text):
                    result = self.service.predict("unigram", text)
                    self.assertIn(result["predicted_sentiment"], LABELS.values())
                    self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

            punctuation_result = self.service.predict("unigram", "!!!")
            self.assertEqual(punctuation_result["explanations"], [])

    def test_unavailable_model_is_reported(self) -> None:
        with patch.object(
            self.service,
            "_get_model",
            side_effect=ModelUnavailableError("Model 'unigram' is not trained."),
        ):
            with self.assertRaisesRegex(ModelUnavailableError, "is not trained"):
                self.service.predict("unigram", "A valid input")


if __name__ == "__main__":
    unittest.main()
