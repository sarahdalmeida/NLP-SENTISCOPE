from collections import Counter
from hashlib import sha256
from threading import Event, Lock
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_dataset

from backend.app.config import (
    CACHE_DIR,
    DATASET_CONFIG,
    DATASET_ID,
    LABELS,
    SPLITS,
)


class DatasetUnavailableError(RuntimeError):
    """Raised when the official dataset cannot be loaded."""


class DatasetService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._loaded = Event()
        self._datasets: DatasetDict | None = None
        self._state = "not_loaded"
        self._error: str | None = None
        self._fingerprint: str | None = None
        self._summary: dict[str, Any] | None = None
        self._preview_frames: dict[str, pd.DataFrame] | None = None

    def load(self, retry: bool = False) -> dict[str, Any]:
        already_available = False
        with self._lock:
            if self._datasets is not None:
                already_available = True
                wait_for_existing_load = False
            elif self._state == "loading":
                wait_for_existing_load = True
            elif self._state == "unavailable" and not retry:
                raise DatasetUnavailableError(self._error or "Dataset loading failed.")
            else:
                self._state = "loading"
                self._error = None
                self._fingerprint = None
                self._loaded.clear()
                wait_for_existing_load = False

        if already_available:
            return self.status()
        if wait_for_existing_load:
            if not self._loaded.wait(timeout=600):
                raise DatasetUnavailableError("Timed out waiting for the dataset download.")
            if self._datasets is None:
                raise DatasetUnavailableError(self._error or "Dataset loading failed.")
            return self.status()

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            datasets = load_dataset(
                DATASET_ID,
                DATASET_CONFIG,
                cache_dir=str(CACHE_DIR),
            )
            missing_splits = set(SPLITS).difference(datasets.keys())
            if missing_splits:
                raise ValueError(f"Dataset is missing official splits: {sorted(missing_splits)}")
            feature_names = datasets["train"].features["label"].names
            expected_names = [LABELS[index] for index in range(len(LABELS))]
            if feature_names != expected_names:
                raise ValueError(
                    f"Unexpected label mapping {feature_names!r}; expected {expected_names!r}."
                )
            for split in SPLITS:
                missing_features = {"text", "label"}.difference(datasets[split].features)
                if missing_features:
                    raise ValueError(
                        f"Dataset split '{split}' is missing required fields: "
                        f"{sorted(missing_features)}."
                    )
                label_feature = datasets[split].features["label"]
                split_labels = getattr(label_feature, "names", None)
                if split_labels != expected_names:
                    raise ValueError(
                        f"Unexpected label mapping in '{split}': {split_labels!r}; "
                        f"expected {expected_names!r}."
                    )
                invalid_text_count = sum(
                    not isinstance(text, str) or not text.strip()
                    for text in datasets[split]["text"]
                )
                invalid_label_count = sum(
                    not isinstance(label_id, int) or label_id not in LABELS
                    for label_id in datasets[split]["label"]
                )
                if invalid_text_count or invalid_label_count:
                    raise ValueError(
                        f"Dataset split '{split}' contains {invalid_text_count} "
                        f"missing/empty text records and {invalid_label_count} invalid labels."
                    )
            fingerprint = sha256(
                "|".join(
                    f"{split}:{datasets[split]._fingerprint}:{len(datasets[split])}"
                    for split in SPLITS
                ).encode("utf-8")
            ).hexdigest()
            summary, preview_frames = self._build_summary(datasets)
            with self._lock:
                self._datasets = datasets
                self._summary = summary
                self._preview_frames = preview_frames
                self._fingerprint = fingerprint
                self._state = "available"
                self._error = None
            return self.status()
        except Exception as error:
            message = (
                f"Unable to load {DATASET_ID}/{DATASET_CONFIG}: "
                f"{type(error).__name__}: {error}"
            )
            with self._lock:
                self._state = "unavailable"
                self._error = message
            raise DatasetUnavailableError(message) from error
        finally:
            self._loaded.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            total = (
                self._summary["total_records"]
                if self._summary is not None
                else None
            )
            return {
                "status": self._state,
                "available": self._datasets is not None,
                "total_records": total,
                "error": (
                    "TweetEval is unavailable. Check backend logs and retry."
                    if self._state == "unavailable"
                    else None
                ),
            }

    def fingerprint(self) -> str | None:
        with self._lock:
            return self._fingerprint

    def summary(self) -> dict[str, Any]:
        self.ensure_loaded()
        with self._lock:
            if self._summary is None:
                raise DatasetUnavailableError("The dataset summary is not available.")
            return self._summary

    def preview(
        self,
        split: str,
        limit: int,
        offset: int,
        query: str | None,
        label: str | None,
    ) -> dict[str, Any]:
        self.ensure_loaded()
        normalized_query = (query or "").casefold().strip()
        selected_label = next(
            (label_id for label_id, name in LABELS.items() if name == label),
            None,
        )
        with self._lock:
            if self._preview_frames is None:
                raise DatasetUnavailableError("The dataset preview is not available.")
            frame = self._preview_frames[split]
        selected = frame
        if selected_label is not None:
            selected = selected.loc[selected["label"] == selected_label]
        if normalized_query:
            selected = selected.loc[
                selected["text"].str.contains(normalized_query, case=False, regex=False, na=False)
            ]
        total_matches = len(selected)
        matched = [
            {
                "id": f"{split}-{int(index) + 1}",
                "split": split,
                "text": str(record["text"]),
                "label": int(record["label"]),
                "sentiment": LABELS[int(record["label"])],
            }
            for index, record in selected.iloc[offset : offset + limit].iterrows()
        ]
        return {
            "dataset": DATASET_ID,
            "split": split,
            "offset": offset,
            "limit": limit,
            "total_matches": total_matches,
            "records": matched,
        }

    def splits(self) -> DatasetDict:
        return self.ensure_loaded()

    def ensure_loaded(self) -> DatasetDict:
        with self._lock:
            current_state = self._state
        if current_state == "available":
            with self._lock:
                if self._datasets is not None:
                    return self._datasets
        if current_state == "loading":
            self.load()
        elif current_state == "unavailable":
            raise DatasetUnavailableError(self._error or "Dataset is unavailable.")
        else:
            self.load()
        with self._lock:
            if self._datasets is None:
                raise DatasetUnavailableError(self._error or "Dataset is unavailable.")
            return self._datasets

    @staticmethod
    def _build_summary(
        datasets: DatasetDict,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        preview_frames = {
            split: pd.DataFrame(
                {
                    "text": datasets[split]["text"],
                    "label": datasets[split]["label"],
                    "split": split,
                }
            )
            for split in SPLITS
        }
        frame = pd.concat(list(preview_frames.values()), ignore_index=True)
        counts = Counter(int(label_id) for label_id in frame["label"].tolist())
        distribution = {
            name: {
                "count": counts.get(label_id, 0),
                "proportion": (
                    counts.get(label_id, 0) / len(frame) if len(frame) else 0.0
                ),
            }
            for label_id, name in LABELS.items()
        }
        split_counts = {split: len(datasets[split]) for split in SPLITS}
        normalized_text = frame["text"].fillna("").astype(str).str.strip()
        return (
            {
                "name": "TweetEval Sentiment",
                "source": DATASET_ID,
                "configuration": DATASET_CONFIG,
                "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
                "total_records": len(frame),
                "splits": split_counts,
                "label_mapping": {str(label_id): name for label_id, name in LABELS.items()},
                "distribution": distribution,
                "quality": {
                    "empty_text_records": int(normalized_text.eq("").sum()),
                    "duplicate_text_records": int(normalized_text.duplicated().sum()),
                    "label_values_valid": bool(frame["label"].isin(LABELS).all()),
                },
            },
            preview_frames,
        )


dataset_service = DatasetService()
