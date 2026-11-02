import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.dataset_service import (
    DatasetUnavailableError,
    dataset_service,
)
from backend.app.services.training_service import training_service

router = APIRouter(prefix="/api/dataset", tags=["dataset"])
logger = logging.getLogger("sentiscope.api.dataset")


@router.get("/summary")
def dataset_summary() -> dict:
    try:
        return dataset_service.summary()
    except DatasetUnavailableError as error:
        logger.warning("Dataset summary is unavailable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="TweetEval dataset summary is unavailable. Check backend logs and retry.",
        ) from error


@router.get("/preview")
def dataset_preview(
    split: Literal["train", "validation", "test"] = "train",
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    q: str | None = Query(default=None, max_length=250),
    label: Literal["negative", "neutral", "positive"] | None = None,
) -> dict:
    try:
        return dataset_service.preview(split, limit, offset, q, label)
    except DatasetUnavailableError as error:
        logger.warning("Dataset preview is unavailable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="TweetEval dataset preview is unavailable. Check backend logs and retry.",
        ) from error


@router.post("/reload")
def reload_dataset() -> dict:
    try:
        status = dataset_service.load(retry=True)
        result = training_service.train_all()
        failures = [
            model for model in result["models"]
            if model["training_status"] == "failed"
        ]
        if failures:
            logger.error(
                "Model training failed after dataset reload: %s",
                "; ".join(model["name"] for model in failures),
            )
            raise HTTPException(
                status_code=500,
                detail="Dataset reloaded, but one or more models could not be trained. "
                "Check backend logs.",
            )
        return status
    except DatasetUnavailableError as error:
        logger.warning("Dataset reload failed: %s", error)
        raise HTTPException(
            status_code=503,
            detail="TweetEval could not be reloaded. Check backend connectivity and retry.",
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Dataset reload request failed.")
        raise HTTPException(
            status_code=500,
            detail="Dataset reload failed due to a backend error. Check server logs and retry.",
        ) from error
