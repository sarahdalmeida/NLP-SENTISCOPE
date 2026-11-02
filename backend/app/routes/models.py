import logging

from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import ModelEvaluationResponse, TrainRequest
from backend.app.services.dataset_service import DatasetUnavailableError
from backend.app.services.evaluation_service import (
    EvaluationUnavailableError,
    model_evaluation_service,
)
from backend.app.services.training_service import (
    ModelUnavailableError,
    TrainingInProgressError,
    training_service,
)

router = APIRouter(prefix="/api/models", tags=["models"])
logger = logging.getLogger("sentiscope.api.models")


@router.get("/status")
def model_status() -> dict:
    return {"models": training_service.status()}


@router.get("/evaluation", response_model=ModelEvaluationResponse)
def model_evaluation() -> dict:
    try:
        return model_evaluation_service.get_evaluation()
    except (DatasetUnavailableError, ModelUnavailableError, EvaluationUnavailableError) as error:
        logger.warning("Final evaluation is unavailable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="Final evaluation is unavailable. Check dataset and model readiness.",
        ) from error
    except Exception as error:
        logger.exception("Final model evaluation request failed.")
        raise HTTPException(
            status_code=500,
            detail="Model evaluation failed due to a backend error. Check server logs and try again.",
        ) from error


@router.post("/train")
def train_models(request: TrainRequest = TrainRequest()) -> dict:
    try:
        return training_service.train_all(force=request.force)
    except TrainingInProgressError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except DatasetUnavailableError as error:
        logger.warning("Model training is unavailable because the dataset is unavailable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="Model training is unavailable because TweetEval could not be loaded.",
        ) from error
    except Exception as error:
        logger.exception("Model training request failed.")
        raise HTTPException(
            status_code=500,
            detail="Model training failed due to a backend error. Check server logs and try again.",
        ) from error
