import logging

from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import AnalysisRequest, AnalysisResponse
from backend.app.services.prediction_service import prediction_service
from backend.app.services.training_service import ModelUnavailableError

router = APIRouter(prefix="/api", tags=["analysis"])
logger = logging.getLogger("sentiscope.api.analysis")


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    try:
        result = prediction_service.analyze(request.text, request.model)
    except ModelUnavailableError as error:
        logger.warning("Requested model is unavailable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="The selected model is unavailable. Train the models or check backend logs.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Prediction request failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to a backend error. Check server logs and try again.",
        ) from error
    return AnalysisResponse.model_validate(result)
