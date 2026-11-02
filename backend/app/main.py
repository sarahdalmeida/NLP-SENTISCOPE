import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import CORS_ORIGINS
from backend.app.routes.analysis import router as analysis_router
from backend.app.routes.dataset import router as dataset_router
from backend.app.routes.models import router as models_router
from backend.app.services.dataset_service import (
    DatasetUnavailableError,
    dataset_service,
)
from backend.app.services.training_service import training_service

logger = logging.getLogger("sentiscope")


async def _load_dataset() -> None:
    try:
        await asyncio.to_thread(dataset_service.load)
    except DatasetUnavailableError as error:
        logger.exception("TweetEval dataset loading failed: %s", error)
        return
    try:
        result = await asyncio.to_thread(training_service.train_all)
        for model in result["models"]:
            if model["training_status"] == "failed":
                logger.error("%s training failed: %s", model["name"], model["training_error"])
    except Exception:
        logger.exception("Automatic model training failed after dataset loading.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_task = asyncio.create_task(_load_dataset())
    yield
    if not load_task.done():
        load_task.cancel()


app = FastAPI(
    title="SentiScope NLP API",
    description="Real TweetEval-backed sentiment analysis and model evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(analysis_router)
app.include_router(dataset_router)
app.include_router(models_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "dataset": dataset_service.status(),
        "models": training_service.status(),
    }
