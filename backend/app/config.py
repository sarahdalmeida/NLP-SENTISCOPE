import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = Path(os.getenv("SENTISCOPE_DATA_DIR", str(BACKEND_DIR / "data"))).resolve()
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"

_cors_origins = os.getenv(
    "SENTISCOPE_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
CORS_ORIGINS = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]

DATASET_ID = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_IDS = {name: index for index, name in LABELS.items()}
SPLITS = ("train", "validation", "test")
