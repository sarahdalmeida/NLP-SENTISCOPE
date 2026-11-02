# SentiScope NLP backend

FastAPI service for loading the official [TweetEval sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) dataset and training three scikit-learn sentiment classifiers. Dataset files are cached under `backend/data/cache`; independently trained pipelines and evaluation metadata are stored under `backend/data/models`. These directories are generated at runtime and are not source data or fabricated examples.

## Setup and start

From the repository root, create a Python environment, install dependencies, and start the API:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Python 3.11 or newer is recommended. `GET /health` remains available if Hugging Face is offline; in that case, the response exposes the dataset's unavailable state and error. The API does not silently substitute sample data. After the network is restored, retry with Dataset Explorer or `POST /api/dataset/reload`. After dataset loading, the backend automatically trains missing or incompatible models and reuses compatible persisted artifacts.

Configuration is through environment variables:

| Variable | Default | Description |
|---|---|---|
| `SENTISCOPE_DATA_DIR` | `backend/data` | Hugging Face cache and persisted model directory |
| `SENTISCOPE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed browser origins |

Interactive API documentation is available at `/docs`.

For production, mount `SENTISCOPE_DATA_DIR` on persistent storage and set `SENTISCOPE_CORS_ORIGINS` to the exact frontend origin. The application stores dataset/model readiness in process memory, so start one Uvicorn worker per deployment instance. Protect public deployments with host-level authentication/rate limiting as needed; CORS is not access control. Only load trusted joblib files.

## Dataset and split discipline

The backend loads the `sentiment` configuration from `cardiffnlp/tweet_eval` and its official `train`, `validation`, and `test` splits. Label IDs are `0 = negative`, `1 = neutral`, `2 = positive`; the dataset label names are checked before use. Models are fit **only** on `train`. Vocabulary and inverse-document-frequency weights are learned only during pipeline fitting on `train`. Training-time metrics are calculated on the separate `validation` split. The `test` split is not used for fitting, vocabulary creation, configuration selection, or the training endpoint's evaluation metrics.

The preview endpoint reads actual rows from the official training split. Dataset totals and sentiment counts include all three official splits. Previews and summaries are not model training data substitutes.

## Text preprocessing and models

Shared preprocessing replaces URLs and mentions with stable tokens, lowercases text, keeps hashtag words, normalizes apostrophe-based negation contractions (`can't` to `can not`, `isn't` to `is not`), removes remaining punctuation without dropping words such as `no`, `not`, or `never`, and normalizes whitespace. The unigram and TF-IDF vectorizers use it directly. The bigram analyzer applies the same normalization independently within sentences split at terminal punctuation and line breaks, so adjacent pairs never cross sentence boundaries.

Each persisted scikit-learn `Pipeline` has a separately fitted vectorizer and `MultinomialNB(alpha=1.0)`:

1. `unigram`: `CountVectorizer(ngram_range=(1, 1))`
2. `bigram`: `CountVectorizer` with a sentence-aware analyzer that emits only adjacent two-word pairs; the analyzer defines the feature range, so no `ngram_range` setting is stored for this pipeline
3. `tfidf`: `TfidfVectorizer(ngram_range=(1, 1))`

Training runs automatically after dataset loading for missing or incompatible models. `POST /api/models/train` remains available for manual training; `{"force": true}` requests retraining. Saved artifacts are reused only when their dataset fingerprint, preprocessing version, effective vectorizer configuration (including the bigram analyzer version), and scikit-learn version match the current runtime. Model probabilities come from the fitted classifier's `predict_proba`. Predictions include the selected model name and trained status, plus up to eight feature contributions computed from the fitted vectorizer and classifier log-feature probabilities. For feature `j`, the score is its observed count or fitted TF-IDF weight multiplied by `log P(feature_j | predicted) - log P(feature_j | strongest alternative)`. Positive scores support the predicted class and negative scores oppose it. Their sum is the feature-evidence difference; adding the predicted-versus-alternative class log-prior difference yields the complete joint log-score difference.

## API

- `GET /health`: API, dataset-load state/error, and all model states.
- `POST /api/analyze`: `{ "text": "...", "model": "unigram" }`; requires a trained model.
- `GET /api/dataset/summary`: source, split totals, mapped label counts, class proportions, and dataset quality checks.
- `GET /api/dataset/preview?split=train&limit=10&offset=0&q=...&label=negative`: actual, searchable, optionally filtered records.
- `GET /api/models/status`: model availability, configuration, and validation metrics.
- `GET /api/models/evaluation`: reads the cached validation/final-test report or evaluates and caches the current compatible models when no compatible report exists. Preserve the existing `models/evaluations/*.json` report and its matching dataset/model files in the deployment volume to avoid repeating the held-out test evaluation.
- `POST /api/models/train`: trains missing models; `{"force": true}` retrains.
- `POST /api/dataset/reload`: explicitly retry a failed or unavailable dataset load.

For frontend development, Vite proxies `/api` to `http://127.0.0.1:8000`. Configure `VITE_API_BASE_URL` to change the frontend API base. If the UI and backend are served from different origins, configure `SENTISCOPE_CORS_ORIGINS` for the frontend origin.
