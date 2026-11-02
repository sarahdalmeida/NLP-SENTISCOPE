# SentiScope

SentiScope is an explainable, three-class sentiment analysis and model evaluation application. It trains and compares three scikit-learn Naive Bayes pipelines on the official TweetEval sentiment dataset. Users can inspect the dataset, analyze text with any trained model, review class probabilities and feature contributions, and compare validation and held-out test results.

## Problem and objectives

Short social-media text includes informal spelling, punctuation, hashtags, URLs, mentions, and negation. SentiScope provides a transparent academic implementation to:

- Compare unigram, sentence-aware bigram, and TF-IDF text features with Multinomial Naive Bayes.
- Return negative, neutral, and positive sentiment probabilities from fitted classifiers.
- Explain the evidence that supports or opposes a model's selected class.
- Report model performance on validation data and a separately held-out official test split.
- Keep dataset facts, model states, analysis results, and evaluation results connected to the backend rather than filling the UI with sample values.

## Features

- Dashboard with live dataset totals, class distribution, model statuses, and validation metrics.
- Sentiment Studio with model selection, input counts, sample text, real predictions, class probabilities, and model-specific explanations.
- Model Comparison with persisted validation and test metrics, per-class precision/recall/F1/support, confusion matrices, macro-F1 ranking, and comparisons.
- Dataset Explorer with official TweetEval split statistics, searchable/filterable training-record previews, and retryable dataset loading.
- Methodology explanations for preprocessing, features, Naive Bayes, smoothing, prediction explanations, and evaluation.
- Health, training, dataset, prediction, and evaluation APIs with explicit loading/error states.

## Architecture and technology

The project is a React single-page frontend and a Python API backend:

- **Frontend:** React 19, TypeScript, Vite, CSS.
- **Backend:** Python, FastAPI, Pydantic, scikit-learn, NumPy, pandas, Hugging Face `datasets`, and joblib.
- **Models:** three independent `CountVectorizer`/`TfidfVectorizer` + `MultinomialNB(alpha=1.0)` pipelines.
- **Data:** `cardiffnlp/tweet_eval`, configuration `sentiment`.
- **No LLM service or API key is used.**

```text
.
├── backend/
│   ├── app/
│   │   ├── routes/         # Analysis, dataset, and model APIs
│   │   ├── schemas/        # Pydantic request/response types
│   │   └── services/       # Dataset, preprocessing, training, prediction, evaluation
│   ├── tests/              # Backend regression and API tests
│   ├── requirements.txt
│   └── data/               # Runtime cache, model artifacts, evaluation report (ignored)
├── src/
│   ├── components/         # Dashboard, analyzer, results, model comparison, etc.
│   ├── services/api.ts     # Typed frontend API calls and timeouts
│   ├── App.tsx
│   └── types.ts
├── .env.example
├── package.json
└── vite.config.ts
```

## Dataset and labels

SentiScope loads the official Hugging Face `cardiffnlp/tweet_eval` dataset, `sentiment` configuration, including its original splits:

| Split | Records | Use |
|---|---:|---|
| Train | 45,615 | Fit vectorizers and classifiers |
| Validation | 2,000 | Report validation performance; not used to fit features or classifier parameters |
| Test | 12,284 | Final held-out evaluation only |
| **Total** | **59,899** | |

TweetEval label IDs are mapped and checked as `0 = negative`, `1 = neutral`, `2 = positive`. Loading verifies the official splits, text/label fields, label names, non-empty text, and valid labels. Dataset statistics and previews are derived from these loaded records. If the official dataset is unavailable, the application reports the failure and offers a retry; it does not substitute fabricated data.

## Text processing and classifiers

The same `preprocess_text` implementation is used by training and inference:

- Lowercase and normalize whitespace.
- Replace URLs and mentions with stable `urltoken` and `usertoken` terms.
- Remove the `#` marker while retaining hashtag words.
- Expand common negative contractions (for example, `can't` to `can not`) and preserve negation.
- Remove punctuation without discarding the words it separates.
- For the Bigram model, split sentences at terminal punctuation and line breaks before normalization; emit only adjacent two-word features within each sentence.

The three independently fitted pipelines are:

1. **Unigram Naive Bayes:** `CountVectorizer(ngram_range=(1, 1))` followed by `MultinomialNB(alpha=1.0)`.
2. **Bigram Naive Bayes:** `CountVectorizer` with a callable, versioned sentence-aware analyzer that emits adjacent pairs only, followed by `MultinomialNB(alpha=1.0)`. Since the analyzer controls features, an `ngram_range` setting would be ignored and is not used.
3. **TF-IDF + Naive Bayes:** `TfidfVectorizer(ngram_range=(1, 1))` followed by `MultinomialNB(alpha=1.0)`.

All three models are trained on the training split only. The test split is not used for fitting or validation-based configuration selection.

## Predictions and explainability

Predicted labels are returned by the selected fitted pipeline. The three probabilities are returned from that classifier's `predict_proba` output, mapped by its class IDs; confidence is the probability of the predicted class. Probabilities are model confidence estimates, not guarantees of correctness or a measure of calibrated real-world accuracy.

For each observed feature `j`, the contribution is:

```text
observed_feature_value_j
    × (log P(feature_j | predicted_class)
       - log P(feature_j | strongest_alternative_class))
```

`observed_feature_value` is the input count for count-based models and the actual transformed TF-IDF weight for TF-IDF. Positive contributions support the predicted class relative to the strongest alternative; negative contributions oppose it. Bigram features are actual adjacent two-word phrases. The UI displays only contributions generated by the selected fitted model and states when there is no non-zero feature evidence; it does not use a hand-authored sentiment dictionary.

## Evaluation methodology and actual results

Training metadata retains the original validation metrics. The final evaluation endpoint calculates accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1/support, correct/incorrect totals, and confusion matrices for both validation and test splits. Both are evaluated with the same frozen models and official label order (negative, neutral, positive). Test results are stored as a separate versioned JSON artifact under `backend/data/models/evaluations/`; they do not overwrite validation metadata. The Model Comparison page reads that saved evaluation report and ranks models by **test macro F1**.

Actual final held-out test results in the saved report:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Unigram Naive Bayes | 58.29% | 57.09% | 58.80% | **57.69%** |
| Bigram Naive Bayes | 49.02% | 50.16% | 47.93% | 43.59% |
| TF-IDF + Naive Bayes | 51.91% | 58.33% | 47.31% | 40.78% |

**Unigram Naive Bayes is the best model by test macro F1.** Bigram's negative-class recall is low (15.76%), while TF-IDF's negative-class recall is particularly low (2.97%); their stronger neutral-class recall illustrates class-specific errors hidden by overall accuracy. These are descriptive results, not a significance test.

Validation results are separate and must not be presented as test metrics:

| Model | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| Unigram Naive Bayes | 62.60% | 58.66% |
| Bigram Naive Bayes | 61.50% | 52.23% |
| TF-IDF + Naive Bayes | 61.55% | 45.35% |

Do not retrain or call the evaluation endpoint without the existing saved evaluation artifact merely to repopulate a demonstration: if a matching report is not present, the endpoint can perform the final test evaluation and cache it.

## Requirements and installation

Prerequisites:

- Python 3.11 or newer (3.12 is a suitable environment choice).
- Node.js and npm.
- Network access to Hugging Face the first time the dataset is loaded, unless the dataset cache has been pre-populated.
- Approximately 26 MiB for the current model artifacts, 16 KiB for the evaluation report, and 7 MiB for the current TweetEval cache. Allow additional room for Python packages, Hugging Face cache metadata, logs, and future artifacts.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
npm install
```

The backend requirements are listed in [backend/requirements.txt](./backend/requirements.txt); frontend dependencies are locked in `package-lock.json`.

## Configuration

The Vite frontend reads `VITE_API_BASE_URL` at build time. The root [.env.example](./.env.example) contains the development default:

```dotenv
VITE_API_BASE_URL=/api
```

With that default, the development Vite proxy forwards `/api` and `/health` to `http://127.0.0.1:8000`. In production, either configure the web server/reverse proxy to forward `/api` and `/health` to FastAPI, or set `VITE_API_BASE_URL` to the deployed backend API prefix (for example, `https://<your-api-host>/api`) before running the production build.

Backend process environment:

| Variable | Default | Description |
|---|---|---|
| `SENTISCOPE_DATA_DIR` | `backend/data` | Root directory for Hugging Face cache, trained model files/metadata, and saved evaluations. Use persistent storage in deployment. |
| `SENTISCOPE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated browser origins permitted by FastAPI. Set the exact production frontend origin(s); do not use `*` as a substitute for deployment configuration. |

For local development in PowerShell:

```powershell
$env:SENTISCOPE_DATA_DIR = "$PWD\backend\data"
$env:SENTISCOPE_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
```

No secrets or API keys are required. The application does not load a backend `.env` file automatically; configure backend variables in the process environment or hosting provider.

## Run locally

Start the backend from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). The backend health endpoint is `http://127.0.0.1:8000/health`; interactive API documentation is at `http://127.0.0.1:8000/docs`.

At startup, FastAPI loads the official dataset in the background. It reuses compatible model artifacts and trains missing/incompatible models on the train split. Wait for `/health` to show the dataset as available and all three models as trained before analysis. If the dataset could not load, restore connectivity and retry from Dataset Explorer or with `POST /api/dataset/reload`.

## API reference

| Method and path | Purpose |
|---|---|
| `GET /health` | API, dataset, and model readiness |
| `POST /api/analyze` | Predict from `{ "text": "...", "model": "unigram" }` |
| `GET /api/dataset/summary` | Actual dataset totals, split counts, distribution, and quality |
| `GET /api/dataset/preview` | Search/filter official records |
| `POST /api/dataset/reload` | Retry official dataset loading, then train missing/incompatible models |
| `GET /api/models/status` | Model status and training-time validation metrics |
| `POST /api/models/train` | Train missing/incompatible models; `{ "force": true }` retrains |
| `GET /api/models/evaluation` | Load the saved final evaluation report, or evaluate/cache when no compatible report exists |

The prediction API accepts 1–5,000 characters, rejects whitespace-only input, and returns a meaningful error when the model is unavailable. The frontend applies request timeouts, avoids submitting empty input, clears old results on input/model changes, and ignores stale in-flight prediction responses.

## Testing and production build

Run the complete backend tests and Python compilation check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
.\.venv\Scripts\python.exe -m compileall -q backend
```

Build the frontend for production:

```powershell
npm run build
```

The current backend tests cover preprocessing and bigram boundaries, classifier pipelines, probabilities and explanations, evaluation calculations/caching, and API error handling. The production build runs TypeScript project checks before bundling.

## Deployment readiness and manual steps

No deployment provider, production domain, container image, or hosting configuration is included in this repository. Deployment has **not** been performed.

For a compatible deployment:

1. Build the frontend with `npm ci` and `npm run build`. Configure `VITE_API_BASE_URL` at build time, or configure the static-site web server to proxy `/api` and `/health` to the backend.
2. Install backend packages with `python -m pip install -r backend/requirements.txt`.
3. Start FastAPI with `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1` (adjust the port to the provider's runtime environment). Use one worker: dataset/model readiness and training locks are process-local.
4. Configure `SENTISCOPE_DATA_DIR` on a persistent volume and `SENTISCOPE_CORS_ORIGINS` to the exact deployed frontend origin.
5. Ensure the deployment volume contains `cache/` (or allow startup network access to download TweetEval), `models/*.joblib`, their matching `models/*.json` metadata, and the saved `models/evaluations/*.json` report. Preserve the report and matching models: if artifact timestamps change during a move, the backend checks saved model/dataset provenance to reuse a compatible report before considering a new test evaluation.
6. Verify `GET /health`, the three model statuses, `GET /api/dataset/summary`, `GET /api/models/evaluation`, and a prediction before directing users to the frontend.
7. Protect a public deployment with the hosting platform's authentication/rate limiting as appropriate. CORS is not authentication. Only load trusted joblib model artifacts; joblib files are executable pickle-based data.

The current persisted model and dataset files are generated runtime data, are excluded from Git by `.gitignore`, and will **not** automatically be present in a clean deployment. Decide whether to seed them in a secure release artifact or prewarm a persistent volume. If they are absent, dataset download and/or model training will take place during startup; keep outbound Hugging Face access enabled or pre-populate its cache. Do not copy model metadata without its corresponding model artifact.

## Limitations and future work

- Naive Bayes uses a conditional-independence/bag-of-features assumption and does not model long-range context, sarcasm, or domain shift.
- Performance is evaluated on one fixed TweetEval split; the scores do not guarantee performance on other domains.
- Confidence values come from `predict_proba`; calibration was not separately measured.
- The current serving API has no built-in authentication or rate limiter.
- Dataset loading and model training occur during backend startup if persisted artifacts are absent; plan persistent storage and startup capacity accordingly.
- A production deployment still requires a provider, domains, TLS, reverse proxy/static hosting, persistent-volume configuration, and provider-specific health/startup settings.

Possible future work includes probability calibration, stronger contextual baselines, broader-domain evaluation, statistical uncertainty analysis, and deployment-specific observability and access control.
