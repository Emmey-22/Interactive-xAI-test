# Interactive XAI Healthcare Risk Screening

## Backend (FastAPI)

Run from project root:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:
- `http://127.0.0.1:8000/docs`

## Frontend (React)

Run from `frontend/`:

```bash
npm.cmd install
npm.cmd run dev
```

Frontend URL:
- `http://127.0.0.1:5173`

Set API URL with `frontend/.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Optional (if backend auth is enabled):

```bash
VITE_API_TOKEN=<your-api-token>
```

## Backend Tests (Route-Level)

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

## Security Configuration (Phase 2)

By default, auth is disabled for local development (`AUTH_REQUIRED=false`).

Enable token auth with:

```bash
AUTH_REQUIRED=true
USER_TOKENS=user_a:token_a,user_b:token_b
```

Alternative JSON format:

```bash
USER_TOKENS_JSON={"user_a":"token_a","user_b":"token_b"}
```

Rate limiting defaults:

```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MIN=60
```

## Feedback Loop (Phase 3)

- `predict` and `explain` now return a real `case_id`.
- Feature-level feedback (`relevant`, `irrelevant`, `confusing`) is case-scoped and requires both:
  - `feature_name`
  - `case_id`
- Explanations no longer hard-hide disputed features; they remain visible and are marked as disputed.

## UI/Model Ops (Phase 4)

- Risk command center now uses threshold-aware tone and ring fill (based on latest predict/explain output).
- Added model metadata endpoint:
  - `GET /model/info`
- Predict response now includes:
  - `case_id`
  - `model_version`
- Training/calibration/SHAP scripts now use a shared artifact directory:
  - `ARTIFACT_DIR` (default: `artifacts`)
  - optional `MODEL_VERSION`

Example:

```bash
ARTIFACT_DIR=artifacts
MODEL_VERSION=20260301T120000Z
python framingham_xgb_train.py
python next_step_screening_calibrate.py
python framingham_step2_shap.py
```

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` runs:
- backend compile checks + pytest
- frontend build check

## Quick Deploy (Testing)

### 1) Deploy Backend to Render

1. In Render, choose **New +** -> **Blueprint**.
2. Connect this GitHub repo and select branch `main`.
3. Render will read `render.yaml` and create `interactive-xai-api`.
4. In Render service environment variables, set:
   - `CORS_ORIGINS=https://<your-frontend-domain>`
   - Optional for Vercel preview URLs: `CORS_ORIGIN_REGEX=^https://.*\\.vercel\\.app$`
5. Deploy and copy backend URL, for example:
   - `https://interactive-xai-api.onrender.com`

### 2) Deploy Frontend to Vercel

1. Import the same repo in Vercel.
2. Set **Root Directory** to `frontend`.
3. Add environment variable:
   - `VITE_API_BASE_URL=https://<your-render-backend-url>`
4. Deploy.

### 3) Verify

1. Open frontend URL.
2. Enter a `User ID`.
3. Run **Predict Risk** then **Explain Drivers**.
4. Confirm analytics cards are user-specific.
