# Interactive XAI Healthcare Risk Screening

## Backend (FastAPI)

Run from project root:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:
- `http://127.0.0.1:8000/docs`

Auth configuration (required for API calls except `/`):

```bash
# format: token:user_id[,token2:user2]
export AUTH_TOKENS="dev-token:dev_user"
```

Then use `Authorization: Bearer <token>` from the frontend/API client.

Validation notes:
- `preferences.top_k` must be between `1` and `10`.
- `preferences.style` must be `simple` or `detailed`.
- Feature-level feedback (`relevant`, `irrelevant`, `confusing`) requires `feature_name` that matches a model feature.

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

## Backend Tests (Route-Level)

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

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


## Phase 4 explainability notes
- Explanations are SHAP attributions from the base XGBoost model.
- Displayed risk is produced by a calibrated model (`xgb_calibrated_screening`).
- The API now returns `meta.uncertainty_label` and `meta.top_k_semantics` to improve interpretation at the UI layer.
