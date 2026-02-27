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

## Backend Tests (Route-Level)

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```
