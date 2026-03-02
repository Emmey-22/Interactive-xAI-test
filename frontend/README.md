# Frontend (React + Vite)

## Run locally

1. Install dependencies:
   - `npm.cmd install`
2. Set API URL:
   - copy `.env.example` to `.env`
   - optionally set `VITE_API_TOKEN` when backend auth is enabled
3. Start dev server:
   - `npm.cmd run dev`

The app expects FastAPI backend at `http://127.0.0.1:8000` by default.

Notes:
- Run **Predict** or **Explain** to establish an active `case_id`.
- Feature-level feedback is submitted with that active case automatically.
