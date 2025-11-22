# Explo MSI — Quickstart

## Pre-reqs
- Docker & docker-compose
- Or Python 3.10+ and Node 18 if running locally

## Steps (Docker)
1. Place trained pipelines in `backend/app/models/`:
   - ucs_pipeline.joblib
   - slope_failure_pipeline.joblib
2. From repo root run:
   docker-compose up --build
3. Frontend: http://localhost:3000
   Backend: http://localhost:8000

## Steps (Local dev)

# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm start
