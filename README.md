# US Accidents — Severity & Duration API

FastAPI service serving the two production models from the graduation project:
XGBoost severity classifier + XGBoost duration regressor.

## Why this is a separate project, not notebook cells

The notebook is for training (raw CSV, SMOTE, plotting, RandomizedSearchCV —
heavy, one-time). This service only needs to *load pickled artifacts and
transform one request at a time* — it should stay small, fast to cold-start,
and testable. Both this API and your Streamlit app import the exact same
`app/preprocessing.py`, so severity/duration logic is defined in one place,
not duplicated.

## Project structure

```
us-accidents-api/
├── app/
│   ├── main.py            # FastAPI routes: /health, /predict/severity, /predict/duration
│   ├── schemas.py          # Pydantic request/response models (also = auto-generated docs)
│   ├── preprocessing.py    # Feature engineering, mirrors the notebook step-by-step
│   └── model_loader.py     # Loads all .pkl artifacts once at startup, caches them
├── saved_models/           # Copy your exported .pkl files here (see below)
├── tests/
│   └── test_api.py         # Smoke tests (pytest)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── 00_ADD_TO_NOTEBOOK_export_artifacts.py   # paste this cell into your notebook
```

## Step 1 — Export the missing artifacts from the notebook

Your current save cell (cell 69) only dumps the models and scalers. Inference
also needs the frequency-encoding maps and the exact trained column order —
add `00_ADD_TO_NOTEBOOK_export_artifacts.py` as a new cell **after** your
models are trained (end of the notebook) and re-run it. It writes 9 files
into `saved_models/`:

```
severity_xgb_model.pkl          duration_xgb_model.pkl
severity_logreg_model.pkl       duration_scaler.pkl
severity_scaler.pkl             freq_maps_duration.pkl
freq_maps_severity.pkl          feature_columns_duration.pkl
feature_columns_severity.pkl    cols_to_scale_duration.pkl
cols_to_scale_severity.pkl      severity_decision_threshold.pkl
```

Copy all of them into this project's `saved_models/` folder.

**Version pinning matters here**: pickle/joblib artifacts are tied to the
scikit-learn/xgboost versions that created them (you already hit this once —
"scikit-learn version incompatibilities" — during training). Run
`pip freeze | grep -E "scikit-learn|xgboost"` in the notebook's environment
and match those exact versions in `requirements.txt` before deploying.

## Step 2 — Run locally

```bash
cd us-accidents-api
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI, or:

```bash
curl -X POST http://127.0.0.1:8000/predict/severity \
  -H "Content-Type: application/json" \
  -d '{"Start_Time":"2026-01-15T08:30:00","End_Time":"2026-01-15T09:30:00",
       "Distance(mi)":1.2,"Temperature(F)":45,"Wind_Chill(F)":40,"Humidity(%)":80,
       "Pressure(in)":29.8,"Visibility(mi)":6,"Wind_Speed(mph)":12,
       "Precipitation(in)":0,"Weather_Condition":"Cloudy","Wind_Direction":"W",
       "State":"CA","Sunrise_Sunset":"Day","Civil_Twilight":"Day",
       "Junction":true,"Traffic_Signal":true}'
```

Run tests:
```bash
pytest tests/ -v
```

## Step 3 — Deploy

Pick based on what you need for the submission:

**Render / Railway (easiest, free tier, good for a grad project demo)**
1. Push this repo to GitHub (models included, or pulled from a release/Git LFS if >100MB).
2. New Web Service → connect repo → build command `pip install -r requirements.txt`,
   start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

**Docker anywhere (Fly.io, a VM, etc.)**
```bash
docker build -t us-accidents-api .
docker run -p 8000:8000 us-accidents-api
```

**Connecting Streamlit to this API**
In your Streamlit app, call it like any REST API instead of loading the
`.pkl` files directly in Streamlit:
```python
import requests
resp = requests.post(f"{API_URL}/predict/severity", json=payload)
result = resp.json()
```
This also means Streamlit Cloud doesn't need xgboost/sklearn installed at all
if you split it this way — only `requests`. If you'd rather keep everything
in one Streamlit deployment (simpler for a single submission), you can skip
FastAPI and call `app/preprocessing.py` + `model_loader.py` directly from
Streamlit instead — the module is deployment-target agnostic either way.

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + confirms models loaded |
| `/predict/severity` | POST | Returns High/Low severity + probability |
| `/predict/duration` | POST | Returns predicted blockage duration (minutes) |
| `/docs` | GET | Auto-generated Swagger UI |
