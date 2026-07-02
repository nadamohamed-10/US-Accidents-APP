import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AccidentInput, SeverityPrediction, DurationPrediction, HealthResponse
from app.preprocessing import build_feature_vector
from app.model_loader import get_artifacts

app = FastAPI(
    title="US Accidents Severity & Duration API",
    description="Predicts accident severity (High/Low) and expected traffic-blocking duration.",
    version="1.0.0",
)

# Loosen for local dev / Streamlit Cloud; tighten to your Streamlit app's
# origin before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "US Accidents API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.on_event("startup")
def _warm_up():
    # Forces model loading at boot instead of on first request, so the
    # first real user doesn't pay the cold-start latency.
    get_artifacts()


@app.get("/health", response_model=HealthResponse)
def health():
    ...
