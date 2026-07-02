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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_up():
    # Forces model loading at boot instead of on first request, so the
    # first real user doesn't pay the cold-start latency.
    get_artifacts()


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_artifacts()
        return HealthResponse(status="ok", models_loaded=True)
    except Exception:
        return HealthResponse(status="degraded", models_loaded=False)


@app.post("/predict/severity", response_model=SeverityPrediction)
def predict_severity(payload: AccidentInput):
    artifacts = get_artifacts()
    record = payload.model_dump(by_alias=True)

    try:
        X = build_feature_vector(
            record,
            task="severity",
            feature_columns=artifacts.severity_feature_columns,
            freq_maps=artifacts.severity_freq_maps,
            cols_to_scale=artifacts.severity_cols_to_scale,
            scaler=artifacts.severity_scaler,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    proba = float(artifacts.severity_model.predict_proba(X)[0, 1])
    threshold = artifacts.severity_threshold
    label = "High Severity" if proba >= threshold else "Low Severity"

    return SeverityPrediction(
        predicted_class=label,
        high_severity_probability=round(proba, 4),
        threshold_used=threshold,
    )


@app.post("/predict/duration", response_model=DurationPrediction)
def predict_duration(payload: AccidentInput):
    artifacts = get_artifacts()
    record = payload.model_dump(by_alias=True)

    X = build_feature_vector(
        record,
        task="duration",
        feature_columns=artifacts.duration_feature_columns,
        freq_maps=artifacts.duration_freq_maps,
        cols_to_scale=artifacts.duration_cols_to_scale,
        scaler=artifacts.duration_scaler,
    )

    pred_log = artifacts.duration_model.predict(X)[0]
    pred_minutes = float(np.expm1(pred_log))

    return DurationPrediction(predicted_duration_minutes=round(pred_minutes, 1))
