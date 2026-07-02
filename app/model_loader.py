import joblib
from pathlib import Path
from functools import lru_cache

MODELS_DIR = Path(__file__).resolve().parent.parent / "saved_models"


class Artifacts:
    def __init__(self):
        self.severity_model = joblib.load(MODELS_DIR / "severity_xgb_model.pkl")
        self.severity_scaler = joblib.load(MODELS_DIR / "severity_scaler.pkl")
        self.severity_freq_maps = joblib.load(MODELS_DIR / "freq_maps_severity.pkl")
        self.severity_feature_columns = joblib.load(MODELS_DIR / "feature_columns_severity.pkl")
        self.severity_cols_to_scale = joblib.load(MODELS_DIR / "cols_to_scale_severity.pkl")
        self.severity_threshold = joblib.load(MODELS_DIR / "severity_decision_threshold.pkl")

        self.duration_model = joblib.load(MODELS_DIR / "duration_xgb_model.pkl")
        self.duration_scaler = joblib.load(MODELS_DIR / "duration_scaler.pkl")
        self.duration_freq_maps = joblib.load(MODELS_DIR / "freq_maps_duration.pkl")
        self.duration_feature_columns = joblib.load(MODELS_DIR / "feature_columns_duration.pkl")
        self.duration_cols_to_scale = joblib.load(MODELS_DIR / "cols_to_scale_duration.pkl")


@lru_cache(maxsize=1)
def get_artifacts() -> Artifacts:
    """Loaded once per process (cold start), then reused for every request."""
    return Artifacts()
