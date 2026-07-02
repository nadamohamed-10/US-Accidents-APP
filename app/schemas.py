from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class AccidentInput(BaseModel):
    """
    Raw-ish accident features, close to the original US_Accidents columns.
    All engineered features (Start_Hour, Season, Is_Rush_Hour, cyclical
    encodings, etc.) are derived server-side inside preprocessing.py, so the
    client never has to compute them.
    """
    Start_Time: datetime = Field(..., description="Accident start timestamp")
    End_Time: Optional[datetime] = Field(
        None, description="Accident end timestamp. Required only for the "
                           "severity model (Duration_Min is a feature there). "
                           "Not needed for duration prediction (that's what "
                           "you're predicting)."
    )

    Distance_mi: float = Field(..., alias="Distance(mi)")
    Temperature_F: float = Field(..., alias="Temperature(F)")
    Wind_Chill_F: Optional[float] = Field(None, alias="Wind_Chill(F)")
    Humidity_pct: float = Field(..., alias="Humidity(%)")
    Pressure_in: float = Field(..., alias="Pressure(in)")
    Visibility_mi: float = Field(..., alias="Visibility(mi)")
    Wind_Speed_mph: float = Field(..., alias="Wind_Speed(mph)")
    Precipitation_in: float = Field(0.0, alias="Precipitation(in)")

    Weather_Condition: str = "Clear"
    Wind_Direction: str = "CALM"
    State: str
    Sunrise_Sunset: str = Field("Day", description="'Day' or 'Night'")
    Civil_Twilight: str = Field("Day", description="'Day' or 'Night'")

    # Point-of-interest booleans (used to compute Road_Complexity)
    Amenity: bool = False
    Bump: bool = False
    Crossing: bool = False
    Give_Way: bool = False
    Junction: bool = False
    No_Exit: bool = False
    Railway: bool = False
    Roundabout: bool = False
    Station: bool = False
    Stop: bool = False
    Traffic_Calming: bool = False
    Traffic_Signal: bool = False

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Start_Time": "2026-01-15T08:30:00",
                "End_Time": "2026-01-15T09:30:00",
                "Distance(mi)": 1.2,
                "Temperature(F)": 45.0,
                "Wind_Chill(F)": 40.0,
                "Humidity(%)": 80.0,
                "Pressure(in)": 29.8,
                "Visibility(mi)": 6.0,
                "Wind_Speed(mph)": 12.0,
                "Precipitation(in)": 0.0,
                "Weather_Condition": "Cloudy",
                "Wind_Direction": "W",
                "State": "CA",
                "Sunrise_Sunset": "Day",
                "Civil_Twilight": "Day",
                "Junction": True,
                "Traffic_Signal": True,
            }
        }


class SeverityPrediction(BaseModel):
    predicted_class: str          # "Low Severity" / "High Severity"
    high_severity_probability: float
    threshold_used: float


class DurationPrediction(BaseModel):
    predicted_duration_minutes: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
