"""
Feature engineering for inference.

This intentionally mirrors, step by step, the transformations from the
training notebook (engineer_features -> add_environment_features -> OHE ->
frequency encoding -> scaling), so a single accident record produces the
same feature vector shape/values a training row would have had.

Kept as plain functions (no sklearn Pipeline) on purpose: it matches the
notebook's structure almost 1:1, which makes it easy to audit against the
notebook if the model is retrained later.
"""
import numpy as np
import pandas as pd

POI_COLS = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
            "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
            "Traffic_Signal"]

OHE_COLS = ["Season", "Lighting_Phase", "Time_of_Day"]
FREQ_COLS = ["State", "Weather_Condition", "Wind_Direction"]


def _base_engineering(df: pd.DataFrame, need_duration: bool) -> pd.DataFrame:
    df = df.copy()

    df["Start_Hour"] = df["Start_Time"].dt.hour.fillna(0).astype(int)
    df["Time_of_Day"] = pd.cut(
        df["Start_Hour"], bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"], right=False
    )

    if need_duration:
        # Severity model uses Duration_Min as an input feature.
        df["Duration_Min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
        df["Impact_Ratio"] = df["Distance(mi)"] / (df["Duration_Min"] + 1e-6)
    # else: nothing to do — the duration model no longer uses Impact_Ratio at all,
    # since it was leaking the target it's trying to predict.

    df["Is_Weekend"] = df["Start_Time"].dt.dayofweek >= 5
    df["Is_Rush_Hour"] = ((~df["Is_Weekend"]) & (df["Start_Hour"].isin([7, 8, 9, 16, 17, 18]))).astype(int)
    df["Is_Freezing"] = (df["Temperature(F)"] <= 32.0).astype(int)
    df["Is_Bad_Weather"] = (
        (df["Visibility(mi)"] < 3.0) | (df["Wind_Speed(mph)"] > 25.0) | (df["Precipitation(in)"] > 0.1)
    ).astype(int)

    month = df["Start_Time"].dt.month
    df["Season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"], default="Fall"
    )

    conditions = [
        (df["Sunrise_Sunset"] == "Day") & (df["Civil_Twilight"] == "Day"),
        (df["Sunrise_Sunset"] == "Night") & (df["Civil_Twilight"] == "Night"),
    ]
    df["Lighting_Phase"] = np.select(conditions, ["Day", "Night"], default="Twilight")

    existing_poi = [c for c in POI_COLS if c in df.columns]
    df["Road_Complexity"] = df[existing_poi].astype(int).sum(axis=1)

    df["Start_Hour_Sin"] = np.sin(2 * np.pi * df["Start_Hour"] / 24.0)
    df["Start_Hour_Cos"] = np.cos(2 * np.pi * df["Start_Hour"] / 24.0)
    df["Month_Sin"] = np.sin(2 * np.pi * month / 12.0)
    df["Month_Cos"] = np.cos(2 * np.pi * month / 12.0)
    day_series = df["Start_Time"].dt.dayofweek
    df["Day_Sin"] = np.sin(2 * np.pi * day_series / 7.0)
    df["Day_Cos"] = np.cos(2 * np.pi * day_series / 7.0)

    return df


def build_feature_vector(
    record: dict,
    *,
    task: str,
    feature_columns: list,
    freq_maps: dict,
    cols_to_scale: list,
    scaler,
) -> pd.DataFrame:
    """
    task: "severity" or "duration"
    Returns a single-row DataFrame with columns exactly matching
    `feature_columns` (the training-time column order), ready for
    model.predict().
    """
    need_duration = task == "severity"
    if need_duration and record.get("End_Time") is None:
        raise ValueError("End_Time is required for severity prediction (used to derive Duration_Min).")

    df = pd.DataFrame([record])
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    if need_duration:
        df["End_Time"] = pd.to_datetime(df["End_Time"])

    df = _base_engineering(df, need_duration=need_duration)

    # One-hot encode, matching training's drop_first=True behavior. Since we
    # can't drop_first on a single row reliably, we instead build every
    # possible dummy column implied by feature_columns and set the right one
    # to 1 -- equivalent result, robust for a single record.
    for base_col in OHE_COLS:
        val = df.at[0, base_col]
        dummy_col = f"{base_col}_{val}"
        for col in feature_columns:
            if col.startswith(f"{base_col}_"):
                df[col] = 1 if col == dummy_col else 0

    # Frequency encoding using the maps captured at training time. Unseen
    # categories fall back to the mean frequency, same as the notebook does
    # for the test set.
    for col in FREQ_COLS:
        fmap = freq_maps.get(col, {})
        mean_freq = float(np.mean(list(fmap.values()))) if fmap else 0.0
        raw_val = df.at[0, col]
        df[f"{col}_Freq"] = fmap.get(raw_val, mean_freq)

    # Scale numeric columns with the fitted scaler (same columns/order used at fit time).
    present_scale_cols = [c for c in cols_to_scale if c in df.columns]
    df[present_scale_cols] = scaler.transform(df[present_scale_cols])

    # Reindex to the exact trained column set/order. Anything the pipeline
    # didn't produce (e.g. a dummy for a category never seen in training)
    # is filled with 0, mirroring how pd.get_dummies would have handled it.
    out = df.reindex(columns=feature_columns, fill_value=0)
    return out
