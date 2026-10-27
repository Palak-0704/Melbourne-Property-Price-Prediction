"""Recreate the training-time preprocessing pipeline from analysis.ipynb.

The saved CatBoost model was trained on a DataFrame whose object columns were
LabelEncoded and whose numeric columns were KNN-imputed and IQR/z-score clipped.
To score a single user request the same way, we replay those steps on the full
training CSV, capture the resulting encoders, imputers and clipping bounds, and
expose a `transform` function that maps raw user input to the model's feature
vector.

The artifacts are cached to `backend/artifacts.joblib` so the API starts fast.
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "Dataset", "melb_data.csv")
ARTIFACT_PATH = os.path.join(HERE, "artifacts.joblib")

# Column order the model expects (matches model.feature_names_).
FEATURE_ORDER = [
    "Suburb", "Rooms", "Type", "Method", "SellerG", "Distance", "Postcode",
    "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "CouncilArea",
    "Lattitude", "Longtitude", "Regionname", "Propertycount", "Month_sold",
    "Year_sold", "day_sold",
]

# Numeric columns as defined after feature extraction in the notebook
# (YearBuilt is imputed then dropped; kept here so KNN imputation sees the same
# neighbourhood structure as training).
NUM_COLS = [
    "Rooms", "Price", "Distance", "Postcode", "Bedroom2", "Bathroom", "Car",
    "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
    "Propertycount", "Month_sold", "Year_sold",
]
# Object columns that get LabelEncoded (Address/Date dropped, YearBuilt dropped).
OBJ_COLS = ["Suburb", "Type", "Method", "SellerG", "CouncilArea", "Regionname", "day_sold"]

# Columns the notebook drops before training.
DROP_COLS = ["Address", "Date", "YearBuilt"]


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Month_sold"] = df["Date"].dt.month
    df["Year_sold"] = df["Date"].dt.year
    df["day_sold"] = df["Date"].dt.day_name()
    df["Date"] = df["Date"].astype(str)
    return df


def _zscore_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Clipping bounds per numeric column using the notebook's z_score method
    (mean +/- 3 standard deviations), matching replace(df, col, "z_score")."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in NUM_COLS:
        mean = df[col].mean()
        std = df[col].std()
        bounds[col] = (mean - 3 * std, mean + 3 * std)
    return bounds


def build_artifacts() -> dict[str, Any]:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = _feature_engineer(df)

    # KNN-impute numeric columns (this also creates the "mean-like" values used
    # to fill missing Landsize/BuildingArea/Car for the training set).
    knn = KNNImputer(n_neighbors=5)
    df[NUM_COLS] = knn.fit_transform(df[NUM_COLS])

    # CouncilArea fill by suburb mode, else global mode.
    global_mode = df["CouncilArea"].mode()[0]
    df["CouncilArea"] = df.groupby("Suburb")["CouncilArea"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else global_mode)
    )

    # Clip outliers with z_score bounds (matches replace(df, col, "z_score")).
    bounds = _zscore_bounds(df)
    for col, (lo, hi) in bounds.items():
        df[col] = df[col].clip(lower=lo, upper=hi)

    df = df.drop(columns=DROP_COLS)

    # Keep a name-based snapshot for per-suburb form defaults before encoding.
    raw_for_stats = df[["Suburb", "CouncilArea", "Regionname", "Distance",
                        "Postcode", "Lattitude", "Longtitude", "Propertycount",
                        "Rooms", "Bedroom2", "Bathroom", "Car", "Landsize",
                        "BuildingArea"]].copy()

    encoders: dict[str, LabelEncoder] = {}
    for col in OBJ_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Numeric fallbacks (median) used when a user leaves an optional field blank.
    medians = {
        col: float(df[col].median())
        for col in NUM_COLS
        if col not in DROP_COLS
    }

    artifacts = {
        "encoders": encoders,
        "bounds": bounds,
        "medians": medians,
        "categorical_options": {
            "Suburb": sorted(encoders["Suburb"].classes_.tolist()),
            "Type": encoders["Type"].classes_.tolist(),
            "Method": encoders["Method"].classes_.tolist(),
            "SellerG": sorted(encoders["SellerG"].classes_.tolist()),
            "CouncilArea": sorted(encoders["CouncilArea"].classes_.tolist()),
            "Regionname": sorted(encoders["Regionname"].classes_.tolist()),
            "day_sold": encoders["day_sold"].classes_.tolist(),
        },
        "suburb_stats": _suburb_stats(raw_for_stats),
        "feature_ranges": {
            col: {"min": float(df[col].min()), "max": float(df[col].max()),
                  "median": float(df[col].median())}
            for col in FEATURE_ORDER
        },
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    return artifacts


def _suburb_stats(raw: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Median values per suburb (keyed by suburb name) to pre-fill the form and
    keep inputs realistic once the user picks a location. `raw` is the frame
    before label-encoding, so council/region names stay human readable."""
    stats: dict[str, dict[str, Any]] = {}
    for suburb, group in raw.groupby("Suburb"):
        council = group["CouncilArea"].mode()
        region = group["Regionname"].mode()
        stats[str(suburb)] = {
            "median_distance": float(group["Distance"].median()),
            "median_postcode": float(group["Postcode"].median()),
            "median_lat": float(group["Lattitude"].median()),
            "median_lon": float(group["Longtitude"].median()),
            "median_propertycount": float(group["Propertycount"].median()),
            "median_rooms": float(group["Rooms"].median()),
            "median_bedroom2": float(group["Bedroom2"].median()),
            "median_bathroom": float(group["Bathroom"].median()),
            "median_car": float(group["Car"].median()),
            "median_land": float(group["Landsize"].median()),
            "median_building": float(group["BuildingArea"].median()),
            "council": str(council.iloc[0]) if not council.empty else "",
            "region": str(region.iloc[0]) if not region.empty else "",
        }
    return stats


def load_artifacts(force: bool = False) -> dict[str, Any]:
    if force or not os.path.exists(ARTIFACT_PATH):
        return build_artifacts()
    return joblib.load(ARTIFACT_PATH)


def _encode(artifacts: dict[str, Any], col: str, value: Any) -> int:
    le: LabelEncoder = artifacts["encoders"][col]
    value = str(value)
    classes = list(le.classes_)
    if value in classes:
        return int(le.transform([value])[0])
    # Fall back to the most common class so a novel value never crashes inference.
    return int(np.argmax(np.bincount(le.transform(classes))))


def transform(payload: dict[str, Any], artifacts: dict[str, Any]) -> pd.DataFrame:
    """Map a raw user payload into the model's 20-feature row."""
    medians = artifacts["medians"]

    def num(key: str, default: float | None = None) -> float:
        raw = payload.get(key)
        if raw is None or raw == "":
            return float(medians.get(key, default if default is not None else 0.0))
        return float(raw)

    date_value = payload.get("Date") or "1/01/2017"
    date = pd.to_datetime(date_value, dayfirst=True, errors="coerce")
    if pd.isna(date):
        date = pd.Timestamp("2017-01-01")

    row: dict[str, Any] = {
        "Suburb": _encode(artifacts, "Suburb", payload.get("Suburb", "")),
        "Rooms": num("Rooms"),
        "Type": _encode(artifacts, "Type", payload.get("Type", "h")),
        "Method": _encode(artifacts, "Method", payload.get("Method", "S")),
        "SellerG": _encode(artifacts, "SellerG", payload.get("SellerG", "")),
        "Distance": num("Distance"),
        "Postcode": num("Postcode"),
        "Bedroom2": num("Bedroom2"),
        "Bathroom": num("Bathroom"),
        "Car": num("Car"),
        "Landsize": num("Landsize"),
        "BuildingArea": num("BuildingArea"),
        "CouncilArea": _encode(artifacts, "CouncilArea", payload.get("CouncilArea", "")),
        "Lattitude": num("Lattitude"),
        "Longtitude": num("Longtitude"),
        "Regionname": _encode(artifacts, "Regionname", payload.get("Regionname", "")),
        "Propertycount": num("Propertycount"),
        "Month_sold": int(date.month),
        "Year_sold": int(date.year),
        "day_sold": _encode(artifacts, "day_sold", date.day_name()),
    }

    # Keep values inside the training distribution (same clipping as training).
    for col, (lo, hi) in artifacts["bounds"].items():
        if col in row:
            row[col] = float(np.clip(row[col], lo, hi))

    return pd.DataFrame([row], columns=FEATURE_ORDER)


if __name__ == "__main__":
    arts = build_artifacts()
    print("artifacts written to", ARTIFACT_PATH)
    print("suburbs:", len(arts["categorical_options"]["Suburb"]))
