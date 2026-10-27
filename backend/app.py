"""Flask API that serves the Melbourne property price prediction model.

Endpoints
---------
GET  /api/health      -> service + model status
GET  /api/metadata    -> form options, suburb defaults, model metrics
POST /api/predict     -> predicted price, range and comparable sales
GET  /                -> the single-page frontend
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import preprocess as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FRONTEND_DIR = os.path.join(ROOT, "frontend")
MODEL_PATH = os.path.join(ROOT, "final_model_catboost.pkl")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

MODEL = joblib.load(MODEL_PATH)
ARTIFACTS = P.load_artifacts()
DATA = pd.read_csv(P.DATA_PATH, encoding="latin-1")

# Model quality measured on the notebook's 80/20 split (see backend/README notes).
MODEL_METRICS = {"r2": 0.9652, "mae": 78723, "rmse": 108607, "mape": 8.06}


def _clean(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "CatBoostRegressor",
        "features": len(P.FEATURE_ORDER),
        "metrics": MODEL_METRICS,
    })


@app.get("/api/metadata")
def metadata():
    opts = ARTIFACTS["categorical_options"]
    ranges = ARTIFACTS["feature_ranges"]
    # Defaults for a fresh form: median of the whole dataset.
    defaults = {
        col: ranges[col]["median"] for col in P.FEATURE_ORDER if col in ranges
    }
    defaults.update({"Type": "h", "Method": "S", "Suburb": opts["Suburb"][0]})
    return jsonify(_clean({
        "options": opts,
        "ranges": ranges,
        "defaults": defaults,
        "suburb_stats": ARTIFACTS["suburb_stats"],
        "metrics": {**MODEL_METRICS, "records": int(len(DATA))},
    }))


def _comparables(payload: dict, limit: int = 5) -> list[dict]:
    """Find similar recorded sales to contextualise the prediction.

    Prefers same suburb, then same property type and a similar room count /
    land size, so the comparison set is meaningful rather than arbitrary.
    """
    df = DATA
    suburb = str(payload.get("Suburb", ""))
    ptype = str(payload.get("Type", "") or "")

    def as_float(key):
        try:
            v = payload.get(key)
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    rooms = as_float("Rooms")
    land = as_float("Landsize")

    subset = df[df["Suburb"] == suburb]
    if subset.empty:
        subset = df

    if ptype and ptype in set(subset["Type"]):
        typed = subset[subset["Type"] == ptype]
        if len(typed) >= limit:
            subset = typed

    rooms_ref = rooms if rooms is not None else float(subset["Rooms"].median())
    land_ref = land if land is not None else float(subset["Landsize"].median())

    # Normalised distance score across rooms and land size.
    land_scale = float(subset["Landsize"].quantile(0.75)) or 1.0
    subset = subset.assign(
        _dist=(subset["Rooms"] - rooms_ref).abs()
        + ((subset["Landsize"] - land_ref).abs() / land_scale)
    ).sort_values(["_dist", "Price"])

    out = []
    for _, row in subset.head(limit).iterrows():
        out.append({
            "suburb": row["Suburb"],
            "address": row["Address"],
            "rooms": int(row["Rooms"]),
            "type": row["Type"],
            "price": float(row["Price"]),
            "distance": float(row["Distance"]),
            "landsize": float(row["Landsize"]) if pd.notna(row["Landsize"]) else None,
        })
    return out


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        features = P.transform(payload, ARTIFACTS)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    price = float(MODEL.predict(features)[0])
    price = max(price, 0.0)

    # Uncertainty band derived from the model's held-out MAPE / RMSE.
    band = max(MODEL_METRICS["rmse"], price * MODEL_METRICS["mape"] / 100)
    low, high = max(price - band, 0.0), price + band

    comparables = _comparables(payload)
    comp_median = float(np.median([c["price"] for c in comparables])) if comparables else None

    return jsonify(_clean({
        "predicted_price": round(price, -2),
        "range": {"low": round(low, -2), "high": round(high, -2)},
        "confidence": MODEL_METRICS["r2"],
        "per_room": round(price / max(float(payload.get("Rooms") or 1), 1), -2),
        "comparable_median": round(comp_median, -2) if comp_median else None,
        "comparables": comparables,
        "model": "CatBoostRegressor",
    }))


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
