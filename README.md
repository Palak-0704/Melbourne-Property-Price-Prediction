# Melbourne Property Price Prediction

An end-to-end property valuation website powered by a gradient-boosted
**CatBoost** regression model. Enter a dwelling's details in the browser and the
model returns an estimated sale price with a confidence range and comparable
recent sales.

## What's here

| Path | Purpose |
| --- | --- |
| `analysis.ipynb` | Original notebook: EDA, feature engineering, model comparison, hyper-parameter tuning, final CatBoost model. |
| `Dataset/melb_data.csv` | 13,580 recorded Melbourne property sales. |
| `final_model_catboost.pkl` | Trained CatBoost model (depth 7, 1,000 trees, RMSE loss). |
| `backend/preprocess.py` | Rebuilds the notebook's preprocessing (feature extraction, KNN imputation, z-score outlier clipping, label encoding) and exposes `transform()` to turn raw form input into the model's 20-feature vector. |
| `backend/app.py` | Flask API + static frontend server. |
| `frontend/` | Single-page website (`index.html`, `styles.css`, `app.js`). |
| `run.sh` | One-command launcher. |

## Model

The saved model is a `CatBoostRegressor` trained on 20 engineered features:

```
Suburb, Rooms, Type, Method, SellerG, Distance, Postcode, Bedroom2, Bathroom,
Car, Landsize, BuildingArea, CouncilArea, Lattitude, Longtitude, Regionname,
Propertycount, Month_sold, Year_sold, day_sold
```

Held-out performance on the notebook's 80/20 split:

| Metric | Value |
| --- | --- |
| R² | 0.965 |
| MAE | $78,723 |
| RMSE | $108,607 |
| MAPE | 8.1% |

`backend/preprocess.py` faithfully replays the training pipeline (KNN imputation
with `n_neighbors=5`, mean ± 3σ outlier clipping, `LabelEncoder` per categorical
column) and caches the resulting encoders, clipping bounds and per-suburb medians
to `backend/artifacts.joblib`, so inference matches training exactly.

## Running it

```bash
# 1. install dependencies
pip install -r backend/requirements.txt

# 2. start the website (builds artifacts on first run)
./run.sh
# or pick a port: PORT=12000 ./run.sh
```

Then open <http://localhost:8000>.

## API

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service and model status. |
| `GET` | `/api/metadata` | Form options, per-suburb defaults, feature ranges, model metrics. |
| `POST` | `/api/predict` | Predicted price, confidence range, per-room value and comparable sales. |
| `GET` | `/` | The website. |

Example request:

```bash
curl -X POST localhost:8000/api/predict -H 'Content-Type: application/json' -d '{
  "Suburb":"Richmond","Rooms":3,"Type":"h","Method":"S","SellerG":"Biggin",
  "Distance":3.0,"Postcode":3121,"Bedroom2":3,"Bathroom":2,"Car":1,
  "Landsize":300,"BuildingArea":150,"CouncilArea":"Yarra",
  "Lattitude":-37.82,"Longtitude":145.0,
  "Regionname":"Northern Metropolitan","Propertycount":14949,"Date":"2017-06-01"
}'
```

Any omitted field falls back to the dataset median, so partial payloads still
produce a prediction. Values outside the training distribution are clipped to the
same bounds used during training.

## How the frontend connects to the model

1. On load, `app.js` fetches `/api/metadata` to populate suburb autocomplete,
   region/council/agency/type/method selects and the headline model metrics.
2. Choosing a suburb pre-fills realistic location defaults from that suburb's
   recorded medians.
3. Submitting the form `POST`s the payload to `/api/predict`; the Flask handler
   calls `preprocess.transform()` and then `model.predict()`.
4. The response renders the price, a confidence band (derived from the model's
   held-out MAPE/RMSE), per-room value and a table of comparable sales.

## Notes

- Estimates are indicative only and based on 2016–2017 Melbourne sales data.
- The confidence band reflects the model's average error, not a per-property
  guarantee.
