"""
EcoWatt – Renewable Energy Forecasting Engine
==============================================
ML-based production forecasting for wind and solar plants.
Supports XGBoost ensemble models with advanced feature engineering.

Data sources:
  - Open-Meteo (free weather API, no key required)
  - Optional: custom database via env vars (see README)

Usage:
  python ecowatt.py train   --plant_id PLANT_A --start 2023-01-01 --end 2023-12-31
  python ecowatt.py forecast --plant_id PLANT_A --days 7
  python ecowatt.py schedule   # starts the automatic scheduling daemon
"""

import os
import sys
import json
import time
import logging
import argparse
import urllib.parse
from datetime import datetime, timedelta, date, timezone
from threading import Thread

import numpy as np
import pandas as pd
import requests
import requests_cache
import joblib
import schedule
import psutil
import xgboost as xgb
import openmeteo_requests
from retry_requests import retry
from scipy.stats import uniform

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import ProgrammingError
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EcoWatt")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DATABASE (optional – configure via environment variables)
# ---------------------------------------------------------------------------
# Set ECOWATT_DB_CONN_STR in your environment to point to a real database.
# If not set, database-dependent functions are skipped and a warning is shown.
#
# Example connection string (SQL Server via pyodbc):
#   DRIVER={ODBC Driver 17 for SQL Server};SERVER=<host>;DATABASE=<db>;UID=<user>;PWD=<pass>
#
_DB_ENGINE = None

def _get_db_engine():
    """Lazily initialise the SQLAlchemy engine from the environment variable."""
    global _DB_ENGINE
    if _DB_ENGINE is not None:
        return _DB_ENGINE
    conn_str = os.getenv("ECOWATT_DB_CONN_STR")
    if not conn_str:
        logger.warning(
            "ECOWATT_DB_CONN_STR not set – database features are disabled. "
            "Using CSV/demo data instead."
        )
        return None
    if not _DB_AVAILABLE:
        logger.warning("SQLAlchemy not installed – database features disabled.")
        return None
    try:
        _DB_ENGINE = create_engine(
            f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
        )
        logger.info("Database engine initialised successfully.")
        return _DB_ENGINE
    except Exception as exc:
        logger.error(f"Could not create DB engine: {exc}")
        return None

# ---------------------------------------------------------------------------
# PLANT CONFIGURATION
# ---------------------------------------------------------------------------

def load_plant_config(config_path: str = None) -> list[dict]:
    """Load plant definitions from a JSON file.

    The JSON should be a list of objects with the following keys:
        plant_id   : str  – unique identifier (used for model file names)
        lat        : float
        lon        : float
        plant_type : "wind" | "solar"
        rated_power_kw : float  – nameplate capacity in kW
        forecast_horizon_days : int (optional, default 7)

    See plants_config.json for a template.
    """
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "plants_config.json")
    try:
        with open(config_path, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.error(f"Plant config not found: {config_path}")
        return []
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON in plant config: {exc}")
        return []

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-6
    if not mask.any():
        return 100.0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

# ---------------------------------------------------------------------------
# DATA UTILITIES
# ---------------------------------------------------------------------------

def clean_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf and NaN with column medians."""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        med = df[col].median()
        df[col] = df[col].fillna(med if not np.isnan(med) else 0.0)
    return df


def compute_shear_exponent(df: pd.DataFrame) -> pd.DataFrame:
    """Wind shear exponent α = log(v80/v10) / log(80/10)."""
    if {"wind_speed_80m", "wind_speed_10m"}.issubset(df.columns):
        mask = (df["wind_speed_10m"] > 0.1) & (df["wind_speed_80m"] > 0.1)
        df["shear_exponent"] = np.nan
        df.loc[mask, "shear_exponent"] = (
            np.log(df.loc[mask, "wind_speed_80m"] / df.loc[mask, "wind_speed_10m"])
            / np.log(80 / 10)
        )
    return df


def merge_forecast_real(df_meteo: pd.DataFrame, df_real: pd.DataFrame) -> pd.DataFrame:
    """Inner-left merge on Timestamp after stripping timezone info."""
    for df in (df_meteo, df_real):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(None)
    return pd.merge(df_meteo, df_real, on="Timestamp", how="left")

# ---------------------------------------------------------------------------
# WIND POWER CURVE
# ---------------------------------------------------------------------------

def compute_theoretical_power(
    wind_speed: np.ndarray,
    rated_power_kw: float,
    cut_in: float = 3.5,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
) -> np.ndarray:
    """Theoretical power output (kWh per 15-min period) from a simplified power curve.

    Cubic ramp between cut-in and rated speed; flat at rated between rated
    speed and cut-out; zero elsewhere.
    """
    power = np.zeros_like(wind_speed, dtype=float)
    startup = (wind_speed >= cut_in) & (wind_speed < rated_speed)
    if startup.any():
        norm = (wind_speed[startup] - cut_in) / (rated_speed - cut_in)
        power[startup] = rated_power_kw * norm**3
    power[(wind_speed >= rated_speed) & (wind_speed <= cut_out)] = rated_power_kw
    return power * 0.25  # → kWh/quarter-hour


def add_power_curve_features(df: pd.DataFrame, rated_power_kw: float) -> pd.DataFrame:
    """Append theoretical power, efficiency and wind-regime bin to df."""
    if "wind_speed_80m" not in df.columns:
        return df
    df["theoretical_power"] = compute_theoretical_power(
        df["wind_speed_80m"].values, rated_power_kw
    )
    p95 = df["wind_speed_80m"].quantile(0.95)
    df["wind_efficiency"] = np.where(
        df["wind_speed_80m"] > 3.5,
        df["wind_speed_80m"] ** 3 / max(p95 ** 3, 1e-6),
        0,
    )
    df["wind_regime"] = pd.cut(
        df["wind_speed_80m"],
        bins=[0, 3.5, 7, 12, 18, 25, 50],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float)
    return df

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def feature_engineering(
    df: pd.DataFrame, plant_type: str, rated_power_kw: float
) -> pd.DataFrame:
    """Add temporal, meteorological and physics-based features.

    Plant-type-specific additions:
      wind  – power-curve features, wind-direction encoding, rolling stats,
               shear exponent
      solar – solar geometry (hour/month cyclical encoding covers this at
               basic level; extend with pvlib if desired)
    """
    logger.info("Applying feature engineering …")

    # Cyclic temporal features
    df["hour_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["day_of_year_norm"] = df["Timestamp"].dt.dayofyear / 365.0

    if plant_type == "wind":
        df = add_power_curve_features(df, rated_power_kw)
        df = compute_shear_exponent(df)
        if "wind_direction_80m" in df.columns:
            df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction_80m"]))
            df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction_80m"]))
        if "wind_speed_80m" in df.columns:
            df["wind_speed_ma3"] = (
                df["wind_speed_80m"].rolling(3, center=True, min_periods=1).mean()
            )
            df["wind_speed_std6"] = (
                df["wind_speed_80m"].rolling(6, center=True, min_periods=1).std()
            )

    df = clean_inf_nan(df)
    logger.info(f"Feature engineering done – shape: {df.shape}")
    return df


_FEATURE_MAP = {
    "wind": [
        "wind_speed_80m", "theoretical_power", "wind_efficiency", "wind_regime",
        "temperature_2m", "surface_pressure", "relative_humidity_2m",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_norm",
        # optional – added if present
        "wind_dir_sin", "wind_dir_cos", "wind_speed_ma3", "wind_speed_std6",
        "wind_gusts_10m", "shear_exponent",
    ],
    "solar": [
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "temperature_2m", "relative_humidity_2m",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_norm",
    ],
}


def select_features(plant_type: str, df: pd.DataFrame) -> list[str]:
    """Return the subset of desired features that are actually in df."""
    wanted = _FEATURE_MAP.get(plant_type, [])
    available = [f for f in wanted if f in df.columns]
    logger.info(f"Selected {len(available)} features out of {len(wanted)} desired.")
    return available

# ---------------------------------------------------------------------------
# DATA VALIDATION
# ---------------------------------------------------------------------------

def validate_and_clean(
    df: pd.DataFrame, target_col: str = "value", rated_power_kw: float = None
) -> pd.DataFrame:
    """Drop duplicates, clip out-of-range production values, impute NaNs."""
    logger.info(f"Data validation – initial shape: {df.shape}")
    df = df.drop_duplicates(subset=["Timestamp"], keep="first").sort_values("Timestamp")

    if target_col in df.columns and rated_power_kw:
        upper = rated_power_kw * 1.1  # 10 % tolerance above nameplate
        valid = (df[target_col] >= 0) & (df[target_col] <= upper)
        dropped = (~valid & df[target_col].notna()).sum()
        if dropped:
            logger.warning(f"Dropped {dropped} out-of-range production values.")
        df = df[valid | df[target_col].isna()].copy()

    df = clean_inf_nan(df)
    logger.info(f"Data validation done – final shape: {df.shape}")
    return df

# ---------------------------------------------------------------------------
# WEATHER DATA (Open-Meteo, no API key required)
# ---------------------------------------------------------------------------

_HOURLY_WIND = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_speed_80m", "wind_direction_80m", "wind_gusts_10m", "surface_pressure",
]
_HOURLY_SOLAR = [
    "temperature_2m", "relative_humidity_2m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "surface_pressure",
]


def _openmeteo_client():
    cache = requests_cache.CachedSession(".cache", expire_after=3600)
    return openmeteo_requests.Client(session=retry(cache, retries=5, backoff_factor=0.2))


def download_weather(
    start: date,
    end: date,
    lat: float,
    lon: float,
    plant_type: str,
    historical: bool = False,
) -> pd.DataFrame:
    """Download hourly weather from Open-Meteo.

    Uses the historical-forecast API when ``historical=True`` or when the
    requested period lies entirely in the past; otherwise uses the standard
    forecast API.
    """
    today = date.today()

    if historical or end < today:
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        logger.info(f"Using historical-forecast API ({start} → {end})")
    elif start >= today:
        url = "https://api.open-meteo.com/v1/forecast"
        logger.info(f"Using forecast API ({start} → {end})")
    else:
        # Mixed window: recurse for each part
        hist_df = download_weather(start, today - timedelta(1), lat, lon, plant_type, True)
        fut_df = download_weather(today, end, lat, lon, plant_type, False)
        parts = [df for df in (hist_df, fut_df) if not df.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    variables = _HOURLY_WIND if plant_type == "wind" else _HOURLY_SOLAR
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": variables,
    }

    try:
        client = _openmeteo_client()
        resp = client.weather_api(url, params=params)[0]
        hourly = resp.Hourly()
        data = {
            "Timestamp": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }
        for i, var in enumerate(variables):
            data[var] = hourly.Variables(i).ValuesAsNumpy()
        df = pd.DataFrame(data)
        logger.info(f"Downloaded {len(df)} hourly rows from Open-Meteo.")
        return df
    except Exception as exc:
        logger.error(f"Open-Meteo error: {exc}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# PRODUCTION DATA (database or CSV demo)
# ---------------------------------------------------------------------------

def load_production_db(
    plant_id: str, start: str, end: str
) -> pd.DataFrame:
    """Load measured production from the configured database.

    Returns an empty DataFrame (with a warning) when the database is not
    configured, so the pipeline can continue in demo mode.

    Expected table / view schema (adapt via ECOWATT_QUERY env var):
        Timestamp  datetime
        value      float   [kWh per quarter-hour]
    """
    engine = _get_db_engine()
    if engine is None:
        logger.warning("No database configured – returning empty production DataFrame.")
        return pd.DataFrame(columns=["Timestamp", "value"])

    custom_query = os.getenv("ECOWATT_QUERY")
    if custom_query:
        query_str = custom_query
    else:
        query_str = """
            SELECT DataMisura AS Timestamp, MisuraKWh AS value
            FROM production_view
            WHERE plant_id = :plant_id
              AND DataMisura BETWEEN :start AND :end
            ORDER BY DataMisura
        """

    try:
        df = pd.read_sql(
            text(query_str), engine,
            params={"plant_id": plant_id, "start": start, "end": f"{end} 23:59:59"},
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        # Resample to hourly averages for model training
        df = (
            df.set_index("Timestamp")["value"]
            .resample("1h").mean()
            .reset_index()
        )
        logger.info(f"Loaded {len(df)} production rows from database.")
        return df
    except Exception as exc:
        logger.error(f"Database read error: {exc}")
        return pd.DataFrame(columns=["Timestamp", "value"])


def load_production_csv(csv_path: str) -> pd.DataFrame:
    """Load production data from a CSV file (alternative to database).

    Expected columns: Timestamp, value
    Timestamp format: YYYY-MM-DD HH:MM(:SS)
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        logger.info(f"Loaded {len(df)} rows from {csv_path}.")
        return df
    except Exception as exc:
        logger.error(f"CSV load error: {exc}")
        return pd.DataFrame(columns=["Timestamp", "value"])

# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------

def _sample_weights(y: np.ndarray, rated_power_kw: float) -> np.ndarray:
    """Up-weight high-production periods to counteract under-estimation bias."""
    w = np.ones(len(y))
    w[y > rated_power_kw * 0.30 * 0.25] *= 3.0
    w[y > rated_power_kw * 0.60 * 0.25] *= 5.0
    return w


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    rated_power_kw: float,
):
    """Train an XGBoost regressor with sample weighting and early stopping.

    Returns (model, scaler, y_pred_val).
    """
    logger.info("Training XGBoost model …")
    X_train = clean_inf_nan(X_train)
    X_val = clean_inf_nan(X_val)

    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    Xva = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

    w = _sample_weights(y_train.values, rated_power_kw)
    dtrain = xgb.DMatrix(Xtr, label=y_train, weight=w)
    dval = xgb.DMatrix(Xva, label=y_val)

    params = {
        "learning_rate": 0.005,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "min_child_weight": 100,
        "gamma": 0.1,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    y_pred = np.clip(model.predict(dval), 0, None)
    logger.info("XGBoost training complete.")
    return model, scaler, y_pred


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rated_power_kw: float,
) -> dict:
    """Compute MAE, RMSE, R², sMAPE, bias and normalised variants."""
    r = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }
    cap = rated_power_kw * 0.25
    r["nrmse"] = r["rmse"] / cap * 100
    r["bias_pct"] = r["bias"] / cap * 100
    thr = cap * 0.05
    mask = y_true > thr
    if mask.sum() > 0:
        r["mape_significant"] = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )
    return r


def train(
    plant_id: str,
    lat: float,
    lon: float,
    plant_type: str,
    rated_power_kw: float,
    start: date,
    end: date,
    production_csv: str = None,
    validation_start: date = None,
) -> bool:
    """Full training pipeline: download weather → merge → feature engineering
    → split → train → evaluate → save model artifacts.
    """
    logger.info(f"Training [{plant_id}]  {start} → {end}")

    df_meteo = download_weather(start, end, lat, lon, plant_type, historical=True)
    if df_meteo.empty:
        logger.error("No weather data – aborting.")
        return False

    if production_csv:
        df_prod = load_production_csv(production_csv)
    else:
        df_prod = load_production_db(plant_id, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if df_prod.empty:
        logger.error("No production data – aborting.")
        return False

    merged = merge_forecast_real(df_meteo, df_prod)
    merged = validate_and_clean(merged, "value", rated_power_kw)
    merged = merged.dropna(subset=["value"])

    if len(merged) < 1000:
        logger.error(f"Insufficient data ({len(merged)} rows) – aborting.")
        return False

    merged = feature_engineering(merged, plant_type, rated_power_kw)
    feature_cols = select_features(plant_type, merged)
    if not feature_cols:
        logger.error("No valid features – aborting.")
        return False

    X = merged[feature_cols]
    y = merged["value"]

    # Train / validation split
    if validation_start and validation_start <= end:
        val_mask = merged["Timestamp"] >= pd.to_datetime(validation_start)
    else:
        split = merged["Timestamp"].max() - timedelta(days=30)
        val_mask = merged["Timestamp"] > split

    X_tr, y_tr = X[~val_mask], y[~val_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    logger.info(f"Train: {len(X_tr)} rows | Val: {len(X_va)} rows")

    try:
        model, scaler, y_pred_va = train_xgboost(X_tr, y_tr, X_va, y_va, rated_power_kw)
    except Exception as exc:
        logger.error(f"Training error: {exc}")
        return False

    metrics = evaluate(y_va.values, y_pred_va, rated_power_kw)
    _print_metrics(metrics, rated_power_kw)

    # Persist artifacts
    joblib.dump(scaler, f"scaler_{plant_id}.pkl")
    model.save_model(f"model_{plant_id}.xgb")
    meta = {
        "plant_id": plant_id,
        "plant_type": plant_type,
        "rated_power_kw": rated_power_kw,
        "features": feature_cols,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "training_period": f"{start} / {end}",
        "created_at": datetime.now().isoformat(),
    }
    with open(f"metadata_{plant_id}.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(f"✅ Training complete – model saved as model_{plant_id}.xgb")
    return True


def _print_metrics(m: dict, rated_power_kw: float):
    sep = "=" * 70
    print(f"\n{sep}\nMODEL PERFORMANCE\n{sep}")
    print(f"  MAE   : {m['mae']:.1f} kWh  ({m['mae'] / rated_power_kw * 100:.2f}% Pn)")
    print(f"  RMSE  : {m['rmse']:.1f} kWh  (nRMSE {m['nrmse']:.1f}%)")
    print(f"  R²    : {m['r2']:.3f}")
    print(f"  sMAPE : {m['smape']:.1f}%")
    print(f"  Bias  : {m['bias']:.1f} kWh  ({m['bias_pct']:.2f}%)")
    if "mape_significant" in m:
        print(f"  MAPE  : {m['mape_significant']:.1f}%  (>5% Pn)")
    print(sep)

# ---------------------------------------------------------------------------
# FORECASTING
# ---------------------------------------------------------------------------

def forecast(
    plant_id: str,
    lat: float,
    lon: float,
    plant_type: str,
    rated_power_kw: float,
    start: date,
    end: date,
) -> bool:
    """Run inference for the requested horizon and write results to output/."""
    logger.info(f"Forecasting [{plant_id}]  {start} → {end}")

    # Load model artifacts
    model_path = f"model_{plant_id}.xgb"
    scaler_path = f"scaler_{plant_id}.pkl"
    meta_path = f"metadata_{plant_id}.json"

    if not all(os.path.exists(p) for p in (model_path, scaler_path, meta_path)):
        logger.error("Model artifacts missing – run 'train' first.")
        return False

    scaler = joblib.load(scaler_path)
    model = xgb.Booster()
    model.load_model(model_path)
    with open(meta_path) as fh:
        meta = json.load(fh)
    feature_cols = meta["features"]

    # Download weather and build features
    today = date.today()
    df_meteo = download_weather(start, end, lat, lon, plant_type, historical=(end < today))
    if df_meteo.empty:
        logger.error("No weather data.")
        return False

    df_feat = feature_engineering(df_meteo.copy(), plant_type, rated_power_kw)
    missing = [f for f in feature_cols if f not in df_feat.columns]
    if missing:
        logger.warning(f"Missing features: {missing} – they will be filled with 0.")
        for f in missing:
            df_feat[f] = 0.0

    X = df_feat[feature_cols]
    X_scaled = scaler.transform(clean_inf_nan(X))
    dmat = xgb.DMatrix(X_scaled, feature_names=feature_cols)
    y_pred = np.clip(model.predict(dmat), 0, None)

    df_out = df_meteo[["Timestamp"]].copy()
    df_out["Timestamp"] = df_out["Timestamp"].dt.tz_convert(None)
    df_out["forecast_kwh"] = y_pred
    df_out["plant_id"] = plant_id

    # Save CSV output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"forecast_{plant_id}_{ts}.csv")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Forecast saved to {csv_path}")

    # Generate plot
    _plot_forecast(df_out, plant_id, ts)

    logger.info("✅ Forecast complete.")
    return True


def _plot_forecast(df: pd.DataFrame, plant_id: str, ts: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Timestamp"], df["forecast_kwh"], color="#2ecc71", linewidth=1.5,
            label="Forecast (kWh/h)")
    ax.set_title(f"Production Forecast – {plant_id}", fontsize=14)
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"forecast_plot_{plant_id}_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved to {path}")

# ---------------------------------------------------------------------------
# SCHEDULED DAEMON
# ---------------------------------------------------------------------------

def _scheduled_task(forecast_hour: int = 0):
    logger.info(f"Scheduled task – run @ {forecast_hour:02d}:00")
    plants = load_plant_config()
    if not plants:
        logger.warning("No plants configured.")
        return
    today = date.today()
    for plant in plants:
        horizon = min(plant.get("forecast_horizon_days", 7), 14)
        try:
            forecast(
                plant_id=plant["plant_id"],
                lat=plant["lat"],
                lon=plant["lon"],
                plant_type=plant["plant_type"],
                rated_power_kw=plant["rated_power_kw"],
                start=today,
                end=today + timedelta(days=horizon),
            )
        except Exception as exc:
            logger.error(f"Task error for {plant['plant_id']}: {exc}")


def run_scheduler():
    """Start the automatic forecasting daemon (runs in the foreground)."""
    logger.info("Starting scheduler …")

    # Run immediately at startup, then every 3 hours
    _scheduled_task(0)

    for h in [3, 6, 9, 12, 15, 18, 21]:
        schedule.every().day.at(f"{h:02d}:05").do(_scheduled_task, forecast_hour=h)

    logger.info("Scheduler active. Press Ctrl-C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="EcoWatt – ML-based renewable energy forecasting engine"
    )
    sub = p.add_subparsers(dest="command")

    # ---- train ----
    tr = sub.add_parser("train", help="Train a forecasting model")
    tr.add_argument("--plant_id",  required=True, help="Unique plant identifier")
    tr.add_argument("--lat",       required=True, type=float)
    tr.add_argument("--lon",       required=True, type=float)
    tr.add_argument("--plant_type", required=True, choices=["wind", "solar"])
    tr.add_argument("--rated_power_kw", required=True, type=float, help="Nameplate capacity (kW)")
    tr.add_argument("--start",     required=True, help="Training start date YYYY-MM-DD")
    tr.add_argument("--end",       required=True, help="Training end date YYYY-MM-DD")
    tr.add_argument("--production_csv", default=None,
                    help="Path to CSV with Timestamp,value columns (overrides DB)")
    tr.add_argument("--validation_start", default=None,
                    help="Start of validation window YYYY-MM-DD (default: last 30 days)")

    # ---- forecast ----
    fc = sub.add_parser("forecast", help="Run a production forecast")
    fc.add_argument("--plant_id",  required=True)
    fc.add_argument("--lat",       required=True, type=float)
    fc.add_argument("--lon",       required=True, type=float)
    fc.add_argument("--plant_type", required=True, choices=["wind", "solar"])
    fc.add_argument("--rated_power_kw", required=True, type=float)
    fc.add_argument("--days", default=7, type=int, help="Forecast horizon (days)")

    # ---- schedule ----
    sub.add_parser("schedule", help="Start the automatic scheduling daemon")

    return p.parse_args()


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "train":
        ok = train(
            plant_id=args.plant_id,
            lat=args.lat,
            lon=args.lon,
            plant_type=args.plant_type,
            rated_power_kw=args.rated_power_kw,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            production_csv=args.production_csv,
            validation_start=_parse_date(args.validation_start) if args.validation_start else None,
        )
        sys.exit(0 if ok else 1)

    elif args.command == "forecast":
        today = date.today()
        ok = forecast(
            plant_id=args.plant_id,
            lat=args.lat,
            lon=args.lon,
            plant_type=args.plant_type,
            rated_power_kw=args.rated_power_kw,
            start=today,
            end=today + timedelta(days=args.days),
        )
        sys.exit(0 if ok else 1)

    elif args.command == "schedule":
        run_scheduler()

    else:
        print("Run  python ecowatt.py --help  for usage.")
        sys.exit(1)
