"""
demo.py  –  Self-contained demonstration of the EcoWatt forecasting pipeline
=============================================================================
Generates synthetic wind-production data, trains a model on one year of it,
and produces a 7-day forecast – all without any external database or API key.

Usage:
    python demo.py

The script writes CSV and PNG outputs to the ./output/ directory.
"""

import os
import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EcoWattDemo")

# ---------------------------------------------------------------------------
# 1. Synthetic production data generator
# ---------------------------------------------------------------------------

def generate_synthetic_wind_production(
    start: date,
    end: date,
    rated_power_kw: float = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with hourly synthetic wind production (kWh/h).

    The synthetic model combines:
      - seasonal variation (higher in winter)
      - diurnal variation
      - Weibull-distributed wind-speed noise
      - a simplified cubic power curve
      - random zero-production periods (maintenance / low-wind events)
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    n = len(idx)

    day_of_year = idx.day_of_year / 365.0
    hour = idx.hour / 24.0

    # Seasonal and diurnal wind-speed envelope (m/s)
    seasonal = 8 + 4 * np.cos(2 * np.pi * (day_of_year - 0.1))  # ~12 m/s in winter
    diurnal  = 1 + 0.5 * np.sin(2 * np.pi * (hour - 0.25))
    base_speed = seasonal * diurnal

    # Weibull noise (shape=2 → Rayleigh, roughly realistic for wind)
    noise = rng.weibull(2, n) * 3
    wind_speed = np.clip(base_speed + noise - 3, 0, 30)

    # Simplified cubic power curve (kWh per hour)
    cut_in, rated_speed, cut_out = 3.5, 12.0, 25.0
    power = np.zeros(n)
    startup = (wind_speed >= cut_in) & (wind_speed < rated_speed)
    rated   = (wind_speed >= rated_speed) & (wind_speed <= cut_out)
    power[startup] = rated_power_kw * ((wind_speed[startup] - cut_in) / (rated_speed - cut_in)) ** 3
    power[rated]   = rated_power_kw

    # Random zero events (maintenance, curtailment)
    zero_mask = rng.random(n) < 0.02  # ~2% of hours
    power[zero_mask] = 0.0

    df = pd.DataFrame({"Timestamp": idx, "value": power})
    df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
    return df


# ---------------------------------------------------------------------------
# 2. Run the demo
# ---------------------------------------------------------------------------

def main():
    from ecowatt import train, forecast

    PLANT_ID       = "DEMO_WIND"
    PLANT_TYPE     = "wind"
    RATED_POWER_KW = 5000.0
    LAT, LON       = 44.0, 11.0   # Northern Italy (generic)

    os.makedirs("output", exist_ok=True)

    # --- Generate synthetic data and save to CSV ---
    train_start = date(2023, 1, 1)
    train_end   = date(2023, 12, 31)

    logger.info("Generating synthetic production data …")
    df_prod = generate_synthetic_wind_production(train_start, train_end, RATED_POWER_KW)
    csv_path = "output/demo_production.csv"
    df_prod.to_csv(csv_path, index=False)
    logger.info(f"Synthetic data saved to {csv_path}  ({len(df_prod)} rows)")

    # --- Train ---
    logger.info("Starting training …")
    ok = train(
        plant_id=PLANT_ID,
        lat=LAT,
        lon=LON,
        plant_type=PLANT_TYPE,
        rated_power_kw=RATED_POWER_KW,
        start=train_start,
        end=train_end,
        production_csv=csv_path,
    )
    if not ok:
        logger.error("Training failed – check logs above.")
        return

    # --- Forecast ---
    today = date.today()
    logger.info("Running 7-day forecast …")
    ok = forecast(
        plant_id=PLANT_ID,
        lat=LAT,
        lon=LON,
        plant_type=PLANT_TYPE,
        rated_power_kw=RATED_POWER_KW,
        start=today,
        end=today + timedelta(days=7),
    )
    if ok:
        logger.info("✅ Demo complete – check the output/ directory for results.")
    else:
        logger.error("Forecast step failed.")


if __name__ == "__main__":
    main()
