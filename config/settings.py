"""
RouteIQ Platform — Global Configuration
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models" / "artifacts"
LOGS_DIR = ROOT_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Simulation Parameters ────────────────────────────────────────────────────
SIMULATION = {
    "n_orders": 1_000_000,
    "n_riders": 4_500,
    "n_dark_stores": 48,
    "n_cities": 8,
    "start_date": "2023-01-01",
    "end_date": "2024-06-30",
    "random_seed": 42,
}

# ─── Cities & Zones ───────────────────────────────────────────────────────────
CITIES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "zones": 12, "weight": 0.22},
    "Delhi": {"lat": 28.7041, "lon": 77.1025, "zones": 14, "weight": 0.20},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "zones": 10, "weight": 0.18},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "zones": 8, "weight": 0.12},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "zones": 7, "weight": 0.10},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "zones": 6, "weight": 0.08},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "zones": 5, "weight": 0.06},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "zones": 6, "weight": 0.04},
}

# ─── SLA Thresholds ──────────────────────────────────────────────────────────
SLA = {
    "express_minutes": 30,
    "standard_minutes": 60,
    "scheduled_minutes": 120,
    "breach_penalty_inr": 50,
    "critical_breach_threshold": 0.15,  # 15% breach rate triggers alert
}

# ─── Operational Thresholds ──────────────────────────────────────────────────
OPS = {
    "rider_max_daily_deliveries": 40,
    "rider_utilization_target": 0.75,
    "dark_store_max_capacity": 500,
    "peak_hour_multiplier": 1.8,
    "congestion_threshold_kmh": 15,
    "weather_delay_minutes": {"rain": 12, "heavy_rain": 28, "fog": 8, "storm": 45},
}

# ─── ML Model Settings ───────────────────────────────────────────────────────
MODEL = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "target_accuracy": 0.86,
    "xgb_params": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    },
}

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD = {
    "page_title": "RouteIQ — Delivery Operations Intelligence",
    "page_icon": "🚚",
    "layout": "wide",
    "theme_primary": "#0EA5E9",
    "theme_accent": "#F59E0B",
    "theme_danger": "#EF4444",
    "theme_success": "#10B981",
}
