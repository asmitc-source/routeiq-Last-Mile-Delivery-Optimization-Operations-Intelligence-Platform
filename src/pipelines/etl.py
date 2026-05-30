"""
RouteIQ — ETL Pipeline
Loads raw data, validates schemas, applies feature engineering,
and writes production-ready analytical tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import PROCESSED_DIR, SAMPLE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ETL] %(message)s")
log = logging.getLogger(__name__)


class RouteIQETL:
    """
    Production ETL pipeline for RouteIQ analytics platform.
    Handles: ingestion → validation → enrichment → feature engineering → export
    """

    def __init__(self, use_sample: bool = True):
        self.data_dir = SAMPLE_DIR if use_sample else PROCESSED_DIR
        self.datasets = {}

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> "RouteIQETL":
        log.info("Loading datasets from %s", self.data_dir)
        for name in ["orders", "riders", "dark_stores", "weather"]:
            path = self.data_dir / f"{name}.parquet"
            if path.exists():
                self.datasets[name] = pd.read_parquet(path)
                log.info("  ✓ %s: %s rows", name, f"{len(self.datasets[name]):,}")
            else:
                log.warning("  ✗ %s not found — run data_generator first", name)
        return self

    # ── Validate ──────────────────────────────────────────────────────────────

    def validate(self) -> "RouteIQETL":
        orders = self.datasets.get("orders")
        if orders is None:
            raise ValueError("Orders dataset missing")

        required_cols = [
            "order_id", "created_at", "city", "rider_id", "dark_store_id",
            "actual_delivery_min", "is_late", "delivery_distance_km"
        ]
        missing = [c for c in required_cols if c not in orders.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Dedup
        before = len(orders)
        orders = orders.drop_duplicates("order_id")
        log.info("Dedup: %d → %d rows", before, len(orders))

        # Null checks
        null_pct = (orders.isnull().sum() / len(orders) * 100).round(2)
        for col, pct in null_pct[null_pct > 0].items():
            log.warning("  Null %s: %.1f%%", col, pct)

        self.datasets["orders"] = orders
        log.info("✓ Validation passed")
        return self

    # ── Feature Engineering ───────────────────────────────────────────────────

    def engineer_features(self) -> "RouteIQETL":
        orders = self.datasets["orders"].copy()
        riders = self.datasets.get("riders", pd.DataFrame())

        # Time features
        orders["hour_sin"] = np.sin(2 * np.pi * orders["hour_of_day"] / 24)
        orders["hour_cos"] = np.cos(2 * np.pi * orders["hour_of_day"] / 24)
        orders["day_sin"] = np.sin(2 * np.pi * orders["day_of_week"] / 7)
        orders["day_cos"] = np.cos(2 * np.pi * orders["day_of_week"] / 7)
        orders["week_of_year"] = orders["created_at"].dt.isocalendar().week.astype(int)
        orders["quarter"] = orders["created_at"].dt.quarter

        # Operational flags
        orders["is_express"] = (orders["order_type"] == "express").astype(int)
        orders["is_high_distance"] = (orders["delivery_distance_km"] > 5).astype(int)
        orders["distance_per_min"] = (
            orders["delivery_distance_km"] / orders["actual_delivery_min"].clip(1)
        ).round(4)
        orders["delay_ratio"] = (
            orders["delay_minutes"] / orders["sla_minutes"].clip(1)
        ).round(4)
        orders["efficiency_gap"] = (
            orders["route_efficiency_score"] - 1.0
        ).round(4)

        # Rider-level rolling stats (join)
        if len(riders) > 0:
            rider_stats = riders[["rider_id", "experience_months", "base_efficiency_score", "rating"]].copy()
            orders = orders.merge(rider_stats, on="rider_id", how="left")
        else:
            orders["experience_months"] = 24
            orders["base_efficiency_score"] = 0.75
            orders["rating"] = 4.0

        # Store load index (orders per store per day)
        orders["date"] = orders["created_at"].dt.date
        store_day_load = (
            orders.groupby(["dark_store_id", "date"])
            .size()
            .reset_index(name="store_daily_load")
        )
        orders = orders.merge(store_day_load, on=["dark_store_id", "date"], how="left")
        del orders["date"]

        # City-level congestion index
        city_hour_congestion = (
            orders.groupby(["city", "hour_of_day"])["congestion_factor"]
            .mean()
            .reset_index(name="city_hour_avg_congestion")
        )
        orders = orders.merge(city_hour_congestion, on=["city", "hour_of_day"], how="left")

        # Late probability score (engineered)
        orders["late_risk_score"] = (
            orders["congestion_factor"] * 0.35
            + orders["route_efficiency_score"] * 0.25
            + (orders["weather_delay_min"] / 30) * 0.20
            + orders["is_peak_hour"] * 0.20
        ).round(4)

        log.info("✓ Feature engineering complete: %d features", len(orders.columns))
        self.datasets["orders_enriched"] = orders
        return self

    # ── Aggregations ──────────────────────────────────────────────────────────

    def build_aggregations(self) -> "RouteIQETL":
        orders = self.datasets.get("orders_enriched", self.datasets.get("orders"))

        # ── City Daily KPIs ──
        city_daily = (
            orders.groupby(["city", orders["created_at"].dt.date.rename("date")])
            .agg(
                total_orders=("order_id", "count"),
                late_orders=("is_late", "sum"),
                avg_delivery_min=("actual_delivery_min", "mean"),
                avg_delay_min=("delay_minutes", "mean"),
                total_cost_inr=("total_cost_inr", "sum"),
                avg_congestion=("congestion_factor", "mean"),
                avg_route_efficiency=("route_efficiency_score", "mean"),
            )
            .reset_index()
        )
        city_daily["late_rate"] = (city_daily["late_orders"] / city_daily["total_orders"]).round(4)
        city_daily["cost_per_order"] = (city_daily["total_cost_inr"] / city_daily["total_orders"]).round(2)
        self.datasets["city_daily_kpis"] = city_daily

        # ── Rider Metrics ──
        rider_perf = (
            orders.groupby("rider_id")
            .agg(
                total_deliveries=("order_id", "count"),
                late_deliveries=("is_late", "sum"),
                avg_delivery_min=("actual_delivery_min", "mean"),
                avg_distance_km=("delivery_distance_km", "mean"),
                total_earnings_inr=("total_cost_inr", "sum"),
                avg_rating=("customer_rating", "mean"),
            )
            .reset_index()
        )
        rider_perf["late_rate"] = (rider_perf["late_deliveries"] / rider_perf["total_deliveries"]).round(4)
        rider_perf["utilization_score"] = np.clip(
            rider_perf["total_deliveries"] / (rider_perf["total_deliveries"].max() * 0.8), 0, 1
        ).round(4)
        self.datasets["rider_performance"] = rider_perf

        # ── Dark Store Load ──
        store_metrics = (
            orders.groupby("dark_store_id")
            .agg(
                total_orders=("order_id", "count"),
                late_orders=("is_late", "sum"),
                avg_prep_time=("prep_time_min", "mean"),
                peak_load=("store_daily_load", "max") if "store_daily_load" in orders.columns else ("order_id", "count"),
            )
            .reset_index()
        )
        store_metrics["sla_breach_rate"] = (store_metrics["late_orders"] / store_metrics["total_orders"]).round(4)
        stores = self.datasets.get("dark_stores", pd.DataFrame())
        if len(stores) > 0:
            store_metrics = store_metrics.merge(
                stores[["store_id", "city", "latitude", "longitude", "max_capacity", "tier"]],
                left_on="dark_store_id", right_on="store_id", how="left"
            )
        self.datasets["store_performance"] = store_metrics

        # ── Hourly Operational Stress ──
        hourly_stress = (
            orders.groupby(["city", "hour_of_day"])
            .agg(
                orders_per_hour=("order_id", "count"),
                late_rate=("is_late", "mean"),
                avg_congestion=("congestion_factor", "mean"),
                avg_weather_delay=("weather_delay_min", "mean"),
            )
            .reset_index()
        )
        hourly_stress["stress_score"] = (
            hourly_stress["orders_per_hour"] / hourly_stress["orders_per_hour"].max() * 0.4
            + hourly_stress["late_rate"] * 0.35
            + (hourly_stress["avg_congestion"] - 1) / 0.8 * 0.25
        ).round(4)
        self.datasets["hourly_stress"] = hourly_stress

        log.info("✓ Aggregations built: %d tables", 4)
        return self

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, out_dir: Path = None) -> "RouteIQETL":
        out = out_dir or self.data_dir
        save_keys = [
            "orders_enriched", "city_daily_kpis",
            "rider_performance", "store_performance", "hourly_stress"
        ]
        for key in save_keys:
            if key in self.datasets:
                path = out / f"{key}.parquet"
                self.datasets[key].to_parquet(path, index=False)
                log.info("  ✓ %s → %s (%.1f MB)", key, path.name,
                         path.stat().st_size / 1e6)
        return self

    # ── Run all ───────────────────────────────────────────────────────────────

    def run(self) -> dict:
        return (
            self
            .load()
            .validate()
            .engineer_features()
            .build_aggregations()
            .save()
            .datasets
        )


if __name__ == "__main__":
    etl = RouteIQETL(use_sample=True)
    datasets = etl.run()
    print("\nAvailable tables:", list(datasets.keys()))
