"""
RouteIQ — Synthetic Data Generator
Generates production-scale logistics data: 1M+ delivery records with
realistic operational patterns, geo-coordinates, weather, and rider data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import SIMULATION, CITIES, SLA, OPS


class RouteIQDataGenerator:
    """
    Generates realistic last-mile delivery operational data.
    Simulates: orders, riders, dark-stores, routes, weather, GPS traces.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        np.random.seed(seed)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _date_range(self):
        start = pd.Timestamp(SIMULATION["start_date"])
        end = pd.Timestamp(SIMULATION["end_date"])
        return start, end

    def _sample_timestamps(self, n: int):
        start, end = self._date_range()
        span_secs = int((end - start).total_seconds())
        offsets = self.rng.integers(0, span_secs, size=n)
        base = start.value // 10**9
        ts = pd.to_datetime(base + offsets, unit="s")
        return ts

    def _hour_weight(self, hours: np.ndarray) -> np.ndarray:
        """Realistic delivery demand curve — peaks at lunch & dinner."""
        weights = np.ones(len(hours), dtype=float)
        # Morning ramp
        weights[(hours >= 8) & (hours < 10)] = 1.5
        # Lunch peak
        weights[(hours >= 11) & (hours < 14)] = 2.8
        # Afternoon lull
        weights[(hours >= 14) & (hours < 17)] = 1.2
        # Dinner peak
        weights[(hours >= 18) & (hours < 21)] = 3.2
        # Night tail
        weights[(hours >= 21) & (hours < 23)] = 1.8
        # Deep night
        weights[(hours >= 23) | (hours < 6)] = 0.3
        return weights

    # ── Dark Stores ───────────────────────────────────────────────────────────

    def generate_dark_stores(self) -> pd.DataFrame:
        n = SIMULATION["n_dark_stores"]
        rows = []
        store_id = 1
        for city, meta in CITIES.items():
            n_stores = max(2, int(n * meta["weight"]))
            for _ in range(n_stores):
                lat_offset = self.rng.uniform(-0.15, 0.15)
                lon_offset = self.rng.uniform(-0.15, 0.15)
                rows.append({
                    "store_id": f"DS{store_id:04d}",
                    "city": city,
                    "latitude": round(meta["lat"] + lat_offset, 6),
                    "longitude": round(meta["lon"] + lon_offset, 6),
                    "max_capacity": self.rng.integers(300, 700),
                    "area_sqft": self.rng.integers(1500, 6000),
                    "operational_since": pd.Timestamp("2022-01-01") + timedelta(
                        days=int(self.rng.integers(0, 365))
                    ),
                    "zone_coverage_km": round(self.rng.uniform(2.5, 8.0), 1),
                    "tier": self.rng.choice(["Tier-1", "Tier-2", "Tier-3"],
                                             p=[0.4, 0.4, 0.2]),
                })
                store_id += 1
        return pd.DataFrame(rows)

    # ── Riders ────────────────────────────────────────────────────────────────

    def generate_riders(self, dark_stores: pd.DataFrame) -> pd.DataFrame:
        n = SIMULATION["n_riders"]
        store_ids = dark_stores["store_id"].tolist()
        city_map = dict(zip(dark_stores["store_id"], dark_stores["city"]))

        assigned_stores = self.rng.choice(store_ids, size=n)
        experience_months = self.rng.integers(1, 60, size=n)

        df = pd.DataFrame({
            "rider_id": [f"R{i:06d}" for i in range(1, n + 1)],
            "assigned_store": assigned_stores,
            "city": [city_map[s] for s in assigned_stores],
            "experience_months": experience_months,
            "vehicle_type": self.rng.choice(
                ["2W-Petrol", "2W-EV", "Cycle", "3W-EV"],
                size=n, p=[0.55, 0.25, 0.10, 0.10]
            ),
            "rating": np.round(self.rng.uniform(3.2, 5.0, size=n), 2),
            "join_date": [
                pd.Timestamp("2022-01-01") + timedelta(days=int(d))
                for d in self.rng.integers(0, 540, size=n)
            ],
            "shift_type": self.rng.choice(
                ["Morning", "Afternoon", "Evening", "Night", "Full-Day"],
                size=n, p=[0.25, 0.20, 0.25, 0.10, 0.20]
            ),
            "is_active": self.rng.choice([True, False], size=n, p=[0.88, 0.12]),
        })
        # Experience-based performance bonus
        df["base_efficiency_score"] = np.clip(
            0.55 + experience_months / 120 + self.rng.normal(0, 0.05, size=n), 0.4, 1.0
        ).round(3)
        return df

    # ── Weather Events ────────────────────────────────────────────────────────

    def generate_weather_events(self) -> pd.DataFrame:
        start, end = self._date_range()
        dates = pd.date_range(start, end, freq="h")
        n = len(dates)

        # Monsoon seasons (Jun–Sep) have higher rain probability
        month = dates.month
        is_monsoon = month.isin([6, 7, 8, 9])

        conditions = []
        for i, (d, mon) in enumerate(zip(dates, is_monsoon)):
            r = self.rng.random()
            if mon:
                if r < 0.35:
                    cond = "heavy_rain"
                elif r < 0.60:
                    cond = "rain"
                else:
                    cond = "clear"
            elif month[i] in [11, 12, 1]:
                if r < 0.08:
                    cond = "fog"
                elif r < 0.15:
                    cond = "storm"
                else:
                    cond = "clear"
            else:
                cond = "rain" if r < 0.10 else "clear"
            conditions.append(cond)

        delay_map = {"clear": 0, "rain": 12, "heavy_rain": 28, "fog": 8, "storm": 45}
        visibility_map = {"clear": 10, "rain": 5, "heavy_rain": 2, "fog": 0.5, "storm": 1}

        df = pd.DataFrame({
            "timestamp": dates,
            "condition": conditions,
            "temperature_c": np.round(
                self.rng.uniform(18, 38, n) - (np.array(conditions) == "rain").astype(int) * 5, 1
            ),
            "humidity_pct": np.round(self.rng.uniform(40, 95, n), 1),
            "wind_kmh": np.round(self.rng.uniform(5, 50, n), 1),
            "visibility_km": [visibility_map[c] for c in conditions],
            "delay_impact_min": [delay_map[c] + self.rng.integers(0, 5) for c in conditions],
        })
        return df

    # ── Orders (Main Fact Table) ──────────────────────────────────────────────

    def generate_orders(
        self,
        dark_stores: pd.DataFrame,
        riders: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        n = SIMULATION["n_orders"]
        print(f"  Generating {n:,} order records...")

        # Sample order creation timestamps with realistic demand curve
        raw_ts = self._sample_timestamps(n)
        hours = raw_ts.hour
        weights = self._hour_weight(np.array(hours))
        # Resample to enforce hourly weight distribution
        probs = weights / weights.sum()
        indices = self.rng.choice(len(raw_ts), size=n, replace=True, p=probs)
        order_ts = raw_ts[indices]

        # City distribution
        cities = list(CITIES.keys())
        city_weights = [CITIES[c]["weight"] for c in cities]
        order_cities = self.rng.choice(cities, size=n, p=city_weights)

        # Assign dark stores by city
        store_city_map = dark_stores.groupby("city")["store_id"].apply(list).to_dict()
        assigned_stores = [
            self.rng.choice(store_city_map.get(c, dark_stores["store_id"].tolist()))
            for c in order_cities
        ]

        # Assign riders by store
        rider_store_map = riders.groupby("assigned_store")["rider_id"].apply(list).to_dict()
        all_rider_ids = riders["rider_id"].tolist()
        assigned_riders = [
            self.rng.choice(rider_store_map.get(s, all_rider_ids))
            for s in assigned_stores
        ]

        # Order types & SLA
        order_types = self.rng.choice(
            ["express", "standard", "scheduled"],
            size=n, p=[0.45, 0.42, 0.13]
        )
        sla_map = {"express": 30, "standard": 60, "scheduled": 120}
        sla_mins = np.array([sla_map[t] for t in order_types])

        # Distance & route
        delivery_distance_km = np.round(self.rng.gamma(2.5, 1.5, size=n), 2)
        delivery_distance_km = np.clip(delivery_distance_km, 0.3, 12.0)

        # Base delivery time computation
        # Base speed: ~25 km/h, adjusted for hour, weather, distance
        hour_arr = np.array(order_ts.hour)
        is_peak = ((hour_arr >= 11) & (hour_arr <= 14)) | ((hour_arr >= 18) & (hour_arr <= 21))
        congestion_factor = self.rng.uniform(1.0, 1.8, n)
        congestion_factor[is_peak] *= 1.3

        # Weather join (approximate by hour)
        weather_hour = weather.set_index("timestamp")["delay_impact_min"]
        weather_delays = np.zeros(n)
        for i in range(0, n, 50000):
            batch_ts = order_ts[i:i+50000].floor("h")
            w_vals = weather_hour.reindex(batch_ts, method="nearest").fillna(0).values
            weather_delays[i:i+len(w_vals)] = w_vals

        base_travel_min = (delivery_distance_km / 25) * 60  # 25 km/h avg
        prep_time_min = self.rng.integers(3, 18, size=n)
        actual_delivery_min = (
            base_travel_min * congestion_factor
            + prep_time_min
            + weather_delays
            + self.rng.normal(0, 4, n)
        ).clip(5, 180).round(1)

        # SLA breach
        is_late = actual_delivery_min > sla_mins
        late_pct_noise = self.rng.random(n)
        # Inject ~14% overall late rate realistically
        forced_late_mask = late_pct_noise < 0.02  # 2% random failures
        is_late = is_late | forced_late_mask

        delay_minutes = np.where(
            is_late,
            (actual_delivery_min - sla_mins).clip(0),
            0.0
        ).round(1)

        # Geo coordinates for delivery locations
        store_lat_map = dict(zip(dark_stores["store_id"], dark_stores["latitude"]))
        store_lon_map = dict(zip(dark_stores["store_id"], dark_stores["longitude"]))

        delivery_lats = np.array([
            store_lat_map.get(s, 19.0) + self.rng.uniform(-0.08, 0.08)
            for s in assigned_stores
        ])
        delivery_lons = np.array([
            store_lon_map.get(s, 72.8) + self.rng.uniform(-0.08, 0.08)
            for s in assigned_stores
        ])

        # Route efficiency (1 = optimal, >1 = inefficient detour)
        route_efficiency = np.round(
            self.rng.gamma(1.2, 0.15, n).clip(1.0, 2.5), 3
        )
        route_efficiency[is_peak] *= self.rng.uniform(1.05, 1.35, is_peak.sum())

        # Cost
        base_cost = 25 + delivery_distance_km * 3.5
        surge_cost = np.where(is_peak, base_cost * 1.25, base_cost)
        total_cost = np.round(surge_cost + self.rng.uniform(0, 10, n), 2)

        df = pd.DataFrame({
            "order_id": [f"ORD{i:09d}" for i in range(1, n + 1)],
            "created_at": order_ts,
            "city": order_cities,
            "dark_store_id": assigned_stores,
            "rider_id": assigned_riders,
            "order_type": order_types,
            "sla_minutes": sla_mins,
            "delivery_distance_km": delivery_distance_km,
            "actual_delivery_min": actual_delivery_min,
            "prep_time_min": prep_time_min,
            "travel_time_min": np.round(actual_delivery_min - prep_time_min, 1),
            "is_late": is_late.astype(int),
            "delay_minutes": delay_minutes,
            "weather_delay_min": np.round(weather_delays, 1),
            "congestion_factor": np.round(congestion_factor, 3),
            "route_efficiency_score": route_efficiency,
            "is_peak_hour": is_peak.astype(int),
            "delivery_lat": np.round(delivery_lats, 6),
            "delivery_lon": np.round(delivery_lons, 6),
            "total_cost_inr": total_cost,
            "customer_rating": np.where(
                is_late,
                np.round(self.rng.uniform(1.0, 3.5, n), 1),
                np.round(self.rng.uniform(3.5, 5.0, n), 1)
            ),
            "attempt_count": self.rng.choice([1, 2, 3], size=n, p=[0.88, 0.10, 0.02]),
            "status": np.where(
                self.rng.random(n) < 0.02, "failed",
                np.where(is_late, "late", "on_time")
            ),
        })

        # Day-of-week & hour features
        df["hour_of_day"] = df["created_at"].dt.hour
        df["day_of_week"] = df["created_at"].dt.dayofweek
        df["month"] = df["created_at"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        print(f"  Late delivery rate: {df['is_late'].mean():.1%}")
        print(f"  Avg delivery time: {df['actual_delivery_min'].mean():.1f} min")
        return df

    # ── Run all ───────────────────────────────────────────────────────────────

    def run(self, save: bool = True, sample_only: bool = False) -> dict:
        """
        Generate all datasets. Returns dict of DataFrames.
        If sample_only=True, generates 50k records for fast testing.
        """
        if sample_only:
            SIMULATION["n_orders"] = 50_000
            SIMULATION["n_riders"] = 500
            SIMULATION["n_dark_stores"] = 12
            print("⚡ Sample mode: 50k records")
        else:
            print("🏭 Full mode: 1M records")

        print("\n[1/4] Generating dark stores...")
        stores = self.generate_dark_stores()

        print("[2/4] Generating riders...")
        riders = self.generate_riders(stores)

        print("[3/4] Generating weather events...")
        weather = self.generate_weather_events()

        print("[4/4] Generating orders (this may take ~30s)...")
        orders = self.generate_orders(stores, riders, weather)

        datasets = {
            "orders": orders,
            "riders": riders,
            "dark_stores": stores,
            "weather": weather,
        }

        if save:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
            from config.settings import PROCESSED_DIR as PD, SAMPLE_DIR as SD
            out = SD if sample_only else PD

            for name, df in datasets.items():
                path = out / f"{name}.parquet"
                df.to_parquet(path, index=False)
                print(f"  ✓ Saved {name}.parquet ({len(df):,} rows, {path.stat().st_size/1e6:.1f} MB)")

        print("\n✅ Data generation complete.")
        return datasets


if __name__ == "__main__":
    gen = RouteIQDataGenerator()
    # Use sample_only=True for fast runs; False for full 1M
    gen.run(save=True, sample_only=True)
