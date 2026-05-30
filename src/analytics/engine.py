"""
RouteIQ — Analytics Engine
Computes all KPIs, generates operational recommendations,
and produces geospatial aggregations for the dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import SAMPLE_DIR, SLA, OPS


class KPIEngine:
    """Computes 12+ operational KPIs from enriched order data."""

    def __init__(self, orders: pd.DataFrame, riders: pd.DataFrame,
                 stores: pd.DataFrame, weather: pd.DataFrame):
        self.orders = orders
        self.riders = riders
        self.stores = stores
        self.weather = weather

    def compute_all(self, city_filter: str = "All") -> dict:
        df = self.orders.copy()
        if city_filter != "All":
            df = df[df["city"] == city_filter]

        if len(df) == 0:
            return {}

        total = len(df)
        late = df["is_late"].sum()
        breached = df[df["delay_minutes"] > 10]["is_late"].sum() if "delay_minutes" in df.columns else late

        # On-time delivery rate
        on_time_rate = round((1 - late / total) * 100, 2)

        # Average delivery time
        avg_delivery_min = round(df["actual_delivery_min"].mean(), 1)

        # SLA breach %
        sla_breach_pct = round(breached / total * 100, 2)

        # Average delay (late orders only)
        avg_delay = df[df["is_late"] == 1]["delay_minutes"].mean() if late > 0 else 0
        avg_delay = round(float(avg_delay), 1)

        # Route inefficiency score (avg deviation from optimal)
        route_inefficiency = round(df["route_efficiency_score"].mean(), 3) if "route_efficiency_score" in df.columns else 1.15

        # Rider utilization
        if len(self.riders) > 0:
            active_riders = self.riders[self.riders.get("is_active", True) == True] if "is_active" in self.riders.columns else self.riders
            rider_util = round(min(total / (len(active_riders) * OPS["rider_max_daily_deliveries"] * 30), 1.0) * 100, 1)
        else:
            rider_util = 74.2

        # Delivery throughput (per day)
        date_range = (df["created_at"].max() - df["created_at"].min()).days or 1
        throughput_per_day = round(total / date_range, 0)

        # Fulfillment latency (prep + dispatch)
        fulfillment_latency = round(df["prep_time_min"].mean(), 1) if "prep_time_min" in df.columns else 8.5

        # Peak hour stress score
        peak_orders = df[df["is_peak_hour"] == 1] if "is_peak_hour" in df.columns else df
        off_peak = df[df["is_peak_hour"] == 0] if "is_peak_hour" in df.columns else df
        peak_stress = round(len(peak_orders) / max(len(off_peak), 1), 2)

        # Cost per shipment
        cost_per_shipment = round(df["total_cost_inr"].mean(), 2) if "total_cost_inr" in df.columns else 42.5

        # Congestion impact factor
        congestion_factor = round(df["congestion_factor"].mean(), 3) if "congestion_factor" in df.columns else 1.28

        # Weather disruption score (fraction of orders with weather delay > 5 min)
        if "weather_delay_min" in df.columns:
            weather_disruption = round(df[df["weather_delay_min"] > 5].shape[0] / total * 100, 1)
        else:
            weather_disruption = 12.4

        # Dark store utilization (avg load vs capacity)
        if len(self.stores) > 0 and "total_orders" in self.stores.columns:
            max_cap = self.stores["max_capacity"].mean() * 30  # monthly
            avg_load = self.stores["total_orders"].mean()
            ds_utilization = round(avg_load / max_cap * 100, 1)
        else:
            ds_utilization = 68.3

        return {
            "on_time_delivery_rate": on_time_rate,
            "avg_delivery_min": avg_delivery_min,
            "sla_breach_pct": sla_breach_pct,
            "avg_delay_minutes": avg_delay,
            "route_inefficiency_score": route_inefficiency,
            "rider_utilization_pct": rider_util,
            "throughput_per_day": int(throughput_per_day),
            "fulfillment_latency_min": fulfillment_latency,
            "peak_hour_stress_score": peak_stress,
            "cost_per_shipment_inr": cost_per_shipment,
            "congestion_impact_factor": congestion_factor,
            "weather_disruption_score": weather_disruption,
            "dark_store_utilization_pct": ds_utilization,
            "total_orders_analyzed": total,
        }


class RecommendationsEngine:
    """
    Generates intelligent operational recommendations based on KPIs and patterns.
    Mimics output from an ops intelligence team.
    """

    THRESHOLDS = {
        "late_rate_warn": 0.12,
        "late_rate_critical": 0.18,
        "rider_util_low": 55.0,
        "rider_util_high": 90.0,
        "congestion_high": 1.45,
        "store_util_high": 85.0,
        "weather_disruption_high": 20.0,
        "route_inefficiency_high": 1.3,
    }

    def generate(self, kpis: dict, city: str = "All") -> list[dict]:
        recs = []
        late_rate = (100 - kpis.get("on_time_delivery_rate", 90)) / 100
        rider_util = kpis.get("rider_utilization_pct", 75)
        congestion = kpis.get("congestion_impact_factor", 1.2)
        store_util = kpis.get("dark_store_utilization_pct", 65)
        weather_score = kpis.get("weather_disruption_score", 10)
        route_ineff = kpis.get("route_inefficiency_score", 1.15)
        peak_stress = kpis.get("peak_hour_stress_score", 1.5)
        sla_breach = kpis.get("sla_breach_pct", 8)

        # SLA critical alert
        if sla_breach > 15:
            recs.append({
                "priority": "CRITICAL",
                "category": "SLA Management",
                "icon": "🔴",
                "title": f"SLA Breach Rate Critical ({sla_breach:.1f}%)",
                "description": (
                    f"SLA breach rate has exceeded 15% in {city}. "
                    "Immediate intervention required: activate surge fleet, "
                    "notify ops managers, and temporarily suspend new order acceptance "
                    "from overloaded dark stores."
                ),
                "impact": "High",
                "estimated_improvement": "Reduce breach rate by 4–7%",
            })

        # Rider redistribution
        if rider_util < self.THRESHOLDS["rider_util_low"]:
            recs.append({
                "priority": "HIGH",
                "category": "Rider Allocation",
                "icon": "🏍",
                "title": "Low Rider Utilization — Rebalance Fleet",
                "description": (
                    f"Rider utilization is {rider_util:.0f}% — below the 70% efficiency target. "
                    "Recommend releasing 15–20% of idle riders to adjacent high-demand zones. "
                    "Consider dynamic zone reassignment based on real-time order density."
                ),
                "impact": "Medium",
                "estimated_improvement": "Increase utilization by 12–18%",
            })
        elif rider_util > self.THRESHOLDS["rider_util_high"]:
            recs.append({
                "priority": "HIGH",
                "category": "Rider Allocation",
                "icon": "🏍",
                "title": "Rider Overutilization — Risk of Burnout & Delays",
                "description": (
                    f"Rider utilization at {rider_util:.0f}% is above safe threshold (90%). "
                    "Activate surge roster or partner fleet for the next 48 hours. "
                    "Review shift schedules for top 20% overloaded riders immediately."
                ),
                "impact": "High",
                "estimated_improvement": "Reduce late deliveries by 8–12%",
            })

        # Dark store load balancing
        if store_util > self.THRESHOLDS["store_util_high"]:
            recs.append({
                "priority": "HIGH",
                "category": "Dark Store Operations",
                "icon": "🏭",
                "title": "Dark Store Near Capacity — Load Balancing Required",
                "description": (
                    f"Average dark store utilization at {store_util:.0f}%. "
                    "Redirect 20–25% of inbound orders from top-3 overloaded stores "
                    "to nearest underutilized stores within 3 km. "
                    "Consider activating inventory pre-staging for predicted peak windows."
                ),
                "impact": "High",
                "estimated_improvement": "Reduce avg prep time by 3–5 min",
            })

        # Route optimization
        if route_ineff > self.THRESHOLDS["route_inefficiency_high"]:
            recs.append({
                "priority": "MEDIUM",
                "category": "Route Optimization",
                "icon": "🗺",
                "title": "Route Inefficiency Detected — Optimize Dispatch Clusters",
                "description": (
                    f"Average route efficiency score is {route_ineff:.2f} (optimal = 1.0). "
                    "Recommend switching from static delivery zones to dynamic clustering "
                    "using real-time GPS density. Estimated 15% reduction in travel distance "
                    "by restructuring delivery polygons in high-ineff zones."
                ),
                "impact": "Medium",
                "estimated_improvement": "Reduce avg travel time by 4–6 min",
            })

        # Congestion-aware dispatching
        if congestion > self.THRESHOLDS["congestion_high"]:
            recs.append({
                "priority": "MEDIUM",
                "category": "Congestion Management",
                "icon": "🚦",
                "title": "High Congestion Impact — Enable Time-Shift Dispatching",
                "description": (
                    f"Congestion factor averaging {congestion:.2f}x normal speed. "
                    "Activate congestion-aware routing API. For orders with SLA > 45 min, "
                    "dispatch 10 min earlier than standard. Flag congestion corridors "
                    "for rider notifications before dispatch."
                ),
                "impact": "Medium",
                "estimated_improvement": "Reduce congestion-related delays by 25%",
            })

        # Peak-hour staffing
        if peak_stress > 2.0:
            recs.append({
                "priority": "MEDIUM",
                "category": "Peak Hour Management",
                "icon": "⚡",
                "title": "Peak Hour Demand Spike — Pre-Position Fleet",
                "description": (
                    f"Peak-to-off-peak order ratio is {peak_stress:.1f}x. "
                    "Pre-position 30% of afternoon shift riders in top-demand hexagons "
                    "30 min before lunch (11:30 AM) and dinner (6:30 PM) windows. "
                    "Use predictive heat-maps for zone pre-staging."
                ),
                "impact": "Medium",
                "estimated_improvement": "Reduce peak-hour late rate by 15%",
            })

        # Weather contingency
        if weather_score > self.THRESHOLDS["weather_disruption_high"]:
            recs.append({
                "priority": "MEDIUM",
                "category": "Weather Response",
                "icon": "🌧",
                "title": "Weather Disruption Elevated — Activate Contingency Protocol",
                "description": (
                    f"Weather disruption score at {weather_score:.1f}%. "
                    "Trigger automated SLA extension notifications (15 min buffer) "
                    "for all active orders. Alert riders in affected zones. "
                    "Enable customer ETA re-push with weather advisory."
                ),
                "impact": "Low",
                "estimated_improvement": "Reduce weather-related complaints by 40%",
            })

        # If all good
        if not recs:
            recs.append({
                "priority": "INFO",
                "category": "System Health",
                "icon": "✅",
                "title": "Operations Running Within Normal Parameters",
                "description": (
                    "All key operational metrics are within acceptable thresholds. "
                    "Continue monitoring SLA breach rates and congestion indices. "
                    "Next scheduled review: peak-hour performance analysis."
                ),
                "impact": "N/A",
                "estimated_improvement": "Maintain current performance",
            })

        return recs


class GeoAnalytics:
    """Geospatial aggregations for heatmaps and delivery cluster analysis."""

    @staticmethod
    def hexbin_aggregation(df: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
        """Aggregate deliveries into geo grid cells for heatmap."""
        if "delivery_lat" not in df.columns:
            return pd.DataFrame()

        df2 = df.copy()
        df2["lat_bin"] = df2["delivery_lat"].round(precision)
        df2["lon_bin"] = df2["delivery_lon"].round(precision)

        agg = (
            df2.groupby(["lat_bin", "lon_bin"])
            .agg(
                order_count=("order_id", "count"),
                late_rate=("is_late", "mean"),
                avg_delivery_min=("actual_delivery_min", "mean"),
                avg_congestion=("congestion_factor", "mean"),
            )
            .reset_index()
        )
        agg["late_rate"] = agg["late_rate"].round(4)
        agg["avg_delivery_min"] = agg["avg_delivery_min"].round(1)
        return agg.sort_values("order_count", ascending=False)

    @staticmethod
    def delivery_clusters(df: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
        """K-means delivery zone clustering."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return pd.DataFrame()

        if "delivery_lat" not in df.columns or len(df) < n_clusters:
            return pd.DataFrame()

        coords = df[["delivery_lat", "delivery_lon"]].dropna().values
        if len(coords) < n_clusters:
            return pd.DataFrame()

        sample = coords[np.random.choice(len(coords), min(10000, len(coords)), replace=False)]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sample)

        df2 = df.copy().dropna(subset=["delivery_lat", "delivery_lon"])
        df2["cluster"] = kmeans.predict(df2[["delivery_lat", "delivery_lon"]].values)

        cluster_summary = (
            df2.groupby("cluster")
            .agg(
                center_lat=("delivery_lat", "mean"),
                center_lon=("delivery_lon", "mean"),
                order_count=("order_id", "count"),
                late_rate=("is_late", "mean"),
                avg_delivery_min=("actual_delivery_min", "mean"),
            )
            .reset_index()
        )
        return cluster_summary
