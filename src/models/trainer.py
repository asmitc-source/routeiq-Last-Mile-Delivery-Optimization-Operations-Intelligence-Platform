"""
RouteIQ — Predictive Models
Trains and evaluates ML models for:
  - Late delivery prediction (XGBoost classifier)
  - SLA breach probability (Gradient Boosting)
  - Delivery delay estimation (Random Forest regressor)
  - Route congestion risk scoring
Target: ~86% accuracy on test set
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging, sys, os, json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    mean_absolute_error, r2_score, confusion_matrix
)
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not found — using GradientBoosting fallback")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import SAMPLE_DIR, MODELS_DIR, MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MODEL] %(message)s")
log = logging.getLogger(__name__)


FEATURE_COLS = [
    "delivery_distance_km", "prep_time_min", "congestion_factor",
    "route_efficiency_score", "is_peak_hour", "weather_delay_min",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "is_express", "is_high_distance", "sla_minutes",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "experience_months", "base_efficiency_score",
    "store_daily_load", "city_hour_avg_congestion",
    "late_risk_score", "efficiency_gap",
]

CITY_LABEL_COL = "city"


class RouteIQModelTrainer:

    def __init__(self, data_dir: Path = SAMPLE_DIR, model_dir: Path = MODELS_DIR):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.le_city = LabelEncoder()

    def _load_data(self) -> pd.DataFrame:
        path = self.data_dir / "orders_enriched.parquet"
        if not path.exists():
            path = self.data_dir / "orders.parquet"
        df = pd.read_parquet(path)
        log.info("Loaded %s rows from %s", f"{len(df):,}", path.name)
        return df

    def _prepare_features(self, df: pd.DataFrame):
        df = df.copy()

        # City label encode
        df["city_enc"] = self.le_city.fit_transform(df["city"])

        available_features = [c for c in FEATURE_COLS if c in df.columns]
        available_features.append("city_enc")

        # Fill any missing features with sensible defaults
        for col in available_features:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        X = df[available_features].astype(float)
        return X, available_features

    # ── Late Delivery Classifier ──────────────────────────────────────────────

    def train_late_delivery_model(self, df: pd.DataFrame):
        log.info("\n=== Training Late Delivery Classifier ===")
        X, feat_cols = self._prepare_features(df)
        y = df["is_late"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL["test_size"],
            random_state=MODEL["random_state"], stratify=y
        )

        if HAS_XGB:
            params = MODEL["xgb_params"].copy()
            params.pop("use_label_encoder", None)
            params.pop("eval_metric", None)
            model = xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8,
                random_state=MODEL["random_state"]
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        log.info("  Accuracy:  %.4f", acc)
        log.info("  ROC-AUC:   %.4f", auc)
        log.info("  Precision: %.4f", report["1"]["precision"])
        log.info("  Recall:    %.4f", report["1"]["recall"])
        log.info("  F1:        %.4f", report["1"]["f1-score"])

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feat_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
        else:
            fi = pd.DataFrame({"feature": feat_cols, "importance": [0] * len(feat_cols)})

        # Save
        model_path = self.model_dir / "late_delivery_model.pkl"
        joblib.dump({"model": model, "features": feat_cols, "label_encoder": self.le_city}, model_path)

        self.results["late_delivery"] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "precision": round(report["1"]["precision"], 4),
            "recall": round(report["1"]["recall"], 4),
            "f1": round(report["1"]["f1-score"], 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "top_features": fi.head(10).to_dict("records"),
        }
        return model, fi

    # ── SLA Breach Probability ────────────────────────────────────────────────

    def train_sla_breach_model(self, df: pd.DataFrame):
        log.info("\n=== Training SLA Breach Probability Model ===")
        # SLA breach: late AND delay > 10 min
        df = df.copy()
        df["sla_breach"] = ((df["is_late"] == 1) & (df["delay_minutes"] > 10)).astype(int)

        X, feat_cols = self._prepare_features(df)
        y = df["sla_breach"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL["test_size"],
            random_state=MODEL["random_state"], stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            n_jobs=-1, random_state=MODEL["random_state"]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        log.info("  Accuracy: %.4f | AUC: %.4f", acc, auc)

        model_path = self.model_dir / "sla_breach_model.pkl"
        joblib.dump({"model": model, "features": feat_cols}, model_path)

        self.results["sla_breach"] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
        }
        return model

    # ── Delay Estimation (Regression) ────────────────────────────────────────

    def train_delay_estimator(self, df: pd.DataFrame):
        log.info("\n=== Training Delivery Delay Estimator (Regressor) ===")
        df_late = df[df["is_late"] == 1].copy()
        if len(df_late) < 1000:
            df_late = df.sample(min(len(df), 5000), random_state=42)

        X, feat_cols = self._prepare_features(df_late)
        y = df_late["delay_minutes"].clip(0, 120)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL["test_size"], random_state=MODEL["random_state"]
        )

        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            n_jobs=-1, random_state=MODEL["random_state"]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        log.info("  MAE: %.2f min | R²: %.4f", mae, r2)

        model_path = self.model_dir / "delay_estimator.pkl"
        joblib.dump({"model": model, "features": feat_cols}, model_path)

        self.results["delay_estimation"] = {"mae_minutes": round(mae, 2), "r2": round(r2, 4)}
        return model

    # ── Route Congestion Risk ─────────────────────────────────────────────────

    def train_congestion_risk_model(self, df: pd.DataFrame):
        log.info("\n=== Training Route Congestion Risk Model ===")
        df = df.copy()
        df["high_congestion"] = (df["congestion_factor"] > 1.5).astype(int)

        X, feat_cols = self._prepare_features(df)
        y = df["high_congestion"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL["test_size"],
            random_state=MODEL["random_state"], stratify=y
        )

        if HAS_XGB:
            params = MODEL["xgb_params"].copy()
            params.pop("use_label_encoder", None)
            params.pop("eval_metric", None)
            model = xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=6, n_jobs=-1, random_state=42
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        log.info("  Accuracy: %.4f | AUC: %.4f", acc, auc)

        model_path = self.model_dir / "congestion_risk_model.pkl"
        joblib.dump({"model": model, "features": feat_cols}, model_path)

        self.results["congestion_risk"] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)}
        return model

    # ── Run all ───────────────────────────────────────────────────────────────

    def run(self):
        df = self._load_data()
        self.train_late_delivery_model(df)
        self.train_sla_breach_model(df)
        self.train_delay_estimator(df)
        self.train_congestion_risk_model(df)

        results_path = self.model_dir / "model_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        log.info("\n✅ All models trained. Results saved to %s", results_path)
        for name, res in self.results.items():
            acc = res.get("accuracy") or res.get("r2") or "N/A"
            log.info("  %-25s → acc/r2: %s", name, acc)

        return self.results


class RouteIQPredictor:
    """
    Inference wrapper for serving predictions in the dashboard.
    """

    def __init__(self, model_dir: Path = MODELS_DIR):
        self.model_dir = model_dir
        self._models = {}

    def _load(self, name: str):
        if name not in self._models:
            path = self.model_dir / f"{name}.pkl"
            if path.exists():
                self._models[name] = joblib.load(path)
        return self._models.get(name)

    def predict_late_delivery(self, features: dict) -> dict:
        bundle = self._load("late_delivery_model")
        if bundle is None:
            return {"probability": 0.14, "is_late": False, "confidence": "low"}

        model = bundle["model"]
        feat_cols = bundle["features"]
        le = bundle.get("label_encoder")

        row = {}
        for col in feat_cols:
            if col == "city_enc" and le is not None:
                row[col] = le.transform([features.get("city", "Mumbai")])[0]
            else:
                row[col] = features.get(col, 0)

        X = pd.DataFrame([row])
        prob = model.predict_proba(X)[0][1]
        return {
            "probability": round(float(prob), 4),
            "is_late": prob > 0.5,
            "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
        }

    def score_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch score all orders with late delivery probability."""
        bundle = self._load("late_delivery_model")
        if bundle is None:
            df["late_prob"] = 0.14
            return df

        model = bundle["model"]
        feat_cols = bundle["features"]
        le = bundle.get("label_encoder")

        df2 = df.copy()
        if "city_enc" in feat_cols and le is not None:
            df2["city_enc"] = le.transform(df2["city"].fillna("Mumbai"))

        available = [c for c in feat_cols if c in df2.columns]
        X = df2[available].fillna(0).astype(float)
        df2["late_prob"] = model.predict_proba(X)[:, 1].round(4)
        return df2


if __name__ == "__main__":
    trainer = RouteIQModelTrainer()
    results = trainer.run()
