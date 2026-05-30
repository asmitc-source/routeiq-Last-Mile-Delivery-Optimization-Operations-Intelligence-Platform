# RouteIQ — Interview Q&A Guide

## Technical Architecture Questions

**Q: Walk me through the data pipeline.**
A: The pipeline has three stages. First, `data_generator.py` creates synthetic but statistically realistic records — it applies a real demand curve with lunch/dinner peaks, monsoon weather seasonality, and rider experience-based performance variance. Second, `etl.py` validates, deduplicates, applies feature engineering (cyclical time encoding, composite risk scores, rolling load indices), and builds 5 aggregation tables. Third, `trainer.py` trains four ML models using the enriched dataset and serializes them with joblib.

**Q: Why did you choose XGBoost over simpler models?**
A: Late delivery prediction is a non-linear problem — the interaction between peak hour, congestion factor, and rider experience creates decision boundaries that linear models can't capture well. XGBoost handles this naturally, supports missing values, and gives feature importances that are operationally interpretable. We also tried Random Forest for SLA breach (because class imbalance is severe there and RF handles it better with class_weight).

**Q: How did you handle the class imbalance in SLA breach prediction?**
A: SLA breach (delay > 10 min while late) is a rarer event than general lateness. The Random Forest classifier handles this through bootstrap sampling, and we verified that precision/recall were acceptable — not just accuracy. For production, we'd add `class_weight='balanced'` or SMOTE oversampling if breach rate drops below 5%.

**Q: What does the route inefficiency score mean?**
A: It's the ratio of actual route distance to optimal straight-line distance, normalized. Score of 1.0 = perfectly optimal. Score of 1.3 means the rider took a path 30% longer than necessary — could be due to traffic, one-way roads, or poor dispatch routing. We aggregate this at city and zone level to identify structural inefficiencies vs. transient congestion.

**Q: How does the Recommendations Engine scale?**
A: Currently it's threshold-based rules over KPI outputs — O(1) per city evaluation. For production, we'd replace static thresholds with a dynamic baseline computed from rolling 30-day percentiles, making it adaptive to seasonal patterns. The recommendation categories (rider redistribution, store load balancing, etc.) would each become a separate microservice with its own SLA.

## Business/Domain Questions

**Q: What's the most impactful KPI in last-mile delivery?**
A: On-time delivery rate because it directly affects customer retention. But it's a lagging indicator. The leading indicators that predict it are congestion factor (real-time) and rider utilization (structural) — these are what ops teams should monitor proactively. A 5% drop in on-time rate is predictable 30-60 minutes earlier if you watch congestion spikes.

**Q: How would you present this to a non-technical ops manager?**
A: I'd focus on the three alert levels — green/yellow/red — for each metric. The recommendations section is specifically designed for ops: it says "Rider utilization is 91% — activate surge roster for next 48 hours" rather than showing a utilization distribution chart. The goal is zero cognitive load between seeing the dashboard and knowing what action to take.

**Q: How do dark stores affect last-mile efficiency?**
A: Dark stores are the fulcrum. Prep time (picking + packing) is 20-40% of total delivery time on express orders. A store operating at 90%+ capacity has elevated prep times due to crowding, which cascades into SLA breaches even when rider performance is good. The store load balancing recommendation redirects orders to nearby underutilized stores to prevent this.

## Data Science Questions

**Q: What feature had the highest importance in the late delivery model?**
A: `congestion_factor` and `late_risk_score` (our engineered composite) dominate. `hour_of_day` encoded cyclically (`hour_sin`, `hour_cos`) ranks third. This confirms that time-of-day and real-time traffic are the primary drivers — more so than rider-level features, which suggests systemic/route issues dominate over individual performance variance.

**Q: How did you validate the synthetic data is realistic?**
A: Four checks: (1) Late delivery rate of ~14% matches industry benchmarks for quick-commerce. (2) Average delivery time of 31 min is consistent with 10-min dark store prep + 20-min travel at 25 km/h average for 4 km median distance. (3) Monsoon months (Jun-Sep) show elevated weather delay scores. (4) Lunch (12-13h) and dinner (19-20h) peaks are clearly visible in hourly heatmaps.

**Q: How would you deploy the ML models in production?**
A: Wrap the predictor in a FastAPI service with `/predict/late-delivery` and `/batch-score` endpoints. Store models in MLflow Model Registry with versioning. Trigger retraining weekly via Airflow when model drift is detected (PSI > 0.2 on feature distributions). Use Redis to cache batch predictions so the dashboard doesn't need to call the model for every page load.
