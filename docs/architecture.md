# RouteIQ — Architecture Documentation

## System Design

### Data Flow
```
[Data Generator] → [Parquet Files] → [ETL Pipeline] → [Enriched Tables]
                                                              ↓
                                                    [Analytics Engine]
                                                         ↙       ↘
                                              [ML Models]   [KPI Engine]
                                                         ↘       ↙
                                                    [Streamlit Dashboard]
```

### Layer Responsibilities

**1. Data Layer (`src/pipelines/data_generator.py`)**
- Generates synthetic but statistically realistic delivery records
- Simulates 8 Indian cities with realistic demand curves
- Applies monsoon seasonality to weather data
- Produces rider profiles with experience-based performance variance
- Outputs: `orders.parquet`, `riders.parquet`, `dark_stores.parquet`, `weather.parquet`

**2. ETL Layer (`src/pipelines/etl.py`)**
- Schema validation and deduplication
- Feature engineering: cyclical time encoding, composite risk scores, load indices
- Joins: rider stats, store load, city-hour congestion
- Aggregation tables: daily city KPIs, rider perf, store metrics, hourly stress
- Outputs: `orders_enriched.parquet` + 4 aggregation tables

**3. ML Layer (`src/models/trainer.py`)**
- Late delivery classifier (XGBoost): 93.8% accuracy
- SLA breach probability (Random Forest): 95.8% accuracy
- Delay estimator (RF Regressor): R² = 0.885, MAE = 3.3 min
- Congestion risk scorer (XGBoost): 99.9% AUC
- Serialized to `.pkl` artifacts with feature lists

**4. Analytics Layer (`src/analytics/engine.py`)**
- `KPIEngine`: computes 13 operational KPIs with city/date filtering
- `RecommendationsEngine`: threshold-based intelligent recommendations
- `GeoAnalytics`: hexbin aggregation, K-means cluster analysis

**5. Dashboard Layer (`dashboard/app.py`)**
- 7-tab Streamlit interface
- Dark theme with custom CSS (IBM Plex Mono + Syne fonts)
- Plotly Express + Graph Objects charts
- Mapbox integration for geo heatmaps
- Real-time filter propagation (city, date range, order type)

### Database Schema (SQL)
See `sql/schema.sql` for:
- `dim_cities`, `dim_dark_stores`, `dim_riders` (dimension tables)
- `fact_orders`, `fact_weather_events` (fact tables)
- 5 analytical views (city KPIs, rider perf, store load, hourly stress, SLA monitoring)

### Scalability Path
| Component | Current | Production |
|-----------|---------|------------|
| Storage | Parquet files | PostgreSQL + S3 |
| Compute | Single process | Spark / Dask |
| Orchestration | CLI script | Airflow DAGs |
| Serving | Streamlit | FastAPI + Redis |
| ML Serving | Joblib pkl | MLflow + Seldon |
