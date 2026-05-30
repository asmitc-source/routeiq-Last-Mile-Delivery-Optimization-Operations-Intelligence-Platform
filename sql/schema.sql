-- ============================================================
-- RouteIQ Platform — SQL Schema
-- Last-Mile Delivery Operations Intelligence Platform
-- Compatible with: PostgreSQL 14+ / DuckDB 0.9+
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- DIMENSION TABLES
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_cities (
    city_id         SERIAL PRIMARY KEY,
    city_name       VARCHAR(100) NOT NULL UNIQUE,
    state           VARCHAR(100),
    latitude        DECIMAL(9, 6) NOT NULL,
    longitude       DECIMAL(9, 6) NOT NULL,
    timezone        VARCHAR(50) DEFAULT 'Asia/Kolkata',
    tier            VARCHAR(20) CHECK (tier IN ('Metro', 'Tier-1', 'Tier-2')),
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_dark_stores (
    store_id            VARCHAR(20) PRIMARY KEY,
    city                VARCHAR(100) REFERENCES dim_cities(city_name),
    latitude            DECIMAL(9, 6) NOT NULL,
    longitude           DECIMAL(9, 6) NOT NULL,
    max_capacity        INT NOT NULL DEFAULT 400,
    area_sqft           INT,
    zone_coverage_km    DECIMAL(4, 1),
    tier                VARCHAR(20) CHECK (tier IN ('Tier-1', 'Tier-2', 'Tier-3')),
    operational_since   DATE,
    is_active           BOOLEAN DEFAULT TRUE,
    manager_name        VARCHAR(200),
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_riders (
    rider_id                VARCHAR(20) PRIMARY KEY,
    assigned_store          VARCHAR(20) REFERENCES dim_dark_stores(store_id),
    city                    VARCHAR(100),
    experience_months       INT DEFAULT 0,
    vehicle_type            VARCHAR(30) CHECK (vehicle_type IN ('2W-Petrol','2W-EV','Cycle','3W-EV')),
    shift_type              VARCHAR(20) CHECK (shift_type IN ('Morning','Afternoon','Evening','Night','Full-Day')),
    rating                  DECIMAL(3, 2),
    is_active               BOOLEAN DEFAULT TRUE,
    base_efficiency_score   DECIMAL(5, 3),
    join_date               DATE,
    created_at              TIMESTAMP DEFAULT NOW(),
    updated_at              TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
-- FACT TABLES
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_orders (
    order_id                VARCHAR(30) PRIMARY KEY,
    created_at              TIMESTAMP NOT NULL,
    city                    VARCHAR(100) NOT NULL,
    dark_store_id           VARCHAR(20) REFERENCES dim_dark_stores(store_id),
    rider_id                VARCHAR(20) REFERENCES dim_riders(rider_id),
    order_type              VARCHAR(20) CHECK (order_type IN ('express','standard','scheduled')),
    sla_minutes             INT NOT NULL,
    delivery_distance_km    DECIMAL(6, 2),
    actual_delivery_min     DECIMAL(7, 1),
    prep_time_min           INT,
    travel_time_min         DECIMAL(7, 1),
    is_late                 SMALLINT DEFAULT 0,
    delay_minutes           DECIMAL(6, 1) DEFAULT 0,
    weather_delay_min       DECIMAL(5, 1) DEFAULT 0,
    congestion_factor       DECIMAL(5, 3),
    route_efficiency_score  DECIMAL(5, 3),
    is_peak_hour            SMALLINT DEFAULT 0,
    delivery_lat            DECIMAL(9, 6),
    delivery_lon            DECIMAL(9, 6),
    total_cost_inr          DECIMAL(8, 2),
    customer_rating         DECIMAL(3, 1),
    attempt_count           SMALLINT DEFAULT 1,
    status                  VARCHAR(20) CHECK (status IN ('on_time','late','failed')),
    hour_of_day             SMALLINT,
    day_of_week             SMALLINT,
    month                   SMALLINT,
    is_weekend              SMALLINT DEFAULT 0
);

CREATE INDEX idx_orders_created_at    ON fact_orders(created_at);
CREATE INDEX idx_orders_city          ON fact_orders(city);
CREATE INDEX idx_orders_rider         ON fact_orders(rider_id);
CREATE INDEX idx_orders_store         ON fact_orders(dark_store_id);
CREATE INDEX idx_orders_is_late       ON fact_orders(is_late);
CREATE INDEX idx_orders_date_city     ON fact_orders(created_at, city);

CREATE TABLE IF NOT EXISTS fact_weather_events (
    event_id        SERIAL PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    city            VARCHAR(100),
    condition       VARCHAR(30) CHECK (condition IN ('clear','rain','heavy_rain','fog','storm')),
    temperature_c   DECIMAL(4, 1),
    humidity_pct    DECIMAL(4, 1),
    wind_kmh        DECIMAL(5, 1),
    visibility_km   DECIMAL(4, 1),
    delay_impact_min SMALLINT DEFAULT 0
);

CREATE INDEX idx_weather_timestamp ON fact_weather_events(timestamp);

-- ─────────────────────────────────────────────────────────────
-- ANALYTICAL VIEWS
-- ─────────────────────────────────────────────────────────────

-- City Daily KPIs
CREATE OR REPLACE VIEW vw_city_daily_kpis AS
SELECT
    city,
    DATE_TRUNC('day', created_at)           AS date,
    COUNT(*)                                AS total_orders,
    SUM(is_late)                            AS late_orders,
    ROUND(AVG(actual_delivery_min)::NUMERIC, 1)     AS avg_delivery_min,
    ROUND(AVG(delay_minutes)::NUMERIC, 1)           AS avg_delay_min,
    ROUND(SUM(total_cost_inr)::NUMERIC, 2)          AS total_revenue_inr,
    ROUND(AVG(congestion_factor)::NUMERIC, 3)       AS avg_congestion,
    ROUND(AVG(route_efficiency_score)::NUMERIC, 3)  AS avg_route_efficiency,
    ROUND(SUM(is_late)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) AS late_rate_pct,
    ROUND(SUM(total_cost_inr)::NUMERIC / NULLIF(COUNT(*), 0), 2) AS cost_per_order
FROM fact_orders
GROUP BY city, DATE_TRUNC('day', created_at);

-- Rider Performance Summary
CREATE OR REPLACE VIEW vw_rider_performance AS
SELECT
    r.rider_id,
    r.city,
    r.experience_months,
    r.vehicle_type,
    r.shift_type,
    r.rating                                AS profile_rating,
    COUNT(o.order_id)                       AS total_deliveries,
    SUM(o.is_late)                          AS late_deliveries,
    ROUND(AVG(o.actual_delivery_min)::NUMERIC, 1)   AS avg_delivery_min,
    ROUND(AVG(o.delivery_distance_km)::NUMERIC, 2)  AS avg_distance_km,
    ROUND(SUM(o.total_cost_inr)::NUMERIC, 2)        AS total_earnings_inr,
    ROUND(AVG(o.customer_rating)::NUMERIC, 2)       AS avg_customer_rating,
    ROUND(SUM(o.is_late)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) AS late_rate_pct
FROM dim_riders r
LEFT JOIN fact_orders o ON r.rider_id = o.rider_id
GROUP BY r.rider_id, r.city, r.experience_months,
         r.vehicle_type, r.shift_type, r.rating;

-- Dark Store Load Analysis
CREATE OR REPLACE VIEW vw_store_load_analysis AS
SELECT
    ds.store_id,
    ds.city,
    ds.tier,
    ds.max_capacity,
    ds.zone_coverage_km,
    COUNT(o.order_id)                       AS total_orders,
    SUM(o.is_late)                          AS total_late,
    ROUND(AVG(o.prep_time_min)::NUMERIC, 1)         AS avg_prep_time_min,
    ROUND(AVG(o.congestion_factor)::NUMERIC, 3)     AS avg_congestion,
    ROUND(SUM(o.is_late)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) AS sla_breach_rate_pct,
    ROUND(COUNT(o.order_id)::NUMERIC / (ds.max_capacity * 30) * 100, 1) AS monthly_capacity_util_pct
FROM dim_dark_stores ds
LEFT JOIN fact_orders o ON ds.store_id = o.dark_store_id
GROUP BY ds.store_id, ds.city, ds.tier, ds.max_capacity, ds.zone_coverage_km;

-- Hourly Operational Stress
CREATE OR REPLACE VIEW vw_hourly_stress AS
SELECT
    city,
    hour_of_day,
    COUNT(*)                                AS orders_per_hour_avg,
    ROUND(AVG(is_late)::NUMERIC * 100, 2)           AS late_rate_pct,
    ROUND(AVG(congestion_factor)::NUMERIC, 3)       AS avg_congestion,
    ROUND(AVG(weather_delay_min)::NUMERIC, 1)       AS avg_weather_delay_min,
    ROUND(
        (COUNT(*) * 1.0 / MAX(COUNT(*)) OVER (PARTITION BY city)) * 0.4
        + AVG(is_late::NUMERIC) * 0.35
        + (AVG(congestion_factor) - 1) / 0.8 * 0.25
    , 4) AS stress_index
FROM fact_orders
GROUP BY city, hour_of_day;

-- SLA Monitoring Dashboard View
CREATE OR REPLACE VIEW vw_sla_monitoring AS
SELECT
    city,
    order_type,
    DATE_TRUNC('week', created_at)          AS week,
    COUNT(*)                                AS total_orders,
    SUM(is_late)                            AS breached,
    ROUND(SUM(is_late)::NUMERIC / COUNT(*) * 100, 2) AS breach_rate_pct,
    ROUND(AVG(CASE WHEN is_late = 1 THEN delay_minutes ELSE NULL END)::NUMERIC, 1) AS avg_breach_delay_min,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY actual_delivery_min)::NUMERIC, 1) AS p95_delivery_min
FROM fact_orders
GROUP BY city, order_type, DATE_TRUNC('week', created_at);

-- ─────────────────────────────────────────────────────────────
-- OPERATIONAL ANALYTICS QUERIES
-- ─────────────────────────────────────────────────────────────

-- Q1: Top congested delivery zones (hexbin)
-- SELECT
--     ROUND(delivery_lat, 3) AS lat_bin,
--     ROUND(delivery_lon, 3) AS lon_bin,
--     COUNT(*)               AS order_count,
--     AVG(is_late)           AS late_rate,
--     AVG(congestion_factor) AS avg_congestion
-- FROM fact_orders
-- WHERE created_at >= NOW() - INTERVAL '30 days'
-- GROUP BY lat_bin, lon_bin
-- HAVING COUNT(*) > 50
-- ORDER BY avg_congestion DESC
-- LIMIT 20;

-- Q2: Riders with high late rate (needs attention)
-- SELECT rider_id, city, total_deliveries, late_rate_pct, avg_delivery_min
-- FROM vw_rider_performance
-- WHERE total_deliveries >= 50
-- AND late_rate_pct > 20
-- ORDER BY late_rate_pct DESC;

-- Q3: Peak hour SLA breach rate
-- SELECT hour_of_day, late_rate_pct, stress_index
-- FROM vw_hourly_stress
-- WHERE city = 'Mumbai'
-- ORDER BY hour_of_day;
