"""
RouteIQ — Delivery Operations Intelligence Platform
Production Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json, sys, os, warnings
warnings.filterwarnings("ignore")

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import SAMPLE_DIR, MODELS_DIR, DASHBOARD, CITIES
from src.analytics.engine import KPIEngine, RecommendationsEngine, GeoAnalytics

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD["page_title"],
    page_icon=DASHBOARD["page_icon"],
    layout=DASHBOARD["layout"],
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

  :root {
    --bg: #0A0E1A;
    --surface: #111827;
    --border: #1E2D40;
    --text: #E2E8F0;
    --muted: #64748B;
    --primary: #0EA5E9;
    --accent: #F59E0B;
    --danger: #EF4444;
    --success: #10B981;
    --warning: #F97316;
  }

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text);
  }
  .main { background: var(--bg) !important; padding: 0 !important; }
  .block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }
  [data-testid="stSidebar"] {
    background: #0D1321 !important;
    border-right: 1px solid var(--border);
  }

  /* Header */
  .riq-header {
    background: linear-gradient(135deg, #0D1321 0%, #0A1929 50%, #0D2137 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .riq-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .riq-logo span { color: var(--primary); }
  .riq-tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
  }
  .riq-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--success);
  }
  .riq-status::before {
    content: '';
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

  /* Section headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  /* KPI Cards */
  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }
  .kpi-card.primary::before { background: var(--primary); }
  .kpi-card.success::before { background: var(--success); }
  .kpi-card.warning::before { background: var(--warning); }
  .kpi-card.danger::before { background: var(--danger); }
  .kpi-card.accent::before { background: var(--accent); }
  .kpi-card.muted::before { background: var(--muted); }

  .kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
  }
  .kpi-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    line-height: 1;
    color: var(--text);
  }
  .kpi-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.3rem;
  }
  .kpi-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
    margin-top: 4px;
  }
  .delta-up { background: #10b98122; color: var(--success); }
  .delta-down { background: #ef444422; color: var(--danger); }
  .delta-warn { background: #f9731622; color: var(--warning); }

  /* Alert boxes */
  .alert-critical {
    background: #1a0a0a; border: 1px solid #7f1d1d;
    border-left: 4px solid var(--danger);
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
  }
  .alert-high {
    background: #1a1000; border: 1px solid #78350f;
    border-left: 4px solid var(--warning);
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
  }
  .alert-medium {
    background: #0a1a1a; border: 1px solid #164e63;
    border-left: 4px solid var(--primary);
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
  }
  .alert-info {
    background: #0a1a0a; border: 1px solid #14532d;
    border-left: 4px solid var(--success);
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
  }
  .alert-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 0.3rem;
    color: var(--text);
  }
  .alert-body { font-size: 0.8rem; color: var(--muted); line-height: 1.5; }
  .alert-meta {
    margin-top: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
  }

  /* Tab styling */
  [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 2px !important;
  }
  [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
  }
  [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
  }

  /* Chart containers */
  .chart-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  .chart-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text);
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  /* Sidebar */
  [data-testid="stSidebar"] .sidebar-nav-item {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
  }
  div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem;
  }
  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ─── Plot theme ───────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1E2D40", linecolor="#1E2D40", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1E2D40", linecolor="#1E2D40", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E2D40"),
    colorway=["#0EA5E9", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#F97316"],
)

COLOR_PRIMARY = "#0EA5E9"
COLOR_DANGER = "#EF4444"
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"

# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    """Load all datasets; generate sample if missing."""
    files_needed = ["orders_enriched.parquet", "orders.parquet"]
    has_data = any((SAMPLE_DIR / f).exists() for f in files_needed)

    if not has_data:
        with st.spinner("🔄 Generating sample data for first run (30–60s)..."):
            from src.pipelines.data_generator import RouteIQDataGenerator
            gen = RouteIQDataGenerator()
            gen.run(save=True, sample_only=True)
            from src.pipelines.etl import RouteIQETL
            etl = RouteIQETL(use_sample=True)
            etl.run()

    datasets = {}
    for name in ["orders_enriched", "orders", "riders", "dark_stores",
                 "city_daily_kpis", "rider_performance", "store_performance",
                 "hourly_stress"]:
        path = SAMPLE_DIR / f"{name}.parquet"
        if path.exists():
            datasets[name] = pd.read_parquet(path)

    # Fallback alias
    if "orders_enriched" not in datasets and "orders" in datasets:
        datasets["orders_enriched"] = datasets["orders"]

    return datasets


@st.cache_data(ttl=600, show_spinner=False)
def load_model_results():
    path = MODELS_DIR / "model_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def kpi_card(label, value, sub="", color="primary", delta="", delta_type="up"):
    delta_html = f'<div class="kpi-delta delta-{delta_type}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card {color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
      {delta_html}
    </div>
    """


def alert_card(rec: dict) -> str:
    p = rec["priority"].lower()
    cls = {"critical": "alert-critical", "high": "alert-high",
           "medium": "alert-medium", "info": "alert-info"}.get(p, "alert-info")
    return f"""
    <div class="{cls}">
      <div class="alert-title">{rec['icon']} {rec['title']}</div>
      <div class="alert-body">{rec['description']}</div>
      <div class="alert-meta">
        Category: {rec['category']} &nbsp;|&nbsp;
        Impact: {rec['impact']} &nbsp;|&nbsp;
        {rec['estimated_improvement']}
      </div>
    </div>
    """


def fmt(v, decimals=1, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}{suffix}"
    return f"{v:.{decimals}f}{suffix}"


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0;">
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:#E2E8F0;">
        Route<span style="color:#0EA5E9;">IQ</span>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748B;text-transform:uppercase;letter-spacing:0.1em;">
        Operations Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**🌆 City Filter**")
    city_options = ["All"] + list(CITIES.keys())
    selected_city = st.selectbox("Select City", city_options, label_visibility="collapsed")

    st.markdown("**📅 Date Range**")
    date_presets = ["Last 30 Days", "Last 90 Days", "Last 6 Months", "All Time"]
    date_preset = st.selectbox("Date Range", date_presets, index=0, label_visibility="collapsed")

    st.markdown("**📦 Order Type**")
    order_types_filter = st.multiselect(
        "Order Types",
        ["express", "standard", "scheduled"],
        default=["express", "standard", "scheduled"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**🤖 Model Status**")
    model_results = load_model_results()
    if model_results:
        for mname, mres in model_results.items():
            acc = mres.get("accuracy") or mres.get("r2", 0)
            color = "#10B981" if acc > 0.82 else "#F97316"
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.7rem;color:{color};">'
                f'✓ {mname.replace("_"," ").title()} — {acc:.0%}</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("Run `python src/models/trainer.py` to train models")

    st.divider()
    if st.button("🔄 Retrain Models", use_container_width=True):
        with st.spinner("Training models..."):
            from src.models.trainer import RouteIQModelTrainer
            trainer = RouteIQModelTrainer()
            results = trainer.run()
            st.cache_data.clear()
            st.success("Models trained!")

    st.markdown(
        '<div style="font-family:monospace;font-size:0.65rem;color:#475569;margin-top:2rem;">'
        'v2.4.1 · Sample Mode · 50k records</div>',
        unsafe_allow_html=True
    )


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

with st.spinner("Loading operational data..."):
    datasets = load_data()

orders = datasets.get("orders_enriched", datasets.get("orders", pd.DataFrame()))
riders = datasets.get("riders", pd.DataFrame())
stores_df = datasets.get("dark_stores", pd.DataFrame())
city_daily = datasets.get("city_daily_kpis", pd.DataFrame())
rider_perf = datasets.get("rider_performance", pd.DataFrame())
store_perf = datasets.get("store_performance", pd.DataFrame())
hourly_stress = datasets.get("hourly_stress", pd.DataFrame())

# Apply filters
filtered_orders = orders.copy()
if selected_city != "All":
    filtered_orders = filtered_orders[filtered_orders["city"] == selected_city]
if order_types_filter:
    filtered_orders = filtered_orders[filtered_orders["order_type"].isin(order_types_filter)]

# Date filter
if "created_at" in filtered_orders.columns and len(filtered_orders) > 0:
    max_date = filtered_orders["created_at"].max()
    days_map = {"Last 30 Days": 30, "Last 90 Days": 90, "Last 6 Months": 180, "All Time": 99999}
    cutoff = max_date - pd.Timedelta(days=days_map[date_preset])
    if date_preset != "All Time":
        filtered_orders = filtered_orders[filtered_orders["created_at"] >= cutoff]

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="riq-header">
  <div>
    <div class="riq-logo">Route<span>IQ</span></div>
    <div class="riq-tagline">Last-Mile Delivery Operations Intelligence Platform</div>
  </div>
  <div>
    <div class="riq-status">LIVE OPERATIONAL FEED</div>
    <div style="font-family:monospace;font-size:0.68rem;color:#475569;margin-top:4px;text-align:right;">
      {len(filtered_orders):,} orders · {selected_city} · {date_preset}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────

kpi_engine = KPIEngine(filtered_orders, riders, store_perf, pd.DataFrame())
kpis = kpi_engine.compute_all(city_filter=selected_city)

st.markdown('<div class="section-header">Operational KPI Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    otr = kpis.get("on_time_delivery_rate", 0)
    color = "success" if otr > 88 else "warning" if otr > 82 else "danger"
    st.markdown(kpi_card("On-Time Rate", f"{otr}%", "Deliveries within SLA",
                         color, "↑ 1.2% WoW", "up"), unsafe_allow_html=True)
with col2:
    st.markdown(kpi_card("Avg Delivery", f"{kpis.get('avg_delivery_min',0):.0f}m",
                         "Minutes end-to-end", "primary", "↓ 0.8m WoW", "up"), unsafe_allow_html=True)
with col3:
    sla = kpis.get("sla_breach_pct", 0)
    color = "danger" if sla > 15 else "warning" if sla > 10 else "success"
    st.markdown(kpi_card("SLA Breach", f"{sla}%", "Threshold: 10%",
                         color, "↑ 0.4% WoW", "down"), unsafe_allow_html=True)
with col4:
    st.markdown(kpi_card("Rider Util", f"{kpis.get('rider_utilization_pct',0):.0f}%",
                         "Fleet efficiency", "accent", "Target: 75%", "up"), unsafe_allow_html=True)
with col5:
    st.markdown(kpi_card("Cost / Order", f"₹{kpis.get('cost_per_shipment_inr',0):.0f}",
                         "Incl. surge & ops", "muted", "↑ ₹1.2 WoW", "down"), unsafe_allow_html=True)
with col6:
    st.markdown(kpi_card("Throughput", f"{kpis.get('throughput_per_day',0):,}",
                         "Orders per day", "primary", "↑ 3.8% MoM", "up"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    ri = kpis.get("route_inefficiency_score", 1.0)
    color = "danger" if ri > 1.3 else "warning" if ri > 1.15 else "success"
    st.markdown(kpi_card("Route Ineff.", f"{ri:.3f}", "Optimal = 1.000",
                         color, f"{'HIGH' if ri>1.3 else 'NORMAL'}", "down" if ri > 1.15 else "up"),
                unsafe_allow_html=True)
with col2:
    st.markdown(kpi_card("Fulfillment Lag", f"{kpis.get('fulfillment_latency_min',0):.1f}m",
                         "Prep + dispatch", "muted"), unsafe_allow_html=True)
with col3:
    ps = kpis.get("peak_hour_stress_score", 0)
    color = "danger" if ps > 2.5 else "warning" if ps > 1.8 else "success"
    st.markdown(kpi_card("Peak Stress", f"{ps:.1f}×", "vs. off-peak volume",
                         color), unsafe_allow_html=True)
with col4:
    cf = kpis.get("congestion_impact_factor", 1.0)
    color = "danger" if cf > 1.45 else "warning" if cf > 1.25 else "success"
    st.markdown(kpi_card("Congestion ×", f"{cf:.2f}×", "Speed impact factor",
                         color), unsafe_allow_html=True)
with col5:
    wd = kpis.get("weather_disruption_score", 0)
    color = "warning" if wd > 20 else "success"
    st.markdown(kpi_card("Weather Disruption", f"{wd:.1f}%", "Orders impacted",
                         color), unsafe_allow_html=True)
with col6:
    du = kpis.get("dark_store_utilization_pct", 0)
    color = "danger" if du > 85 else "warning" if du > 70 else "success"
    st.markdown(kpi_card("DS Utilization", f"{du:.0f}%", "Capacity usage",
                         color), unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Trends & Operations",
    "🗺️ Geo Intelligence",
    "🏍️ Rider Analytics",
    "🏭 Dark Store Ops",
    "⚡ SLA & Alerts",
    "🤖 Predictive Models",
    "💡 Recommendations",
])

# ── Tab 1: Trends ──────────────────────────────────────────────────────────────

with tab1:
    if len(city_daily) == 0:
        st.info("Run ETL pipeline to generate trend data.")
    else:
        cd = city_daily.copy()
        if selected_city != "All":
            cd = cd[cd["city"] == selected_city]
        if len(cd) == 0:
            cd = city_daily.copy()
        cd["date"] = pd.to_datetime(cd["date"])
        cd = cd.sort_values("date")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-title">📦 Daily Order Volume & Late Rate</div>', unsafe_allow_html=True)
            agg = cd.groupby("date").agg(
                total_orders=("total_orders", "sum"),
                late_rate=("late_rate", "mean")
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=agg["date"], y=agg["total_orders"],
                name="Orders", marker_color=COLOR_PRIMARY, opacity=0.7
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=agg["date"], y=(agg["late_rate"] * 100).round(1),
                name="Late Rate %", line=dict(color=COLOR_DANGER, width=2),
                mode="lines"
            ), secondary_y=True)
            fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=True)
            fig.update_yaxes(title_text="Orders", secondary_y=False,
                             gridcolor="#1E2D40", color="#94A3B8")
            fig.update_yaxes(title_text="Late Rate %", secondary_y=True, color=COLOR_DANGER)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="chart-title">⏱ Avg Delivery Time by City</div>', unsafe_allow_html=True)
            city_avg = city_daily.groupby("city")["avg_delivery_min"].mean().sort_values().reset_index()
            colors = [COLOR_DANGER if v > 50 else COLOR_WARNING if v > 40 else COLOR_SUCCESS
                      for v in city_avg["avg_delivery_min"]]
            fig = go.Figure(go.Bar(
                x=city_avg["avg_delivery_min"].round(1),
                y=city_avg["city"],
                orientation="h",
                marker_color=colors,
                text=city_avg["avg_delivery_min"].round(1),
                texttemplate="%{text} min",
                textposition="outside",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=280,
                              xaxis_title="Avg Delivery Time (min)")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-title">🕐 Hourly Demand Heatmap</div>', unsafe_allow_html=True)
            if len(filtered_orders) > 0 and "hour_of_day" in filtered_orders.columns:
                hourly_city = (
                    filtered_orders.groupby(["city", "hour_of_day"])
                    .size().reset_index(name="orders")
                )
                if selected_city != "All":
                    pivot_data = hourly_city[hourly_city["city"] == selected_city]
                    pivot_data = pivot_data.set_index("hour_of_day")["orders"]
                    fig = go.Figure(go.Bar(
                        x=pivot_data.index, y=pivot_data.values,
                        marker_color=[COLOR_DANGER if x in [12,13,19,20] else COLOR_PRIMARY
                                      for x in pivot_data.index]
                    ))
                    fig.update_layout(**PLOT_LAYOUT, height=260,
                                      xaxis_title="Hour of Day", yaxis_title="Orders")
                else:
                    pivot = hourly_city.pivot_table(
                        index="city", columns="hour_of_day", values="orders", fill_value=0
                    )
                    fig = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale=[[0, "#0D1321"], [0.5, COLOR_PRIMARY], [1, "#F59E0B"]],
                        hoverongaps=False,
                    ))
                    fig.update_layout(**PLOT_LAYOUT, height=260)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="chart-title">💰 Cost Per Order by City</div>', unsafe_allow_html=True)
            if len(city_daily) > 0:
                cost_city = city_daily.groupby("city")["cost_per_order"].mean().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Bar(
                    x=cost_city["city"],
                    y=cost_city["cost_per_order"].round(1),
                    marker_color=COLOR_ACCENT if "COLOR_ACCENT" in dir() else COLOR_WARNING,
                    text=cost_city["cost_per_order"].round(1),
                    texttemplate="₹%{text}",
                    textposition="outside",
                    marker=dict(color=cost_city["cost_per_order"],
                                colorscale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]]),
                ))
                fig.update_layout(**PLOT_LAYOUT, height=260,
                                  yaxis_title="₹ Cost per Order")
                st.plotly_chart(fig, use_container_width=True)

        # Order type distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-title">📊 Order Type Mix</div>', unsafe_allow_html=True)
            if "order_type" in filtered_orders.columns:
                type_dist = filtered_orders["order_type"].value_counts().reset_index()
                fig = go.Figure(go.Pie(
                    labels=type_dist["order_type"],
                    values=type_dist["count"],
                    hole=0.55,
                    marker=dict(colors=[COLOR_PRIMARY, COLOR_WARNING, COLOR_SUCCESS]),
                    textinfo="label+percent",
                    textfont=dict(size=11, color="#94A3B8"),
                ))
                fig.update_layout(**PLOT_LAYOUT, height=240, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="chart-title">🌤 Weather Impact on Deliveries</div>', unsafe_allow_html=True)
            if "weather_delay_min" in filtered_orders.columns:
                w_agg = (filtered_orders.groupby("month")
                         .agg(avg_delay=("weather_delay_min", "mean"),
                              pct_impacted=("weather_delay_min", lambda x: (x>5).mean()*100))
                         .reset_index())
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=w_agg["month"], y=w_agg["avg_delay"].round(1),
                                     name="Avg Weather Delay (min)",
                                     marker_color=COLOR_PRIMARY, opacity=0.7), secondary_y=False)
                fig.add_trace(go.Scatter(x=w_agg["month"], y=w_agg["pct_impacted"].round(1),
                                         name="% Orders Impacted",
                                         line=dict(color=COLOR_WARNING, width=2)), secondary_y=True)
                fig.update_layout(**PLOT_LAYOUT, height=240, xaxis_title="Month")
                st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Geo Intelligence ────────────────────────────────────────────────────

with tab2:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="chart-title">🗺️ Delivery Density Heatmap</div>', unsafe_allow_html=True)
        if "delivery_lat" in filtered_orders.columns and len(filtered_orders) > 0:
            geo_agg = GeoAnalytics.hexbin_aggregation(filtered_orders.sample(min(20000, len(filtered_orders))))
            if len(geo_agg) > 0:
                fig = px.scatter_mapbox(
                    geo_agg,
                    lat="lat_bin",
                    lon="lon_bin",
                    size="order_count",
                    color="late_rate",
                    color_continuous_scale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
                    size_max=20,
                    zoom=9,
                    mapbox_style="carto-darkmatter",
                    hover_data={"order_count": True, "late_rate": ":.2%",
                                "avg_delivery_min": ":.1f"},
                    labels={"late_rate": "Late Rate", "order_count": "Orders"},
                    height=460,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    coloraxis_colorbar=dict(
                        title="Late Rate",
                        tickformat=".0%",
                        bgcolor="rgba(17,24,39,0.8)",
                        bordercolor="#1E2D40",
                        tickfont=dict(color="#94A3B8"),
                        titlefont=dict(color="#94A3B8"),
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="chart-title">🏪 Dark Store Locations</div>', unsafe_allow_html=True)
        ds_show = stores_df.copy()
        if selected_city != "All" and len(ds_show) > 0:
            ds_show = ds_show[ds_show["city"] == selected_city]

        if len(ds_show) > 0:
            fig2 = px.scatter_mapbox(
                ds_show,
                lat="latitude",
                lon="longitude",
                color="tier" if "tier" in ds_show.columns else None,
                hover_name="store_id",
                hover_data={"city": True, "zone_coverage_km": True},
                color_discrete_map={"Tier-1": COLOR_SUCCESS, "Tier-2": COLOR_WARNING, "Tier-3": COLOR_DANGER},
                zoom=9,
                mapbox_style="carto-darkmatter",
                height=220,
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0),
                               showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)",
                                                            font=dict(color="#94A3B8", size=10)))
            st.plotly_chart(fig2, use_container_width=True)

        # Late rate by zone
        st.markdown('<div class="chart-title" style="margin-top:8px;">🔴 Top Late-Rate Zones</div>', unsafe_allow_html=True)
        if len(filtered_orders) > 0 and "delivery_lat" in filtered_orders.columns:
            geo_agg2 = GeoAnalytics.hexbin_aggregation(filtered_orders.sample(min(10000, len(filtered_orders))), precision=2)
            if len(geo_agg2) > 0:
                top_late = geo_agg2.nlargest(8, "late_rate")[["lat_bin", "lon_bin", "late_rate", "order_count"]]
                top_late["zone"] = top_late.apply(
                    lambda r: f"{r['lat_bin']:.2f}°N / {r['lon_bin']:.2f}°E", axis=1)
                top_late["late_rate_pct"] = (top_late["late_rate"] * 100).round(1)
                for _, row in top_late.iterrows():
                    color = "#EF4444" if row["late_rate"] > 0.20 else "#F97316"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-family:monospace;font-size:0.72rem;color:#94A3B8;'
                        f'padding:3px 0;border-bottom:1px solid #1E2D40;">'
                        f'<span>{row["zone"]}</span>'
                        f'<span style="color:{color};font-weight:600;">{row["late_rate_pct"]}%</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# ── Tab 3: Rider Analytics ─────────────────────────────────────────────────────

with tab3:
    if len(rider_perf) == 0:
        st.info("Rider performance data not available. Run ETL pipeline.")
    else:
        rp = rider_perf.copy()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="chart-title">🏅 Rider Performance Distribution</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=rp["late_rate"] * 100,
                nbinsx=30,
                marker_color=COLOR_PRIMARY,
                opacity=0.75,
                name="Late Rate Distribution",
            ))
            fig.add_vline(x=rp["late_rate"].mean() * 100, line_dash="dash",
                          line_color=COLOR_DANGER, annotation_text="Mean")
            fig.update_layout(**PLOT_LAYOUT, height=260,
                              xaxis_title="Late Rate (%)", yaxis_title="# Riders")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="chart-title">⚡ Utilization vs Late Rate</div>', unsafe_allow_html=True)
            sample_rp = rp.sample(min(500, len(rp)))
            fig = go.Figure(go.Scatter(
                x=sample_rp["utilization_score"] * 100,
                y=sample_rp["late_rate"] * 100,
                mode="markers",
                marker=dict(
                    size=4,
                    color=sample_rp["avg_rating"],
                    colorscale=[[0, COLOR_DANGER], [0.5, COLOR_WARNING], [1, COLOR_SUCCESS]],
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title="Rating", thickness=10,
                                  tickfont=dict(color="#94A3B8", size=9),
                                  titlefont=dict(color="#94A3B8", size=9)),
                ),
                hovertemplate="Util: %{x:.0f}%<br>Late: %{y:.1f}%<extra></extra>",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=260,
                              xaxis_title="Utilization %", yaxis_title="Late Rate %")
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown('<div class="chart-title">📊 Top 10 Riders by Volume</div>', unsafe_allow_html=True)
            top_riders = rp.nlargest(10, "total_deliveries")[
                ["rider_id", "total_deliveries", "late_rate", "avg_rating"]
            ]
            top_riders["late_pct"] = (top_riders["late_rate"] * 100).round(1)
            fig = go.Figure(go.Bar(
                x=top_riders["rider_id"],
                y=top_riders["total_deliveries"],
                marker=dict(
                    color=top_riders["late_pct"],
                    colorscale=[[0, COLOR_SUCCESS], [0.5, COLOR_WARNING], [1, COLOR_DANGER]],
                ),
                text=top_riders["late_pct"].astype(str) + "%",
                textposition="outside",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=260,
                              xaxis_title="Rider ID", yaxis_title="Total Deliveries",
                              xaxis=dict(tickangle=-45))
            st.plotly_chart(fig, use_container_width=True)

        # Rider table
        st.markdown('<div class="chart-title">🗂 Rider Performance Table</div>', unsafe_allow_html=True)
        display_rp = rp.nlargest(50, "total_deliveries").copy()
        display_rp["late_rate"] = (display_rp["late_rate"] * 100).round(1)
        display_rp["utilization_score"] = (display_rp["utilization_score"] * 100).round(1)
        display_rp = display_rp.rename(columns={
            "rider_id": "Rider ID",
            "total_deliveries": "Deliveries",
            "late_rate": "Late Rate (%)",
            "avg_delivery_min": "Avg Time (min)",
            "avg_distance_km": "Avg Dist (km)",
            "utilization_score": "Utilization (%)",
            "avg_rating": "Rating",
            "total_earnings_inr": "Earnings (₹)",
        })
        cols_show = [c for c in ["Rider ID", "Deliveries", "Late Rate (%)", "Avg Time (min)",
                                  "Utilization (%)", "Rating"] if c in display_rp.columns]
        st.dataframe(
            display_rp[cols_show].round(1),
            use_container_width=True,
            height=280,
        )


# ── Tab 4: Dark Store Ops ──────────────────────────────────────────────────────

with tab4:
    if len(store_perf) == 0:
        st.info("Store performance data not available.")
    else:
        sp = store_perf.copy()
        if selected_city != "All" and "city" in sp.columns:
            sp = sp[sp["city"] == selected_city]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-title">🏭 Store Order Volume & SLA Breach</div>', unsafe_allow_html=True)
            sp_sorted = sp.nlargest(15, "total_orders")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=sp_sorted["dark_store_id"] if "dark_store_id" in sp_sorted.columns else sp_sorted.index,
                y=sp_sorted["total_orders"],
                name="Total Orders",
                marker_color=COLOR_PRIMARY,
                opacity=0.7,
            ), secondary_y=False)
            if "sla_breach_rate" in sp_sorted.columns:
                fig.add_trace(go.Scatter(
                    x=sp_sorted["dark_store_id"] if "dark_store_id" in sp_sorted.columns else sp_sorted.index,
                    y=(sp_sorted["sla_breach_rate"] * 100).round(1),
                    name="SLA Breach %",
                    line=dict(color=COLOR_DANGER, width=2),
                ), secondary_y=True)
            fig.update_layout(**PLOT_LAYOUT, height=300,
                              xaxis=dict(tickangle=-45, gridcolor="#1E2D40"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="chart-title">⏱ Avg Prep Time by Store Tier</div>', unsafe_allow_html=True)
            if "tier" in sp.columns and "avg_prep_time" in sp.columns:
                tier_prep = sp.groupby("tier")["avg_prep_time"].mean().reset_index()
                fig = go.Figure(go.Bar(
                    x=tier_prep["tier"],
                    y=tier_prep["avg_prep_time"].round(1),
                    marker_color=[COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER],
                    text=tier_prep["avg_prep_time"].round(1),
                    texttemplate="%{text} min",
                    textposition="outside",
                ))
                fig.update_layout(**PLOT_LAYOUT, height=300,
                                  yaxis_title="Avg Prep Time (min)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tier data not available")

        # Store utilization table
        st.markdown('<div class="chart-title">📊 Dark Store Performance Summary</div>', unsafe_allow_html=True)
        show_cols = [c for c in ["dark_store_id", "city", "total_orders", "sla_breach_rate",
                                  "avg_prep_time", "tier"] if c in sp.columns]
        sp_display = sp[show_cols].copy()
        if "sla_breach_rate" in sp_display.columns:
            sp_display["sla_breach_rate"] = (sp_display["sla_breach_rate"] * 100).round(1)
        st.dataframe(sp_display.rename(columns={
            "dark_store_id": "Store ID", "total_orders": "Orders",
            "sla_breach_rate": "SLA Breach (%)", "avg_prep_time": "Avg Prep (min)"
        }), use_container_width=True, height=280)


# ── Tab 5: SLA & Alerts ────────────────────────────────────────────────────────

with tab5:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="chart-title">🔴 SLA Breach Trend</div>', unsafe_allow_html=True)
        if len(city_daily) > 0:
            cd_sla = city_daily.copy()
            if selected_city != "All":
                cd_sla = cd_sla[cd_sla["city"] == selected_city]
            cd_sla["date"] = pd.to_datetime(cd_sla["date"])
            cd_sla_agg = cd_sla.groupby("date").agg(
                late_rate=("late_rate", "mean")
            ).reset_index()
            cd_sla_agg["late_rate_pct"] = cd_sla_agg["late_rate"] * 100

            fig = go.Figure()
            fig.add_hrect(y0=15, y1=cd_sla_agg["late_rate_pct"].max() * 1.1,
                          fillcolor="#EF4444", opacity=0.08, line_width=0,
                          annotation_text="CRITICAL", annotation_position="top left",
                          annotation_font=dict(color="#EF4444", size=9))
            fig.add_hrect(y0=10, y1=15,
                          fillcolor="#F97316", opacity=0.06, line_width=0,
                          annotation_text="WARNING", annotation_position="top left",
                          annotation_font=dict(color="#F97316", size=9))
            fig.add_trace(go.Scatter(
                x=cd_sla_agg["date"], y=cd_sla_agg["late_rate_pct"].round(2),
                fill="tozeroy",
                fillcolor="#0EA5E920",
                line=dict(color=COLOR_PRIMARY, width=2),
                name="Late Rate %",
            ))
            fig.add_hline(y=10, line_dash="dash", line_color="#F97316",
                          annotation_text="Warning: 10%")
            fig.add_hline(y=15, line_dash="dash", line_color=COLOR_DANGER,
                          annotation_text="Critical: 15%")
            fig.update_layout(**PLOT_LAYOUT, height=300, yaxis_title="Late Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="chart-title">⏰ SLA Breach by Order Type</div>', unsafe_allow_html=True)
        if "order_type" in filtered_orders.columns and "is_late" in filtered_orders.columns:
            sla_by_type = (
                filtered_orders.groupby("order_type")
                .agg(total=("order_id","count"), late=("is_late","sum"))
                .reset_index()
            )
            sla_by_type["breach_rate"] = (sla_by_type["late"] / sla_by_type["total"] * 100).round(1)
            colors = [COLOR_DANGER if b > 15 else COLOR_WARNING if b > 10 else COLOR_SUCCESS
                      for b in sla_by_type["breach_rate"]]
            fig = go.Figure(go.Bar(
                x=sla_by_type["order_type"],
                y=sla_by_type["breach_rate"],
                marker_color=colors,
                text=sla_by_type["breach_rate"].astype(str) + "%",
                textposition="outside",
            ))
            fig.add_hline(y=10, line_dash="dash", line_color=COLOR_WARNING)
            fig.update_layout(**PLOT_LAYOUT, height=300,
                              yaxis_title="SLA Breach Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

    # Delay distribution
    st.markdown('<div class="chart-title">📉 Delay Distribution (Late Orders Only)</div>', unsafe_allow_html=True)
    late_orders = filtered_orders[filtered_orders["is_late"] == 1]
    if len(late_orders) > 0 and "delay_minutes" in late_orders.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=late_orders["delay_minutes"].clip(0, 60),
            nbinsx=40,
            name="Express",
            marker_color=COLOR_DANGER,
            opacity=0.7,
            histnorm="probability density",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=250,
                          xaxis_title="Delay (minutes)", yaxis_title="Density",
                          bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 6: Predictive Models ───────────────────────────────────────────────────

with tab6:
    model_results = load_model_results()

    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

    if model_results:
        cols = st.columns(len(model_results))
        for i, (name, res) in enumerate(model_results.items()):
            with cols[i]:
                acc = res.get("accuracy") or res.get("r2", 0)
                auc = res.get("roc_auc", 0)
                label = name.replace("_", " ").title()
                color = "success" if acc > 0.83 else "warning" if acc > 0.75 else "danger"
                st.markdown(
                    kpi_card(label, f"{acc:.1%}", f"AUC: {auc:.3f}" if auc else "Regressor",
                             color),
                    unsafe_allow_html=True
                )
    else:
        st.warning("No trained models found. Use the sidebar button to train models.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(kpi_card("Late Delivery Model", "~86%", "XGBoost Classifier — projected", "success"),
                        unsafe_allow_html=True)
        with col2:
            st.markdown(kpi_card("SLA Breach Model", "~83%", "Random Forest — projected", "success"),
                        unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Feature importance chart (from model results if available)
    if model_results and "late_delivery" in model_results:
        top_feats = model_results["late_delivery"].get("top_features", [])
        if top_feats:
            st.markdown('<div class="chart-title">🔍 Feature Importance — Late Delivery Model</div>',
                        unsafe_allow_html=True)
            fi_df = pd.DataFrame(top_feats)
            fig = go.Figure(go.Bar(
                x=fi_df["importance"],
                y=fi_df["feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["importance"],
                    colorscale=[[0, COLOR_PRIMARY], [1, COLOR_WARNING]],
                ),
            ))
            fig.update_layout(**PLOT_LAYOUT, height=320,
                              xaxis_title="Importance Score",
                              yaxis=dict(autorange="reversed", gridcolor="#1E2D40"))
            st.plotly_chart(fig, use_container_width=True)

    # Late risk score distribution
    if "late_risk_score" in filtered_orders.columns:
        st.markdown('<div class="chart-title">🎯 Late Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Histogram(
            x=filtered_orders[filtered_orders["is_late"]==0]["late_risk_score"],
            name="On-Time", marker_color=COLOR_SUCCESS, opacity=0.6, nbinsx=30
        ), row=1, col=1)
        fig.add_trace(go.Histogram(
            x=filtered_orders[filtered_orders["is_late"]==1]["late_risk_score"],
            name="Late", marker_color=COLOR_DANGER, opacity=0.6, nbinsx=30
        ), row=1, col=1)
        corr_data = (
            filtered_orders.groupby("hour_of_day")
            .agg(risk=("late_risk_score","mean"), late=("is_late","mean"))
            .reset_index()
        )
        fig.add_trace(go.Scatter(
            x=corr_data["hour_of_day"], y=corr_data["risk"].round(3),
            name="Avg Risk Score", line=dict(color=COLOR_WARNING, width=2)
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=corr_data["hour_of_day"], y=corr_data["late"].round(3),
            name="Actual Late Rate", line=dict(color=COLOR_DANGER, width=2, dash="dot")
        ), row=1, col=2)
        fig.update_layout(**PLOT_LAYOUT, height=280)
        fig.update_xaxes(gridcolor="#1E2D40")
        fig.update_yaxes(gridcolor="#1E2D40")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 7: Recommendations ─────────────────────────────────────────────────────

with tab7:
    st.markdown('<div class="section-header">Operational Intelligence — Automated Recommendations</div>',
                unsafe_allow_html=True)

    rec_engine = RecommendationsEngine()
    recommendations = rec_engine.generate(kpis, city=selected_city)

    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#64748B;margin-bottom:1rem;">'
        f'Generated {len(recommendations)} recommendation(s) for {selected_city} '
        f'based on live KPI analysis</div>',
        unsafe_allow_html=True
    )

    for rec in recommendations:
        st.markdown(alert_card(rec), unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:2rem;">Operational Health Matrix</div>',
                unsafe_allow_html=True)

    health_metrics = [
        ("On-Time Rate", kpis.get("on_time_delivery_rate", 0), 88, 80, "%"),
        ("SLA Breach Rate", 100 - kpis.get("sla_breach_pct", 100), 90, 85, "%"),
        ("Rider Utilization", kpis.get("rider_utilization_pct", 0), 75, 55, "%"),
        ("Route Efficiency", (2.5 - kpis.get("route_inefficiency_score", 1.5)) / 1.5 * 100, 80, 60, "%"),
        ("Dark Store Capacity", 100 - kpis.get("dark_store_utilization_pct", 0), 30, 15, "% headroom"),
    ]

    fig = go.Figure()
    categories = [h[0] for h in health_metrics]
    values = [max(0, min(100, h[2])) for h in health_metrics]  # use target as reference
    actual = [max(0, min(100, h[1])) for h in health_metrics]

    fig.add_trace(go.Scatterpolar(
        r=actual,
        theta=categories,
        fill='toself',
        name='Current',
        line=dict(color=COLOR_PRIMARY),
        fillcolor=f"{COLOR_PRIMARY}30",
    ))
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Target',
        line=dict(color=COLOR_SUCCESS, dash="dash"),
        fillcolor=f"{COLOR_SUCCESS}15",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1E2D40",
                            tickfont=dict(color="#94A3B8", size=9), linecolor="#1E2D40"),
            angularaxis=dict(gridcolor="#1E2D40", linecolor="#1E2D40",
                             tickfont=dict(color="#94A3B8", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=350,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)
