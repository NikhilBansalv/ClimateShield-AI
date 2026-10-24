from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==================================================
# CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
ANNUAL_DIR = BASE_DIR / "outputs" / "annual_data"
ML_DIR = BASE_DIR / "outputs" / "monthly_ml"

CLIMATE_FILE = ANNUAL_DIR / "ClimateShield_DA1_Cleaned_Dataset_1991_2025.csv"
RISK_FILE = ANNUAL_DIR / "ClimateShield_Risk_Assessment.csv"
MONTHLY_FILE = ML_DIR / "monthly_cleaned_dataset_1991_2025.csv"
METRICS_FILE = ML_DIR / "model_evaluation_metrics.csv"
SELECTION_FILE = ML_DIR / "model_selection_summary.csv"
TEST_PRED_FILE = ML_DIR / "test_period_predictions.csv"
PROJECTION_FILE = ML_DIR / "future_projections_2026_2035.csv"
PROJ_CHECK_FILE = ML_DIR / "projection_validation.csv"
IMPORTANCE_FILE = ML_DIR / "feature_importance.csv"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
UNITS = {"Temperature": "°C", "Precipitation": "mm"}
LEVEL_COLOURS = {"High": "#d62728", "Moderate": "#ff9f1c", "Low": "#2ca02c"}
BASELINE = "Seasonal Climatology Baseline"
FEATURE_LABELS = {
    "Lag_1": "Previous month", "Lag_2": "2 months ago", "Lag_3": "3 months ago",
    "Lag_12": "Same month last year", "Roll_3": "Last 3-month average",
    "Roll_12": "Last 12-month average", "Month_sin": "Season (sin)",
    "Month_cos": "Season (cos)", "Climatology": "Normal for the month",
}

st.set_page_config(page_title="ClimateShield AI", page_icon="🌍", layout="wide")


# ==================================================
# DATA LOADING
# ==================================================
@st.cache_data
def load_annual():
    return pd.read_csv(CLIMATE_FILE), pd.read_csv(RISK_FILE)


@st.cache_data
def load_ml():
    return {
        "monthly": pd.read_csv(MONTHLY_FILE, parse_dates=["Date"]),
        "metrics": pd.read_csv(METRICS_FILE),
        "selection": pd.read_csv(SELECTION_FILE),
        "test": pd.read_csv(TEST_PRED_FILE, parse_dates=["Date"]),
        "future": pd.read_csv(PROJECTION_FILE, parse_dates=["Date"]),
        "proj_check": pd.read_csv(PROJ_CHECK_FILE),
        "importance": (pd.read_csv(IMPORTANCE_FILE)
                       if IMPORTANCE_FILE.exists() else pd.DataFrame()),
    }


def missing(paths):
    return [p for p in paths if not p.exists()]


def style(fig, height=360, **kwargs):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=50, b=10), **kwargs)
    return fig


# ==================================================
# HEADER
# ==================================================
st.title("🌍 ClimateShield AI")
st.caption("Climate trends, regional exposure scores and climate estimates "
           "for 35 Indian regions (1991–2025, World Bank CCKP).")

gaps = missing([CLIMATE_FILE, RISK_FILE])
if gaps:
    st.error("Annual outputs are missing. Run `python main.py` first.\n\n"
             + "\n".join(f"- `{p.relative_to(BASE_DIR)}`" for p in gaps))
    st.stop()

climate_df, risk_df = load_annual()
overview_tab, predict_tab = st.tabs(["📊 Regional Overview", "🔮 Climate Predictions"])

# ==================================================
# TAB 1: REGIONAL OVERVIEW
# ==================================================
with overview_tab:
    region = st.selectbox("Region", sorted(climate_df["name"].unique()),
                          key="overview_region")
    series = climate_df[climate_df["name"] == region].sort_values("Year")
    risk = risk_df[risk_df["Region"] == region].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ClimateShield score", f"{risk['ClimateShield_Score']:.0f} / 100",
              risk["Risk_Level"], delta_color="off")
    k2.metric("Average max temperature", f"{risk['Mean_Temperature']:.1f} °C")
    k3.metric("Warming trend", f"{risk['Temperature_Trend']:+.2f} °C / decade")
    k4.metric("Rainfall variability", f"{risk['Rainfall_Variability']:.0f} %",
              help="Coefficient of variation of annual rainfall: higher means "
                   "less predictable year-to-year rainfall.")

    left, right = st.columns(2)
    with left:
        slope, intercept = np.polyfit(series["Year"], series["Max_Temperature"], 1)
        fig = go.Figure()
        fig.add_scatter(x=series["Year"], y=series["Max_Temperature"],
                        mode="lines+markers", name="Observed",
                        line=dict(color="#d62728"))
        fig.add_scatter(x=series["Year"], y=slope * series["Year"] + intercept,
                        mode="lines", name="Linear trend",
                        line=dict(color="#333", dash="dash"))
        style(fig, title="Annual maximum temperature (°C)",
              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(go.Bar(x=series["Year"], y=series["Precipitation"],
                               marker_color="#1f77b4"))
        fig.add_hline(y=series["Precipitation"].mean(), line_dash="dash",
                      line_color="#333", annotation_text="Average")
        style(fig, title="Annual precipitation (mm)")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### How does this region compare?")
    ranking = risk_df.sort_values("ClimateShield_Score")
    fig = go.Figure(go.Bar(
        x=ranking["ClimateShield_Score"], y=ranking["Region"], orientation="h",
        marker_color=[LEVEL_COLOURS[v] for v in ranking["Risk_Level"]],
        marker_line_color="black",
        marker_line_width=[2.5 if r == region else 0 for r in ranking["Region"]],
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    style(fig, height=700, xaxis_title="ClimateShield score (0–100)")
    st.plotly_chart(fig, width="stretch")
    st.caption("🟥 High · 🟧 Moderate · 🟩 Low. The selected region has a black outline. "
               "Scores are project-defined relative indicators (heat 70%, rainfall "
               "variability 30%), not validated disaster probabilities.")

# ==================================================
# TAB 2: CLIMATE PREDICTIONS
# ==================================================
with predict_tab:
    ml_gaps = missing([MONTHLY_FILE, METRICS_FILE, SELECTION_FILE, TEST_PRED_FILE,
                       PROJECTION_FILE, PROJ_CHECK_FILE])
    if ml_gaps:
        st.warning("Prediction outputs are missing. Run `python monthly_ml.py` first.\n\n"
                   + "\n".join(f"- `{p.relative_to(BASE_DIR)}`" for p in ml_gaps))
        st.stop()
    ml = load_ml()

    c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
    target = c1.radio("Variable", ["Temperature", "Precipitation"], key="target")
    pred_region = c2.selectbox("Region", sorted(ml["monthly"]["name"].unique()),
                               key="pred_region")
    year = c3.selectbox("Year", list(range(2026, 2036)), key="pred_year")
    month_name = c4.selectbox("Month", MONTHS, index=5, key="pred_month")
    month = MONTHS.index(month_name) + 1
    unit = UNITS[target]
    label = "temperature" if target == "Temperature" else "rainfall"
    color = "#d62728" if target == "Temperature" else "#1f77b4"

    # ---------- A. The estimate ----------
    future = ml["future"]
    future_line = future[(future["name"] == pred_region) & (future["Target"] == target)
                         & (future["Month"] == month)].sort_values("Year")
    row = future_line[future_line["Year"] == year].iloc[0]
    history = (ml["monthly"][(ml["monthly"]["name"] == pred_region)
                             & (ml["monthly"]["Month"] == month)]
               .dropna(subset=[target]).sort_values("Year"))
    difference = row["Prediction"] - history[target].mean()

    st.markdown(f"### {pred_region} · {month_name} {year}")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Estimated {label}", f"{row['Prediction']:.1f} {unit}")
    m2.metric("Likely range", f"{row['Lower_80']:.1f} – {row['Upper_80']:.1f} {unit}")
    m3.metric(f"vs. 1991–2025 average {month_name}", f"{difference:+.1f} {unit}",
              delta_color="off")

    fig = go.Figure()
    fig.add_scatter(x=future_line["Year"], y=future_line["Upper_80"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=future_line["Year"], y=future_line["Lower_80"], mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(120,120,120,0.2)", name="Likely range")
    fig.add_scatter(x=history["Year"], y=history[target], mode="lines+markers",
                    name=f"Observed {month_name}s", line=dict(color=color))
    fig.add_scatter(x=future_line["Year"], y=future_line["Prediction"],
                    mode="lines+markers", name="Estimate",
                    line=dict(color="#333", dash="dash"))
    fig.add_scatter(x=[year], y=[row["Prediction"]], mode="markers",
                    marker=dict(size=14, color="#333", symbol="star"),
                    name="Selected", showlegend=False)
    style(fig, height=380, title=f"{month_name} {label}: observed and estimated",
          yaxis_title=f"{target} ({unit})", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    check = ml["proj_check"][ml["proj_check"]["Target"] == target].iloc[0]
    if target == "Temperature":
        st.caption(
            "**Method:** the month's normal plus a warming trend (regional trend "
            "blended with the all-India trend). In a 2021–2025 backtest its error was "
            f"{check['Projection_MAE_2021_2025']:.2f} °C vs "
            f"{check['Climatology_MAE_2021_2025']:.2f} °C for the plain monthly normal, "
            f"and the range contained {check['Interval_Coverage_80pct']:.0%} of actual months.")
    else:
        st.caption(
            "**Method:** the region's normal rainfall for that month. No trend is "
            "applied because none of the tested models beat the monthly normal for "
            "rainfall. In a 2021–2025 backtest the range contained "
            f"{check['Interval_Coverage_80pct']:.0%} of actual months.")
    st.info("Long-range climate estimates, not weather forecasts. Use them for "
            "academic exploration, not safety decisions.", icon="ℹ️")

    # ---------- B. How good is the ML? ----------
    st.divider()
    st.markdown("### How accurate are the models?")
    st.caption("Next-month prediction tested on 2021–2025, which the models never saw "
               "during training. Models predict how far a month will deviate from "
               "normal, using the deviations of the previous 1, 2, 3 and 12 months.")

    metrics = ml["metrics"][ml["metrics"]["Target"] == target].copy()
    selected = ml["selection"].set_index("Target").loc[target, "Selected_Approach"]
    chosen = metrics[metrics["Approach"] == selected].iloc[0]

    a1, a2, a3 = st.columns(3)
    a1.metric("Selected approach", selected)
    a2.metric("Average error (MAE)", f"{chosen['MAE']:.2f} {unit}")
    a3.metric("Improvement over monthly normal",
              f"{chosen['Skill_vs_Baseline_%']:+.1f}%", delta_color="off")

    if selected == BASELINE:
        st.warning(
            f"**Finding:** no machine-learning model predicted monthly {label} better "
            "than the region's normal for that month. Monthly rainfall is dominated "
            "by unpredictable weather, so the app uses the normal. Reporting this "
            "honestly is part of good model validation.")
    else:
        st.success(
            f"**Finding:** {selected} reduces the error by "
            f"{chosen['Skill_vs_Baseline_%']:.0f}% compared with the monthly normal. "
            "Recent conditions carry useful information about the next month.")

    table = (metrics[["Approach", "MAE", "RMSE", "Skill_vs_Baseline_%"]]
             .rename(columns={"Skill_vs_Baseline_%": "Improvement vs normal (%)"})
             .sort_values("MAE").round(3))
    st.dataframe(table, hide_index=True, width="stretch")

    test = ml["test"][(ml["test"]["Target"] == target)
                      & (ml["test"]["name"] == pred_region)].sort_values("Date")
    fig = go.Figure()
    fig.add_scatter(x=test["Date"], y=test["Actual"], name="Actual",
                    line=dict(color="#333"))
    fig.add_scatter(x=test["Date"], y=test["Predicted"], name=f"Predicted ({selected})",
                    line=dict(color=color))
    if selected != BASELINE:
        fig.add_scatter(x=test["Date"], y=test["Baseline"], name="Monthly normal",
                        line=dict(color="#999", dash="dot"))
    style(fig, title=f"Actual vs predicted, {pred_region} (2021–2025)",
          yaxis_title=f"{target} ({unit})", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    importance = ml["importance"]
    if not importance.empty and (importance["Target"] == target).any():
        with st.expander("What does the model rely on?"):
            imp = importance[importance["Target"] == target].copy()
            imp["Predictor"] = imp["Feature"].map(FEATURE_LABELS)
            imp = imp.sort_values("Importance")
            fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Predictor"],
                                   orientation="h", marker_color=color))
            style(fig, height=320, xaxis_title=f"Error increase if shuffled ({unit})")
            st.plotly_chart(fig, width="stretch")
            st.caption("The previous month's deviation matters most: warm months "
                       "tend to follow warm months.")

st.divider()
st.caption("Source: World Bank Climate Change Knowledge Portal (CCKP), CRU historical "
           "data. Academic prototype: scores and estimates are preliminary.")