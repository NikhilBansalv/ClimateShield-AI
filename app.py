from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================================================
# CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
ANNUAL_DIR = OUTPUT_DIR / "annual_data"
MONTHLY_DIR = OUTPUT_DIR / "monthly_ml"

ANNUAL_DATA_FILE = ANNUAL_DIR / "ClimateShield_DA1_Cleaned_Dataset_1991_2025.csv"
RISK_FILE = ANNUAL_DIR / "ClimateShield_Risk_Assessment.csv"

MONTHLY_DATA_FILE = MONTHLY_DIR / "monthly_cleaned_dataset_1991_2025.csv"
MONTHLY_METRICS_FILE = MONTHLY_DIR / "model_evaluation_metrics.csv"
MONTHLY_PREDICTIONS_FILE = MONTHLY_DIR / "test_period_predictions.csv"
MONTHLY_SELECTION_FILE = MONTHLY_DIR / "model_selection_summary.csv"

st.set_page_config(
    page_title="ClimateShield AI",
    page_icon="🌍",
    layout="wide",
)

# ==================================================
# LOADERS
# ==================================================
@st.cache_data
def load_annual():
    climate = pd.read_csv(ANNUAL_DATA_FILE)
    risk = pd.read_csv(RISK_FILE)

    climate["Year"] = pd.to_numeric(climate["Year"], errors="coerce")
    for col in ["Max_Temperature", "Precipitation"]:
        climate[col] = pd.to_numeric(climate[col], errors="coerce")

    climate = climate.dropna(
        subset=["name", "Year", "Max_Temperature", "Precipitation"]
    ).copy()
    climate["Year"] = climate["Year"].astype(int)
    risk["ClimateShield_Score"] = pd.to_numeric(
        risk["ClimateShield_Score"], errors="coerce"
    )
    return climate, risk


@st.cache_data
def load_monthly():
    monthly = pd.read_csv(MONTHLY_DATA_FILE, parse_dates=["Date"])
    predictions = pd.read_csv(MONTHLY_PREDICTIONS_FILE, parse_dates=["Date"])
    metrics = pd.read_csv(MONTHLY_METRICS_FILE)
    selection = (
        pd.read_csv(MONTHLY_SELECTION_FILE)
        if MONTHLY_SELECTION_FILE.exists()
        else pd.DataFrame()
    )
    return monthly, predictions, metrics, selection


def missing_files(paths):
    return [p for p in paths if not p.exists()]


def make_forecast_features(frame):
    """Create seasonal features and a time trend for monthly forecasting."""
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Year"] = frame["Date"].dt.year
    frame["Month"] = frame["Date"].dt.month
    frame["Year_Trend"] = frame["Year"] - 1991
    frame["Month_sin"] = np.sin(2 * np.pi * frame["Month"] / 12)
    frame["Month_cos"] = np.cos(2 * np.pi * frame["Month"] / 12)
    return frame[["Year_Trend", "Month_sin", "Month_cos"]]


def forecast_one_region(monthly_data, region_name, target_name, target_year, target_month):
    """Evaluate on 2021–2025, then refit on all available history for a future estimate."""
    region_data = monthly_data[monthly_data["name"] == region_name].copy()
    region_data["Date"] = pd.to_datetime(region_data["Date"], errors="coerce")
    region_data[target_name] = pd.to_numeric(region_data[target_name], errors="coerce")
    region_data = region_data.dropna(subset=["Date", target_name]).sort_values("Date")

    train = region_data[region_data["Date"] < pd.Timestamp("2021-01-01")]
    test = region_data[region_data["Date"].between("2021-01-01", "2025-12-31")]
    if len(train) < 60 or test.empty:
        raise ValueError("Not enough historical data to evaluate this regional model.")

    model_eval = LinearRegression()
    model_eval.fit(make_forecast_features(train), train[target_name])
    test_predictions = model_eval.predict(make_forecast_features(test))
    mae = mean_absolute_error(test[target_name], test_predictions)
    rmse = mean_squared_error(test[target_name], test_predictions) ** 0.5

    model_final = LinearRegression()
    model_final.fit(make_forecast_features(region_data), region_data[target_name])
    future = pd.DataFrame({"Date": [pd.Timestamp(target_year, target_month, 1)]})
    prediction = float(model_final.predict(make_forecast_features(future))[0])
    if target_name == "Precipitation":
        prediction = max(0.0, prediction)
    return prediction, mae, rmse, len(region_data), region_data["Date"].max().year


# ==================================================
# HEADER
# ==================================================
st.title("🌍 ClimateShield AI")
st.caption(
    "Regional historical climate analysis, preliminary exposure scoring, "
    "and monthly model evaluation for 35 Indian regions."
)

annual_missing = missing_files([ANNUAL_DATA_FILE, RISK_FILE])
if annual_missing:
    st.error(
        "Annual outputs are missing. Run `python main.py` first.\n\n"
        + "\n".join(f"- `{p.relative_to(BASE_DIR)}`" for p in annual_missing)
    )
    st.stop()

try:
    climate_df, risk_df = load_annual()
except Exception as exc:
    st.error(f"Could not load annual data: {exc}")
    st.stop()

annual_tab, ml_tab = st.tabs(["📊 Annual Climate", "🤖 Monthly ML Predictions"])

# ==================================================
# TAB 1: ANNUAL CLIMATE
# ==================================================
with annual_tab:
    regions = sorted(climate_df["name"].dropna().unique())
    years = sorted(climate_df["Year"].unique())

    f1, f2 = st.columns([2, 1])
    with f1:
        region = st.selectbox("Region", regions, key="annual_region")
    with f2:
        year = st.selectbox(
            "Year",
            years,
            index=len(years) - 1,
            key="annual_year",
        )

    regional = climate_df[climate_df["name"] == region].sort_values("Year")
    selected = regional[regional["Year"] == year]
    risk_match = risk_df[risk_df["Region"] == region]

    a, b, c, d = st.columns(4)
    a.metric("Regions", climate_df["name"].nunique())
    a1, a2 = int(climate_df["Year"].min()), int(climate_df["Year"].max())
    b.metric("Study period", f"{a1}–{a2}")
    if not selected.empty:
        row = selected.iloc[0]
        c.metric("Max temperature", f"{row['Max_Temperature']:.1f} °C")
        d.metric("Annual precipitation", f"{row['Precipitation']:.0f} mm")
    else:
        c.metric("Max temperature", "N/A")
        d.metric("Annual precipitation", "N/A")

    if not risk_match.empty:
        score = risk_match.iloc[0]
        st.markdown("#### Preliminary regional exposure")
        s1, s2, s3 = st.columns(3)
        s1.metric("ClimateShield score", f"{score['ClimateShield_Score']:.1f}/100")
        s2.metric("Heat score", f"{score['Heat_Risk_Score']:.1f}/100")
        s3.metric("Rainfall variability", f"{score['Rainfall_Risk_Score']:.1f}/100")
        st.caption(
            f"Project-defined relative category: {score['Risk_Level']}. "
            "This score is not a validated disaster probability."
        )

    left, right = st.columns(2)
    with left:
        fig_t = px.line(
            regional,
            x="Year",
            y="Max_Temperature",
            markers=True,
            title=f"Annual maximum temperature — {region}",
            labels={"Max_Temperature": "Temperature (°C)", "Year": "Year"},
        )
        fig_t.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig_t, use_container_width=True)

    with right:
        fig_p = px.line(
            regional,
            x="Year",
            y="Precipitation",
            markers=True,
            title=f"Annual precipitation — {region}",
            labels={"Precipitation": "Precipitation (mm)", "Year": "Year"},
        )
        fig_p.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig_p, use_container_width=True)

    with st.expander("Compare regional exposure scores"):
        ranking = risk_df.sort_values("ClimateShield_Score", ascending=True)
        fig = px.bar(
            ranking,
            x="ClimateShield_Score",
            y="Region",
            color="Risk_Level",
            orientation="h",
            title="ClimateShield regional comparison",
            labels={"ClimateShield_Score": "Score (0–100)"},
            hover_data=["Heat_Risk_Score", "Rainfall_Risk_Score"],
        )
        fig.update_layout(template="plotly_white", height=650)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download regional scores",
            data=risk_df.to_csv(index=False).encode("utf-8"),
            file_name="ClimateShield_Risk_Assessment.csv",
            mime="text/csv",
        )

# ==================================================
# TAB 2: MONTHLY ML PREDICTIONS
# ==================================================
with ml_tab:
    st.markdown("### Model predictions on unseen historical months")
    st.caption(
        "Models are evaluated on 2021–2025, which is held out from training. "
        "This section shows historical test predictions, not forecasts for 2026 onward."
    )

    monthly_required = [
        MONTHLY_DATA_FILE,
        MONTHLY_METRICS_FILE,
        MONTHLY_PREDICTIONS_FILE,
    ]
    missing = missing_files(monthly_required)

    if missing:
        st.warning(
            "Monthly ML outputs are missing. Run `python monthly_ml.py` first.\n\n"
            + "\n".join(f"- `{p.relative_to(BASE_DIR)}`" for p in missing)
        )
    else:
        monthly_df, predictions_df, metrics_df, selection_df = load_monthly()

        target = st.radio(
            "Climate variable",
            ["Temperature", "Precipitation"],
            horizontal=True,
            key="ml_target",
        )
        unit = "°C" if target == "Temperature" else "mm"

        # Future monthly estimate: model is evaluated on 2021–2025 before refitting.
        st.markdown("### Predict a future month")
        st.caption(
            "Choose a region and month to estimate its temperature or precipitation. "
            "The model learns a linear time trend plus seasonal patterns from historical monthly data."
        )
        forecast_regions = sorted(monthly_df["name"].dropna().unique())
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            forecast_region = st.selectbox(
                "Indian region/state", forecast_regions, key="forecast_region"
            )
        with fc2:
            forecast_year = st.selectbox(
                "Forecast year", list(range(2026, 2036)), key="forecast_year"
            )
        with fc3:
            month_names = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            forecast_month_name = st.selectbox(
                "Month", month_names, key="forecast_month"
            )
            forecast_month = month_names.index(forecast_month_name) + 1

        if st.button("Generate monthly prediction", type="primary"):
            try:
                prediction, forecast_mae, forecast_rmse, history_count, last_year = forecast_one_region(
                    monthly_df, forecast_region, target, forecast_year, forecast_month
                )
                st.markdown(f"#### {forecast_region} · {forecast_month_name} {forecast_year}")
                result_col, mae_col, rmse_col = st.columns(3)
                result_col.metric(f"Estimated {target.lower()}", f"{prediction:.1f} {unit}")
                mae_col.metric("Historical test MAE", f"{forecast_mae:.2f} {unit}")
                rmse_col.metric("Historical test RMSE", f"{forecast_rmse:.2f} {unit}")
                st.caption(
                    f"Model: seasonal linear-trend regression · trained on {history_count:,} available monthly records through {last_year}. "
                    "MAE/RMSE are measured on 2021–2025 after training on data before 2021."
                )
                st.warning(
                    "This is an experimental long-term climate estimate, not a weather forecast. "
                    "Uncertainty increases for years farther beyond the historical record; use it for academic exploration, not safety decisions."
                )
            except Exception as exc:
                st.error(f"Could not generate the prediction: {exc}")

        st.divider()
        st.markdown("### Historical model evaluation")
        st.caption(
            "The charts below compare predictions for unseen historical months (2021–2025) against observed values."
        )
        target_metrics = metrics_df[metrics_df["Target"] == target].copy()
        if target_metrics.empty:
            st.info("No metrics are available for this variable.")
        else:
            selected_approach = "Not available"
            if not selection_df.empty:
                match = selection_df[selection_df["Target"] == target]
                if not match.empty:
                    selected_approach = str(
                        match.iloc[0]["Selected_Approach"]
                    )

            selected_rows = target_metrics[
                target_metrics["Approach"] == selected_approach
            ]
            if not selected_rows.empty:
                best = selected_rows.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("Selected approach", selected_approach)
                m2.metric("Test MAE", f"{best['MAE']:.3f} {unit}")
                m3.metric("Test RMSE", f"{best['RMSE']:.3f} {unit}")
            else:
                st.info(f"Selected approach: {selected_approach}")

            if selected_approach == "Seasonal Climatology Baseline":
                st.info(
                    "The seasonal baseline was selected on validation data "
                    "because it outperformed the tested ML models on validation MAE."
                )
            else:
                st.info(
                    f"{selected_approach} had the lowest validation MAE "
                    "among the approaches tested for this variable."
                )

            # Compare models with compact metrics.
            comparison = target_metrics[
                ["Approach", "MAE", "RMSE", "R2", "Test_Records"]
            ].copy()
            for col in ["MAE", "RMSE", "R2"]:
                comparison[col] = pd.to_numeric(
                    comparison[col], errors="coerce"
                ).round(3)

            with st.expander("Compare model performance"):
                fig_metrics = px.bar(
                    target_metrics.sort_values("MAE"),
                    x="Approach",
                    y="MAE",
                    title=f"{target} test MAE (lower is better)",
                    labels={"MAE": f"MAE ({unit})"},
                    hover_data=["RMSE", "R2"],
                )
                fig_metrics.update_layout(template="plotly_white")
                st.plotly_chart(fig_metrics, use_container_width=True)
                st.dataframe(
                    comparison, use_container_width=True, hide_index=True
                )

            region_options = ["All regions (average)"] + sorted(
                predictions_df["name"].dropna().unique()
            )
            selected_prediction_region = st.selectbox(
                "Show predictions for",
                region_options,
                key="ml_region",
            )

            actual_column = target
            plot_df = predictions_df.copy()
            if "Selected_Approach" in plot_df.columns:
                plot_df = plot_df[
                    plot_df["Selected_Approach"] == selected_approach
                ]

            if selected_prediction_region == "All regions (average)":
                plot_df = (
                    plot_df.groupby("Date")[[actual_column, "Predicted"]]
                    .mean()
                    .reset_index()
                    .sort_values("Date")
                )
                plot_title = f"{target}: actual vs predicted regional average"
            else:
                plot_df = plot_df[
                    plot_df["name"] == selected_prediction_region
                ].sort_values("Date")
                plot_title = (
                    f"{target}: actual vs predicted — "
                    f"{selected_prediction_region}"
                )

            if not plot_df.empty:
                fig_pred = px.line(
                    plot_df,
                    x="Date",
                    y=[actual_column, "Predicted"],
                    title=plot_title,
                    labels={
                        "Date": "Month",
                        "value": f"{target} ({unit})",
                        "variable": "Series",
                    },
                )
                fig_pred.update_layout(template="plotly_white", height=380)
                st.plotly_chart(fig_pred, use_container_width=True)

            with st.expander("View prediction records"):
                st.dataframe(
                    predictions_df[
                        predictions_df.get(
                            "Selected_Approach",
                            pd.Series(index=predictions_df.index, dtype=str),
                        ) == selected_approach
                    ].head(100),
                    use_container_width=True,
                    hide_index=True,
                )

            download_df = predictions_df.copy()
            if "Selected_Approach" in download_df.columns:
                download_df = download_df[
                    download_df["Selected_Approach"] == selected_approach
                ]
            st.download_button(
                f"Download {target.lower()} test predictions",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{target.lower()}_test_predictions.csv",
                mime="text/csv",
            )

            with st.expander("Data coverage"):
                st.write(
                    f"{len(monthly_df):,} regional-month rows across "
                    f"{monthly_df['name'].nunique()} regions."
                )
                st.write(
                    "Missing values:",
                    monthly_df[["Temperature", "Precipitation"]]
                    .isna()
                    .sum()
                    .to_dict(),
                )

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "Source: World Bank Climate Change Knowledge Portal (CCKP), historical CRU data. "
    "ClimateShield AI is an academic prototype; risk scores are preliminary."
)
