
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

CLEAN_FILE = OUTPUT_DIR / "ClimateShield_DA1_Cleaned_Dataset_1991_2025.csv"
RISK_FILE = OUTPUT_DIR / "ClimateShield_Risk_Assessment.csv"

st.set_page_config(
    page_title="ClimateShield AI",
    page_icon="🌍",
    layout="wide",
)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    climate = pd.read_csv(CLEAN_FILE)
    risk = pd.read_csv(RISK_FILE)

    climate["Year"] = pd.to_numeric(
        climate["Year"], errors="coerce"
    )
    risk["ClimateShield_Score"] = pd.to_numeric(
        risk["ClimateShield_Score"], errors="coerce"
    )

    climate = climate.dropna(
        subset=["Year", "Max_Temperature", "Precipitation"]
    )
    climate["Year"] = climate["Year"].astype(int)

    return climate, risk


st.title("🌍 ClimateShield AI")
st.subheader("Personalized Climate Risk Assessment and Adaptation Planner")

st.caption(
    "Explore historical climate conditions and preliminary climate-exposure "
    "scores across 35 Indian regions (1991–2025)."
)

if not CLEAN_FILE.exists() or not RISK_FILE.exists():
    st.error(
        "Required output files were not found. Run main.py first "
        "and check that the cleaned dataset and risk assessment CSV "
        "exist inside the outputs folder."
    )
    st.stop()

try:
    climate_df, risk_df = load_data()
except Exception as exc:
    st.error(f"Unable to load the project data: {exc}")
    st.stop()

required_climate = {
    "name", "Year", "Max_Temperature", "Precipitation"
}
required_risk = {
    "Region", "Heat_Risk_Score",
    "Rainfall_Risk_Score", "ClimateShield_Score", "Risk_Level"
}

if not required_climate.issubset(climate_df.columns):
    st.error("The cleaned dataset is missing required columns.")
    st.stop()

if not required_risk.issubset(risk_df.columns):
    st.error("The risk assessment file is missing required columns.")
    st.stop()

# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Explore Climate Data")

regions = sorted(climate_df["name"].dropna().unique())
selected_region = st.sidebar.selectbox(
    "Select an Indian region",
    regions,
)

years = sorted(climate_df["Year"].unique())
selected_year = st.sidebar.slider(
    "Select year",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=int(max(years)),
)

region_data = climate_df[
    climate_df["name"] == selected_region
].sort_values("Year")

selected_data = region_data[
    region_data["Year"] == selected_year
]

risk_match = risk_df[
    risk_df["Region"] == selected_region
]

# --------------------------------------------------
# 4. OVERVIEW METRICS
# --------------------------------------------------
st.markdown("### Climate Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Regions Analyzed", f"{climate_df['name'].nunique()}")
c2.metric(
    "Historical Period",
    f"{int(climate_df['Year'].min())}–{int(climate_df['Year'].max())}",
)
c3.metric("Records Analyzed", f"{len(climate_df):,}")
c4.metric("Selected Region", selected_region)

st.divider()

# --------------------------------------------------
# 5. SELECTED REGION INDICATORS
# --------------------------------------------------
st.markdown(f"### Regional Profile: {selected_region}")

if not selected_data.empty:
    row = selected_data.iloc[0]

    m1, m2 = st.columns(2)

    m1.metric(
        f"Average Daily Maximum Temperature ({selected_year})",
        f"{row['Max_Temperature']:.2f} °C",
    )

    m2.metric(
        f"Annual Precipitation ({selected_year})",
        f"{row['Precipitation']:.2f} mm",
    )
else:
    st.info("No observations are available for the selected year.")

if not risk_match.empty:
    result = risk_match.iloc[0]

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "ClimateShield Score",
        f"{result['ClimateShield_Score']:.2f}/100",
    )
    r2.metric(
        "Heat Exposure Score",
        f"{result['Heat_Risk_Score']:.2f}/100",
    )
    r3.metric(
        "Rainfall Variability Stress",
        f"{result['Rainfall_Risk_Score']:.2f}/100",
    )

    st.info(
        f"Preliminary category: **{result['Risk_Level']}**. "
        "This is a relative, indicator-based score, not a validated "
        "prediction of disaster probability or individual vulnerability."
    )

# --------------------------------------------------
# 6. HISTORICAL CLIMATE TRENDS
# --------------------------------------------------
st.divider()
st.markdown("### Historical Climate Trends")

left, right = st.columns(2)

with left:
    fig_temp = px.line(
        region_data,
        x="Year",
        y="Max_Temperature",
        markers=True,
        title=f"Annual Average Daily Maximum Temperature — {selected_region}",
        labels={
            "Year": "Year",
            "Max_Temperature": "Temperature (°C)",
        },
    )
    fig_temp.update_layout(
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with right:
    fig_rain = px.line(
        region_data,
        x="Year",
        y="Precipitation",
        markers=True,
        title=f"Annual Precipitation — {selected_region}",
        labels={
            "Year": "Year",
            "Precipitation": "Precipitation (mm)",
        },
    )
    fig_rain.update_layout(
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_rain, use_container_width=True)

# --------------------------------------------------
# 7. COMPARE ALL REGIONS
# --------------------------------------------------
st.divider()
st.markdown("### ClimateShield Regional Comparison")

ranking = risk_df.sort_values(
    "ClimateShield_Score", ascending=False
)

fig_risk = px.bar(
    ranking,
    x="ClimateShield_Score",
    y="Region",
    color="Risk_Level",
    orientation="h",
    title="Preliminary ClimateShield Scores Across Indian Regions",
    labels={
        "ClimateShield_Score": "ClimateShield Score (0–100)",
        "Region": "Region",
        "Risk_Level": "Preliminary Category",
    },
    hover_data=["Heat_Risk_Score", "Rainfall_Risk_Score"],
)
fig_risk.update_layout(
    template="plotly_white",
    height=max(500, len(ranking) * 23),
    yaxis={"categoryorder": "total ascending"},
)
st.plotly_chart(fig_risk, use_container_width=True)

# --------------------------------------------------
# 8. RULE-BASED ADAPTATION PLANNER
# --------------------------------------------------
st.divider()
st.markdown("### 🌱 Climate Adaptation Planner")

st.write(
    "The suggestions below are preliminary, indicator-based guidance. "
    "They are not generated by a trained AI model."
)

recommendations = []

if not risk_match.empty:
    result = risk_match.iloc[0]

    if result["Heat_Risk_Score"] >= 66.67:
        recommendations.extend([
            "Prioritize heat-awareness measures and access to shaded, "
            "cool environments.",
            "Plan hydration and heat-safety practices during hot periods.",
            "Consider local heat-action plans when available.",
        ])
    elif result["Heat_Risk_Score"] >= 33.33:
        recommendations.extend([
            "Monitor seasonal temperature conditions and heat advisories.",
            "Review access to shade, drinking water and cool indoor spaces.",
        ])
    else:
        recommendations.extend([
            "Continue monitoring temperature trends and local advisories.",
            "Maintain basic heat preparedness during unusually hot periods.",
        ])

    if result["Rainfall_Risk_Score"] >= 66.67:
        recommendations.extend([
            "Monitor rainfall forecasts and official weather warnings.",
            "Review water-storage, drainage and water-conservation practices "
            "according to local conditions.",
        ])
    elif result["Rainfall_Risk_Score"] >= 33.33:
        recommendations.extend([
            "Track seasonal rainfall variation and local water availability.",
            "Plan water use according to local rainfall and storage conditions.",
        ])
    else:
        recommendations.extend([
            "Continue monitoring rainfall and water availability.",
            "Maintain appropriate water-conservation practices.",
        ])

    for item in recommendations:
        st.markdown(f"- {item}")

    st.caption(
        "The score thresholds are project-defined categories. "
        "They do not represent official government risk thresholds."
    )

# --------------------------------------------------
# 9. DATA TABLE AND DOWNLOAD
# --------------------------------------------------
st.divider()
st.markdown("### Regional Risk Assessment Table")

display_columns = [
    "Region",
    "Heat_Risk_Score",
    "Rainfall_Risk_Score",
    "ClimateShield_Score",
    "Risk_Level",
]

st.dataframe(
    ranking[display_columns].round(2),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Risk Assessment CSV",
    data=ranking.to_csv(index=False).encode("utf-8"),
    file_name="ClimateShield_Risk_Assessment.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 10. METHODOLOGY AND LIMITATIONS
# --------------------------------------------------
with st.expander("Scoring methodology"):
    st.markdown("""
    **Heat Exposure Score**

    - 50%: normalized mean maximum temperature
    - 30%: normalized historical temperature trend
    - 20%: normalized recent temperature change

    **Rainfall Variability Stress Score**

    - 100%: normalized rainfall coefficient of variation

    **Overall ClimateShield Score**

    - 70%: Heat Exposure Score
    - 30%: Rainfall Variability Stress Score

    Each component is min–max normalized to a 0–100 scale across the
    35 analyzed regions. The score is intended for relative comparison.
    """)

with st.expander("Limitations"):
    st.markdown("""
    - The assessment uses historical temperature and precipitation data.
    - Annual indicators do not capture every extreme weather event.
    - Population exposure, infrastructure and socioeconomic vulnerability
      are not incorporated.
    - The score has not been validated against observed disaster impacts.
    - Recommendations are rule-based and are not personalized to an
      individual's health, assets or circumstances.
    - Historical patterns should not be interpreted as future forecasts.
    """)

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Data source: World Bank Climate Change Knowledge Portal (CCKP), "
    "historical CRU climate data. "
    "ClimateShield AI — academic prototype."
)
