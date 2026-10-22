import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import linregress
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ANNUAL_OUTPUT_DIR = BASE_DIR / "outputs" / "annual_data"
GRAPHS_DIR = ANNUAL_OUTPUT_DIR / "graphs"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD CCKP DATASETS
temp = pd.read_csv(DATA_DIR / "gw_tsmax.csv")
rain = pd.read_csv(DATA_DIR / "gw_pr.csv")

print("Temperature dataset shape:", temp.shape)
print("Rainfall dataset shape:", rain.shape)

# 2. CLEAN COLUMN NAMES AND IDENTIFIERS
for df in (temp, rain):
    df.columns = df.columns.astype(str).str.strip()
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

# 3. CONVERT WIDE FORMAT → LONG FORMAT
def to_long(df, value_name):

    # Identify year/date columns such as 1901-07, 1902-07, etc.
    year_cols = [
        c for c in df.columns
        if str(c).split("-")[0].isdigit()
    ]

    out = df.melt(
        id_vars=["code", "name"],
        value_vars=year_cols,
        var_name="Date_Label",
        value_name=value_name
    )

    # Extract year from labels like "1991-07"
    out["Year"] = (
        out["Date_Label"]
        .astype(str)
        .str.extract(r"^(\d{4})")[0]
        .astype(int)
    )

    # Convert climate values to numeric
    out[value_name] = pd.to_numeric(
        out[value_name],
        errors="coerce"
    )

    return out[
        ["code", "name", "Year", value_name]
    ]


temp_long = to_long(temp, "Max_Temperature")
rain_long = to_long(rain, "Precipitation")

# 4. MERGE TEMPERATURE + PRECIPITATION
climate_df = pd.merge(
    temp_long,
    rain_long,
    on=["code", "name", "Year"],
    how="inner",
    validate="one_to_one"
)

# 5. SELECT 1991–2025 FOR DA1 ANALYSIS
climate_df = climate_df[
    climate_df["Year"].between(1991, 2025)
].copy()

# 6. HANDLE MISSING VALUES
numeric_cols = [
    "Max_Temperature",
    "Precipitation"
]

for col in numeric_cols:
    # Fill missing values using the median
    climate_df[col] = climate_df[col].fillna(
        climate_df[col].median()
    )

# 7. REMOVE DUPLICATES
climate_df = climate_df.drop_duplicates(
    subset=["code", "name", "Year"]
).reset_index(drop=True)

# 8. CREATE DERIVED CLIMATE FEATURES

# Temperature anomaly relative to each region's
# 1991–2025 average
climate_df["Temp_Anomaly"] = (
    climate_df["Max_Temperature"]
    - climate_df.groupby("name")["Max_Temperature"]
      .transform("mean")
)

# Rainfall deviation relative to each region's
# 1991–2025 average
climate_df["Rainfall_Deviation"] = (
    climate_df["Precipitation"]
    - climate_df.groupby("name")["Precipitation"]
      .transform("mean")
)

# 9. FINAL DATA QUALITY CHECKS
print("\nFinal dataset shape:", climate_df.shape)

print(
    "Number of regions:",
    climate_df["name"].nunique()
)

print(
    "Year range:",
    climate_df["Year"].min(),
    "to",
    climate_df["Year"].max()
)

print("\nMissing values:")
print(
    climate_df[
        ["Max_Temperature", "Precipitation"]
    ].isnull().sum()
)

print("\nDuplicate rows:")
print(
    climate_df.duplicated(
        subset=["code", "name", "Year"]
    ).sum()
)

print("\nSample of cleaned data:")
print(climate_df.head())

# 10. SAVE CLEANED DATASET
climate_df.to_csv(
    ANNUAL_OUTPUT_DIR / "ClimateShield_DA1_Cleaned_Dataset_1991_2025.csv",
    index=False
)

print(
    "\nCleaned dataset saved successfully!"
)

# 11. EXPLORATORY DATA ANALYSIS (EDA)
# Graphs are saved in outputs/annual_data/graphs/

# 11.1 Average Maximum Temperature by Region
region_temp = (
    climate_df
    .groupby("name")["Max_Temperature"]
    .mean()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 REGIONS BY AVERAGE MAXIMUM TEMPERATURE")
print("=" * 60)

print(region_temp.head(10))


plt.figure(figsize=(10, 6))

region_temp.head(10).sort_values().plot(
    kind="barh"
)

plt.xlabel("Average Maximum Temperature (°C)")
plt.ylabel("Region")
plt.title(
    "Top 10 Indian Regions by Average Maximum Temperature (1991–2025)"
)

plt.tight_layout()

plt.savefig(
    GRAPHS_DIR / "01_average_temperature.png",
    dpi=300
)

plt.show()

# 11.2 Average Annual Precipitation by Region
region_rain = (
    climate_df
    .groupby("name")["Precipitation"]
    .mean()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 REGIONS BY AVERAGE ANNUAL PRECIPITATION")
print("=" * 60)

print(region_rain.head(10))


plt.figure(figsize=(10, 6))

region_rain.head(10).sort_values().plot(
    kind="barh"
)

plt.xlabel("Average Annual Precipitation (mm)")
plt.ylabel("Region")
plt.title(
    "Top 10 Indian Regions by Average Annual Precipitation (1991–2025)"
)

plt.tight_layout()

plt.savefig(
    GRAPHS_DIR / "02_average_rainfall.png",
    dpi=300
)

plt.show()

# 11.3 Annual Temperature Trend
annual_temp = (
    climate_df
    .groupby("Year")["Max_Temperature"]
    .mean()
)

print("\n" + "=" * 60)
print("ANNUAL TEMPERATURE DATA")
print("=" * 60)

print(annual_temp)


plt.figure(figsize=(10, 6))

plt.plot(
    annual_temp.index,
    annual_temp.values,
    marker="o",
    markersize=3
)

plt.xlabel("Year")
plt.ylabel("Average Maximum Temperature (°C)")
plt.title(
    "Average Maximum Temperature Trend Across Indian Regions"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    GRAPHS_DIR / "03_temperature_trend.png",
    dpi=300
)

plt.show()

# 11.4 Annual Rainfall Trend
annual_rain = (
    climate_df
    .groupby("Year")["Precipitation"]
    .mean()
)

print("\n" + "=" * 60)
print("ANNUAL RAINFALL DATA")
print("=" * 60)

print(annual_rain)


plt.figure(figsize=(10, 6))

plt.plot(
    annual_rain.index,
    annual_rain.values,
    marker="o",
    markersize=3
)

plt.xlabel("Year")
plt.ylabel("Average Annual Precipitation (mm)")
plt.title(
    "Average Annual Precipitation Trend Across Indian Regions"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    GRAPHS_DIR / "04_rainfall_trend.png",
    dpi=300
)

plt.show()

# 11.5 Rainfall Variability by Region
rainfall_stats = (
    climate_df
    .groupby("name")["Precipitation"]
    .agg(["mean", "std"])
)

rainfall_stats["CV"] = (
    rainfall_stats["std"]
    / rainfall_stats["mean"]
    * 100
)

rainfall_variability = (
    rainfall_stats["CV"]
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 REGIONS BY RAINFALL VARIABILITY")
print("=" * 60)

print(rainfall_variability.head(10))


plt.figure(figsize=(10, 6))

rainfall_variability.head(10).sort_values().plot(
    kind="barh"
)

plt.xlabel("Coefficient of Variation (%)")
plt.ylabel("Region")
plt.title(
    "Top 10 Indian Regions by Rainfall Variability"
)

plt.tight_layout()

plt.savefig(
    GRAPHS_DIR / "05_rainfall_variability.png",
    dpi=300
)

plt.show()

# 11.6 Save Regional Summary
regional_summary = (
    climate_df
    .groupby("name")
    .agg(
        Average_Max_Temperature=("Max_Temperature", "mean"),
        Average_Precipitation=("Precipitation", "mean"),
        Temperature_Anomaly=("Temp_Anomaly", "mean"),
        Rainfall_Deviation=("Rainfall_Deviation", "mean")
    )
    .reset_index()
)

regional_summary.to_csv(
    ANNUAL_OUTPUT_DIR / "regional_summary.csv",
    index=False
)

print("\nRegional summary saved successfully!")

# 12. PRELIMINARY CLIMATE RISK SCORE
# 12.1 Calculate regional climate indicators
risk_data = []

for region_name, group in climate_df.groupby("name"):

    group = group.sort_values("Year")

    # Average maximum temperature
    mean_temp = group["Max_Temperature"].mean()

    # Temperature trend
    temp_trend = linregress(
        group["Year"],
        group["Max_Temperature"]
    ).slope * 10

    # 1991–2020 baseline
    baseline = group[
        group["Year"].between(1991, 2020)
    ]["Max_Temperature"].mean()

    # 2021–2025 recent period
    recent = group[
        group["Year"].between(2021, 2025)
    ]["Max_Temperature"].mean()

    # Recent temperature change
    recent_change = recent - baseline

    # Rainfall variability
    rainfall_mean = group["Precipitation"].mean()
    rainfall_std = group["Precipitation"].std()

    rainfall_cv = (
        rainfall_std / rainfall_mean
    ) * 100

    risk_data.append({
        "Region": region_name,
        "Mean_Temperature": mean_temp,
        "Temperature_Trend": temp_trend,
        "Recent_Temperature_Change": recent_change,
        "Rainfall_Variability": rainfall_cv
    })


risk_df = pd.DataFrame(risk_data)

# 12.2 Normalize indicators to 0–100
scaler = MinMaxScaler()

risk_df[
    [
        "Mean_Temp_Normalized",
        "Temp_Trend_Normalized",
        "Recent_Change_Normalized",
        "Rainfall_Variability_Normalized"
    ]
] = scaler.fit_transform(
    risk_df[
        [
            "Mean_Temperature",
            "Temperature_Trend",
            "Recent_Temperature_Change",
            "Rainfall_Variability"
        ]
    ]
) * 100

# 12.3 Calculate Heat Risk Score
risk_df["Heat_Risk_Score"] = (
    0.50 * risk_df["Mean_Temp_Normalized"]
    + 0.30 * risk_df["Temp_Trend_Normalized"]
    + 0.20 * risk_df["Recent_Change_Normalized"]
)

# 12.4 Calculate Rainfall Variability Score
risk_df["Rainfall_Risk_Score"] = (
    risk_df["Rainfall_Variability_Normalized"]
)

# 12.5 Calculate Overall ClimateShield Score
risk_df["ClimateShield_Score"] = (
    0.70 * risk_df["Heat_Risk_Score"]
    + 0.30 * risk_df["Rainfall_Risk_Score"]
)

# 12.6 Assign Risk Categories
risk_df["Risk_Level"] = pd.cut(
    risk_df["ClimateShield_Score"],
    bins=[-1, 33.33, 66.67, 100],
    labels=["Low", "Moderate", "High"]
)

# 12.7 Sort by risk
risk_df = risk_df.sort_values(
    "ClimateShield_Score",
    ascending=False
).reset_index(drop=True)

# 12.8 Display results
print("\n" + "=" * 60)
print("CLIMATESHIELD PRELIMINARY RISK RANKING")
print("=" * 60)

print(
    risk_df[
        [
            "Region",
            "Heat_Risk_Score",
            "Rainfall_Risk_Score",
            "ClimateShield_Score",
            "Risk_Level"
        ]
    ].head(10)
)

# 12.9 Save risk assessment
risk_df.to_csv(
    ANNUAL_OUTPUT_DIR / "ClimateShield_Risk_Assessment.csv",
    index=False
)

print(
    "\nRisk assessment saved successfully!"
)

# 13. REGIONAL TEMPERATURE WARMING TREND
temperature_trends = []

for region_name, group in climate_df.groupby("name"):

    group = group.sort_values("Year")

    slope = linregress(
        group["Year"],
        group["Max_Temperature"]
    ).slope

    temperature_trends.append({
        "Region": region_name,
        "Temperature_Trend_C_per_Decade": slope * 10
    })


trend_df = pd.DataFrame(temperature_trends)

trend_df = trend_df.sort_values(
    "Temperature_Trend_C_per_Decade",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 60)
print("TOP 10 REGIONS BY TEMPERATURE WARMING TREND")
print("=" * 60)

print(trend_df.head(10))


# Plot

plt.figure(figsize=(10, 6))

trend_df.head(10).sort_values(
    "Temperature_Trend_C_per_Decade"
).plot(
    x="Region",
    y="Temperature_Trend_C_per_Decade",
    kind="barh",
    legend=False
)

plt.xlabel("Temperature Trend (°C per decade)")
plt.ylabel("Region")

plt.title(
    "Top 10 Indian Regions by Temperature Warming Trend (1991–2025)"
)

plt.tight_layout()

plt.savefig(
    GRAPHS_DIR / "06_temperature_warming_trend.png",
    dpi=300
)

plt.show()


# Save results
trend_df.to_csv(
    ANNUAL_OUTPUT_DIR / "temperature_trends.csv",
    index=False
)

print("\nTemperature trend analysis saved successfully!")

# 14. DISTRIBUTION OF TEMPERATURE WARMING TRENDS
plt.figure(figsize=(10, 6))

plt.hist(
    trend_df["Temperature_Trend_C_per_Decade"],
    bins=8,
    edgecolor="black"
)

plt.xlabel("Temperature Warming Trend (°C per decade)")
plt.ylabel("Number of Regions")

plt.title(
    "Distribution of Temperature Warming Trends Across 35 Indian Regions"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPHS_DIR / "07_temperature_trend_distribution.png",
    dpi=300
)

plt.show()
print(f"\nAll annual outputs saved in: {ANNUAL_OUTPUT_DIR}")
