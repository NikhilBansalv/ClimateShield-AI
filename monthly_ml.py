from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "monthly_ml"
GRAPH_DIR = OUTPUT_DIR / "graphs"

for folder in (OUTPUT_DIR, GRAPH_DIR):
    folder.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("1991-01-01")
END_DATE = pd.Timestamp("2025-12-01")
VALIDATION_START = pd.Timestamp("2016-01-01")
TEST_START = pd.Timestamp("2021-01-01")
FORECAST_YEARS = range(2026, 2036)
BASELINE = "Seasonal Climatology Baseline"
TARGETS = ("Temperature", "Precipitation")
RANDOM_STATE = 42

LAG_FEATURES = ["Lag_1", "Lag_2", "Lag_3", "Lag_12", "Roll_3", "Roll_12"]
FEATURES = LAG_FEATURES + ["Month_sin", "Month_cos", "Climatology"]

def find_regional_file(variable):
    candidates = sorted(DATA_DIR.glob(f"*{variable}*monthly*.xlsx"))

    if not candidates:
        raise FileNotFoundError(
            f"No monthly Excel file matching '*{variable}*monthly*.xlsx' "
            f"was found in: {DATA_DIR}"
        )

    valid = []
    for path in candidates:
        try:
            preview = pd.read_excel(path, sheet_name=0)
            preview.columns = preview.columns.astype(str).str.strip()

            if (
                {"code", "name"}.issubset(preview.columns)
                and preview["name"].nunique(dropna=True) > 1
            ):
                valid.append((path, len(preview), preview["name"].nunique()))
        except Exception as exc:
            print(f"Could not inspect {path.name}: {exc}")

    if not valid:
        raise ValueError(
            f"No regional dataset was found for '{variable}'. "
            "Check that the ADM1 monthly Excel file is in the data folder."
        )

    valid.sort(key=lambda item: (item[2], item[1]), reverse=True)
    selected_path, row_count, region_count = valid[0]

    if region_count < 30:
        raise ValueError(
            f"{selected_path.name} contains only {region_count} unique regions; "
            "expected approximately 35."
        )

    print(f"Selected {variable} file: {selected_path.name}")
    print(f"Rows: {row_count} | Unique regions: {region_count}")
    return selected_path

def load_monthly_data(file_path, value_name):
    df = pd.read_excel(file_path, sheet_name=0)
    df.columns = df.columns.astype(str).str.strip()

    required = {"code", "name"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{file_path.name} must contain 'code' and 'name' columns."
        )

    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    date_columns = [
        col for col in df.columns
        if re.fullmatch(r"\d{4}-\d{2}", col)
    ]

    if not date_columns:
        date_columns = [
            col for col in df.columns
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", col)
        ]

    if not date_columns:
        raise ValueError(
            f"No monthly date columns were found in {file_path.name}. "
            "Expected labels such as '1991-01'."
        )

    long_df = df[["code", "name"] + date_columns].melt(
        id_vars=["code", "name"],
        value_vars=date_columns,
        var_name="Date",
        value_name=value_name,
    )

    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df = long_df.dropna(subset=["Date"])

    long_df = long_df[
        long_df["Date"].between(START_DATE, END_DATE)
    ].copy()

    duplicate_count = long_df.duplicated(["code", "Date"]).sum()
    if duplicate_count:
        print(
            f"Warning: {duplicate_count} duplicate region-month rows "
            f"found in {file_path.name}; keeping the first."
        )
        long_df = long_df.drop_duplicates(["code", "Date"], keep="first")

    return long_df[["code", "name", "Date", value_name]]

def make_dataset(data, target, climatology_end):
    df = data[["code", "name", "Date", "Year", "Month", target]].copy()
    df = df.sort_values(["name", "Date"]).reset_index(drop=True)

    climatology = (
        df[df["Date"] < climatology_end]
        .groupby(["name", "Month"])[target].mean()
        .rename("Climatology")
    )
    df = df.join(climatology, on=["name", "Month"])
    df["Anomaly"] = df[target] - df["Climatology"]

    by_region = df.groupby("name")["Anomaly"]
    for lag in (1, 2, 3, 12):
        df[f"Lag_{lag}"] = by_region.shift(lag)
    df["Roll_3"] = by_region.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )
    df["Roll_12"] = by_region.transform(
        lambda s: s.shift(1).rolling(12, min_periods=8).mean()
    )
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df.dropna(subset=FEATURES + ["Anomaly", target]).copy()


def model_factories():
    return {
        "Ridge Regression": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=10.0)
        ),
        "Extra Trees": lambda: ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def to_value(frame, anomaly_prediction, target):
    value = frame["Climatology"].to_numpy() + anomaly_prediction
    return np.clip(value, 0, None) if target == "Precipitation" else value

def score(actual, predicted):
    return (
        mean_absolute_error(actual, predicted),
        float(np.sqrt(mean_squared_error(actual, predicted))),
        r2_score(actual, predicted),
    )

def project(history, target, future_dates, trend_weight=0.5, widen=1.0):
    hist = history.dropna(subset=[target]).copy()
    hist["t"] = hist["Year"] + (hist["Month"] - 1) / 12
    clim = hist.groupby(["name", "Month"])[target].transform("mean")
    hist["anom"] = hist[target] - clim

    if target == "Temperature":
        national = hist.groupby("t")["anom"].mean()
        national_slope = np.polyfit(national.index, national.values, 1)[0]
        slopes = {
            n: trend_weight * np.polyfit(g["t"], g["anom"], 1)[0]
               + (1 - trend_weight) * national_slope
            for n, g in hist.groupby("name")
        }
    else:
        slopes = {n: 0.0 for n in hist["name"].unique()}
    centre = hist.groupby("name")["t"].mean()

    fitted = clim + hist["name"].map(slopes) * (hist["t"] - hist["name"].map(centre))
    hist["resid"] = hist[target] - fitted
    q = (
        hist.groupby(["name", "Month"])["resid"]
        .quantile([0.10, 0.90]).unstack()
        .rename(columns={0.10: "q_lo", 0.90: "q_hi"})
    )
    month_mean = hist.groupby(["name", "Month"])[target].mean().rename("base")

    out = pd.DataFrame(
        [(n, d) for n in sorted(slopes) for d in future_dates],
        columns=["name", "Date"],
    )
    out["Month"] = out["Date"].dt.month
    out["t"] = out["Date"].dt.year + (out["Month"] - 1) / 12
    out = out.join(month_mean, on=["name", "Month"]).join(q, on=["name", "Month"])
    out["Prediction"] = out["base"] + out["name"].map(slopes) * (
        out["t"] - out["name"].map(centre)
    )
    out["Lower_80"] = out["Prediction"] + widen * out["q_lo"]
    out["Upper_80"] = out["Prediction"] + widen * out["q_hi"]
    if target == "Precipitation":
        for col in ("Prediction", "Lower_80", "Upper_80"):
            out[col] = out[col].clip(lower=0)
    out["Target"] = target
    out["Year"] = out["Date"].dt.year
    return out[["name", "Target", "Date", "Year", "Month",
                "Prediction", "Lower_80", "Upper_80"]]


def coverage(projection, data, target):
    actual = data[["name", "Date", target]].dropna()
    merged = projection.merge(actual, on=["name", "Date"])
    inside = merged[target].between(merged["Lower_80"], merged["Upper_80"])
    return merged, float(inside.mean())


def calibrate_widen(data, target):
    pre = data[data["Date"] < VALIDATION_START]
    dates = pd.date_range(VALIDATION_START, TEST_START - pd.offsets.MonthBegin(1), freq="MS")
    best_k, best_gap = 1.0, 9.0
    for k in np.arange(1.0, 1.61, 0.05):
        _, cov = coverage(project(pre, target, dates, widen=k), data, target)
        if abs(cov - 0.80) < best_gap:
            best_k, best_gap = float(k), abs(cov - 0.80)
    return best_k

def main():
    print("\n" + "=" * 64)
    print("CLIMATESHIELD AI — MONTHLY DATA AND ML PIPELINE")
    print("=" * 64)

    temp_path = find_regional_file("tsmax")
    rain_path = find_regional_file("pr")

    temperature = load_monthly_data(temp_path, "Temperature")
    precipitation = load_monthly_data(rain_path, "Precipitation")

    data = pd.merge(
        temperature, precipitation,
        on=["code", "name", "Date"], how="outer", validate="one_to_one",
    ).sort_values(["name", "Date"]).reset_index(drop=True)
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month

    print(f"\nMerged shape: {data.shape} | Regions: {data['name'].nunique()}")
    print("Missing target values:")
    print(data[["Temperature", "Precipitation"]].isna().sum())

    data.to_csv(OUTPUT_DIR / "monthly_cleaned_dataset_1991_2025.csv", index=False)

    (data.groupby("name")[list(TARGETS)]
         .agg(lambda c: int(c.isna().sum())).reset_index()
         .to_csv(OUTPUT_DIR / "missing_values_report.csv", index=False))

    (data.groupby("Month").agg(
        Mean_Temperature=("Temperature", "mean"),
        Mean_Precipitation=("Precipitation", "mean"),
        Temperature_Std=("Temperature", "std"),
        Precipitation_Std=("Precipitation", "std"),
        Temperature_Observations=("Temperature", "count"),
        Precipitation_Observations=("Precipitation", "count"),
    ).reset_index().to_csv(OUTPUT_DIR / "monthly_climate_summary.csv", index=False))

    pd.Series(FEATURES, name="Feature").to_csv(
        OUTPUT_DIR / "model_features.csv", index=False
    )

    all_metrics, all_validation, all_predictions = [], [], []
    selection_summary, importance_rows, projection_rows = [], [], []
    all_forecasts = []

    future_dates = pd.date_range(
        f"{FORECAST_YEARS[0]}-01-01", f"{FORECAST_YEARS[-1]}-12-01", freq="MS"
    )
    test_dates = pd.date_range(TEST_START, END_DATE, freq="MS")

    for target in TARGETS:
        print("\n" + "-" * 64)
        print(f"TARGET: {target.upper()}")
        print("-" * 64)
        unit = "degC" if target == "Temperature" else "mm"

        ds = make_dataset(data, target, VALIDATION_START)
        train = ds[ds["Date"] < VALIDATION_START]
        valid = ds[(ds["Date"] >= VALIDATION_START) & (ds["Date"] < TEST_START)]
        print(f"Train rows: {len(train)} | Validation rows: {len(valid)}")

        validation_scores = [{
            "Target": target, "Approach": BASELINE,
            "Validation_MAE": mean_absolute_error(valid[target], valid["Climatology"]),
        }]
        validation_models = {}
        for name, factory in model_factories().items():
            model = factory().fit(train[FEATURES], train["Anomaly"])
            validation_models[name] = model
            predicted = to_value(valid, model.predict(valid[FEATURES]), target)
            validation_scores.append({
                "Target": target, "Approach": name,
                "Validation_MAE": mean_absolute_error(valid[target], predicted),
            })
        all_validation.extend(validation_scores)

        best_mae = min(r["Validation_MAE"] for r in validation_scores)
        complexity = [BASELINE, "Ridge Regression", "Extra Trees", "Hist Gradient Boosting"]
        eligible = [r for r in validation_scores
                    if r["Validation_MAE"] <= best_mae * 1.01]
        best = min(eligible, key=lambda r: complexity.index(r["Approach"]))
        selected = best["Approach"]
        baseline_val = validation_scores[0]["Validation_MAE"]
        selection_summary.append({
            "Target": target, "Selected_Approach": selected,
            "Validation_MAE": best["Validation_MAE"],
            "Baseline_Validation_MAE": baseline_val,
        })
        print(f"Selected on validation: {selected}")

        if selected != BASELINE:
            sample = valid.sample(min(4000, len(valid)), random_state=RANDOM_STATE)
            imp = permutation_importance(
                validation_models[selected], sample[FEATURES], sample["Anomaly"],
                n_repeats=5, random_state=RANDOM_STATE,
                scoring="neg_mean_absolute_error",
            )
            for feat, value in zip(FEATURES, imp.importances_mean):
                importance_rows.append({
                    "Target": target, "Model": selected, "Feature": feat,
                    "Importance": float(value),
                })

        ds = make_dataset(data, target, TEST_START)
        pretest = ds[ds["Date"] < TEST_START]
        test = ds[ds["Date"] >= TEST_START].copy()
        print(f"Final test rows (2021–2025): {len(test)}")

        baseline_mae, baseline_rmse, _ = score(test[target], test["Climatology"])
        all_metrics.append({
            "Target": target, "Approach": BASELINE,
            "MAE": baseline_mae, "RMSE": baseline_rmse, "R2": np.nan,
            "Skill_vs_Baseline_%": 0.0, "Test_Records": len(test),
            "Selected_On_Validation": selected == BASELINE,
        })
        test_predictions = {BASELINE: test["Climatology"].to_numpy()}

        for name, factory in model_factories().items():
            model = factory().fit(pretest[FEATURES], pretest["Anomaly"])
            predicted = to_value(test, model.predict(test[FEATURES]), target)
            test_predictions[name] = predicted
            mae, rmse, r2 = score(test[target], predicted)
            all_metrics.append({
                "Target": target, "Approach": name, "MAE": mae, "RMSE": rmse,
                "R2": r2,
                "Skill_vs_Baseline_%": 100 * (1 - mae / baseline_mae),
                "Test_Records": len(test),
                "Selected_On_Validation": selected == name,
            })
            print(f"  {name:<24} MAE {mae:.3f} {unit} "
                  f"(skill {100 * (1 - mae / baseline_mae):+.1f}%)")
        print(f"  {BASELINE:<24} MAE {baseline_mae:.3f} {unit}")

        out = test[["code", "name", "Date", "Year", "Month"]].copy()
        out["Target"] = target
        out["Actual"] = test[target].to_numpy()
        out["Predicted"] = test_predictions[selected]
        out["Baseline"] = test_predictions[BASELINE]
        out["Absolute_Error"] = (out["Actual"] - out["Predicted"]).abs()
        out["Selected_Approach"] = selected
        all_predictions.append(out)

        pre = data[data["Date"] < TEST_START]
        widen = calibrate_widen(data, target)
        backtest = project(pre, target, test_dates, widen=widen)
        merged, band_coverage = coverage(backtest, data, target)
        clim_only = merged.merge(
            pre.groupby(["name", "Month"])[target].mean().rename("clim").reset_index(),
            on=["name", "Month"],
        )
        projection_rows.append({
            "Target": target,
            "Projection_MAE_2021_2025": mean_absolute_error(
                merged[target], merged["Prediction"]),
            "Climatology_MAE_2021_2025": mean_absolute_error(
                clim_only[target], clim_only["clim"]),
            "Interval_Widening_Factor": widen,
            "Interval_Coverage_80pct": band_coverage,
            "Test_Records": len(merged),
        })
        all_forecasts.append(project(data, target, future_dates, widen=widen))
        ylabel = "Temperature (°C)" if target == "Temperature" else "Precipitation (mm)"
        avg = out.groupby("Date")[["Actual", "Predicted", "Baseline"]].mean()
        plt.figure(figsize=(13, 5))
        plt.plot(avg.index, avg["Actual"], label="Actual", linewidth=1.8)
        plt.plot(avg.index, avg["Predicted"], label=f"Selected: {selected}", linewidth=1.8)
        plt.plot(avg.index, avg["Baseline"], "--", label=BASELINE, linewidth=1.2, alpha=0.7)
        plt.title(f"{target}: actual vs predicted, regional average, 2021–2025")
        plt.xlabel("Month"); plt.ylabel(ylabel)
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(GRAPH_DIR / f"{target.lower()}_actual_vs_predicted.png", dpi=200)
        plt.close()

        cmp_df = pd.DataFrame([m for m in all_metrics if m["Target"] == target]).sort_values("MAE")
        plt.figure(figsize=(9, 5))
        plt.bar(cmp_df["Approach"], cmp_df["MAE"])
        plt.title(f"{target}: test MAE by approach (lower is better)")
        plt.ylabel(f"MAE ({unit})"); plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(GRAPH_DIR / f"{target.lower()}_model_comparison.png", dpi=200)
        plt.close()

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)
    pd.DataFrame(all_validation).to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame(selection_summary).to_csv(OUTPUT_DIR / "model_selection_summary.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "test_period_predictions.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(OUTPUT_DIR / "projection_validation.csv", index=False)
    pd.concat(all_forecasts, ignore_index=True).to_csv(
        OUTPUT_DIR / "future_projections_2026_2035.csv", index=False)

    with open(OUTPUT_DIR / "run_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("ClimateShield AI — Monthly ML pipeline\n")
        handle.write(f"Study period: {START_DATE.date()} to {END_DATE.date()}\n")
        handle.write("Training: 1991–2015 | Validation: 2016–2020 | Final test: 2021–2025\n\n")
        for row in selection_summary:
            handle.write(f"{row['Target']}: selected approach = {row['Selected_Approach']}\n")
        handle.write("\nOne-month-ahead models predict the anomaly from lagged anomalies.\n"
                     "Projections for 2026–2035 use climatology + a shrunk warming trend.\n")

    print("\n" + "=" * 64)
    print("PIPELINE COMPLETED")
    print("=" * 64)
    print(metrics_df.round(3).to_string(index=False))
    print("\nProjection back-test (fit < 2021, predict 2021–2025):")
    print(pd.DataFrame(projection_rows).round(3).to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()