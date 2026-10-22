from pathlib import Path
import re
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. PATHS AND SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "monthly_ml"
GRAPH_DIR = OUTPUT_DIR / "graphs"
MODEL_DIR = OUTPUT_DIR / "models"

for folder in (OUTPUT_DIR, GRAPH_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("1991-01-01")
END_DATE = pd.Timestamp("2025-12-01")
VALIDATION_START = pd.Timestamp("2016-01-01")
TEST_START = pd.Timestamp("2021-01-01")

# =========================================================
# 2. FIND THE REGIONAL EXCEL FILES
# =========================================================

def find_regional_file(variable):
    """
    Find a monthly Excel file for a variable.
    Prefer a file with code/name columns and approximately
    35 regional rows. A country-only India file is rejected.
    """
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

    # Prefer the file with the most unique regions, then most rows.
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


# =========================================================
# 3. WIDE EXCEL DATA TO LONG MONTHLY DATA
# =========================================================

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

    # CCKP monthly columns normally use YYYY-MM labels.
    date_columns = [
        col for col in df.columns
        if re.fullmatch(r"\d{4}-\d{2}", col)
    ]

    if not date_columns:
        # Accept columns such as YYYY-MM-DD, if present.
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


# =========================================================
# 4. LOAD, MERGE, AND SAVE CLEANED DATA
# =========================================================

def main():
    print("\n" + "=" * 64)
    print("CLIMATESHIELD AI — MONTHLY DATA AND ML PIPELINE")
    print("=" * 64)

    temp_path = find_regional_file("tsmax")
    rain_path = find_regional_file("pr")

    temperature = load_monthly_data(temp_path, "Temperature")
    precipitation = load_monthly_data(rain_path, "Precipitation")

    print(f"\nTemperature rows: {len(temperature)}")
    print(f"Precipitation rows: {len(precipitation)}")

    data = pd.merge(
        temperature,
        precipitation,
        on=["code", "name", "Date"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["name", "Date"]).reset_index(drop=True)

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month

    print(f"\nMerged shape: {data.shape}")
    print(f"Regions: {data['name'].nunique()}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print("\nMissing target values:")
    print(data[["Temperature", "Precipitation"]].isna().sum())

    # Preserve raw missing values in the cleaned dataset; do not invent rainfall.
    cleaned_path = OUTPUT_DIR / "monthly_cleaned_dataset_1991_2025.csv"
    data.to_csv(cleaned_path, index=False)

    missing_report = (
        data.groupby("name")[["Temperature", "Precipitation"]]
        .agg(lambda column: int(column.isna().sum()))
        .reset_index()
    )
    missing_report.to_csv(OUTPUT_DIR / "missing_values_report.csv", index=False)

    monthly_summary = (
        data.groupby("Month")
        .agg(
            Mean_Temperature=("Temperature", "mean"),
            Mean_Precipitation=("Precipitation", "mean"),
            Temperature_Std=("Temperature", "std"),
            Precipitation_Std=("Precipitation", "std"),
            Temperature_Observations=("Temperature", "count"),
            Precipitation_Observations=("Precipitation", "count"),
        )
        .reset_index()
    )
    monthly_summary.to_csv(
        OUTPUT_DIR / "monthly_climate_summary.csv", index=False
    )

    # Region one-hot encoding plus seasonal cyclical features.
    region_dummies = pd.get_dummies(
        data["name"], prefix="Region", dtype=int
    )
    data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)

    feature_columns = [
        "Year", "Month_sin", "Month_cos"
    ] + list(region_dummies.columns)
    model_data = pd.concat([data, region_dummies], axis=1)

    # Save a transparent record of the feature names.
    pd.Series(feature_columns, name="Feature").to_csv(
        OUTPUT_DIR / "model_features.csv", index=False
    )

    model_factories = {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": lambda: ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    all_metrics = []
    all_predictions = []
    all_validation_scores = []
    selected_models = {}
    selection_summary = []

    # =====================================================
    # 5. TRAIN, VALIDATE, SELECT, AND TEST EACH TARGET
    # =====================================================

    for target in ("Temperature", "Precipitation"):
        print("\n" + "-" * 64)
        print(f"TARGET: {target.upper()}")
        print("-" * 64)

        target_df = model_data.dropna(subset=[target]).copy()
        target_df = target_df.sort_values("Date")

        train = target_df[target_df["Date"] < VALIDATION_START]
        validation = target_df[
            (target_df["Date"] >= VALIDATION_START)
            & (target_df["Date"] < TEST_START)
        ]
        test = target_df[target_df["Date"] >= TEST_START]

        if train.empty or validation.empty or test.empty:
            raise ValueError(
                f"Train/validation/test split is empty for {target}. "
                "Check the downloaded date range."
            )

        print(f"Training rows (1991–2015): {len(train)}")
        print(f"Validation rows (2016–2020): {len(validation)}")
        print(f"Test rows (2021–2025): {len(test)}")

        # Seasonal climatology: region-specific average for each calendar month.
        # Fit using training data only during model selection.
        train_climatology = train.groupby(["name", "Month"])[target].mean()

        validation_baseline = np.array([
            train_climatology.get((row.name, row.Month), np.nan)
            for row in validation[["name", "Month"]].itertuples(index=False)
        ], dtype=float)

        valid_baseline_mask = np.isfinite(validation_baseline)
        if not valid_baseline_mask.any():
            raise ValueError(f"No valid climatology baseline for {target}.")

        baseline_validation_mae = mean_absolute_error(
            validation[target].to_numpy()[valid_baseline_mask],
            validation_baseline[valid_baseline_mask],
        )

        validation_scores = [{
            "Target": target,
            "Approach": "Seasonal Climatology Baseline",
            "Validation_MAE": baseline_validation_mae,
        }]

        fitted_validation_models = {}
        print(f"Baseline validation MAE: {baseline_validation_mae:.4f}")

        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(train[feature_columns], train[target])
            val_predictions = model.predict(validation[feature_columns])
            val_mae = mean_absolute_error(
                validation[target], val_predictions
            )

            validation_scores.append({
                "Target": target,
                "Approach": model_name,
                "Validation_MAE": val_mae,
            })
            fitted_validation_models[model_name] = model
            print(f"{model_name} validation MAE: {val_mae:.4f}")

        # Select using validation MAE only; final test data is not used
        # to choose the preferred approach.
        all_validation_scores.extend(validation_scores)
        best_validation = min(
            validation_scores, key=lambda row: row["Validation_MAE"]
        )
        selected_name = best_validation["Approach"]
        selection_summary.append({
            "Target": target,
            "Selected_Approach": selected_name,
            "Validation_MAE": best_validation["Validation_MAE"],
            "Baseline_Validation_MAE": baseline_validation_mae,
        })
        print(f"Selected using validation: {selected_name}")

        # Refit candidate models on all pre-test observations (1991–2020).
        pretest = target_df[target_df["Date"] < TEST_START]
        final_train_climatology = pretest.groupby(["name", "Month"])[target].mean()

        test_baseline = np.array([
            final_train_climatology.get((row.name, row.Month), np.nan)
            for row in test[["name", "Month"]].itertuples(index=False)
        ], dtype=float)
        valid_test_baseline = np.isfinite(test_baseline)

        baseline_test_predictions = test_baseline.copy()
        baseline_test_mae = mean_absolute_error(
            test[target].to_numpy()[valid_test_baseline],
            baseline_test_predictions[valid_test_baseline],
        )
        baseline_test_rmse = np.sqrt(mean_squared_error(
            test[target].to_numpy()[valid_test_baseline],
            baseline_test_predictions[valid_test_baseline],
        ))

        all_metrics.append({
            "Target": target,
            "Approach": "Seasonal Climatology Baseline",
            "Split": "Final test 2021–2025",
            "MAE": baseline_test_mae,
            "RMSE": baseline_test_rmse,
            "R2": np.nan,
            "Test_Records": int(valid_test_baseline.sum()),
            "Selected_On_Validation": selected_name
                == "Seasonal Climatology Baseline",
        })

        # Evaluate every ML model on final test for comparison, but do not
        # use final test metrics to decide which approach is selected.
        test_predictions_by_model = {}
        for model_name, factory in model_factories.items():
            final_model = factory()
            final_model.fit(pretest[feature_columns], pretest[target])
            predictions = final_model.predict(test[feature_columns])
            test_predictions_by_model[model_name] = predictions

            mae = mean_absolute_error(test[target], predictions)
            rmse = np.sqrt(mean_squared_error(test[target], predictions))
            r2 = r2_score(test[target], predictions)

            all_metrics.append({
                "Target": target,
                "Approach": model_name,
                "Split": "Final test 2021–2025",
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Test_Records": len(test),
                "Selected_On_Validation": selected_name == model_name,
            })

        if selected_name == "Seasonal Climatology Baseline":
            selected_predictions = baseline_test_predictions
            selected_final_model = None
        else:
            selected_final_model = model_factories[selected_name]()
            selected_final_model.fit(pretest[feature_columns], pretest[target])
            selected_predictions = selected_final_model.predict(
                test[feature_columns]
            )
            joblib.dump(
                selected_final_model,
                MODEL_DIR / f"selected_{target.lower()}_model.joblib",
            )

        selected_models[target] = selected_name

        prediction_output = test[
            ["code", "name", "Date", "Year", "Month", target]
        ].copy()
        prediction_output["Predicted"] = selected_predictions
        prediction_output["Absolute_Error"] = np.abs(
            prediction_output[target] - prediction_output["Predicted"]
        )
        prediction_output["Selected_Approach"] = selected_name
        all_predictions.append(prediction_output)

        # Plot monthly averages across regions for actual and selected predictions.
        plot_summary = (
            prediction_output.groupby("Date")[[target, "Predicted"]]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(13, 5))
        plt.plot(
            plot_summary["Date"], plot_summary[target],
            label="Actual", linewidth=1.8,
        )
        plt.plot(
            plot_summary["Date"], plot_summary["Predicted"],
            label=f"Predicted ({selected_name})", linewidth=1.8,
        )
        plt.title(
            f"{target}: Actual vs Selected Approach, 2021–2025"
        )
        plt.xlabel("Month")
        plt.ylabel("Temperature (°C)" if target == "Temperature"
                   else "Precipitation (mm)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            GRAPH_DIR / f"{target.lower()}_actual_vs_predicted.png",
            dpi=200,
        )
        plt.close()

        # Comparison plot for final test MAE.
        target_metric_rows = [
            row for row in all_metrics if row["Target"] == target
        ]
        comparison = pd.DataFrame(target_metric_rows).sort_values("MAE")

        plt.figure(figsize=(9, 5))
        plt.bar(comparison["Approach"], comparison["MAE"])
        plt.title(f"{target}: Final Test MAE Comparison (2021–2025)")
        plt.ylabel("MAE (lower is better)")
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            GRAPH_DIR / f"{target.lower()}_model_comparison.png",
            dpi=200,
        )
        plt.close()

    # =====================================================
    # 6. SAVE RESULTS
    # =====================================================

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        OUTPUT_DIR / "model_evaluation_metrics.csv", index=False
    )

    pd.DataFrame(all_validation_scores).to_csv(
        OUTPUT_DIR / "validation_metrics.csv", index=False
    )
    pd.DataFrame(selection_summary).to_csv(
        OUTPUT_DIR / "model_selection_summary.csv", index=False
    )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(
        OUTPUT_DIR / "test_period_predictions.csv", index=False
    )

    # Save a simple project status report.
    with open(OUTPUT_DIR / "run_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("ClimateShield AI — Monthly ML pipeline\n")
        handle.write(f"Study period: {START_DATE.date()} to {END_DATE.date()}\n")
        handle.write("Training: 1991–2015\nValidation: 2016–2020\n")
        handle.write("Final test: 2021–2025\n\n")
        for target, approach in selected_models.items():
            handle.write(f"{target}: selected approach = {approach}\n")
        handle.write(
            "\nThese are held-out historical predictions, not forecasts "
            "for future years.\n"
        )

    print("\n" + "=" * 64)
    print("PIPELINE COMPLETED")
    print("=" * 64)
    print("\nValidation-based selections:")
    print(pd.DataFrame(selection_summary).to_string(index=False))
    print("\nFinal test metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")
    print(f"Graphs:  {GRAPH_DIR}")
    print(f"Models:  {MODEL_DIR}")
    print(
        "\nNote: Predictions cover historical test months (2021–2025); "
        "they are not future forecasts."
    )


if __name__ == "__main__":
    main()
