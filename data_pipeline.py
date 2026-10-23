from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def load_data():
    data_dir = DATA_DIR
    if not all((data_dir / name).is_file() for name in ("baseline.csv", "followup.csv")):
        data_dir = PROJECT_DIR

    baseline = pd.read_csv(data_dir / "baseline.csv")
    followup = pd.read_csv(data_dir / "followup.csv")

    baseline.columns = baseline.columns.str.lower().str.strip()
    followup.columns = followup.columns.str.lower().str.strip()

    required_baseline = {
        "usubjid", "blgfr", "trt01pn", "blglp1", "strata"
    }
    required_followup = {"usubjid", "aval", "ady"}

    if not required_baseline.issubset(baseline.columns):
        raise ValueError(
            f"Missing baseline columns: "
            f"{required_baseline - set(baseline.columns)}"
        )

    if not required_followup.issubset(followup.columns):
        raise ValueError(
            f"Missing follow-up columns: "
            f"{required_followup - set(followup.columns)}"
        )

    baseline["usubjid"] = baseline["usubjid"].astype(str)
    followup["usubjid"] = followup["usubjid"].astype(str)

    followup["ady"] = pd.to_numeric(
        followup["ady"], errors="coerce"
    )
    followup["aval"] = pd.to_numeric(
        followup["aval"], errors="coerce"
    )

    followup = followup.dropna(
        subset=["usubjid", "ady", "aval"]
    )

    return baseline, followup


def calculate_slope(group):
    if len(group) < 2:
        return np.nan

    days = group["ady"].to_numpy(dtype=float)
    values = group["aval"].to_numpy(dtype=float)

    if np.unique(days).size < 2:
        return np.nan

    # Estimated eGFR change per 365 days.
    return float(np.polyfit(days, values, 1)[0] * 365)


def create_features():
    baseline, followup = load_data()

    rows = []

    for patient_id, group in followup.groupby("usubjid"):
        early = group[group["ady"] <= 365].sort_values("ady")
        future = group[group["ady"] > 365]

        # Require early observations and at least one future observation.
        if early.empty or future.empty:
            continue

        early_values = early["aval"]

        row = {
            "usubjid": patient_id,
            "early_egfr_first": early_values.iloc[0],
            "early_egfr_last": early_values.iloc[-1],
            "early_egfr_mean": early_values.mean(),
            "early_egfr_min": early_values.min(),
            "early_egfr_slope_year": calculate_slope(early),
            "early_visit_count": len(early),
            "days_observed": early["ady"].max(),
            # Research label, not a clinical diagnosis:
            "future_egfr_below_30": int(
                (future["aval"] < 30).any()
            ),
        }

        rows.append(row)

    if not rows:
        raise ValueError(
            "No patients have both early and future measurements. "
            "Check the dataset dates and column names."
        )

    features = pd.DataFrame(rows)

    baseline = baseline.drop_duplicates(
        subset=["usubjid"]
    )

    result = features.merge(
        baseline[
            ["usubjid", "blgfr", "trt01pn", "blglp1", "strata"]
        ],
        on="usubjid",
        how="inner",
        validate="one_to_one",
    )

    result = result.replace([np.inf, -np.inf], np.nan)

    DATA_DIR.mkdir(exist_ok=True)

    output = DATA_DIR / "processed_patients.csv"
    result.to_csv(output, index=False)

    print("Data pipeline completed.")
    print("Patients:", len(result))
    print(
        "Patients with future eGFR below 30:",
        int(result["future_egfr_below_30"].sum()),
    )
    print("Saved:", output)

    return result


if __name__ == "__main__":
    create_features()