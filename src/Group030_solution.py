"""FIT5196 A2 Group030 reproducible cleansing and profiling pipeline.

This source mirrors the executable workflow in the submitted notebook. It
recreates the three cleaned order files and audit log, runs the validation
implemented by each cleaning module, and constructs the Task 3 customer-level
analysis copy without modifying any submitted cleaned value.
"""

from pathlib import Path
import sys

# Support all documented layouts:
#   python -m src.Group030_solution
#   python src/Group030_solution.py
#   python Group030_solution.py        (from the final submission directory)
# The final layout places this entry point beside ``src/``. The development
# layout keeps it inside ``src/``. Resolve the root from those structural facts
# rather than assuming one fixed parent depth.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "src").is_dir() else SCRIPT_DIR.parent
if __package__ in {None, ""} and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.dirty_cleaning import clean_dirty_batch
from src.missing_cleaning import clean_missing_batch
from src.outlier_cleaning import clean_outlier_batch


GROUP_ID = "Group030"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / f"{GROUP_ID}_A2"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DIRTY_INPUT = DATA_DIR / f"{GROUP_ID}_dirty_data.csv"
MISSING_INPUT = DATA_DIR / f"{GROUP_ID}_missing_data.csv"
OUTLIER_INPUT = DATA_DIR / f"{GROUP_ID}_outlier_data.csv"
WAREHOUSES_INPUT = DATA_DIR / "warehouses.csv"
RESOURCE_CONTRACT = DATA_DIR / "resource_contract.json"
NLTK_DATA_DIR = DATA_DIR / "resources" / "nltk_data"

DIRTY_OUTPUT = OUTPUT_DIR / f"{GROUP_ID}_dirty_clean.csv"
MISSING_OUTPUT = OUTPUT_DIR / f"{GROUP_ID}_missing_clean.csv"
OUTLIER_OUTPUT = OUTPUT_DIR / f"{GROUP_ID}_outlier_clean.csv"
AUDIT_OUTPUT = OUTPUT_DIR / "audit_log.csv"

PROFILE_ATTRIBUTES = [
    "order_count",
    "mean_order_value",
    "mean_delivery_distance",
]


def build_customer_profile_analysis() -> dict[str, pd.DataFrame]:
    """Build the in-memory Task 3 analysis from the three cleaned outputs."""
    cleaned_paths = {
        "dirty": DIRTY_OUTPUT,
        "missing": MISSING_OUTPUT,
        "outlier": OUTLIER_OUTPUT,
    }
    cleaned_frames = []
    for source_batch, path in cleaned_paths.items():
        frame = pd.read_csv(
            path,
            dtype={"order_id": "string", "customer_id": "string"},
        )
        frame["source_batch"] = source_batch
        cleaned_frames.append(frame)

    profile_orders = pd.concat(cleaned_frames, ignore_index=True)
    input_summary = pd.DataFrame(
        {
            "source_batch": list(cleaned_paths),
            "retained_rows": [len(frame) for frame in cleaned_frames],
        }
    )
    customer_profiles = (
        profile_orders.groupby("customer_id", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            total_paid=("order_total", "sum"),
            mean_order_value=("order_total", "mean"),
            mean_delivery_distance=("distance_to_nearest_warehouse", "mean"),
        )
    )
    baseline = customer_profiles[["customer_id", *PROFILE_ATTRIBUTES]].copy()

    z_scaled = pd.DataFrame(
        StandardScaler().fit_transform(baseline[PROFILE_ATTRIBUTES]),
        columns=PROFILE_ATTRIBUTES,
        index=baseline.index,
    )
    minmax_scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(baseline[PROFILE_ATTRIBUTES]),
        columns=PROFILE_ATTRIBUTES,
        index=baseline.index,
    )
    variance_by_representation = pd.DataFrame(
        {
            "original": baseline[PROFILE_ATTRIBUTES].var(ddof=0),
            "z_score": z_scaled.var(ddof=0),
            "min_max": minmax_scaled.var(ddof=0),
        }
    )
    distance_contribution = variance_by_representation.div(
        variance_by_representation.sum(axis=0), axis=1
    ) * 100

    transformations = {
        "original": lambda values: values,
        "square_root": np.sqrt,
        "log1p": np.log1p,
    }
    nonlinear_records = []
    for attribute in PROFILE_ATTRIBUTES:
        for method, transform in transformations.items():
            transformed = transform(baseline[attribute])
            nonlinear_records.append(
                {
                    "attribute": attribute,
                    "method": method,
                    "skewness": transformed.skew(),
                    "minimum": transformed.min(),
                    "median": transformed.median(),
                    "maximum": transformed.max(),
                }
            )
    nonlinear_summary = pd.DataFrame(nonlinear_records)

    recommended_base = pd.DataFrame(
        {
            "order_count": baseline["order_count"],
            "mean_order_value": np.sqrt(baseline["mean_order_value"]),
            "mean_delivery_distance": baseline["mean_delivery_distance"],
        }
    )
    recommended_scaled = pd.DataFrame(
        StandardScaler().fit_transform(recommended_base),
        columns=[f"{column}_recommended" for column in PROFILE_ATTRIBUTES],
        index=baseline.index,
    )
    recommended_profiles = pd.concat(
        [baseline[["customer_id"]], recommended_scaled], axis=1
    )
    recommendations = pd.DataFrame(
        [
            {
                "attribute": "order_count",
                "methods_compared": "original, square root, log1p, z-score, min-max",
                "observed_effect": "Skew 0.8793; transforms compress the discrete 1-10 count scale",
                "recommended_treatment": "no nonlinear transformation; z-score standardisation",
                "justification_or_limitation": "Preserves count interpretation before scaling; z-score equal weighting is sample-specific",
            },
            {
                "attribute": "mean_order_value",
                "methods_compared": "original, square root, log1p, z-score, min-max",
                "observed_effect": "Square root reduces skew from 0.9282 to 0.1560",
                "recommended_treatment": "square root first; then z-score standardisation",
                "justification_or_limitation": "Moderates monetary influence; transformed units are less direct and z-scores do not cap extremes",
            },
            {
                "attribute": "mean_delivery_distance",
                "methods_compared": "original, square root, log1p, z-score, min-max",
                "observed_effect": "Transforms worsen original skew from -0.2357",
                "recommended_treatment": "no nonlinear transformation; z-score standardisation",
                "justification_or_limitation": "Retains kilometre ordering; straight-line distance is not travel time and equal weight is a choice",
            },
        ]
    )

    return {
        "input_summary": input_summary,
        "profile_orders": profile_orders,
        "customer_profiles": customer_profiles,
        "baseline_profiles": baseline,
        "z_scaled": z_scaled,
        "minmax_scaled": minmax_scaled,
        "distance_contribution": distance_contribution,
        "nonlinear_summary": nonlinear_summary,
        "recommended_profiles": recommended_profiles,
        "recommendations": recommendations,
    }


def main() -> None:
    """Run the complete Group030 cleansing, validation and profiling workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_result = clean_missing_batch(
        missing_input=MISSING_INPUT,
        warehouses_input=WAREHOUSES_INPUT,
        resource_contract=RESOURCE_CONTRACT,
        nltk_data_dir=NLTK_DATA_DIR,
        missing_output=MISSING_OUTPUT,
        audit_output=AUDIT_OUTPUT,
    )
    missing_reference = pd.read_csv(
        MISSING_INPUT,
        dtype={"order_id": "string", "customer_id": "string"},
    )
    dirty_result = clean_dirty_batch(
        dirty_input=DIRTY_INPUT,
        missing_reference=missing_reference,
        warehouses_input=WAREHOUSES_INPUT,
        dirty_output=DIRTY_OUTPUT,
        audit_output=AUDIT_OUTPUT,
        prices=missing_result.price_map,
        models=missing_result.seasonal_models,
        analyzer=missing_result.vader_analyzer,
    )
    outlier_result = clean_outlier_batch(
        outlier_input=OUTLIER_INPUT,
        missing_reference=missing_reference,
        warehouses_input=WAREHOUSES_INPUT,
        outlier_output=OUTLIER_OUTPUT,
        audit_output=AUDIT_OUTPUT,
        prices=missing_result.price_map,
        models=missing_result.seasonal_models,
        analyzer=missing_result.vader_analyzer,
    )
    print("Missing-data imputation summary")
    print(missing_result.imputation_summary.to_string(index=False))
    print(missing_result.validation[["check", "passed"]].to_string(index=False))
    print("Dirty-data repair summary")
    print(dirty_result.repair_summary.to_string(index=False))
    print(dirty_result.validation[["check", "passed"]].to_string(index=False))
    print("Outlier residual thresholds")
    print(outlier_result.threshold_diagnostics.to_string(index=False))
    print(outlier_result.validation[["check", "passed"]].to_string(index=False))
    profile_analysis = build_customer_profile_analysis()
    print("Customer-profile input summary")
    print(profile_analysis["input_summary"].to_string(index=False))
    print(
        "Combined orders and customers:",
        profile_analysis["profile_orders"]["order_id"].nunique(),
        len(profile_analysis["customer_profiles"]),
    )
    print("Potential Euclidean-distance variance contributions (%)")
    print(profile_analysis["distance_contribution"].round(4).to_string())
    print("Nonlinear transformation comparison")
    print(profile_analysis["nonlinear_summary"].round(4).to_string(index=False))
    print("Recommended analytical treatment")
    print(profile_analysis["recommendations"].to_string(index=False))


if __name__ == "__main__":
    main()
