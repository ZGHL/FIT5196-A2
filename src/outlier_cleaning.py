"""Detect, remove, audit, and validate Group030 delivery-charge outliers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.audit_trail import AUDIT_COLUMNS, AuditTrail
from src.missing_cleaning import (
    MODEL_FEATURES,
    ORDER_COLUMNS,
    SEASON_BY_MONTH,
    calculate_order_price,
    nearest_warehouse,
    parse_cart,
    parse_order_date,
    predict_delivery,
    review_is_happy,
)


@dataclass
class OutlierCleaningResult:
    retained: pd.DataFrame
    outlier_audit: pd.DataFrame
    combined_audit: pd.DataFrame
    threshold_diagnostics: pd.DataFrame
    residual_diagnostics: pd.DataFrame
    validation: pd.DataFrame


def derive_seasonal_residual_thresholds(
    missing_reference: pd.DataFrame,
    final_models: dict[str, LinearRegression],
) -> pd.DataFrame:
    """Derive conservative season-specific cutoffs from reliable residuals.

    Each cutoff is the larger of the maximum reproducible held-out residual and
    the maximum residual after fitting all reliable observations, plus AUD 0.01
    to accommodate source monetary rounding. This is an empirical residual
    rule, not the marking tolerance from the specification.
    """
    reliable = missing_reference.dropna(
        subset=["season", "delivery_charges", *MODEL_FEATURES]
    ).copy()
    rows = []

    for season in ["Summer", "Autumn", "Winter", "Spring"]:
        season_rows = reliable.loc[reliable.season == season]
        x = season_rows[MODEL_FEATURES].astype(float)
        y = season_rows.delivery_charges.astype(float)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=5196
        )
        evaluation_model = LinearRegression().fit(x_train, y_train)
        heldout_absolute = np.abs(y_test - evaluation_model.predict(x_test))
        fitted_absolute = np.abs(y - final_models[season].predict(x))
        empirical_maximum = max(
            float(heldout_absolute.max()), float(fitted_absolute.max())
        )
        cutoff = empirical_maximum + 0.01
        rows.append(
            {
                "season": season,
                "reliable_observations": len(season_rows),
                "heldout_observations": len(x_test),
                "heldout_max_absolute_residual": float(heldout_absolute.max()),
                "all_reliable_max_absolute_residual": float(fitted_absolute.max()),
                "rounding_guard": 0.01,
                "residual_threshold": cutoff,
            }
        )
    return pd.DataFrame(rows)


def calculate_outlier_residuals(
    outlier_original: pd.DataFrame,
    models: dict[str, LinearRegression],
    threshold_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate model expectations and apply the seasonal residual rule."""
    thresholds = threshold_diagnostics.set_index("season")[
        "residual_threshold"
    ].to_dict()
    rows = []
    for row in outlier_original.itertuples(index=False):
        prediction = predict_delivery(
            models[row.season],
            row.distance_to_nearest_warehouse,
            bool(row.is_expedited_delivery),
            bool(row.is_happy_customer),
        )
        residual = row.delivery_charges - prediction
        threshold = thresholds[row.season]
        rows.append(
            {
                "order_id": row.order_id,
                "season": row.season,
                "observed_delivery_charge": row.delivery_charges,
                "predicted_delivery_charge": prediction,
                "residual": residual,
                "absolute_residual": abs(residual),
                "residual_threshold": threshold,
                "remove_as_outlier": abs(residual) > threshold,
            }
        )
    return pd.DataFrame(rows)


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return str(left).lower() == str(right).lower()
    try:
        return bool(np.isclose(float(left), float(right), atol=1e-9, rtol=0))
    except (TypeError, ValueError):
        return str(left) == str(right)


def validate_outlier_output(
    original: pd.DataFrame,
    retained_before_write: pd.DataFrame,
    reread: pd.DataFrame,
    combined_audit: pd.DataFrame,
    source_file: str,
    residual_diagnostics: pd.DataFrame,
    warehouses: pd.DataFrame,
    prices: dict[str, float],
    analyzer: SentimentIntensityAnalyzer,
) -> pd.DataFrame:
    """Validate row removal, preserved values, relationships, and audit records."""
    results: list[dict[str, Any]] = []

    def record(check: str, observed: Any, passed: bool, interpretation: str) -> None:
        results.append(
            {
                "check": check,
                "observed_result": observed,
                "passed": bool(passed),
                "interpretation": interpretation,
            }
        )

    expected_removed = set(
        residual_diagnostics.loc[
            residual_diagnostics.remove_as_outlier, "order_id"
        ]
    )
    expected_retained = set(original.order_id) - expected_removed
    output_ids = set(reread.order_id)

    record(
        "schema_and_column_order",
        list(reread.columns),
        list(reread.columns) == ORDER_COLUMNS,
        "Output uses the exact required 16-column schema.",
    )
    record(
        "input_output_row_reconciliation",
        {
            "input_rows": len(original),
            "removed_rows": len(expected_removed),
            "retained_rows": len(reread),
        },
        len(original) == len(expected_removed) + len(reread),
        "Input rows equal removed plus retained rows.",
    )
    record("unique_retained_order_ids", reread.order_id.nunique(), reread.order_id.is_unique, "Retained IDs remain unique.")
    record(
        "retained_order_id_set",
        {"missing": len(expected_retained - output_ids), "unexpected": len(output_ids - expected_retained)},
        output_ids == expected_retained,
        "Output contains exactly the model-rule retained IDs.",
    )
    record("required_completeness", int(reread.isna().sum().sum()), not reread.isna().any().any(), "No missing values were introduced.")

    # Outlier handling is row removal only. No retained value may be repaired.
    retained_source = original.set_index("order_id").loc[reread.order_id].reset_index()
    retained_differences = 0
    for index in range(len(reread)):
        for column in ORDER_COLUMNS:
            if not _semantic_equal(retained_source.at[index, column], reread.at[index, column]):
                retained_differences += 1
    record(
        "preservation_of_retained_values",
        retained_differences,
        retained_differences == 0,
        "Every retained field is equal in meaning to its source value.",
    )

    retained_residuals = residual_diagnostics.loc[
        ~residual_diagnostics.remove_as_outlier
    ]
    removed_residuals = residual_diagnostics.loc[
        residual_diagnostics.remove_as_outlier
    ]
    retained_rule_failures = int(
        (
            retained_residuals.absolute_residual
            > retained_residuals.residual_threshold
        ).sum()
    )
    removed_rule_failures = int(
        (
            removed_residuals.absolute_residual
            <= removed_residuals.residual_threshold
        ).sum()
    )
    maximum_retained = float(retained_residuals.absolute_residual.max())
    minimum_removed = float(removed_residuals.absolute_residual.min())
    record(
        "residual_rule_reconciliation",
        {
            "retained_rule_failures": retained_rule_failures,
            "removed_rule_failures": removed_rule_failures,
        },
        retained_rule_failures == 0 and removed_rule_failures == 0,
        "Every removal and retention agrees with its seasonal threshold.",
    )
    record(
        "residual_separation",
        {
            "maximum_retained_absolute_residual": maximum_retained,
            "minimum_removed_absolute_residual": minimum_removed,
            "separation_ratio": minimum_removed / maximum_retained,
        },
        minimum_removed > maximum_retained,
        "Removed rows form a clearly separated residual group.",
    )

    # Check the business relationships on retained rows; the only supplied
    # anomaly type was delivery charge, so no other inconsistency is expected.
    carts_valid = True
    try:
        reread.shopping_cart.map(parse_cart)
    except (TypeError, ValueError, json.JSONDecodeError):
        carts_valid = False
    record("shopping_cart_structure", carts_valid, carts_valid, "Retained carts are valid JSON arrays.")
    expected_prices = reread.shopping_cart.map(lambda value: calculate_order_price(value, prices))
    max_price_error = float((expected_prices - reread.order_price).abs().max())
    record("order_price_relationship", max_price_error, max_price_error <= 0.0100001, "Cart equations reproduce retained prices.")
    expected_seasons = reread.date.map(parse_order_date).dt.month.map(SEASON_BY_MONTH)
    season_mismatches = int((expected_seasons != reread.season).sum())
    record("date_season_relationship", season_mismatches, season_mismatches == 0, "Retained dates and seasons agree.")

    warehouse_mismatches = 0
    max_distance_error = 0.0
    for row in reread.itertuples(index=False):
        expected_warehouse, expected_distance = nearest_warehouse(
            row.customer_lat, row.customer_long, warehouses
        )
        warehouse_mismatches += expected_warehouse != row.nearest_warehouse
        max_distance_error = max(
            max_distance_error,
            abs(round(expected_distance, 4) - row.distance_to_nearest_warehouse),
        )
    record("nearest_warehouse_relationship", warehouse_mismatches, warehouse_mismatches == 0, "Retained warehouse assignments agree.")
    record("distance_relationship", max_distance_error, max_distance_error <= 0.0001001, "Retained distances agree.")

    expected_happiness = reread.latest_customer_review.map(
        lambda value: review_is_happy(value, analyzer)
    )
    observed_happiness = reread.is_happy_customer.map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    happiness_mismatches = int((expected_happiness != observed_happiness).sum())
    record("review_happiness_relationship", happiness_mismatches, happiness_mismatches == 0, "Retained VADER labels agree.")
    expected_totals = (
        reread.order_price * (1 - reread.coupon_discount / 100)
        + reread.delivery_charges
    ).round(2)
    max_total_error = float((expected_totals - reread.order_total).abs().max())
    record("order_total_relationship", max_total_error, max_total_error <= 0.0100001, "Retained monetary totals agree.")

    outlier_audit = combined_audit.loc[
        (combined_audit.source_file == source_file)
        & (combined_audit.issue_type == "outlier_row")
    ].copy()
    audit_removed = set(outlier_audit.record_key)
    audit_values_ok = bool(
        (outlier_audit.field_name == "delivery_charges").all()
        and (outlier_audit.corrected_value == "ROW_REMOVED").all()
        and outlier_audit.record_key.is_unique
    )
    original_charge = original.set_index("order_id").delivery_charges
    original_values_match = all(
        _semantic_equal(row.original_value, original_charge.at[row.record_key])
        for row in outlier_audit.itertuples(index=False)
    )
    record(
        "outlier_audit_reconciliation",
        {
            "removed_ids": len(expected_removed),
            "audit_rows": len(outlier_audit),
            "missing_from_audit": len(expected_removed - audit_removed),
            "extra_in_audit": len(audit_removed - expected_removed),
            "original_values_match": original_values_match,
        },
        expected_removed == audit_removed and audit_values_ok and original_values_match,
        "Every removed row has one correct delivery-charge audit record.",
    )
    record(
        "combined_audit_schema",
        list(combined_audit.columns),
        list(combined_audit.columns) == AUDIT_COLUMNS,
        "Combined audit retains the exact required schema.",
    )

    return pd.DataFrame(results)


def clean_outlier_batch(
    *,
    outlier_input: Path,
    missing_reference: pd.DataFrame,
    warehouses_input: Path,
    outlier_output: Path,
    audit_output: Path,
    prices: dict[str, float],
    models: dict[str, LinearRegression],
    analyzer: SentimentIntensityAnalyzer,
) -> OutlierCleaningResult:
    """Remove anomalous-charge rows, append audit records, and validate output."""
    original = pd.read_csv(
        outlier_input, dtype={"order_id": "string", "customer_id": "string"}
    )
    if list(original.columns) != ORDER_COLUMNS or len(original) != 500:
        raise RuntimeError("Unexpected outlier-data input structure")
    warehouses = pd.read_csv(warehouses_input)
    threshold_diagnostics = derive_seasonal_residual_thresholds(
        missing_reference, models
    )
    residual_diagnostics = calculate_outlier_residuals(
        original, models, threshold_diagnostics
    )
    removed_ids = set(
        residual_diagnostics.loc[
            residual_diagnostics.remove_as_outlier, "order_id"
        ]
    )
    retained = original.loc[~original.order_id.isin(removed_ids)].copy()

    outlier_trail = AuditTrail()
    diagnostic_by_id = residual_diagnostics.set_index("order_id")
    for row in original.loc[original.order_id.isin(removed_ids)].itertuples(index=False):
        evidence = diagnostic_by_id.loc[row.order_id]
        outlier_trail.add_removed_row(
            source_file=outlier_input.name,
            record_key=row.order_id,
            original_delivery_charge=row.delivery_charges,
            method="Remove row when seasonal-model absolute residual exceeds empirical threshold",
            evidence=(
                f"O01: {row.season} observed={row.delivery_charges:.2f}, "
                f"predicted={evidence.predicted_delivery_charge:.4f}, "
                f"abs residual={evidence.absolute_residual:.4f}, "
                f"threshold={evidence.residual_threshold:.4f}"
            ),
            confidence="high",
        )

    outlier_output.parent.mkdir(parents=True, exist_ok=True)
    retained = retained[ORDER_COLUMNS]
    retained.to_csv(outlier_output, index=False, encoding="utf-8")

    outlier_audit = outlier_trail.to_frame()
    if audit_output.exists():
        prior_audit = pd.read_csv(audit_output, keep_default_na=False)
        prior_audit = prior_audit.loc[prior_audit.source_file != outlier_input.name]
    else:
        prior_audit = pd.DataFrame(columns=AUDIT_COLUMNS)
    combined_audit = pd.concat([prior_audit, outlier_audit], ignore_index=True)[
        AUDIT_COLUMNS
    ]
    combined_audit.to_csv(audit_output, index=False, encoding="utf-8")

    reread = pd.read_csv(
        outlier_output, dtype={"order_id": "string", "customer_id": "string"}
    )
    reread_audit = pd.read_csv(audit_output, keep_default_na=False)
    validation = validate_outlier_output(
        original,
        retained,
        reread,
        reread_audit,
        outlier_input.name,
        residual_diagnostics,
        warehouses,
        prices,
        analyzer,
    )
    if not validation.passed.all():
        failures = validation.loc[~validation.passed, "check"].tolist()
        raise RuntimeError(f"Outlier-output validation failed: {failures}")

    return OutlierCleaningResult(
        retained=retained,
        outlier_audit=outlier_audit,
        combined_audit=reread_audit,
        threshold_diagnostics=threshold_diagnostics,
        residual_diagnostics=residual_diagnostics,
        validation=validation,
    )

