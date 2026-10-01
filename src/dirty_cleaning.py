"""Diagnose, repair, audit, and validate the Group030 dirty-data batch."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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


@dataclass(frozen=True)
class RepairDecision:
    field_name: str
    original_value: Any
    corrected_value: Any
    method: str
    evidence: str
    confidence: str = "high"


@dataclass
class DirtyCleaningResult:
    cleaned: pd.DataFrame
    dirty_audit: pd.DataFrame
    combined_audit: pd.DataFrame
    repair_summary: pd.DataFrame
    validation: pd.DataFrame
    reliable_model_residual_limit: float


def reliable_delivery_residual_limit(
    missing_reference: pd.DataFrame,
    models: dict[str, LinearRegression],
) -> float:
    """Derive a conservative model-consistency limit from reliable observations."""
    reliable = missing_reference.dropna(
        subset=["season", "delivery_charges", *MODEL_FEATURES]
    )
    residuals = []
    for row in reliable.itertuples(index=False):
        prediction = predict_delivery(
            models[row.season],
            row.distance_to_nearest_warehouse,
            bool(row.is_expedited_delivery),
            bool(row.is_happy_customer),
        )
        residuals.append(abs(row.delivery_charges - prediction))
    # The extra cent accommodates the published monetary rounding sequence.
    return float(max(residuals) + 0.01)


def diagnose_dirty_repairs(
    original: pd.DataFrame,
    warehouses: pd.DataFrame,
    prices: dict[str, float],
    models: dict[str, LinearRegression],
    analyzer: SentimentIntensityAnalyzer,
    model_residual_limit: float,
) -> dict[int, RepairDecision]:
    """Identify one defensible field repair per affected row from independent evidence."""
    repairs: dict[int, RepairDecision] = {}

    def register(index: int, decision: RepairDecision) -> None:
        if index in repairs:
            previous = repairs[index]
            raise RuntimeError(
                f"Row {index} has competing repairs: "
                f"{previous.field_name} and {decision.field_name}"
            )
        repairs[index] = decision

    valid_warehouses = set(warehouses["warehouse"])

    # Direct calendar, coordinate, cart, and review evidence.
    for index, row in original.iterrows():
        parsed_date = parse_order_date(row["date"])
        canonical_date = parsed_date.strftime("%Y-%m-%d")
        if str(row["date"]) != canonical_date:
            register(
                index,
                RepairDecision(
                    "date",
                    row["date"],
                    canonical_date,
                    "Parse calendar date using published year-first/day-first rules",
                    "D01: valid date represented in required YYYY-MM-DD format",
                ),
            )
            continue

        expected_season = SEASON_BY_MONTH[parsed_date.month]
        if row["season"] != expected_season:
            register(
                index,
                RepairDecision(
                    "season",
                    row["season"],
                    expected_season,
                    "Australian season derived from order-date month",
                    "D02: published month-to-season mapping",
                ),
            )
            continue

        # A positive latitude is independently inconsistent with Melbourne.
        # Skip downstream geography checks so one bad coordinate is not treated
        # as multiple errors.
        if row["customer_lat"] > 0:
            corrected_latitude = -row["customer_lat"]
            register(
                index,
                RepairDecision(
                    "customer_lat",
                    row["customer_lat"],
                    corrected_latitude,
                    "Restore Melbourne latitude sign while retaining precision",
                    "D03: supplied warehouses and all reliable customers have southern latitude",
                ),
            )
            continue

        expected_warehouse, expected_distance = nearest_warehouse(
            row["customer_lat"], row["customer_long"], warehouses
        )
        if (
            row["nearest_warehouse"] not in valid_warehouses
            or row["nearest_warehouse"] != expected_warehouse
        ):
            register(
                index,
                RepairDecision(
                    "nearest_warehouse",
                    row["nearest_warehouse"],
                    expected_warehouse,
                    "Select minimum Haversine distance to supplied warehouses",
                    "D04: 6,371 km Haversine comparison and canonical warehouse name",
                ),
            )
            continue

        rounded_distance = round(expected_distance, 4)
        if abs(row["distance_to_nearest_warehouse"] - rounded_distance) > 0.0001001:
            register(
                index,
                RepairDecision(
                    "distance_to_nearest_warehouse",
                    row["distance_to_nearest_warehouse"],
                    rounded_distance,
                    "Recalculate Haversine distance to the nearest warehouse",
                    "D05: 6,371 km radius and four-decimal reconstruction",
                ),
            )
            continue

        expected_price = calculate_order_price(row["shopping_cart"], prices)
        if abs(row["order_price"] - expected_price) > 0.0100001:
            register(
                index,
                RepairDecision(
                    "order_price",
                    row["order_price"],
                    expected_price,
                    "Recalculate cart total from inferred current unit prices",
                    "D06: rank-12 price system with zero held-out cart-equation error",
                ),
            )
            continue

        expected_happiness = review_is_happy(row["latest_customer_review"], analyzer)
        if bool(row["is_happy_customer"]) != expected_happiness:
            register(
                index,
                RepairDecision(
                    "is_happy_customer",
                    row["is_happy_customer"],
                    expected_happiness,
                    "Apply frozen local VADER compound threshold and sentinel rule",
                    "D07: verified lexicon; compound >= 0.05",
                ),
            )

    # A failed total equation can arise from one bad order price, delivery
    # charge, or total. Price errors were already resolved independently above.
    for index, row in original.iterrows():
        calculated_total = round(
            row["order_price"] * (1 - row["coupon_discount"] / 100)
            + row["delivery_charges"],
            2,
        )
        if abs(calculated_total - row["order_total"]) <= 0.0100001:
            continue
        if index in repairs:
            if repairs[index].field_name != "order_price":
                raise RuntimeError(
                    f"Unexpected total failure alongside {repairs[index].field_name}"
                )
            continue

        predicted_charge = predict_delivery(
            models[row["season"]],
            row["distance_to_nearest_warehouse"],
            bool(row["is_expedited_delivery"]),
            bool(row["is_happy_customer"]),
        )
        implied_charge = round(
            row["order_total"]
            - row["order_price"] * (1 - row["coupon_discount"] / 100),
            2,
        )
        observed_charge_residual = abs(row["delivery_charges"] - predicted_charge)
        implied_charge_residual = abs(implied_charge - predicted_charge)

        if (
            observed_charge_residual <= model_residual_limit
            and implied_charge_residual > model_residual_limit
        ):
            register(
                index,
                RepairDecision(
                    "order_total",
                    row["order_total"],
                    calculated_total,
                    "Recalculate total from independently validated price, discount, and charge",
                    f"D08: observed-charge model residual={observed_charge_residual:.4f}; published total formula",
                ),
            )
        elif (
            implied_charge_residual <= model_residual_limit
            and observed_charge_residual > model_residual_limit
        ):
            register(
                index,
                RepairDecision(
                    "delivery_charges",
                    row["delivery_charges"],
                    implied_charge,
                    "Back-calculate charge from validated total, price, and discount",
                    f"D09: implied-charge model residual={implied_charge_residual:.4f}; observed residual={observed_charge_residual:.4f}",
                ),
            )
        else:
            raise RuntimeError(
                f"Could not isolate monetary error for order {row['order_id']}"
            )

    # For otherwise consistent rows, compare the two possible expedited states.
    for index, row in original.iterrows():
        if index in repairs:
            continue
        regular_prediction = predict_delivery(
            models[row["season"]],
            row["distance_to_nearest_warehouse"],
            False,
            bool(row["is_happy_customer"]),
        )
        expedited_prediction = predict_delivery(
            models[row["season"]],
            row["distance_to_nearest_warehouse"],
            True,
            bool(row["is_happy_customer"]),
        )
        regular_residual = abs(row["delivery_charges"] - regular_prediction)
        expedited_residual = abs(row["delivery_charges"] - expedited_prediction)
        current_residual = (
            expedited_residual if bool(row["is_expedited_delivery"]) else regular_residual
        )
        alternative_residual = (
            regular_residual if bool(row["is_expedited_delivery"]) else expedited_residual
        )
        if (
            alternative_residual <= model_residual_limit
            and current_residual > model_residual_limit
        ):
            register(
                index,
                RepairDecision(
                    "is_expedited_delivery",
                    row["is_expedited_delivery"],
                    not bool(row["is_expedited_delivery"]),
                    "Choose 0/1 state with smaller seasonal-model absolute residual",
                    f"D10: current residual={current_residual:.4f}; alternative residual={alternative_residual:.4f}",
                ),
            )

    return repairs


def _semantic_equal(left: Any, right: Any) -> bool:
    """Compare audit/output scalar values without relying on CSV dtype inference."""
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return str(left).lower() == str(right).lower()
    try:
        return bool(np.isclose(float(left), float(right), atol=1e-9, rtol=0))
    except (TypeError, ValueError):
        return str(left) == str(right)


def validate_dirty_output(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    reread: pd.DataFrame,
    combined_audit: pd.DataFrame,
    source_file: str,
    warehouses: pd.DataFrame,
    prices: dict[str, float],
    models: dict[str, LinearRegression],
    analyzer: SentimentIntensityAnalyzer,
    model_residual_limit: float,
) -> pd.DataFrame:
    """Validate schema, relationships, preservation, and audit reconciliation."""
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

    record(
        "schema_and_column_order",
        list(reread.columns),
        list(reread.columns) == ORDER_COLUMNS,
        "Output uses the required 16-column schema.",
    )
    record("row_count", len(reread), len(reread) == 500, "All dirty-batch rows retained.")
    record("unique_order_ids", reread.order_id.nunique(), reread.order_id.is_unique, "Order IDs remain unique.")
    record(
        "preserved_order_id_set",
        {"missing": len(set(original.order_id) - set(reread.order_id)), "unexpected": len(set(reread.order_id) - set(original.order_id))},
        set(original.order_id) == set(reread.order_id),
        "No order IDs were lost or introduced.",
    )
    record("required_completeness", int(reread.isna().sum().sum()), not reread.isna().any().any(), "No missing cells exist.")

    canonical_dates = reread.date.map(lambda value: parse_order_date(value).strftime("%Y-%m-%d"))
    noncanonical_dates = int((canonical_dates != reread.date).sum())
    record("canonical_dates", noncanonical_dates, noncanonical_dates == 0, "All dates use YYYY-MM-DD.")
    expected_seasons = reread.date.map(parse_order_date).dt.month.map(SEASON_BY_MONTH)
    season_mismatches = int((expected_seasons != reread.season).sum())
    record("date_season_relationship", season_mismatches, season_mismatches == 0, "All seasons match order month.")

    melbourne_coordinate_issues = int(
        (~reread.customer_lat.between(-38.0, -37.0)).sum()
        + (~reread.customer_long.between(144.0, 146.0)).sum()
    )
    record("melbourne_coordinates", melbourne_coordinate_issues, melbourne_coordinate_issues == 0, "Coordinates are consistent with Melbourne.")

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
    record("nearest_warehouse_relationship", warehouse_mismatches, warehouse_mismatches == 0, "Nearest warehouses agree with coordinates.")
    record("distance_relationship", max_distance_error, max_distance_error <= 0.0001001, "Distances agree with Haversine calculations.")

    carts_valid = True
    try:
        reread.shopping_cart.map(parse_cart)
    except (TypeError, ValueError, json.JSONDecodeError):
        carts_valid = False
    record("shopping_cart_structure", carts_valid, carts_valid, "All shopping carts are valid JSON arrays.")
    expected_prices = reread.shopping_cart.map(lambda value: calculate_order_price(value, prices))
    max_price_error = float((expected_prices - reread.order_price).abs().max())
    record("order_price_relationship", max_price_error, max_price_error <= 0.0100001, "Cart equations reproduce all order prices.")

    expected_happiness = reread.latest_customer_review.map(
        lambda value: review_is_happy(value, analyzer)
    )
    observed_happiness = reread.is_happy_customer.map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    happiness_mismatches = int((expected_happiness != observed_happiness).sum())
    record("review_happiness_relationship", happiness_mismatches, happiness_mismatches == 0, "VADER and sentinel rules agree.")

    expected_totals = (
        reread.order_price * (1 - reread.coupon_discount / 100)
        + reread.delivery_charges
    ).round(2)
    max_total_error = float((expected_totals - reread.order_total).abs().max())
    record("order_total_relationship", max_total_error, max_total_error <= 0.0100001, "Published total formula agrees.")

    delivery_residuals = []
    for row in reread.itertuples(index=False):
        prediction = predict_delivery(
            models[row.season],
            row.distance_to_nearest_warehouse,
            bool(row.is_expedited_delivery),
            bool(row.is_happy_customer),
        )
        delivery_residuals.append(abs(row.delivery_charges - prediction))
    max_delivery_residual = float(max(delivery_residuals))
    record(
        "seasonal_delivery_relationship",
        {"maximum_absolute_residual": max_delivery_residual, "reliable_limit": model_residual_limit},
        max_delivery_residual <= model_residual_limit,
        "Every repaired/retained dirty row follows the seasonal delivery relationship.",
    )

    actual_changes: list[tuple[str, str]] = []
    changes_per_order: dict[str, int] = {}
    for index, source_row in original.iterrows():
        for column in ORDER_COLUMNS:
            if not _semantic_equal(source_row[column], cleaned.at[index, column]):
                order_id = str(source_row.order_id)
                actual_changes.append((order_id, column))
                changes_per_order[order_id] = changes_per_order.get(order_id, 0) + 1
    max_changes_per_order = max(changes_per_order.values(), default=0)
    record(
        "one_repair_per_affected_order",
        {"changed_cells": len(actual_changes), "affected_orders": len(changes_per_order), "maximum_per_order": max_changes_per_order},
        max_changes_per_order <= 1,
        "No row has more than one changed field.",
    )

    dirty_audit = combined_audit.loc[
        (combined_audit.source_file == source_file)
        & (combined_audit.issue_type == "dirty")
    ].copy()
    audit_keys = list(zip(dirty_audit.record_key, dirty_audit.field_name))
    key_reconciliation = set(actual_changes) == set(audit_keys) and len(audit_keys) == len(set(audit_keys))
    corrected_values_match = True
    output_by_id = reread.set_index("order_id")
    for row in dirty_audit.itertuples(index=False):
        if not _semantic_equal(row.corrected_value, output_by_id.at[row.record_key, row.field_name]):
            corrected_values_match = False
            break
    record(
        "dirty_audit_reconciliation",
        {
            "actual_changes": len(actual_changes),
            "dirty_audit_rows": len(dirty_audit),
            "missing_from_audit": len(set(actual_changes) - set(audit_keys)),
            "extra_in_audit": len(set(audit_keys) - set(actual_changes)),
            "corrected_values_match": corrected_values_match,
        },
        key_reconciliation and corrected_values_match,
        "Actual dirty differences match one audit row each, including corrected values.",
    )
    record(
        "combined_audit_schema",
        list(combined_audit.columns),
        list(combined_audit.columns) == AUDIT_COLUMNS,
        "Combined missing and dirty audit uses the exact required schema.",
    )

    return pd.DataFrame(results)


def clean_dirty_batch(
    *,
    dirty_input: Path,
    missing_reference: pd.DataFrame,
    warehouses_input: Path,
    dirty_output: Path,
    audit_output: Path,
    prices: dict[str, float],
    models: dict[str, LinearRegression],
    analyzer: SentimentIntensityAnalyzer,
) -> DirtyCleaningResult:
    """Repair, audit, export, read back, and validate the dirty batch."""
    original = pd.read_csv(
        dirty_input, dtype={"order_id": "string", "customer_id": "string"}
    )
    if list(original.columns) != ORDER_COLUMNS or len(original) != 500:
        raise RuntimeError("Unexpected dirty-data input structure")
    warehouses = pd.read_csv(warehouses_input)
    residual_limit = reliable_delivery_residual_limit(missing_reference, models)
    repairs = diagnose_dirty_repairs(
        original,
        warehouses,
        prices,
        models,
        analyzer,
        residual_limit,
    )

    cleaned = original.copy(deep=True)
    dirty_trail = AuditTrail()
    for index, decision in repairs.items():
        order_id = str(original.at[index, "order_id"])
        cleaned.at[index, decision.field_name] = decision.corrected_value
        dirty_trail.add_change(
            source_file=dirty_input.name,
            record_key=order_id,
            field_name=decision.field_name,
            issue_type="dirty",
            original_value=decision.original_value,
            corrected_value=decision.corrected_value,
            method=decision.method,
            evidence=decision.evidence,
            confidence=decision.confidence,
        )

    dirty_output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = cleaned[ORDER_COLUMNS]
    cleaned.to_csv(dirty_output, index=False, encoding="utf-8")

    dirty_audit = dirty_trail.to_frame()
    if audit_output.exists():
        prior_audit = pd.read_csv(audit_output, keep_default_na=False)
        prior_audit = prior_audit.loc[prior_audit.source_file != dirty_input.name]
    else:
        prior_audit = pd.DataFrame(columns=AUDIT_COLUMNS)
    combined_audit = pd.concat([prior_audit, dirty_audit], ignore_index=True)[AUDIT_COLUMNS]
    combined_audit.to_csv(audit_output, index=False, encoding="utf-8")

    reread = pd.read_csv(
        dirty_output, dtype={"order_id": "string", "customer_id": "string"}
    )
    reread_audit = pd.read_csv(audit_output, keep_default_na=False)
    validation = validate_dirty_output(
        original,
        cleaned,
        reread,
        reread_audit,
        dirty_input.name,
        warehouses,
        prices,
        models,
        analyzer,
        residual_limit,
    )
    if not validation.passed.all():
        failures = validation.loc[~validation.passed, "check"].tolist()
        raise RuntimeError(f"Dirty-output validation failed: {failures}")

    repair_summary = (
        dirty_audit.groupby(["field_name", "confidence"], as_index=False)
        .size()
        .rename(columns={"size": "repaired_values"})
    )
    return DirtyCleaningResult(
        cleaned=cleaned,
        dirty_audit=dirty_audit,
        combined_audit=reread_audit,
        repair_summary=repair_summary,
        validation=validation,
        reliable_model_residual_limit=residual_limit,
    )
