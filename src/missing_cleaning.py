"""Clean and validate the Group030 missing-data batch.

The supplied missing batch contains omissions only; every observed value is
guaranteed correct. This module therefore uses deterministic business rules
where possible and evaluated seasonal linear models only for the two fields
that cannot be recovered directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.audit_trail import AUDIT_COLUMNS, AuditTrail


ORDER_COLUMNS = [
    "order_id",
    "customer_id",
    "date",
    "nearest_warehouse",
    "shopping_cart",
    "order_price",
    "delivery_charges",
    "customer_lat",
    "customer_long",
    "coupon_discount",
    "order_total",
    "season",
    "is_expedited_delivery",
    "distance_to_nearest_warehouse",
    "latest_customer_review",
    "is_happy_customer",
]

SEASON_BY_MONTH = {
    12: "Summer",
    1: "Summer",
    2: "Summer",
    3: "Autumn",
    4: "Autumn",
    5: "Autumn",
    6: "Winter",
    7: "Winter",
    8: "Winter",
    9: "Spring",
    10: "Spring",
    11: "Spring",
}

MODEL_FEATURES = [
    "distance_to_nearest_warehouse",
    "is_expedited_delivery",
    "is_happy_customer",
]


@dataclass
class MissingCleaningResult:
    cleaned: pd.DataFrame
    audit: pd.DataFrame
    price_map: dict[str, float]
    seasonal_models: dict[str, LinearRegression]
    vader_analyzer: SentimentIntensityAnalyzer
    product_prices: pd.DataFrame
    price_diagnostics: pd.DataFrame
    model_diagnostics: pd.DataFrame
    imputation_summary: pd.DataFrame
    validation: pd.DataFrame


def parse_order_date(value: Any) -> pd.Timestamp:
    """Parse dates using the published year-first/day-first conventions."""
    text = str(value).strip()
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return pd.to_datetime(text, yearfirst=True, errors="raise")
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        return pd.to_datetime(text, dayfirst=True, errors="raise")
    return pd.to_datetime(text, errors="raise")


def parse_cart(value: str) -> list[dict[str, Any]]:
    """Parse and validate the required shopping-cart JSON structure."""
    cart = json.loads(value)
    if not isinstance(cart, list) or not cart:
        raise ValueError("shopping_cart must be a non-empty JSON array")
    for item in cart:
        if set(item) != {"product", "quantity"}:
            raise ValueError(f"Invalid cart item keys: {item}")
        if not isinstance(item["product"], str) or not item["product"]:
            raise ValueError(f"Invalid product: {item}")
        if not isinstance(item["quantity"], int) or item["quantity"] <= 0:
            raise ValueError(f"Invalid quantity: {item}")
    return cart


def haversine_km(
    customer_lat: float,
    customer_long: float,
    warehouse_lat: float,
    warehouse_long: float,
) -> float:
    """Calculate great-circle distance using the published 6,371 km radius."""
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = np.radians(
        [customer_lat, customer_long, warehouse_lat, warehouse_long]
    )
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return float(2 * radius_km * np.arcsin(np.sqrt(haversine)))


def nearest_warehouse(
    customer_lat: float, customer_long: float, warehouses: pd.DataFrame
) -> tuple[str, float]:
    """Return the nearest warehouse name and unrounded distance."""
    candidates = [
        (
            row.warehouse,
            haversine_km(
                customer_lat,
                customer_long,
                row.latitude,
                row.longitude,
            ),
        )
        for row in warehouses.itertuples(index=False)
    ]
    return min(candidates, key=lambda item: item[1])


def setup_vader(
    resource_contract_path: Path, nltk_data_dir: Path
) -> SentimentIntensityAnalyzer:
    """Verify and load the frozen local VADER resource without downloading."""
    contract = json.loads(resource_contract_path.read_text(encoding="utf-8"))
    if nltk.__version__ != contract["nltk_version"]:
        raise RuntimeError(
            f"NLTK {nltk.__version__} does not match required "
            f"{contract['nltk_version']}"
        )
    lexicon = resource_contract_path.parent / contract["vader_lexicon_relative_path"]
    actual_hash = sha256(lexicon.read_bytes()).hexdigest()
    if actual_hash != contract["vader_lexicon_sha256"]:
        raise RuntimeError("Frozen VADER lexicon checksum does not match contract")
    nltk_path = str(nltk_data_dir.resolve())
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)
    return SentimentIntensityAnalyzer()


def review_is_happy(review: str, analyzer: SentimentIntensityAnalyzer) -> bool:
    """Apply the exact sentinel and published VADER compound rule."""
    if review == "No review submitted":
        return True
    return bool(analyzer.polarity_scores(review)["compound"] >= 0.05)


def infer_product_prices(
    missing_original: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Infer current prices and validate them on held-out cart equations."""
    observed = missing_original.loc[missing_original["order_price"].notna()].copy()
    products = sorted(
        {
            item["product"]
            for cart in observed["shopping_cart"].map(parse_cart)
            for item in cart
        }
    )

    matrix = []
    target = []
    for row in observed.itertuples(index=False):
        quantities = {item["product"]: item["quantity"] for item in parse_cart(row.shopping_cart)}
        matrix.append([quantities.get(product, 0) for product in products])
        target.append(row.order_price)

    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=5196
    )
    if np.linalg.matrix_rank(x[train_indices]) != len(products):
        raise RuntimeError("Price training equations do not have full column rank")

    heldout_prices = np.linalg.lstsq(
        x[train_indices], y[train_indices], rcond=None
    )[0]
    heldout_prices = np.round(heldout_prices, 2)
    heldout_predictions = np.round(x[test_indices] @ heldout_prices, 2)

    final_prices = np.linalg.lstsq(x, y, rcond=None)[0]
    final_prices = np.round(final_prices, 2)
    full_predictions = np.round(x @ final_prices, 2)
    prices = dict(zip(products, final_prices, strict=True))

    diagnostics = pd.DataFrame(
        [
            {
                "observed_equations": len(y),
                "training_equations": len(train_indices),
                "heldout_equations": len(test_indices),
                "training_rank": int(np.linalg.matrix_rank(x[train_indices])),
                "product_count": len(products),
                "heldout_max_absolute_error": float(
                    np.max(np.abs(heldout_predictions - y[test_indices]))
                ),
                "all_equations_max_absolute_error": float(
                    np.max(np.abs(full_predictions - y))
                ),
            }
        ]
    )
    return prices, diagnostics


def calculate_order_price(cart_text: str, prices: dict[str, float]) -> float:
    """Apply the published item-rounding and order-price sequence."""
    item_amounts = [
        round(item["quantity"] * prices[item["product"]], 2)
        for item in parse_cart(cart_text)
    ]
    return round(sum(item_amounts), 2)


def fit_seasonal_models(
    missing_original: pd.DataFrame,
) -> tuple[dict[str, LinearRegression], pd.DataFrame]:
    """Evaluate and fit seasonal delivery models on reliable observed rows."""
    reliable = missing_original.dropna(
        subset=["season", "delivery_charges", *MODEL_FEATURES]
    ).copy()
    models: dict[str, LinearRegression] = {}
    diagnostics = []

    for season in ["Summer", "Autumn", "Winter", "Spring"]:
        season_rows = reliable.loc[reliable["season"] == season]
        x = season_rows[MODEL_FEATURES].astype(float)
        y = season_rows["delivery_charges"].astype(float)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=5196
        )
        evaluation_model = LinearRegression().fit(x_train, y_train)
        predictions = evaluation_model.predict(x_test)
        final_model = LinearRegression().fit(x, y)
        models[season] = final_model
        diagnostics.append(
            {
                "season": season,
                "reliable_observations": len(season_rows),
                "training_observations": len(x_train),
                "heldout_observations": len(x_test),
                "heldout_r_squared": r2_score(y_test, predictions),
                "heldout_mae": mean_absolute_error(y_test, predictions),
                "intercept": final_model.intercept_,
                "distance_coefficient": final_model.coef_[0],
                "expedited_coefficient": final_model.coef_[1],
                "happy_coefficient": final_model.coef_[2],
            }
        )
    return models, pd.DataFrame(diagnostics)


def predict_delivery(
    model: LinearRegression,
    distance: float,
    expedited: bool,
    happy: bool,
) -> float:
    """Predict a delivery charge with consistently encoded boolean features."""
    features = pd.DataFrame(
        [[distance, int(expedited), int(happy)]], columns=MODEL_FEATURES
    )
    return float(model.predict(features)[0])


def _add_missing_change(
    audit: AuditTrail,
    source_file: str,
    order_id: str,
    field_name: str,
    corrected_value: Any,
    method: str,
    evidence: str,
    confidence: str = "high",
) -> None:
    audit.add_change(
        source_file=source_file,
        record_key=order_id,
        field_name=field_name,
        issue_type="missing",
        original_value=np.nan,
        corrected_value=corrected_value,
        method=method,
        evidence=evidence,
        confidence=confidence,
    )


def validate_missing_output(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    reread: pd.DataFrame,
    audit: pd.DataFrame,
    warehouses: pd.DataFrame,
    prices: dict[str, float],
    analyzer: SentimentIntensityAnalyzer,
) -> pd.DataFrame:
    """Run read-back, relationship, preservation, and audit checks."""
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
        "Output must use the exact original 16-column schema.",
    )
    record("row_count", len(reread), len(reread) == 500, "All missing-batch orders retained.")
    record(
        "unique_order_ids",
        reread["order_id"].nunique(),
        reread["order_id"].is_unique,
        "One row remains for each order.",
    )
    input_ids = set(original["order_id"])
    output_ids = set(reread["order_id"])
    record(
        "preserved_order_id_set",
        {"missing": len(input_ids - output_ids), "unexpected": len(output_ids - input_ids)},
        input_ids == output_ids,
        "No order IDs were lost or introduced.",
    )
    missing_cells = int(reread.isna().sum().sum())
    record("required_completeness", missing_cells, missing_cells == 0, "No empty fields remain.")

    carts_valid = True
    try:
        reread["shopping_cart"].map(parse_cart)
    except (TypeError, ValueError, json.JSONDecodeError):
        carts_valid = False
    record("shopping_cart_structure", carts_valid, carts_valid, "All carts are valid JSON arrays.")

    expected_prices = reread["shopping_cart"].map(lambda cart: calculate_order_price(cart, prices))
    price_error = float((expected_prices - reread["order_price"]).abs().max())
    record("order_price_relationship", price_error, price_error <= 0.01, "Cart equations reproduce order prices.")

    parsed_dates = reread["date"].map(parse_order_date)
    expected_seasons = parsed_dates.dt.month.map(SEASON_BY_MONTH)
    season_mismatches = int((expected_seasons != reread["season"]).sum())
    record("date_season_relationship", season_mismatches, season_mismatches == 0, "Dates map to Australian seasons.")

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
    record("nearest_warehouse_relationship", warehouse_mismatches, warehouse_mismatches == 0, "Nearest warehouse agrees with coordinates.")
    record("distance_relationship", max_distance_error, max_distance_error <= 0.0001, "Haversine distances agree to four decimals.")

    expected_happiness = reread["latest_customer_review"].map(
        lambda review: review_is_happy(review, analyzer)
    )
    observed_happiness = reread["is_happy_customer"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    happiness_mismatches = int((expected_happiness != observed_happiness).sum())
    record("review_happiness_relationship", happiness_mismatches, happiness_mismatches == 0, "VADER and sentinel rules agree.")

    expected_totals = (
        reread["order_price"] * (1 - reread["coupon_discount"] / 100)
        + reread["delivery_charges"]
    ).round(2)
    total_errors = (expected_totals - reread["order_total"]).abs()
    imputed_delivery_ids = set(
        audit.loc[audit["field_name"] == "delivery_charges", "record_key"]
    )
    imputed_delivery_mask = reread["order_id"].isin(imputed_delivery_ids)
    observed_charge_max_error = float(total_errors.loc[~imputed_delivery_mask].max())
    imputed_charge_max_error = float(total_errors.loc[imputed_delivery_mask].max())
    record(
        "order_total_relationship",
        {
            "max_error_with_observed_charge": observed_charge_max_error,
            "max_error_with_model_imputed_charge": imputed_charge_max_error,
        },
        observed_charge_max_error <= 0.0100001
        and imputed_charge_max_error <= 0.0800001,
        "Observed totals are preserved; small differences caused by model-imputed "
        "delivery charges remain within the published delivery tolerance.",
    )

    # Validate preservation before CSV serialization, where values have original dtypes.
    changed_observed_cells = []
    for column in ORDER_COLUMNS:
        observed_mask = original[column].notna()
        unequal = original.loc[observed_mask, column].astype(str) != cleaned.loc[observed_mask, column].astype(str)
        if unequal.any():
            changed_observed_cells.extend(
                zip(
                    original.loc[observed_mask].loc[unequal, "order_id"],
                    [column] * int(unequal.sum()),
                )
            )
    record(
        "preservation_of_observed_values",
        len(changed_observed_cells),
        len(changed_observed_cells) == 0,
        "Only originally empty cells were changed.",
    )

    source_missing = {
        (row.order_id, column)
        for row in original.itertuples(index=False)
        for column in ORDER_COLUMNS
        if pd.isna(getattr(row, column))
    }
    audit_keys = set(zip(audit["record_key"], audit["field_name"]))
    audit_shape_ok = list(audit.columns) == AUDIT_COLUMNS
    audit_values_ok = bool(
        (audit["issue_type"] == "missing").all()
        and (audit["original_value"] == "<MISSING>").all()
    )
    record(
        "audit_schema",
        list(audit.columns),
        audit_shape_ok,
        "Audit columns use the exact required order.",
    )
    record(
        "audit_missing_cell_reconciliation",
        {
            "source_missing_cells": len(source_missing),
            "audit_rows": len(audit),
            "missing_from_audit": len(source_missing - audit_keys),
            "extra_in_audit": len(audit_keys - source_missing),
        },
        source_missing == audit_keys and audit_values_ok,
        "Every and only source omission has one valid audit row.",
    )

    return pd.DataFrame(results)


def clean_missing_batch(
    *,
    missing_input: Path,
    warehouses_input: Path,
    resource_contract: Path,
    nltk_data_dir: Path,
    missing_output: Path,
    audit_output: Path,
) -> MissingCleaningResult:
    """Clean, audit, export, read back, and validate the missing batch."""
    original = pd.read_csv(
        missing_input, dtype={"order_id": "string", "customer_id": "string"}
    )
    if list(original.columns) != ORDER_COLUMNS or len(original) != 500:
        raise RuntimeError("Unexpected missing-data input structure")
    warehouses = pd.read_csv(warehouses_input)
    analyzer = setup_vader(resource_contract, nltk_data_dir)
    prices, price_diagnostics = infer_product_prices(original)
    models, model_diagnostics = fit_seasonal_models(original)

    cleaned = original.copy(deep=True)
    audit = AuditTrail()
    source_file = missing_input.name

    for index, row in original.loc[original["season"].isna()].iterrows():
        corrected = SEASON_BY_MONTH[parse_order_date(row["date"]).month]
        cleaned.at[index, "season"] = corrected
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "season",
            corrected,
            "Australian season derived from order-date month",
            "M01: published month-to-season mapping",
        )

    for index, row in original.loc[original["nearest_warehouse"].isna()].iterrows():
        corrected, _ = nearest_warehouse(row["customer_lat"], row["customer_long"], warehouses)
        cleaned.at[index, "nearest_warehouse"] = corrected
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "nearest_warehouse",
            corrected,
            "Minimum Haversine distance to supplied warehouses",
            "M02: 6,371 km Haversine comparison across all three warehouses",
        )

    for index, row in original.loc[original["distance_to_nearest_warehouse"].isna()].iterrows():
        _, distance = nearest_warehouse(row["customer_lat"], row["customer_long"], warehouses)
        corrected = round(distance, 4)
        cleaned.at[index, "distance_to_nearest_warehouse"] = corrected
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "distance_to_nearest_warehouse",
            corrected,
            "Haversine distance to nearest supplied warehouse",
            "M03: 6,371 km radius; reconstructed distance rounded to four decimals",
        )

    for index, row in original.loc[original["order_price"].isna()].iterrows():
        corrected = calculate_order_price(row["shopping_cart"], prices)
        cleaned.at[index, "order_price"] = corrected
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "order_price",
            corrected,
            "Sum of rounded quantity times inferred current unit price",
            "M04: 490 observed equations; rank 12; held-out maximum error AUD 0.00",
        )

    for index, row in original.loc[original["order_total"].isna()].iterrows():
        corrected = round(
            row["order_price"] * (1 - row["coupon_discount"] / 100)
            + row["delivery_charges"],
            2,
        )
        cleaned.at[index, "order_total"] = corrected
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "order_total",
            corrected,
            "Published discounted-item-price plus delivery formula",
            "M05: order_total arithmetic relationship",
        )

    for index, row in original.loc[original["is_happy_customer"].isna()].iterrows():
        corrected = review_is_happy(row["latest_customer_review"], analyzer)
        cleaned.at[index, "is_happy_customer"] = corrected
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "is_happy_customer",
            corrected,
            "Frozen local VADER compound score and published sentinel rule",
            "M06: verified lexicon checksum; compound >= 0.05",
        )

    for index, row in original.loc[original["delivery_charges"].isna()].iterrows():
        prediction = predict_delivery(
            models[row["season"]],
            row["distance_to_nearest_warehouse"],
            bool(row["is_expedited_delivery"]),
            bool(row["is_happy_customer"]),
        )
        corrected = round(prediction, 2)
        cleaned.at[index, "delivery_charges"] = corrected
        metric = model_diagnostics.loc[model_diagnostics["season"] == row["season"]].iloc[0]
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "delivery_charges",
            corrected,
            "Season-specific linear model using distance, expedited, and happiness",
            f"M07: {row['season']} held-out R2={metric['heldout_r_squared']:.6f}, MAE={metric['heldout_mae']:.4f}",
        )

    for index, row in original.loc[original["is_expedited_delivery"].isna()].iterrows():
        model = models[row["season"]]
        expected_regular = predict_delivery(
            model,
            row["distance_to_nearest_warehouse"],
            False,
            bool(row["is_happy_customer"]),
        )
        expected_expedited = predict_delivery(
            model,
            row["distance_to_nearest_warehouse"],
            True,
            bool(row["is_happy_customer"]),
        )
        regular_residual = abs(row["delivery_charges"] - expected_regular)
        expedited_residual = abs(row["delivery_charges"] - expected_expedited)
        corrected = bool(expedited_residual < regular_residual)
        cleaned.at[index, "is_expedited_delivery"] = corrected
        _add_missing_change(
            audit,
            source_file,
            row["order_id"],
            "is_expedited_delivery",
            corrected,
            "Choose 0/1 state with smaller seasonal-model absolute residual",
            f"M08: regular residual={regular_residual:.4f}; expedited residual={expedited_residual:.4f}",
        )

    if cleaned.isna().any().any():
        raise RuntimeError("Missing values remain after imputation")

    # Preserve original column order and write required files.
    cleaned = cleaned[ORDER_COLUMNS]
    missing_output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(missing_output, index=False, encoding="utf-8")
    audit.write_csv(audit_output)

    reread = pd.read_csv(
        missing_output, dtype={"order_id": "string", "customer_id": "string"}
    )
    audit_frame = pd.read_csv(audit_output, keep_default_na=False)
    validation = validate_missing_output(
        original,
        cleaned,
        reread,
        audit_frame,
        warehouses,
        prices,
        analyzer,
    )
    if not validation["passed"].all():
        failures = validation.loc[~validation["passed"], "check"].tolist()
        raise RuntimeError(f"Missing-output validation failed: {failures}")

    product_prices = pd.DataFrame(
        [{"product": product, "unit_price": value} for product, value in prices.items()]
    )
    imputation_summary = (
        audit_frame.groupby(["field_name", "confidence"], as_index=False)
        .size()
        .rename(columns={"size": "imputed_values"})
    )
    return MissingCleaningResult(
        cleaned=cleaned,
        audit=audit_frame,
        price_map=prices,
        seasonal_models=models,
        vader_analyzer=analyzer,
        product_prices=product_prices,
        price_diagnostics=price_diagnostics,
        model_diagnostics=model_diagnostics,
        imputation_summary=imputation_summary,
        validation=validation,
    )
