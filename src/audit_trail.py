"""Field-level audit logging helpers for FIT5196 A2 Group030."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


AUDIT_COLUMNS = [
    "source_file",
    "record_key",
    "field_name",
    "issue_type",
    "original_value",
    "corrected_value",
    "method",
    "evidence",
    "confidence",
]

VALID_ISSUE_TYPES = {"dirty", "missing", "outlier_row"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def _audit_value(value: Any, *, missing: bool = False) -> Any:
    """Return the required audit representation without changing real values."""
    if missing or pd.isna(value):
        return "<MISSING>"
    if isinstance(value, bool):
        return str(value)
    return value


@dataclass(frozen=True)
class AuditRecord:
    source_file: str
    record_key: str
    field_name: str
    issue_type: str
    original_value: Any
    corrected_value: Any
    method: str
    evidence: str
    confidence: str

    def __post_init__(self) -> None:
        if self.issue_type not in VALID_ISSUE_TYPES:
            raise ValueError(f"Unsupported issue_type: {self.issue_type}")
        if self.confidence not in VALID_CONFIDENCE:
            raise ValueError(f"Unsupported confidence: {self.confidence}")
        if not str(self.record_key):
            raise ValueError("record_key must contain the affected order_id")
        if not self.method or not self.evidence:
            raise ValueError("method and evidence must be documented")


class AuditTrail:
    """Collect audit records at the moment each workflow change is applied."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def add_change(
        self,
        *,
        source_file: str,
        record_key: str,
        field_name: str,
        issue_type: str,
        original_value: Any,
        corrected_value: Any,
        method: str,
        evidence: str,
        confidence: str,
    ) -> None:
        """Record one dirty-field repair or missing-value replacement."""
        if issue_type not in {"dirty", "missing"}:
            raise ValueError("add_change only accepts dirty or missing issues")
        self._records.append(
            AuditRecord(
                source_file=source_file,
                record_key=str(record_key),
                field_name=field_name,
                issue_type=issue_type,
                original_value=_audit_value(original_value),
                corrected_value=_audit_value(corrected_value),
                method=method,
                evidence=evidence,
                confidence=confidence,
            )
        )

    def add_removed_row(
        self,
        *,
        source_file: str,
        record_key: str,
        original_delivery_charge: Any,
        method: str,
        evidence: str,
        confidence: str,
    ) -> None:
        """Record one outlier row removed from the submitted output."""
        self._records.append(
            AuditRecord(
                source_file=source_file,
                record_key=str(record_key),
                field_name="delivery_charges",
                issue_type="outlier_row",
                original_value=_audit_value(original_delivery_charge),
                corrected_value="ROW_REMOVED",
                method=method,
                evidence=evidence,
                confidence=confidence,
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """Return records in the exact required column order."""
        return pd.DataFrame(
            [asdict(record) for record in self._records], columns=AUDIT_COLUMNS
        )

    def write_csv(self, path: Path | str) -> None:
        """Write a UTF-8 audit CSV without a pandas index."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output, index=False, encoding="utf-8")

