"""FIT5196 A2 Group030 reproducible pipeline source.

Keep this file aligned with the submitted notebook. Implementation will be
added as the cleaning, validation, and customer-profile sections are completed.
"""

from pathlib import Path

from src.missing_cleaning import clean_missing_batch


GROUP_ID = "Group030"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def main() -> None:
    """Run the currently implemented Group030 workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_result = clean_missing_batch(
        missing_input=MISSING_INPUT,
        warehouses_input=WAREHOUSES_INPUT,
        resource_contract=RESOURCE_CONTRACT,
        nltk_data_dir=NLTK_DATA_DIR,
        missing_output=MISSING_OUTPUT,
        audit_output=AUDIT_OUTPUT,
    )
    print(missing_result.imputation_summary.to_string(index=False))
    print(missing_result.validation[["check", "passed"]].to_string(index=False))


if __name__ == "__main__":
    main()
