# FIT5196 Assessment 2 — Group030

This repository contains the reproducible code and notebook structure for
FIT5196 Assessment 2. The workflow cleans the three allocated order batches,
generates a field-level audit trail, validates the submitted outputs, and
compares transformations for customer profiling.

## Repository layout

```text
FIT5196-A2/
├── data/
│   └── raw/                         # local allocated package; not committed
│       └── Group030_A2/
├── notebooks/
│   └── Group030_solution.ipynb     # main technical report
├── src/
│   └── Group030_solution.py       # matching Python source
├── outputs/                            # reproducible working CSV outputs
├── submission/
│   └── Group030_A2_submission/    # locally assembled final deliverables
├── requirements.txt
└── README.md
```

## Local data setup

The allocated raw package and supplied course files are intentionally excluded
from Git. Extract `Group030_A2.zip` so that the local layout is:

```text
data/raw/Group030_A2/
├── Group030_dirty_data.csv
├── Group030_missing_data.csv
├── Group030_outlier_data.csv
├── warehouses.csv
├── resource_contract.json
└── resources/nltk_data/sentiment/vader_lexicon.zip
```

The notebook keeps `GROUP_ID`, `DATA_DIR`, and `OUTPUT_DIR` in one configuration
cell. Run the notebook from the repository root unless those paths are changed.

## Workflow

1. Inspect schemas, missingness, categories, distributions, keys, and business relationships.
2. Infer current product prices from reliable cart/order-price equations.
3. Repair the dirty batch and record each changed field.
4. Fit and evaluate separate seasonal delivery-charge models.
5. Recover missing values using deterministic rules or evaluated model estimates.
6. Identify delivery-charge outliers from seasonal-model residuals and remove those rows.
7. Export and read back the three cleaned CSVs and `audit_log.csv`.
8. Run executable validation and reconcile output differences with the audit trail.
9. Build customer profiles from the final cleaned outputs and compare transformations.
10. Restart the kernel, Run All, and assemble the required submission folder locally.

## Required final submission files

The local `submission/Group030_A2_submission/` folder must eventually contain:

- `Group030_dirty_clean.csv`
- `Group030_missing_clean.csv`
- `Group030_outlier_clean.csv`
- `audit_log.csv`
- `Group030_solution.ipynb`
- `Group030_solution.py`
- `requirements.txt`
- `Group030_AI_declaration.pdf`
- `AI_records/` when applicable

Generated data and AI records are ignored by default to avoid accidentally
publishing allocated data, assessment answers, signatures, or conversation
records. Review repository visibility and academic-integrity requirements before
changing these ignore rules.

