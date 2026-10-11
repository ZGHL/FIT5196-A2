# FIT5196 Assessment 2 — Group030

This repository contains the reproducible code and notebook structure for
FIT5196 Assessment 2. The workflow cleans the three allocated order batches,
generates a field-level audit trail, validates the submitted outputs, and
compares transformations for customer profiling.

## Repository layout

```text
FIT5196-A2/
├── data/
│   └── raw/                         # extracted allocated package
│       └── Group030_A2/
├── notebooks/
│   └── Group030_solution.ipynb     # main technical report
├── docs/
│   └── Group030_A2_Presentation_Code_Guide.md  # oral defence notes
├── src/
│   └── Group030_solution.py       # matching Python source
├── outputs/                            # reproducible working CSV outputs
├── submission/
│   └── Group030_A2_submission/    # locally assembled final deliverables
├── requirements.txt
└── README.md
```

## Local data setup

The extracted Group030 inputs are tracked in this private repository so all
group members use the same immutable source files. The duplicate source ZIP and
supplied course documents remain excluded from Git. The layout is:

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

Cleaned CSVs and the audit log are tracked under `outputs/` for private-team
review. The final Moodle submission must still exclude all raw inputs and the
frozen lexicon. Signed declarations and AI conversation records remain ignored
to avoid committing signatures or personal records.
