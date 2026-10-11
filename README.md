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

## Suggested five-person work allocation

The notebook sections from 0 to 7 are divided into five sequential work
packages. Each package has one owner; names can be assigned later. Section 8 is
the shared final checklist and is therefore not treated as a separate person's
analytical task.

| Work package | Notebook sections | Main responsibility | Required handoff |
|---|---|---|---|
| 1. Reproducibility and diagnosis | **Sections 0–1** | Maintain paths, dependencies and file-integrity checks; verify schema, keys, missingness, categories, numeric distributions and location evidence. | A confirmed input contract and a short list of reliable/unreliable evidence for Package 2. |
| 2. Shared rules and models | **Section 2** | Explain and verify cart parsing, product-price inference, date/season rules, geography, VADER sentiment and seasonal delivery models, including held-out evaluation. | The validated price map, rule definitions, VADER analyser, seasonal models and diagnostics for Package 3. |
| 3. Missing and dirty cleaning | **Sections 3–4** | Recover the 80 missing values, repair the 80 dirty fields, retain worked examples and run the batch-specific validation checks. | `Group030_missing_clean.csv`, `Group030_dirty_clean.csv`, their audit records and passing validation evidence for Package 4. |
| 4. Outliers, audit and validation | **Sections 5–6** | Apply the seasonal residual rule, remove only defensible outlier orders, consolidate `audit_log.csv`, reconcile actual changes and present the final cross-batch validation summary. | `Group030_outlier_clean.csv`, the complete 184-row audit log and the final validated outputs for Package 5. |
| 5. Customer profiling and transformation | **Section 7** | Build the customer-level analysis copy from the three final cleaned CSVs; compare original values, z-score, min-max, square-root and `log1p`; finalise the recommendation table and limitations. | Final Task 3 narrative, figures and recommendations for the group presentation. |

### Handoff order

```text
Package 1 → Package 2 → Package 3 → Package 4 → Package 5
```

This order follows the notebook's actual dependencies. Package 2 should not
change input assumptions without notifying Package 1. Package 3 must use the
validated rules and models from Package 2. Package 4 validates the exact files
produced by Package 3, and Package 5 must read the final cleaned outputs rather
than recreate or manually alter them.

To minimise merge conflicts, each owner should edit only their assigned notebook
sections and corresponding `src/` functions. Before handing work forward, the
owner should pull the latest `main`, restart the kernel, run their section and
record any changed output assumptions. After all five handoffs, the group should
perform one complete Run All and review Section 8 together.

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
