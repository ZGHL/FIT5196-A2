# Group030 A2 submission validation

Validation date: 20 October 2026 (Australia/Sydney)

## Reproducibility environment

- Python 3.13.12
- Exact package versions from `requirements.txt`
- Fresh test directory containing only the submitted notebook and source
  modules plus a simulated marker-provided `data/raw/Group030_A2/` directory
- Offline local VADER lexicon from the allocated resource package

## Execution results

- `python -m src.Group030_solution`: PASS
- `python Group030_solution.py` from the final submission root: PASS
- Notebook restart/run-all equivalent: PASS
- Notebook code cells executed: 70 of 70
- Notebook error outputs: 0
- Missing-output validation checks: all PASS
- Dirty-output validation checks: all PASS
- Outlier-output validation checks: all PASS
- Task 3 reproduced from the three final cleaned CSVs: PASS

## Final output checks

| File | Rows | Columns | Duplicate order IDs | Blank cells |
|---|---:|---:|---:|---:|
| `Group030_dirty_clean.csv` | 500 | 16 | 0 | 0 |
| `Group030_missing_clean.csv` | 500 | 16 | 0 | 0 |
| `Group030_outlier_clean.csv` | 476 | 16 | 0 | 0 |
| `audit_log.csv` | 184 | 9 | n/a | 0 |

The audit log contains 80 dirty repairs, 80 missing-value replacements and 24
outlier-row removals. Every record uses the required nine-column order. Missing
source values are represented as `<MISSING>` and removed rows as
`ROW_REMOVED`.

The regenerated CSV contents are cell-for-cell equal to the reviewed outputs.
Byte-level hashes can differ when a file is rewritten on Windows because CSV
line endings change from LF to CRLF; this does not change parsed values.

## Submission package checks

- The ZIP has one top-level directory: `Group030_A2_submission/`.
- Raw inputs, the VADER resource, caches, virtual environments and development
  notes are excluded.
- The three cleaned CSVs, audit log, executed notebook, matching Python entry
  point, all imported `src/` helpers and `requirements.txt` are included.
- The AI declaration draft and an English AI-record index are included.
- The Python entry point resolves both the development `src/` layout and the
  final submission-root layout without an absolute path.

## Required human action

Before Moodle upload, every group member must add their own original or
approved electronic signature and date to `Group030_AI_declaration.pdf`.
Members must also add any earlier AI conversation exports not available in this
repository. No signature has been fabricated during packaging.
