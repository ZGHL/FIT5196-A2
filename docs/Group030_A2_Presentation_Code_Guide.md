# Group030 A2 Presentation Code Guide

This guide supports the oral presentation. The submitted notebook remains the
technical report and the authoritative source of results.

## One-minute workflow summary

1. We first verified the allocated files, schema, keys, missingness and business
   relationships without changing the source data.
2. We inferred current product prices from reliable shopping-cart equations and
   evaluated four season-specific delivery-charge regressions on held-out rows.
3. We used deterministic rules where the answer was implied exactly and used an
   evaluated model only where a delivery charge could not be derived directly.
4. We repaired 80 missing cells and 80 dirty fields, then removed 24 orders whose
   delivery-charge residuals were conditionally anomalous.
5. We reconciled all 184 actions against the source/output differences and ran 50
   executable validation checks.
6. We built 471 customer profiles from the 1,476 retained orders and compared
   original, z-score, min-max, square-root and `log1p` representations.

## How to read each notebook code block

Every code cell now starts with `# Purpose:`. The Markdown immediately before a
cell states:

- what question the cell answers;
- what evidence it consumes;
- what object or figure it produces; and
- how that output supports the next decision.

During the presentation, explain the Markdown objective first, then point to only
the two or three code lines that implement the main idea.

## Recommended presentation route

| Stop | What to show | Main message |
|---|---|---|
| Executive summary | Four result cards | 1,476 retained orders, 471 profiles, 184 audited actions, 50/50 checks |
| Section 1.2 | Missingness table and heat map | Missingness is structured and limited to the missing batch |
| Section 2.2 | Product-price diagnostics | Current prices are inferred from full-rank cart equations and checked on held-out equations |
| Section 2.4 | Held-out metrics and prediction plots | Seasonal delivery models generalise to unseen reliable rows |
| Sections 3–5 | One worked example from each batch | Each action follows an independently testable relationship |
| Section 6 | Audit and consolidated validation | Logged actions equal actual source/output differences |
| Section 7.2 | Profile distributions and pairwise plots | The selected features differ greatly in units but are not redundant |
| Sections 7.3–7.5 | Scaling, nonlinear comparison and recommendation table | Transformations are purpose-specific, not applied automatically |

## Likely technical questions and concise answers

### Why are `order_id` and `customer_id` read as strings?

They are identifiers, not measured quantities. String dtype prevents accidental
numeric conversion or loss of formatting and makes key preservation checks exact.

### Why are the three raw files not merged before cleaning?

They have different error contracts: one contains omissions, one contains dirty
values, and one contains conditional delivery-charge outliers. Keeping source
identity prevents applying the wrong operation and is required for the audit log.

### How were product prices obtained?

For order (j), the observed price satisfies

\[
y_j = \sum_i q_{ji}p_i,
\]

where (q_{ji}) is the quantity of product (i) and (p_i) is its unknown
current unit price. The cart quantities form a design matrix. Least squares solves
the simultaneous equations, and full column rank confirms that all product prices
are identifiable. Held-out cart equations check reconstruction before all reliable
equations are used for the final estimates.

### Why use separate delivery models for each season?

The published data-generating relationship is seasonal. Separate models allow the
intercept and predictor effects to differ by season instead of forcing one global
relationship across four regimes.

### What are the delivery-model predictors?

The three predictors are distance to the nearest warehouse, expedited-delivery
status and customer-happiness status. Delivery charge is the response variable.

### What does held-out evaluation mean here?

Within each season, 80% of complete reliable rows train an evaluation model and
20% are kept unseen until prediction. `random_state=5196` makes the split
reproducible. After evaluation, the operational model is refitted on all reliable
rows so no trustworthy evidence is wasted during final cleaning.

### What do R² and MAE mean?

- `R²` measures the proportion of held-out charge variation explained by the
  predictors. Values close to one show strong predictive agreement.
- MAE is the mean absolute prediction error in Australian dollars, so it shows the
  typical error magnitude in the original unit.

High R² alone is not sufficient; the notebook also reports MAE and maximum
absolute residual.

### Why use a model for some missing values but rules for others?

Deterministic evidence is preferred whenever it uniquely determines the value.
Season comes from date, warehouse and distance come from coordinates, happiness
comes from the supplied VADER rule, and monetary fields follow published
arithmetic. A model is used only for delivery charges because they depend on a
learned seasonal relationship rather than an exact algebraic identity.

### How is a dirty monetary field isolated?

The workflow cross-checks independently derived cart price, discount arithmetic,
published total and the seasonal delivery estimate. Because a dirty row contains
at most one intentional error, the inconsistent field is the one whose correction
restores all independent relationships.

### Why are outliers based on residuals rather than raw charges?

A large charge can be legitimate for a distant or expedited delivery. The
residual

\[
r_j = observed_j - predicted_j
\]

measures inconsistency after accounting for the known predictors. This detects
conditionally abnormal charges that a raw-value threshold could miss.

### How is the outlier threshold chosen?

For each season, the workflow uses the largest reproducible residual observed in
reliable evidence and adds an explicit AUD 0.01 rounding guard. The 24 removed rows
are clearly separated from all retained residuals; no fixed percentage or order-ID
pattern is used.

### Why is an outlier order removed instead of replacing its charge?

The task defines the outlier file as containing anomalous delivery charges and
requires an output containing only retained original orders. There is insufficient
evidence to claim one exact replacement charge, so removing the affected order is
more defensible than inventing a corrected value.

### What does audit reconciliation prove?

It proves that the log describes what the program actually changed. Missing and
dirty entries correspond one-to-one with changed cells, and every removed outlier
ID has one `outlier_row` record. Unchanged values are not logged.

### Why were these three customer-profile attributes selected?

- `order_count` represents activity frequency;
- `mean_order_value` represents typical spending; and
- `mean_delivery_distance` represents delivery geography.

`total_paid` was excluded because
`total_paid = order_count × mean_order_value`. Including all three spending fields
would repeatedly weight related information in a distance calculation.

### Why compare z-score and min-max scaling?

Both remove original units, but they imply different distance behaviour. Z-score
gives each feature population variance one. Min-max fixes the observed endpoints
at zero and one but remains sensitive to extreme observed values. The notebook
shows the actual variance shares rather than choosing a method by convention.

### Why is square root recommended only for `mean_order_value`?

Square root reduces its skewness from approximately 0.928 to 0.156 and moderates
the influence of very large purchases. Transforming `order_count` would compress
an already short, interpretable 1–10 scale. Square root and `log1p` make the
delivery-distance skewness worse, so that feature remains untransformed.

### Why is z-score applied after the square-root transformation?

The nonlinear transformation first changes distribution shape and relative gaps.
The scaler is then fitted to that chosen representation so the final features have
comparable variance. Reversing the order would apply square root to negative
z-scores and violate its domain.

### Why was no clustering model fitted?

Task 3 asks for preparation and comparison for possible future distance-based
segmentation. It explicitly does not require clustering, a cluster count or proof
that meaningful customer segments exist.

## Boundaries to state clearly

- The customer profiles describe retained A2 orders only, not complete customer
  histories or lifetime value.
- Straight-line warehouse distance is not travel time or road distance.
- High model accuracy supports the specified charge relationship; it does not prove
  a causal effect.
- Transformations affect only the in-memory Task 3 analysis copy. They do not alter
  the three cleaned CSVs or create audit records.
- The source data are fictional and group-specific; no other group’s data or answer
  list is used.
