# Group030 A2 audit data

The three 500-row files are later incoming batches created by a new staff
member. Audit dirty, missing and outlying values without changing valid cells.
The shopping cart is JSON. Current-period prices differ from the historical A1
catalogue; infer the bounded current price vector from complete cart/order-price
equations before imputing missing order prices.

Each dirty row contains at most one intentional anomaly and has one defensible
correction. The missing file contains omissions only, and the outlier file has
no dirty or missing cells. Preserve every valid value and return dates in
`YYYY-MM-DD` format.

Published derivations:

- order total = order price * (1 - coupon discount / 100) + delivery charge;
- Australian seasons use calendar months; and
- the nearest warehouse is the minimum haversine distance from the customer.

Delivery charge follows a separate linear model in each season using distance,
expedited delivery and prior-customer happiness. Estimate it from valid rows and
inspect residuals. `latest_customer_review` is an English review written before
the current order; when no eligible prior review exists, the customer is treated
as happy. Apply the published VADER threshold (`compound >= 0.05`) to reviews.
Use only the frozen lexicon in `resources/nltk_data/sentiment/vader_lexicon.zip`;
verify it against `resource_contract.json` before analysis.

The outlier file contains delivery-charge outliers only. Remove the identified
outlier rows from the submitted outlier solution; preserve all valid rows and
columns in the dirty and missing solutions.
