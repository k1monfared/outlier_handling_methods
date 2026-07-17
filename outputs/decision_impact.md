# Decision quality and illustrative business impact

Better outlier handling produces better ship / no-ship decisions. The table compares each metric's selected method to the no-handling baseline.

| metric | winner | decision error (none) | decision error (winner) | reduction (pp) | false-ship (none) | false-ship (winner) |
|---|---|---|---|---|---|---|
| Revenue per user (USD/day) | `winsor_99_upper` | 0.427 | 0.333 | 9.3 | 0.000 | 0.000 |
| Engagement events per user (count/day) | `trim_95_upper` | 0.393 | 0.227 | 16.7 | 0.007 | 0.007 |
| Conversion rate per user (bounded 0..1) | `none` | 0.180 | 0.180 | 0.0 | 0.013 | 0.013 |

Mean decision-error reduction across metrics: 8.7 percentage points.

## Confusion matrix per selected method

Ship decision versus the ground-truth-correct action (positive = a real win that should ship), for the selected method against no handling. Precision is TP / (TP + FP), recall is TP / (TP + FN).

| metric | method | true wins shipped (TP) | false ships (FP) | missed wins (FN) | correct holds (TN) | precision | recall |
|---|---|---|---|---|---|---|---|
| Revenue per user (USD/day) | `winsor_99_upper` | 14 | 0 | 50 | 86 | 1.00 | 0.22 |
| Revenue per user (USD/day) | `none` | 0 | 0 | 64 | 86 | 0.00 | 0.00 |
| Engagement events per user (count/day) | `trim_95_upper` | 26 | 1 | 33 | 90 | 0.96 | 0.44 |
| Engagement events per user (count/day) | `none` | 1 | 1 | 58 | 90 | 0.50 | 0.02 |
| Conversion rate per user (bounded 0..1) | `none` | 37 | 2 | 25 | 86 | 0.95 | 0.60 |
| Conversion rate per user (bounded 0..1) | `none` | 37 | 2 | 25 | 86 | 0.95 | 0.60 |

## Illustrative dollar translation

This figure is illustrative and derived transparently from the stated assumptions below. It is not a real financial result.

- Revenue at stake per experiment decision: $10,000,000
- Experiments per year (program scale): 500
- Illustrative annual loss avoided by using the per-metric winners instead of no handling: $81,757,558

Each experiment gates a launch decision affecting revenue_at_stake_per_experiment_usd over one year. A wrong decision (a missed win or a shipped regression) of relative size r misprices that revenue by r. Annual regret for a method = (mean per-experiment regret magnitude) * experiments_per_year * revenue_at_stake_per_experiment_usd. The illustrative avoided loss is the annual regret of no-handling minus that of the selected per-metric winner.