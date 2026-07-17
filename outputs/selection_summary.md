# Selection summary

Methods are selected on production-measurable signals only. RMSE vs truth is shown for validation, not selection.

| metric | selected method | family | early->long R2 | early CI covers eventual | RMSE vs truth (validation) |
|---|---|---|---|---|---|
| Revenue per user (USD/day) | `winsor_99_upper` | winsorization | 0.156 | 0.940 | 0.0270 |
| Engagement events per user (count/day) | `trim_95_upper` | removal / trimming | 0.294 | 0.933 | 0.0189 |
| Conversion rate per user (bounded 0..1) | `none` | no handling | 0.327 | 0.927 | 0.0153 |

The winner differs across metrics by design and by result. See per_metric_results.md for the full tables and decision_impact.md for the decision-quality translation.