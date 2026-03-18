# Scoring

## Open-Ended Questions

Open-ended questions use Brier Skill Score over the submitted outcome-probability pairs.

- Formula: `1 - sum((p_i - y_i)^2)`
- Higher is better.
- `1.0` is perfect.
- `0.0` is no-skill baseline.
- Negative values are worse than uniform.

In multi-agent runs, peer scoring compares your current Brier Skill Score to the average of the other agents' current scores.

## Binary Questions

Binary questions use Brier Score over `Yes` and `No`.

- Let `p` be the probability of `Yes`.
- Let `y` be `1` if the answer resolves to `Yes`, else `0`.
- Formula: `(p - y)^2`
- Lower is better.

Binary peer score uses `100 x (avg others' Brier - your Brier)`, so positive values mean better-than-peer performance.

## Benchmark Mechanics That Matter

- Scores are time-weighted, so earlier predictions matter for longer.
- Final score sums across the questions an agent predicts on, so coverage matters.
- The prompt builders emphasize calibration, not just top-1 accuracy.
- When the environment reviews whether a new forecast improved on the old one, a first prediction is compared against the abstainer baseline `0.0`.
- `daily_metrics.csv` is written once per wakeup session, not once per calendar day.
- When `timegap_days > 1`, active-question metrics use the forecast snapshot that remains in force through the end of that wakeup interval.
- `test_daily_metrics.csv` mirrors the main metrics file but includes only questions tagged with `source_split == "test"`.
