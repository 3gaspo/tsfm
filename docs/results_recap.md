# Finalized experiment recap

## Current evidence

The synchronized full-profile reports select 189 tasks for each completed
family: 945 univariate inputs, 1,512 control inputs, and 756 covariate inputs.
The selected CSV panels for these reports are finite, so the new default
zero-fill policy is inert for those completed computations.

Chronos-2 has lower aggregate nMSE than the best repeat/lookback/mean baseline
on 153 of 189 univariate tasks, with one tie. The median relative improvement
is 24.10%. On the nine Electricity, Solar, and Traffic publication settings,
the improvement ranges from 20.88% to 51.26%. Exchange Rate is the clear
exception: persistence wins all three settings by 0.60% to 4.88%.

Removing constant windows improves only 5 of 189 matched Chronos-2 tasks and
has zero median effect, although those five prevent extreme normalized-error
cases. Instance normalization improves 93 of 189 tasks, also with zero median
effect. These controls therefore do not support a universal intervention.

The identity-covariate capability control beats the same backbone without
covariates on 185 of 189 Chronos-2 tasks and 176 of 189 TS-ICL tasks, with
median relative nMSE improvements of 97.93% and 98.07%. Identity exposes the
target itself and is an oracle control, not a deployable forecasting method.
The synchronized raw results are eligible, but the published covariate
comparison tables must be regenerated: the previous report code incorrectly
looked for a simple baseline inside covariate-only groups and wrote empty
comparison tables.

Foundation replacement job 2964473 resumed the failed evaluation, completed
Weather, Exchange Rate, and all remaining eligible TIME tasks, and regenerated
the full report. The current 512-row grid covers 128 tasks and four models over
64 datasets. TS-ICL wins 51 tasks, Chronos-2 50, Chronos-Bolt 17, and
Chronos-T5 10. Chronos-2 has the best mean rank and beats TS-ICL head-to-head
on 69 of 128 tasks. This supersedes jobs 2962543 and 2964370.

## Finalized conclusions

- Chronos-2 is strong against simple baselines on the completed full grid, but
  the conclusion is not universal because persistence remains best on
  Exchange Rate.
- Constant-window removal and instance normalization are dataset-dependent
  controls, not default performance improvements.
- Identity covariates confirm that both capable backbones use supplied
  covariates, but their oracle nature prevents a practical-method claim.
- The complete foundation grid does not produce one universal winner: TS-ICL
  has the most task wins, while Chronos-2 has the best mean rank and wins their
  head-to-head comparison.

## Next evidence

Deploy the reporting fix and rerun only the full covariate report stage. The
foundation evaluation and report are complete and require no rerun.

The complete evidence boundary remains
[`executive_summary.pdf`](../latex/executive_summary.pdf).
