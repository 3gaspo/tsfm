# Experiment catalog

This page maps workflow fronts to evaluation questions. Exact data, model,
metric, and artifact contracts remain in the
[`experiment guideline`](../latex/experiment_guideline.pdf).

## Cadence-aware profiles

| Mode | Scope |
|---|---|
| `test` | Electricity `504:168`, seed 1 |
| `full` | Primary datasets plus eligible TIME panels at cadence-specific ranges |
| `ultra` | Full scope plus Weather/ETTh1 or wider foundation coverage |

Hourly ranges are `168:24`, `336:48`, and `504:168`; daily ranges are
`7:1`, `14:2`, and `30:7`; 15-minute ranges are `96:4`, `192:8`,
and `672:96`.

## Slurm evaluations

| Front | Scientific question | Models or factors |
|---|---|---|
| `slurm/dgx/main/01_univariate.slurm` | How does Chronos-2 compare with simple non-trainable references? | persistence, expected, repeat, weekly lookback, Chronos-2 |
| `slurm/dgx/main/02_controls.slurm` | How much do target instance normalization and constant-window removal affect results? | model x normalization x constant policy |
| `slurm/dgx/main/03_covariates.slurm` | Do identity or prepared known covariates help capable frozen models? | none, identity, optional known covariates |
| `slurm/dgx/main/04_foundation_models.slurm` | How do official frozen foundation pipelines compare? | Chronos-2, Chronos-Bolt, Chronos-T5, TS-ICL |

Matching fronts under `slurm/selena/main/` execute the same science.
Publishable aggregates are under `outputs/reports/<family>/<mode>/`; detailed
plots and aligned inputs are under `outputs/diagnostics/<family>/<mode>/`.
