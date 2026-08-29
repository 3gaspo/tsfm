# TSFM evaluation

This inference-only benchmark compares official frozen time-series foundation
models with simple non-trainable references on identical query rows. It also
isolates target normalization, constant-window handling, and explicit
covariate use while retaining each upstream model's native inference path.

The evaluator consumes portable wide CSV datasets, produces raw-scale
population-aware metrics, and stores compact publishable reports separately
from per-configuration diagnostics.

## Documentation map

| Need | Document |
|---|---|
| Formal evaluation task, controls, and metrics | [`latex/method_overview.pdf`](latex/method_overview.pdf) |
| Loader, adapter, evaluator, report, and diagnostics flow | [`docs/architecture.md`](docs/architecture.md) |
| Four workflow families and cadence-aware profiles | [`docs/experiment_catalog.md`](docs/experiment_catalog.md) |
| Current evidence boundary and rerun scope | [`docs/results_recap.md`](docs/results_recap.md) |
| Complete reproducibility specification | [`latex/experiment_guideline.pdf`](latex/experiment_guideline.pdf) |
| Full historical and analyzed evidence record | [`latex/executive_summary.pdf`](latex/executive_summary.pdf) |

## Setup

Use the project-managed environment from the repository root:

```bash
uv sync
export PYTHONPATH=src
```

Place wide CSV datasets and adjacent `config.json` files under `datasets/`.
`missing_values` defaults to `zero`, replacing NaNs after aggregation;
`error` rejects them, and infinite values are always rejected.
Place Chronos-2, Chronos-Bolt, Chronos-T5, and TS-ICL checkpoints under
`weights/` using the paths declared by the project configuration. Full
profiles discover eligible TIME panels from `datasets/time/catalog.json`.

## Main executions

Run each narrow test before its publication profile:

```bash
EXPERIMENT_MODE=test sbatch slurm/dgx/main/01_univariate.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/01_univariate.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/02_controls.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/03_covariates.slurm
EXPERIMENT_MODE=test sbatch slurm/dgx/main/04_foundation_models.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/04_foundation_models.slurm
```

The four fronts evaluate reference forecasts, preprocessing controls,
covariates, and cross-foundation-model comparisons respectively. They default
to `STAGES=evaluate,report`; a stage subset is a recovery override. Exact
models, factors, cadence ranges, and dataset coverage are in the
[experiment catalog](docs/experiment_catalog.md).

Matching Selena fronts execute the same science, for example:

```bash
EXPERIMENT_MODE=test sbatch slurm/selena/main/01_univariate_selena.slurm
```

Prepare selected TIME datasets with
`PYTHONPATH=src uv run python -m scripts.prepare_time_csv`; use the command's
help for source and selection arguments.

## Outputs and cluster operations

- Runs: `outputs/<univariate|controls|covariates|foundation_models>/`.
- Publishable aggregates: `outputs/reports/<family>/<mode>/`.
- Detailed plots and aligned inputs: `outputs/diagnostics/<family>/<mode>/`.
- Runtime streams: `logs/`; Selena uses project-specific scratch
  `logs_selena/` and `outputs_selena/` roots.

Existing legacy report subdirectories can be reorganized once without
rerunning inference:

```bash
PYTHONPATH=src uv run python -m scripts.migrate_report_diagnostics \
  --outputs-root outputs_selena
```

Preview and then mirror maintained code from DGX:

```bash
bash sync_code_to_selena.sh --dry-run
bash sync_code_to_selena.sh
```

The preview marks stale maintained files with `*deleting`. Delayed deletion
preserves excluded environments, dependency manifests, datasets, weights,
outputs, and logs.

Pull Selena artifacts from DGX with the smallest useful tier:

```bash
bash sync_results_to_dgx.sh
bash sync_results_to_dgx.sh --size detailed
bash sync_results_to_dgx.sh --size full
```

The default retrieves logs and aggregate reports. `detailed` adds non-binary
runs and diagnostics required by the analysis notebook; `full` adds binary
recovery payloads. Use `bash publish_job.sh <job-id>` for one terminal log pair
or `bash publish_job.sh` for all logs plus aggregate reports.

## Documentation maintenance

```bash
PYTHONPATH=src python -m scripts.build_docs
PYTHONPATH=src python -m scripts.build_docs --render method
PYTHONPATH=src python -m scripts.build_docs --render all
```

The default validates the documentation map and all four DGX fronts. The
method note owns the evaluation formulation, architecture owns implementation,
the catalog owns planned workflows, and the recap plus executive summary own
analyzed evidence.
