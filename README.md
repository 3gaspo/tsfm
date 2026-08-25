# TSFM Evaluation

This repository provides deterministic, inference-only evaluation of frozen
Chronos-2, Chronos-Bolt, TS-ICL, TiRex-2, and in-context TabPFN-TS. The model
files are thin tensor adapters around official packages and checkpoints;
their model architectures and inference pipelines are not reimplemented here.
It contains no forecasting-model training, optimization, PatchTST, or data
generation code. The univariate reference
workflow compares Chronos-2 with the non-trainable references derived from the
original TimeTensor project: persistence, lookback mean (`expected`), the final
`H` observed values (`repeat`), and a periodically aligned lookback window.

## Code organization

```text
src/
  data/             dataset/config loading, TIME preparation, and query construction
  external_models/  isolated frozen TSFM adapters
  model_loading/    non-trainable controls and adapter selection
  evaluation/       deterministic inference and metric computation
  pipeline/         run identity and manifest orchestration
  results/          artifact aggregation, comparisons, and tables
  visualization/    report plots and analysis notebook
  scripts/          evaluation, reporting, and data-preparation fronts
```

The five adapter files and TIME preparation implementation are byte-identical
to the shared evaluation surface in TimeTensors and online adaptation. The
external adapters contain model-specific inference behavior; evaluation
controls only which dates and covariates are accessible. Reporting and plotting
remain downstream of the scientific evaluation path.

## Evaluation contract

Targets are wide univariate panels with shape `(users, 1, dates)`. For a query
date `t`, the input is `X=(t-L,t]` and the target is `Y=(t,t+H]`. Query dates
are ordered deterministically, spaced by `evaluation.stride` (512 by default),
and evaluated for every user. `evaluation.remove_constant=true` removes only
user/query pairs whose lookback standard deviation is at most the configured
epsilon. Evaluation fractions own target dates; a lookback may cross the start
of the selected target interval.

`preprocessing.instance_normalize=true` standardizes each target lookback before
inference and reverses that transform on the prediction. Covariates retain
their source scale, matching the TimeTensors convention. nMSE is
always measured by normalizing prediction and target with the lookback standard
deviation, independently of the model preprocessing switch.

Every run reports MSE, MAE, nMSE, nMAE, and MASE. MASE divides forecast MAE by
the mean absolute in-lookback seasonal difference; seasonality defaults to one
and is controlled by `evaluation.mase_seasonality` or `MASE_SEASONALITY` in the
Slurm workflow. For every error metric the summary contains:

- the mean and population standard deviation across user/windows;
- the equal-user mean and population standard deviation across user means;
- the mean of the worst `ceil(10% * users)` user means;
- per-user means and within-user population standard deviations;
- per-horizon means and population standard deviations.

Total model inference time and time per user/window exclude model and data
loading. Per-user and per-window times are amortized from batched inference;
there is no artificial per-sample timing dispersion. `window_metrics.csv`, `per_user_metrics.csv`, and
`horizon_metrics.csv` preserve the lightweight rows needed for plots and
pairwise comparisons without saving predictions.

The primary ICLR grid is `168:24`, `336:48`, and `504:168` on Electricity,
Traffic, Solar, and Exchange Rate. Evaluation uses the entire eligible
timeline—there is no train/validation/test split—and deterministically retains
one query date every 512 steps by default. `EVAL_STRIDE` changes this speed and
coverage trade-off. Fixed TabPFN periods remain 24 and 168.

The `lookback` baseline is weekly seasonal persistence. With `P` observations
per week, it selects `k=ceil(H/P)` and predicts from
`X[L-kP:L-kP+H]`. This is the most recent weekly aligned `H`-sized history
window that remains inside the lookback. For example, `L=336`, `H=24`, and
hourly `P=168` yields `X[168:192]`. `P` is inferred from dated inputs (168 for
hourly data and 7 for daily data); set `model.lookback_period` for a direct run
or `LOOKBACK_PERIOD_STEPS` for a Slurm workflow to override it. Inputs without
real dates require the explicit override.

## Data and covariates

`data.path` may identify a wide CSV, a directory containing one CSV, or a
tensor directory produced by TimeTensors. The current tensor contract uses
`values.pt`, optional `datetimes.pt` and `individual_ids.pt`, optional
`individual_context.pt` and `global_context.pt`, and
`dataset_metadata.json` for source-ID-to-name mappings. Static context is
broadcast across dates and global context across users before the two are
concatenated. CSV targets default to every non-date, non-covariate column.
Three covariate modes are available:

- `none`: univariate inference;
- `identity`: the complete target window is supplied as a known covariate;
- `known`: use `data.covariate_cols` from the target CSV and/or user-aligned
  wide panels in `data.covariate_paths`.

Each external covariate panel must share the target timeline and either share
its user column names, have the same number/order of users, or contain one
global column. Paths are evaluation inputs; this project does not create them.

The loader discovers `config.json` beside the selected CSV/tensor directory.
Portable fields live at the JSON top level and project-only overrides under
`tsfm_evaluation`. Scoped settings override portable settings, explicit Hydra
values override both, and `drop_users` is merged additively across all levels.
The selected config path and applied keys are logged and recorded in every run
summary.

## Preparing TIME datasets

The Hugging Face `Real-TSF/TIME-ProcessedCSV` snapshot currently contains 2,036
repository files totaling 1,153,874,655 bytes (about 1.075 GiB). It does not
need to be downloaded in full. The preparation utility resolves an immutable
Hugging Face revision and uses filtered `snapshot_download` paths; the default
considers 15-minute (`15T`), hourly (`H`), and daily (`D`) CSV files, retains
source configurations with at most 500 usable series and at most 10,000 dates
in every series, and uses the Hugging Face cache across runs:

```bash
PYTHONPATH=src uv run python -m scripts.prepare_time_csv
```

Restrict it further by TIME dataset name, or select other frequency folders:

```bash
PYTHONPATH=src uv run python -m scripts.prepare_time_csv \
  --datasets CPHL Crypto --frequencies H D

PYTHONPATH=src uv run python -m scripts.prepare_time_csv \
  --frequencies H D --max-series 250 --max-dates-per-series 8000
```

Use `--frequencies all` for every frequency, or process an existing checkout
without network access using `--source-root /path/to/TIME-ProcessedCSV`. The
default eligibility settings are `336:48` and `504:168`, with stride 512;
`--settings` and `--stride` accept the evaluation grid to inventory. On TIME
revision `83e3d0b3`, the default size filters retain five hourly and seven daily
source configurations; no 15-minute source remains because every one contains
at least one series longer than 10,000 dates.

For each source dataset/frequency pair, files with identical complete timestamp
indices are merged as independent univariate series. Non-aligned indices become
separate `*_partNN` datasets. Irregular groups, groups shorter than the largest
requested `L+H`, and non-finite target columns are skipped and recorded. The
outputs live under `datasets/time/<dataset>/` as one wide CSV and a portable
`config.json` directly loadable by this evaluator. `datasets/time/catalog.json`
records provenance, skips, and aggregate inventory. Every dataset config keeps
the number of timestamp samples, series, scalar values, and eligible query-date
and series-window counts for each requested setting at stride 1 and at the
configured evaluation stride.

## One evaluation

Run commands from the repository root with `PYTHONPATH=src` in the
project-managed `uv` environment:

```bash
PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/electricity data.name=electricity \
  task.lags=168 task.horizon=24 \
  model.name=chronos2 model.weights_path=weights/chronos2 \
  data.covariate_mode=none \
  preprocessing.instance_normalize=true \
  evaluation.remove_constant=false evaluation.stride=512
```

Identity-covariate TabPFN-TS:

```bash
PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/electricity data.name=electricity \
  task.lags=168 task.horizon=24 model.name=tabpfn_ts \
  model.weights_path=weights/tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt \
  data.covariate_mode=identity
```

Prepared known covariates, with TabPFN time embeddings removed:

```bash
PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/conso data.name=conso \
  'data.covariate_paths=[datasets/electricity,datasets/solar,datasets/traffic]' \
  data.covariate_mode=known model.name=tabpfn_ts \
  model.use_time_features=false task.lags=168 task.horizon=24
```

Chronos-2 expects a local pretrained directory under `weights/chronos2` by
default. Chronos-Bolt expects `weights/chronos-bolt-base`. TS-ICL expects the
official `tsicl-v1.ckpt` file under
`weights/tsicl/`. TiRex-2 expects the official `model-config.yaml` and
`model.ckpt` together under `weights/tirex2/`. TabPFN-TS expects the v2.5
checkpoint under `weights/tabpfnts/`. Explicit `model.weights_path` values take
precedence. The project pins `chronos-forecasting==2.0.1`,
`tsicl==0.2.0`, `tirex-2==0.2.1`, and `tabpfn==6.3.1`. Their shared
dependency range requires Python 3.12
and PyTorch 2.8 or newer but below 2.10.
TS-ICL code and weights use the upstream non-commercial license; TiRex-2 is
Apache-2.0. Confirm those upstream terms for any use beyond this research.

Every external-model file is a project-owned tensor/checkpoint adapter only:
model architecture, preprocessing, and inference remain in the pinned official
packages. No Chronos, TS-ICL, TiRex-2, or TabPFN architecture is copied into
this repository.

The sole foundation aliases are `chronos2`, `chronos_bolt`, `ts_icl`,
`tirex2`, and `tabpfn_ts`, matching TimeTensors and online adaptation exactly.
Covariate mode is configured independently of model identity. Supplying
covariates to an adapter without native support, including Chronos-Bolt, raises
an error rather than ignoring the input.

TS-ICL and TiRex-2 already own their released preprocessing and test-time
inference behavior. Use raw targets at the TSFM boundary:

```bash
PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/electricity data.name=electricity \
  task.lags=504 task.horizon=168 model.name=ts_icl \
  model.weights_path=weights/tsicl/tsicl-v1.ckpt \
  preprocessing.instance_normalize=false data.covariate_mode=none

PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/electricity data.name=electricity \
  task.lags=504 task.horizon=168 model.name=tirex2 \
  model.weights_path=weights/tirex2 \
  preprocessing.instance_normalize=false data.covariate_mode=none
```

## Cluster workflows

The numbered root launchers are the user-facing submission interface:

1. `01_univariate.slurm` compares persistence, expected, repeat, the weekly
   aligned lookback baseline, and Chronos-2. Lookback settings without enough
   history for the required whole-week offset are skipped.
2. `02_controls.slurm` crosses the models with instance normalization and
   constant-window removal.
3. `03_covariates.slurm` compares univariate and identity-covariate inference;
   set `COVARIATE_MODES_OVERRIDE=known` with prepared covariate paths for other
   covariate experiments.
4. `04_foundation_models.slurm` compares the official Chronos-2,
   Chronos-Bolt, TS-ICL, TiRex-2, and TabPFN-TS inference pipelines on the
   univariate benchmark. Its `full` profile
   is Electricity, Traffic, Solar, Weather, and Exchange Rate at `336:48` and
   `504:168`; its report uses Chronos-2 as the default fixed reference.

Each front is one sequential, resumable allocation. `STAGES=evaluate,report`
is the default; either stage may be selected for recovery. `EXPERIMENT_MODE`
provides:

- `test`: Electricity at `504:168`, seed 1;
- `full`: Electricity, Traffic, Solar, and Exchange Rate at `168:24`, `336:48`,
  and `504:168`;
- `ultra`: additional datasets and the shared `336:48`/`504:168` settings.

The list above describes the first three fronts. The foundation-model front
uses Electricity `504:168` for `test`, the five-dataset/two-setting grid above
for `full`, and adds ETTh1 plus `168:24` for `ultra`.

Use whitespace-separated `DATASETS_OVERRIDE`, `SETTINGS_OVERRIDE`,
`MODELS_OVERRIDE`, and `SEEDS_OVERRIDE` values to narrow a submission. The
weekly lookback period can be overridden with `LOOKBACK_PERIOD_STEPS`. The
controls front also accepts `INSTANCE_NORMS_OVERRIDE` and
`REMOVE_CONSTANT_OVERRIDE`. The covariate front accepts
`COVARIATE_MODES_OVERRIDE`, `TIME_FEATURES_OVERRIDE`,
`COVARIATE_PATHS_OVERRIDE` (an OmegaConf list), and
`COVARIATE_COLS_OVERRIDE`.

```bash
EXPERIMENT_MODE=test sbatch 01_univariate.slurm
EXPERIMENT_MODE=full sbatch 01_univariate.slurm
EXPERIMENT_MODE=full sbatch 02_controls.slurm
EXPERIMENT_MODE=full sbatch 03_covariates.slurm
EXPERIMENT_MODE=test sbatch 04_foundation_models.slurm
EXPERIMENT_MODE=full sbatch 04_foundation_models.slurm
```

## Artifacts

The independently launched workflow roots are `outputs/univariate`,
`outputs/controls`, `outputs/covariates`, and `outputs/foundation_models`. Their
common ordered identity is

```text
dataset/L_H/backbone/covariate_mode/normalization/constant_policy/time_features/run_n/
```

Every model config owns one folder. Batch size, metric/window export controls,
data split and stride, model inference options, and explicit covariate bindings
are pipeline configs in `run_n/manifest.json`; device and scheduler placement
are runtime configs. One seed fixes all stochastic model/inference behavior in
that repetition. Each completed seed contains `summary.json`,
`per_user_metrics.csv`, `horizon_metrics.csv`, optional `window_metrics.csv`,
and `resolved_config.json`.

The manifest contract is `schema_version: 1` with status `not_run`, `running`,
`interrupted`, or `completed`. Resubmission with the default
`RUN_CONFLICT_POLICY=overwrite_exact` skips an identical completed run, resumes
an identical interruption, and creates the next `run_n` for changed pipeline
configs. `overwrite_path` and `new` are explicit alternatives. TSFM launches
one seed per run, so that seed is recorded in the manifest and artifacts remain
directly under `run_n/`; a `seed_n/` leaf is reserved for a future multi-seed
launcher.

Run identity contains only the manifest schema, ordered identity/model configs,
pipeline and experiment parameters, and seeds. Source files, Slurm fronts,
datasets, weights, logs, outputs, and directories are never fingerprinted or hashed.
Plain provenance paths may be recorded but do not affect reuse. Code and data
changes are manual rerun decisions; use `RUN_CONFLICT_POLICY=new` for another
repeat with unchanged parameters. Change `schema_version` only for a deliberate
global artifact-contract break.

Reports read only completed current manifests and are written to
`outputs/reports/<family>/<mode>/`. They support
`TABLE_CONFIG_POLICY=distinct|latest|average`,
`TABLE_REPEAT_POLICY=selected|latest|distinct|average`, explicit
`TABLE_PIPELINE_CONFIGS`, and purpose filters. An explicit pipeline filter must
match even when only one run exists and is the way to choose a pipeline
configuration. Nested pipeline and experiment fields, including embedded
upstream scientific dependencies, use dotted filter keys and participate in
distinct labels. `SELECTED_RUNS.txt` records only automatic or pinned exact
repeats. `report_manifest.json` records requested filters and obtained input
manifests.
`EXPERIMENT_MODE` only selects identity paths and is never part of their path or
computation signature.

The report also writes Chronos-2 strict paired-window win rates against the
best aggregate baseline for every error metric, a plot index, and PDF/PNG
per-user scatter, window-histogram, and
horizon-error plots for every dataset and setting. `TABLE_METRIC` selects the
metric for the marginal CSV/LaTeX tables, including total, amortized per-user,
or amortized per-window inference time. `TABLE_REFERENCE_MODEL` defaults to
`best_baseline` and may name a fixed model such as `repeat`. Marginal tables
average complete configurations equally, once by dataset across settings and
once by setting across datasets.

The artifact-only notebook `src/visualization/tsfm_evaluation_analysis.ipynb` reads the
reports and three lightweight metric CSVs. It can inspect every dataset,
setting, model, metric, win rate, and plot locally or in Colab without loading
a forecasting model.

The evaluator records ready artifacts while its process is still running. They
become `completed` immediately after that `srun` returns successfully. A later
evaluation or report failure preserves completed runs and interrupts only
unfinished work. Once completed, the manifest is authoritative and later reuse
does not hash or revalidate synchronized files.

The schema version, statuses, collision policies, purpose filters, run
selection policies, and report-manifest rules are the thesis-wide experiment
contract. The four workflow roots, ordered identity fields, one-seed layout,
and required TSFM summary/metric/config files are specific to this project.

Runtime and Slurm logs belong in `logs/`. Dataset and model payloads remain in
the ignored `datasets/` and `weights/` directories.

## Synchronizing DGX and Selena

On each machine, store the uppercase NNI beside the existing shared proxy
credentials, outside the project repository:

```bash
mkdir -p "$HOME/codes/.secrets"
printf '%s\n' 'YOUR_NNI' > "$HOME/codes/.secrets/nni"
chmod 600 "$HOME/codes/.secrets/nni"
```

Both synchronization scripts read this file and convert the NNI to lowercase
for SSH usernames and home-directory paths. Because `codes/.secrets/` is
outside `codes/tsfm/`, the NNI is never copied or committed with the project.

After pulling code updates on DGX, synchronize the execution copy on Selena:

```bash
bash sync_code_to_selena.sh
```

The transfer makes Selena's code match DGX while preserving Selena's `.venv`,
`.secrets`, datasets, weights, outputs, and logs. Git metadata is not
transferred; Selena is an execution copy and does not need Git operations.

After Selena jobs finish, transfer their lightweight results back to DGX:

```bash
bash sync_results_to_dgx.sh
```

This copies new or changed files from `outputs/` and `logs/` without deleting
anything already present on DGX.

## Publishing terminal Slurm artifacts

Slurm jobs never submit a publisher or run Git commands. After any job reaches
a terminal state, including failure, cancellation, or timeout, run the manual
publisher from that project's Git root:

```bash
bash publish_job.sh <job-id>
```

The script first verifies `main`, sources `$HOME/codes/proxy.sh`, and runs
`git pull --ff-only origin main`. With a job ID, it selects only the exact
`logs/*_<job-id>.out`/`.err` pair. It force-adds only those paths while excluding
`*.pt`, `*.npy`, and `*.cbm`, commits them, and pushes `origin main`. A
non-fast-forward pull stops without creating a merge commit, and the script
never creates a pull request. Existing unrelated staged paths are excluded from
the commit.

Omit the job ID to force-add, commit, and push the complete `logs/` and
lightweight `outputs/` trees:

```bash
bash publish_job.sh
```

`PROXY_SCRIPT_PATH` overrides the default `$HOME/codes/proxy.sh`. The publisher
sources that script once for both the pull and push and leaves the shell's
existing GitHub credential and askpass context untouched.

## Lightweight checks

```bash
PYTHONPATH=src python src/tests/test_dataset.py
PYTHONPATH=src python src/tests/test_metrics.py
PYTHONPATH=src python src/tests/test_model_adapters.py
PYTHONPATH=src python src/tests/test_prepare_time_csv.py
PYTHONPATH=src python src/tests/test_repeat_smoke.py
PYTHONPATH=src python src/tests/test_slurm_workflow.py
```

## LaTeX documents

`latex/experiment_guideline.tex` defines the inference-only setting, model and
covariate families, controls, metrics, seed policy, workflows, and artifacts.
`latex/executive_summary.tex` records only completed current-contract results;
it currently notes that no remote model evaluation has been synchronized.
Their PDFs are kept beside the sources.

## Maintenance workflow

Every project change is recorded in `PENDING_UPDATES.md` with its scope,
affected contracts, focused checks already completed, deferred integration
coverage, documentation impact, and rerun requirements. Routine edits use only
the smallest relevant smoke check. Periodic maintenance verifies pending entries
against the implementation, runs complementary generic lightweight smoke tests,
reconciles this README and the project LaTeX documents, and renders affected
PDFs before resolving the entries.
