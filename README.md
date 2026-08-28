# TSFM Evaluation

This repository provides deterministic, inference-only evaluation of frozen
Chronos-2, Chronos-Bolt, Chronos-T5, and TS-ICL. The model
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
  pipeline/         cadence-aware profiles, run identity, and manifest orchestration
  results/          artifact aggregation, comparisons, and tables
  visualization/    report plots and analysis notebook
  scripts/          evaluation, reporting, and data-preparation fronts
```

The five adapter files remain byte-identical to the shared evaluation surface
in TimeTensors and online adaptation. TIME preparation retains the shared
conversion contract while TSFM inventories its cadence-specific forecast
ranges. The external adapters contain model-specific inference behavior;
evaluation controls only which dates and covariates are accessible. Reporting
and plotting remain downstream of the scientific evaluation path.

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

The primary grid defines comparable short, mid, and long forecast ranges in
dataset time steps. Hourly data use `168:24`, `336:48`, and `504:168`; daily
data use `7:1`, `14:2`, and `30:7`; 15-minute data use `96:4`, `192:8`, and
`672:96`. Electricity, Traffic, Solar, and Weather are hourly, while Exchange
Rate is daily; TIME datasets take their cadence from `catalog.json`.
Evaluation uses the entire eligible
timeline—there is no train/validation/test split—and deterministically retains
one query date every 512 steps by default. `EVAL_STRIDE` changes this speed and
coverage trade-off.

The `lookback` baseline is weekly seasonal persistence. With `P` observations
per week, it selects `k=ceil(H/P)` and predicts from
`X[L-kP:L-kP+H]`. This is the most recent weekly aligned `H`-sized history
window that remains inside the lookback. For example, `L=336`, `H=24`, and
hourly `P=168` yields `X[168:192]`. `P` is inferred from dated inputs (168 for
hourly data and 7 for daily data); set `model.lookback_period` for a direct run
or `LOOKBACK_PERIOD_STEPS` for a Slurm workflow to override it. The weekly
period is 672 observations for regular 15-minute data. Inputs without
real dates require the explicit override.

## Data and covariates

`data.path` identifies a wide CSV or a directory containing one CSV. TSFM does
not consume TimeTensors `.pt` caches. CSV targets default to every non-date,
non-covariate column.
Three covariate modes are available:

- `none`: univariate inference;
- `identity`: the complete target window is supplied as a known covariate;
- `known`: use `data.covariate_cols` from the target CSV and/or user-aligned
  wide panels in `data.covariate_paths`.

Each external covariate panel must share the target timeline and either share
its user column names, have the same number/order of users, or contain one
global column. Paths are evaluation inputs; this project does not create them.

The loader discovers `config.json` beside the selected CSV.
Portable fields live at the JSON top level and project-only overrides under
`tsfm_evaluation`. Scoped settings override portable settings, explicit Hydra
values override both. For `drop_users`, null or omission inherits, `[]` keeps
every CSV user, and a nonempty override replaces the preceding default.
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
default eligibility settings are the three cadence-specific short/mid/long
ranges above, with stride 512; `--settings` applies an explicit common grid and
`--stride` changes the inventory stride. On TIME
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

Identity-covariate TS-ICL:

```bash
PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/electricity data.name=electricity \
  task.lags=168 task.horizon=24 model.name=ts_icl \
  model.weights_path=weights/tsicl/tsicl-v1.ckpt \
  data.covariate_mode=identity
```

Prepared known covariates with TS-ICL:

```bash
PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/conso data.name=conso \
  'data.covariate_paths=[datasets/electricity,datasets/solar,datasets/traffic]' \
  data.covariate_mode=known model.name=ts_icl \
  model.weights_path=weights/tsicl/tsicl-v1.ckpt task.lags=168 task.horizon=24
```

Chronos-2 expects a local pretrained directory under `weights/chronos2` by
default. Chronos-Bolt expects `weights/chronos-bolt-base`, Chronos-T5 expects
`weights/chronos-t5-base`, and TS-ICL expects the
official `tsicl-v1.ckpt` file under
`weights/tsicl/`. Explicit `model.weights_path` values take precedence. The
project requires `chronos-forecasting>=2.3.1`, `tsicl>=0.2.1`, Python 3.12,
and PyTorch 2.5.1. TS-ICL code and weights use the upstream non-commercial
license; confirm those upstream terms for any use beyond this research.

Every external-model file is a project-owned tensor/checkpoint adapter only:
model architecture, preprocessing, and inference remain in the official
packages. No Chronos or TS-ICL architecture is copied into this repository.
The former TabPFN adapter source is retained but is unregistered and has no
runtime dependency or launcher; the retired TiREx-2 adapter is archived.

The sole foundation aliases are `chronos2`, `chronos_bolt`, `chronos_t5`, and
`ts_icl`, matching TimeTensors and online adaptation exactly.
Covariate mode is configured independently of model identity. Supplying
covariates to an adapter without native support, including Chronos-Bolt, raises
an error rather than ignoring the input.

TS-ICL already owns its released preprocessing and test-time inference
behavior. Use raw targets at the TSFM boundary:

```bash
PYTHONPATH=src uv run python -m scripts.evaluate \
  data.path=datasets/electricity data.name=electricity \
  task.lags=504 task.horizon=168 model.name=ts_icl \
  model.weights_path=weights/tsicl/tsicl-v1.ckpt \
  preprocessing.instance_normalize=false data.covariate_mode=none
```

## Cluster workflows

The numbered launchers under `slurm/dgx/main/` are the DGX submission
interface; matching Selena fronts live under `slurm/selena/main/`:

1. `01_univariate.slurm` compares persistence, expected, repeat, the weekly
   aligned lookback baseline, and Chronos-2. Lookback settings without enough
   history for the required whole-week offset are skipped.
2. `02_controls.slurm` crosses the models with instance normalization and
   constant-window removal.
3. `03_covariates.slurm` compares univariate and identity-covariate inference;
   set `COVARIATE_MODES_OVERRIDE=known` with prepared covariate paths for other
   covariate experiments.
4. `04_foundation_models.slurm` compares the official Chronos-2,
   Chronos-Bolt, Chronos-T5, and TS-ICL inference pipelines on the univariate
   benchmark. Its `full` profile uses the cadence-specific mid
   and long ranges on Electricity, Traffic, Solar, Weather, Exchange Rate, and
   every eligible prepared TIME dataset; its report uses Chronos-2 as the
   default fixed reference.

The four DGX fronts above use partition `h100`. Selena exposes matching
`01_univariate_selena.slurm`, `02_controls_selena.slurm`,
`03_covariates_selena.slurm`, and `04_foundation_models_selena.slurm` fronts.
They run the identical implementations and experiment paths on partition `an`
with QoS `an_preemptable`, an exclusive allocation without disabling the
cluster's requeue behavior, WCKey
`P12CU:DATASCIENCE`, Selena-specific job names, and `selena_`-prefixed launch
IDs. Their Slurm streams and artifacts go under
`/scratch/users/<lowercase-nni>/codes/tsfm/logs_selena/` and
`outputs_selena/`. The relative artifact identity stays the same as DGX, but
synchronized Selena work cannot collide with `logs/` or `outputs/`. The
shared `LOGS_ROOT` and `OUTPUTS_ROOT` variables remain explicitly overridable;
custom Slurm stream paths require matching `sbatch --output` and `--error`
overrides. Use the Selena fronts only through the documented overflow
workflow.

Each front is one sequential, resumable allocation. `STAGES=evaluate,report`
is the default; either stage may be selected for recovery. `EXPERIMENT_MODE`
provides:

- `test`: Electricity at `504:168`, seed 1;
- `full`: Electricity, Traffic, Solar, Exchange Rate, and every dataset in
  `datasets/time/catalog.json`, each at its cadence-specific short, mid, and
  long settings;
- `ultra`: the full profile plus Weather and ETTh1.

The list above describes the first three fronts. The foundation-model front
uses Electricity `504:168` for `test`, the five primary datasets plus TIME at
cadence-specific mid and long settings for `full`, and all three ranges plus
ETTh1 for `ultra`. TIME tasks without at least `L+H` timestamps are omitted.

Use whitespace-separated `DATASETS_OVERRIDE`, `SETTINGS_OVERRIDE`,
`MODELS_OVERRIDE`, and `SEEDS_OVERRIDE` values to narrow a submission. The
dataset and setting overrides replace automatic selection; an explicit dataset
override therefore does not require a TIME catalog. Without an override,
`full` and `ultra` require `datasets/time/catalog.json` and automatically
include its eligible datasets. `DATA_ROOT`, when set, must be the datasets root
that contains both ordinary dataset folders and `time/catalog.json`. The
weekly lookback period can be overridden with `LOOKBACK_PERIOD_STEPS`. The
controls front also accepts `INSTANCE_NORMS_OVERRIDE` and
`REMOVE_CONSTANT_OVERRIDE`. The covariate front accepts
`COVARIATE_MODES_OVERRIDE`,
`COVARIATE_PATHS_OVERRIDE` (an OmegaConf list), and
`COVARIATE_COLS_OVERRIDE`.

For every selected dataset and checkpoint, launchers search the explicit
`DATA_ROOT` or `WEIGHTS_ROOT` when provided, then the project-local directory,
the immediate project parent, and the nested-workspace shared parent. The
first candidate containing that requested resource is used, so an empty or
partially populated project directory does not hide shared cluster resources.

```bash
EXPERIMENT_MODE=test sbatch slurm/dgx/main/01_univariate.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/01_univariate.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/02_controls.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/03_covariates.slurm
EXPERIMENT_MODE=test sbatch slurm/dgx/main/04_foundation_models.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/04_foundation_models.slurm
```

The equivalent Selena submissions replace the filename with its `_selena`
variant, for example:

```bash
EXPERIMENT_MODE=test sbatch slurm/selena/main/01_univariate_selena.slurm
EXPERIMENT_MODE=full sbatch slurm/selena/main/01_univariate_selena.slurm
EXPERIMENT_MODE=full sbatch slurm/selena/main/02_controls_selena.slurm
EXPERIMENT_MODE=full sbatch slurm/selena/main/03_covariates_selena.slurm
EXPERIMENT_MODE=test sbatch slurm/selena/main/04_foundation_models_selena.slurm
EXPERIMENT_MODE=full sbatch slurm/selena/main/04_foundation_models_selena.slurm
```

## Artifacts

The independently launched DGX workflow roots are `outputs/univariate`,
`outputs/controls`, `outputs/covariates`, and `outputs/foundation_models`.
Selena uses the same relative roots under `outputs_selena/`. Their common
ordered identity is

```text
dataset/L_H/backbone/covariate_mode/normalization/constant_policy/run_n/
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
`outputs/reports/<family>/<mode>/` on DGX or
`outputs_selena/reports/<family>/<mode>/` on Selena. They support
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

An `average` policy first aligns and averages the selected summaries and their
per-window, per-user, and per-horizon metric artifacts. Comparisons, Chronos-2
win rates, marginal tables, and plots are then computed from those averages,
so no downstream result silently falls back to one selected repeat.

The report also writes Chronos-2 strict paired-window win rates against the
best aggregate baseline for every error metric, a plot index, and PDF/PNG
per-user scatter, window-histogram, and
horizon-error plots for every dataset and setting. `TABLE_METRIC` selects the
metric for the marginal CSV/LaTeX tables, including total, amortized per-user,
or amortized per-window inference time. `TABLE_REFERENCE_MODEL` defaults to
`best_baseline` and may name a fixed model such as `repeat`. Marginal tables
average complete configurations equally: by dataset across cadence ranges, by
cadence-independent short/mid/long range across datasets, and by literal `L:H`
setting for auditability.

The artifact-only notebook `src/visualization/tsfm_evaluation_analysis.ipynb` reads the
reports and lightweight metric CSVs. It can inspect every dataset,
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

DGX runtime and Slurm logs belong in `logs/`; Selena Slurm streams belong in
`logs_selena/`. Dataset and model payloads remain in the ignored `datasets/`
and `weights/` directories.

## Synchronizing DGX and Selena

On each machine, keep the NNI outside the project repository in a one-line
protected file. Synchronization and Selena runtime helpers strip whitespace
and lowercase it for the SSH account and scratch path:

```bash
mkdir -p "$HOME/codes/.secrets"
printf '%s\n' 'YOUR_NNI' > "$HOME/codes/.secrets/nni"
chmod 600 "$HOME/codes/.secrets/nni"
```

Both synchronization scripts derive the project directory name from their own
checkout and read only this NNI file. Because `codes/.secrets/` is outside the
project, it is never copied or committed with the checkout.

After pulling code updates on DGX, synchronize the execution copy on Selena:

```bash
bash sync_code_to_selena.sh
```

The transfer makes Selena's implementation code match DGX, preserves Selena's
`.venv`, `.secrets`, `pyproject.toml`, `uv.lock`, datasets, weights, and DGX-
named outputs/logs, and creates the scratch `outputs_selena/` and
`logs_selena/` directories. Git metadata and dependency manifests are not
transferred; Selena keeps its user-managed environment and dependency
resolution.

After Selena jobs finish, run the result helper from the TSFM checkout on DGX.
DGX initiates the SSH connection and pulls the lightweight Selena results, so
Selena never needs outbound SSH or SCP access:

```bash
bash sync_results_to_dgx.sh
bash sync_results_to_dgx.sh --size detailed
bash sync_results_to_dgx.sh --size full
```

The default lightweight tier transfers logs, manifests, aggregate metrics,
reports, and ordinary plots while omitting row-level window/user-date/sample
tables and per-run diagnostic plots. `detailed` adds those text and plotting
artifacts; `full` also retrieves cluster-only `.pt`, `.npy`, `.cbm`, and other
binary payloads for recovery or deep debugging. `--job-id ID` restricts the
logs to the exact standard job pair. Transfers never delete DGX files. Do not
run this helper on Selena; returned artifacts remain in the `_selena` trees.

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
lightweight `outputs/` trees plus paired `logs_selena/` and lightweight
`outputs_selena/` trees when present:

```bash
bash publish_job.sh
bash publish_job.sh --size detailed
```

Lightweight is the default and applies the same row-level/per-run diagnostic
omissions as lightweight sync. `detailed` publishes all non-binary diagnostics.
The `*.pt`, `*.npy`, and `*.cbm` exclusions apply to both tiers, so the sync
helper is the only route for those cluster-only payloads. A partial Selena
namespace fails closed; job-ID mode remains scoped to the exact log pair.

Before staging, each selected non-excluded file larger than 100,000,000 bytes
is replaced for publication by `<original>.sample.txt`. Text samples contain
source metadata and the first 10% of content, capped at 10,000,000 bytes;
binary samples contain metadata only. The header retains the first UTC time
when the associated file became stale on Git because of its size. The original
is excluded literally from
both staging and commit selection. `PUBLISH_MAX_FILE_BYTES` and
`PUBLISH_SAMPLE_MAX_BYTES` override the positive byte limits, with the sample
limit required to remain smaller.

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
`latex/executive_summary.tex` separates analyzed historical artifacts from
current-contract evidence. The synchronized tensor-input runs are documented
as pre-contract and are not reusable under the CSV-only replacement contract.
Their PDFs are kept beside the sources.

## Maintenance workflow

Every project change is recorded in `PENDING_UPDATES.md` with its scope,
affected contracts, focused checks already completed, deferred integration
coverage, documentation impact, and rerun requirements. Routine edits use only
the smallest relevant smoke check. Brief daily triage compares stored
fingerprints and updates the queue only for new source, artifact, or external
state; unchanged blockers are carried forward. Broad weekly maintenance verifies
changed entries against the implementation, runs complementary lightweight
integration checks, reconciles this README and the project LaTeX documents, and
renders affected PDFs before resolving entries.
