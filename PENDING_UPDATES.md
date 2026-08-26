# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-26: Moved the four DGX and four Selena submission fronts from the
  project root into `slurm/<cluster>/main/` without changing their resources,
  experiment families, artifact roots, or project-root resolution contract.
  Submissions must still run from the project root so `SLURM_SUBMIT_DIR`
  resolves the repository correctly. The direct Slurm workflow contract
  passed; the prepared runtime lacks pytest, but this test has an equivalent
  direct entry point. README, LaTeX, and cluster handoff command updates are
  deferred to the planned documentation pass. The recursive DGX-to-Selena code
  sync will apply the same hierarchy remotely. No scientific rerun or artifact
  migration is required.

- 2026-08-26: Promoted TSFM's validated Selena transfer/publication behavior to
  the shared standard. The local DGX-initiated pull remains unchanged; the
  unscoped publisher now matches all eight sibling copies and includes paired
  Selena trees under the existing heavy-payload exclusions. Affected contracts:
  publisher regression, README, shared guidance, and cluster handoff. Bash
  syntax passed for all 15 maintained scripts, all five publisher checks and
  the TSFM workflow check passed, and the nine publisher copies plus five
  suffix-result helpers are each byte-identical. No scientific rerun or
  migration is required. The README changed; the guideline's
  all-log/lightweight-output wording remains accurate, so LaTeX/PDF files are
  unchanged. Deferred integration remains one real pull and unscoped
  publication after a Selena test job.

- 2026-08-26: Inverted TSFM result synchronization to accommodate Selena's
  blocked outbound SSH/SCP. `sync_results_to_dgx.sh` now runs on DGX and pulls
  Selena's `outputs_selena/` and `logs_selena/` into the same local names using
  non-deleting rsync; Selena initiates no connection. Affected contracts:
  result helper, focused workflow regression, README, shared cluster guidance,
  and cluster handoff. Git Bash syntax and the direct workflow contract passed.
  No scientific output, identity, or rerun requirement changed. Deferred
  integration: exercise the first real DGX-initiated pull after a Selena test
  job completes.

- 2026-08-26: Isolated every Selena workflow from DGX artifact trees. The four
  existing Selena fronts now write Slurm streams to `logs_selena/` and all
  manifests, results, and reports to `outputs_selena/`; the shared workflow
  exposes overridable `LOGS_ROOT` and `OUTPUTS_ROOT`, defaulting to `logs/` and
  `outputs/` for DGX. The code mirror protects both Selena payload trees, and
  result sync returns them
  into the same named DGX directories without merging or deletion. Affected
  contracts: four Selena fronts, shared workflow, sync pair, ignored
  placeholders, focused regression, README/local/shared guidance, cluster
  handoff, and experiment guideline source/PDF. Git Bash syntax passed across
  all 52 affected TSFM/Online shell files, the TSFM workflow check passed, two clean
  LaTeX passes produced four pages, and all pages were visually inspected with
  no clipping or overlap. DGX behavior and scientific identity are unchanged;
  no existing artifact, migration, or rerun is affected. Deferred integration:
  submit one Selena test front and exercise both sync directions on the real
  clusters.

- 2026-08-26: Removed library-version declarations from the experiment
  guideline and made DGX-to-Selena code synchronization preserve Selena's
  local `pyproject.toml` and `uv.lock` alongside its environment and runtime
  payloads. Affected contracts: guideline source/PDF, shared Selena convention,
  code-sync helper, README, and focused workflow regression. Focused checks:
  active-guideline version scan, Bash syntax, sync exclusion assertions, clean
  LaTeX compilation, and rendered-page inspection. No experiment result or
  scientific contract is invalidated; no inference rerun is required.

- 2026-08-26: Made the user-selected four-model foundation profile explicit:
  Chronos-2, Chronos-Bolt, TS-ICL, and TabPFN-TS launch by default, while the
  TiRex-2 adapter and canonical alias remain implemented but commented out of
  the launcher. Replaced root-wide storage selection with Adaptation-style
  per-resource lookup through an explicit override, project-local storage, the
  immediate project parent used by flat cluster checkouts, and the nested
  workspace shared parent. Direct adapter checkpoint discovery now follows the
  same order. Affected contracts: shared Slurm resource routing, the foundation
  default, five aligned adapter copies, workflow regression, README, guidance,
  and experiment-guideline source. Checks passed: the focused Slurm workflow
  test, Git Bash syntax for both foundation fronts and their implementations, a
  flat `codes/tsfm` fixture resolving datasets and checkpoints from
  `codes/{datasets,weights}`, and SHA-256 parity for all five adapter copies
  across TSFM, TimeTensors, and Online Adaptation. Deferred integration: run
  the four-model test profile against real shared cluster resources; no
  checkpoint was loaded locally. Maintenance rebuilt the guideline PDF and
  visually inspected all four pages. No synchronized result exists, so no rerun is
  invalidated; future foundation runs use the four-model profile.

- 2026-08-26: Added Selena overflow variants for all four TSFM Slurm fronts.
  They reuse the exact DGX workflow implementations while selecting partition
  `an`, exclusive non-requeued allocations, WCKey `P12CU:DATASCIENCE`, distinct
  job names, and `selena_`-prefixed launch IDs. Affected contracts: eight root
  fronts, workflow regression, README, project guidance, and cluster handoff.
  Focused checks: Git Bash syntax passed for all eight fronts, the focused
  static launcher regression passed, direct inspection confirmed every Selena
  scheduling and launch-ID directive, and `git diff --check` passed. Deferred
  integration: submit one Selena test profile and confirm
  scheduling, environment, checkpoint, manifest, log, and result
  synchronization behavior. No existing result is invalidated; Selena runs
  share the current scientific and artifact contract.

- 2026-08-25: Added explicit DGX-to-Selena code synchronization and
  Selena-to-DGX result synchronization scripts. Code synchronization mirrors
  the maintained project while preserving Selena-local environments, secrets,
  datasets, weights, outputs, and logs; result synchronization copies outputs
  and logs without remote deletion. The user-facing scripts live at the
  project root beside the publisher, read the ignored uppercase NNI from
  the first line of shared `$HOME/codes/.secrets/proxy.credentials`, and
  lowercase it only for SSH usernames and home paths without reading the
  password.
  Affected files/contracts: two operational
  scripts and README commands. Focused checks: Git Bash syntax passed for both
  scripts, static inspection confirmed the requested endpoints, protected
  directories, first-line NNI lookup from the shared proxy credentials and
  lowercase conversion without password access, no hardcoded NNI remains, the
  focused Slurm workflow check passed, and `git diff --check` passed. Deferred
  integration: execute each
  direction once between the real hosts. README is updated; no LaTeX change or
  scientific rerun is required.

- 2026-08-20: Inventoried the official TIME Arrow release at revision
  `83e3d0b3be28d11c7182bffcc1892d19b36c4da1` and generated ignored exploratory
  CSV/JSON/PNG artifacts under `outputs/time_inventory/`. The inventory expands
  multivariate targets into the independent univariate series relevant to this
  evaluator and records users, total user-date observations, per-user lengths,
  and exact start/length alignment groups for all 50 dataset-frequency tasks.
  Direct Arrow scanning completed for all 50 files; the exported aggregates
  reconcile to 39 named datasets, 6,633 univariate series, and 54,990,783
  user-date observations, and the histogram image was visually inspected.
  The selected preparation profile now targets 15-minute, hourly, and daily
  configurations with at most 500 series and 10,000 dates per series; the
  current release retains five hourly and seven daily sources and no 15-minute
  source. Deferred integration: prepare and inspect those selected sources and
  reconcile the final benchmark grid in LaTeX. Required runs: none until the
  prepared panels are approved for evaluation.

- 2026-08-20: Added a selective TIME-ProcessedCSV preparation utility for
  expanding the univariate benchmark. It resolves and caches an exact Hugging
  Face dataset revision, defaults to 15-minute/hourly/daily sources capped at
  500 series and 10,000 dates per series, supports dataset and frequency
  subsets or an existing local snapshot, groups files only when
  their full timestamp indices align, splits non-aligned groups, removes
  incomplete series, and skips irregular or too-short groups. Each prepared
  dataset owns a directly loadable wide CSV/config pair; its config and the
  root catalog preserve source provenance, timestamp/series/value counts, and
  eligible query and user-window counts for every requested L/H at stride 1
  and the evaluation stride. Affected files/contracts: new preparation script
  and focused synthetic test, explicit Hugging Face Hub dependency, README,
  and local project guidance. The focused `test_prepare_time_csv.py` check
  passed in the shared thesis runtime, covering aligned merge, non-aligned
  split, frequency and task-size filtering, non-finite-series removal, minimum
  length, count formulas, catalog serialization, and evaluator loading. Python
  compilation and the preparation CLI help/import check also passed. Deferred
  integration: refresh the user-managed environment, run the selected subset
  download, inspect the resulting catalog, and reconcile/render the experiment guideline
  before the expanded benchmark is submitted. Required runs: no existing
  result is invalidated; each newly selected TIME dataset requires new model
  evaluations.

- 2026-08-20: Added thin, inference-only adapters for the official TS-ICL 0.2.0
  and TiRex-2 0.2.1 packages. The adapters preserve the upstream model,
  checkpoint loader, native normalization, quantile head, covariate semantics,
  and TiRex-2 checkpoint test-time augmentation while translating only TSFM
  tensors and selecting the median point forecast. Added the resumable
  `04_foundation_models.slurm` workflow: test runs Electricity `504:168`; full
  runs Chronos-2, TS-ICL, and TiRex-2 on Electricity, Traffic, Solar, Weather,
  and Exchange Rate at `336:48` and `504:168`, with Chronos-2 as report
  reference. Affected files/contracts: two model adapters, factory, Hydra and
  manifest model fields, Python/PyTorch/package requirements, shared weight and
  batch routing, the new root/implementation Slurm pair, focused tests, README,
  project guidance, and cluster handoff. Checks passed in the shared thesis
  runtime: `test_model_adapters.py` and `test_slurm_workflow.py`; Python
  compilation of the changed model/evaluator/test modules; and Git Bash syntax
  for the new front plus changed/new implementation shells. No foundation
  checkpoint was loaded and no model inference ran locally. Deferred
  integration: refresh the user-managed project environment, place official
  checkpoints in the documented paths, run and inspect the test profile on an
  H100, then reconcile and render `latex/experiment_guideline.tex`; the
  executive summary remains unchanged until results exist. Required runs: new
  test then full foundation-model workflows. Existing unsynchronized workflows
  need no migration or schema bump.

- 2026-08-19: Replaced the two absolute-index lookback variants with one weekly
  aligned baseline. It infers observations per week from dated inputs, supports
  an explicit period override, and moves back the minimum whole number of weeks
  required to keep the complete horizon source inside the lookback. The
  univariate model axis, factory, manifest pipeline field, Slurm validity rule,
  tests, README, and cluster handoff now use only `lookback`; `lookback0` and
  `lookback168` are rejected. Checks passed: Python compilation, focused
  cadence/baseline/factory regression, Slurm workflow contract, Git Bash syntax,
  and the repeat end-to-end evaluator/report smoke. No foundation-model
  inference ran locally. Deferred maintenance: reconcile and render the project
  LaTeX documents. Required run: the unsynchronized test and full univariate
  workflows already pending must use this new baseline contract.

- 2026-08-18: Expanded the inference-only reference evaluation for the ICLR
  analysis. The univariate front now runs Chronos-2 with the legacy TimeTensor
  persistence, expected, repeat, and weekly aligned lookback baselines
  over the four-dataset/three-setting primary grid and the complete eligible
  timeline at deterministic stride. Each run now writes MSE, MAE, nMSE, nMAE,
  configurable-seasonality MASE, sample dispersion, equal-user mean/user
  dispersion, W10, per-user within-user dispersion, horizon summaries, and
  inference timing. Reports add best-baseline Chronos win rates, per-user
  mean-versus-std scatters, window histograms, horizon curves, and chosen-metric
  marginal CSV/LaTeX tables by dataset and by setting, including selectable
  total/per-user/per-window inference timing. A Colab-ready artifact
  notebook reads the synchronized reports and metric CSVs without model
  inference. Affected contracts/files: baseline factory, evaluator artifacts,
  Hydra config, univariate/profile/report Slurm paths, report builder, tests,
  notebook, dependency declaration, README, and cluster handoff. Checks passed:
  Python compilation; focused metric, baseline-adapter, repeat end-to-end,
  best-baseline/win-rate reporting, and Slurm workflow tests; Git Bash syntax
  for both changed shell files; and notebook JSON parsing. No foundation-model
  inference ran locally. Deferred maintenance: reconcile and render both project
  LaTeX documents, then inspect the generated test-profile CSV/PDF/PNG artifacts
  on the cluster. Required runs: first `EXPERIMENT_MODE=test sbatch
  01_univariate.slurm`, then the full univariate ICLR grid; there are no older
  synchronized TSFM results to migrate.

- 2026-08-17: Simplify `publish_job.sh`: a numeric job ID now selects only its
  exact stdout/stderr pair, while an omitted ID stages the `logs/` and
  lightweight `outputs/` parent trees directly. Publisher, focused contract
  test, README, and shared guidance changed. The project publisher contract
  test and Git Bash syntax passed, and all nine copies have matching SHA-256
  hashes. No inference rerun or artifact migration is required. Deferred
  maintenance: reconcile and render the experiment guideline; retain the
  existing real-cluster publisher integration check.

- 2026-08-16: Adopt the thesis-standard `publish_job.sh`: source the proxy and
  fast-forward pull `origin/main` before artifact selection, staging, or commit,
  then publish only the lightweight selected paths. Affected contracts:
  publisher, focused contract test, README, and shared experiment guidance.
  Checks passed: Bash syntax for all nine standard copies, matching SHA-256
  hashes, and the TSFM publisher contract test. No inference rerun or artifact
  migration is required. Deferred maintenance: reconcile
  `latex/experiment_guideline.tex` and exercise one real cluster publish with a
  remote update present.

- 2026-08-12: Synchronize Adaptation's terminal lifecycle: remove automatic
  publisher submission, add the manual root `publish_job.sh`, restrict overall
  manifests to `not_run|running|interrupted|completed`, and allow reporting to
  consume ready artifacts only from its own active launch. Affected
  files/contracts: manifest helper, benchmark runner, report reader, publisher
  files, focused tests, README, and parent experiment guidance. Checks passed:
  10 focused lifecycle/publisher/Slurm/repeat-report tests and Bash syntax for
  the runner and manual publisher. No scientific rerun or artifact migration
  is required. Deferred maintenance: reconcile and render
  `latex/experiment_guideline.tex`; cluster-check one successful and one
  failed/cancelled launch, then run the manual publisher once.
  Maintenance 2026-08-13: direct inspection confirmed the four-state overall
  manifest, seed-only `ready`, same-launch report selection, and final exit
  promotion. Dataset, metric, and model-adapter checks passed in the shared
  thesis runtime without running foundation-model inference. The README was
  already current; the guideline now documents the manual publisher. Two
  pdfLaTeX passes completed without warnings, and all three rendered pages
  passed visual inspection. The previously successful lifecycle, publisher,
  Slurm, and repeat-report checks were not repeated because these three checks
  cover complementary evaluation boundaries. Remaining blocker: observe one
  successful and one failed/cancelled cluster launch, then run
  `publish_job.sh` once.

- 2026-08-13: Complete every successful TSFM evaluation immediately, preserve
  it across later evaluation/report failure, interrupt only unfinished runs,
  and retain per-seed artifact lists. Affected contracts: shared manifest
  helper, benchmark runner, tests, README, and experiment guideline. Checks
  passed: 11 lifecycle tests, publisher and four workflow contracts, Python AST
  parsing, Bash syntax, clean LaTeX compilation, and visual inspection of all
  three PDF pages. No artifact migration, inference rerun, or schema bump is
  required. Remaining cluster work: exercise successful and failed/cancelled
  launches and run the manual publisher once.

Maintenance 2026-08-16: no source, configuration, artifact, documentation, or
cluster-handoff file changed after the previous pass. Direct inspection again
found completed-only reuse, same-launch ready report selection, and the manual
publisher contract. The already successful dataset, metric, model-adapter, and
PDF checks were not repeated because there is no changed integration boundary
and foundation-model inference remains out of scope. Live successful/failed
launch observations and one manual publisher run remain the sole blocker; no
scientific rerun is required.

Maintenance 2026-08-17: direct inspection found no new source, artifact, or
cluster-status change and reconfirmed completed-only reuse and same-launch ready
report selection. The README was current; the experiment guideline was
reconciled with the canonical proxy-first, fast-forward-pull publisher. Bash
syntax passed for all nine byte-identical copies. Three pdfLaTeX passes
completed with a clean log, and all three rendered guideline pages passed
visual inspection. The prior dataset, metric, adapter, and lifecycle checks
were not repeated because those boundaries did not change; foundation-model
inference remained out of scope. Live successful/failed launch observations
and one real publisher run remain the blockers; no inference rerun is required.

Maintenance 2026-08-18: direct inspection confirmed that the shared manifest
helper is byte-identical to the canonical schema-1 copies and that this project
has no upstream selector or synchronized current manifest; the already
successful 13 focused manifest tests therefore close that standalone entry.
The README was current, and the experiment guideline was corrected to describe
exact-log job publication and unscoped lightweight-tree publication. Git Bash
syntax passed for all nine byte-identical publishers. Dataset, metric, adapter,
and repeat tests were not repeated because those boundaries did not change;
foundation-model inference remained out of scope. Three pdfLaTeX passes
completed with a clean log, and all three rendered guideline pages passed
visual inspection. Live successful/failed launch observations and one real
publisher run remain the blockers; no inference rerun
is required.

Maintenance 2026-08-19: direct inspection confirmed the five-metric evaluator,
weekly cadence inference and explicit period override, manifest pipeline field,
five-model univariate axis, current four-dataset/three-setting full grid,
report tables/plots, and absence of synchronized remote results. Complementary
`test_dataset.py` and all 13 `test_experiment_runs.py` manifest tests passed in
the shared thesis runtime. The already-successful metric, adapter, reporting,
repeat end-to-end, Slurm, Bash-syntax, compilation, and notebook checks were not
repeated. The experiment guideline was reconciled with the complete model,
metric, grid, artifact, and reporting contracts; three pdfLaTeX passes produced
a clean log, and all three pages passed visual inspection. The executive
summary was left unchanged because no new evidence exists. Required external
work remains the test then full univariate workflows, their artifact inspection,
live success/failure lifecycle observation, and one real publisher run.

Maintenance 2026-08-20: direct timestamp, source, artifact, and cluster-handoff
inspection found no change after the previous pass and reconfirmed the absence
of synchronized remote evaluation results. The README, guideline, and
no-results executive summary remain current, and the publisher remains
byte-identical across all nine projects at SHA-256
`0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
Dataset, metric, adapter, reporting, repeat, manifest, Slurm, Bash-syntax, and
PDF checks were deliberately skipped because no integration boundary changed;
foundation-model inference remained out of scope. Required external work is
still the test then full univariate workflows, artifact inspection, live
lifecycle observations, and one real publisher run.

Maintenance 2026-08-21: direct inspection confirmed the official thin TS-ICL
0.2.0 and TiRex-2 0.2.1 adapters, raw-boundary/model-native preprocessing,
median point selection, foundation-model profile and checkpoint routing,
selective TIME revision/filter/alignment/count contracts, and the absence of
remote evaluation artifacts. Complementary `test_repeat_smoke.py` and
`test_reporting.py` passed in the shared thesis runtime, covering the unchanged
evaluator-to-report integration without loading a foundation checkpoint. The
already-successful adapter, preparation, Slurm, compilation, metric, dataset,
and manifest checks were not repeated because they directly cover the changed
surfaces or unchanged boundaries. Git Bash accepted all 10 byte-identical
publisher copies at SHA-256
`0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
The README was current; the experiment
guideline was reconciled with the two official models, fourth workflow root,
foundation grid, checkpoints, and selective TIME preparation. Three pdfLaTeX
passes produced a clean three-page PDF, and all pages passed visual inspection.
The stale literal `$outDir` auxiliary/render directory and current transient
LaTeX/render files were removed. The executive summary remains unchanged
because no new result exists. Required
external work remains environment refresh and selected TIME preparation, then
the foundation and univariate test/full workflows, artifact inspection, live
lifecycle observations, and one real publisher run; no existing result is
invalidated.

Maintenance 2026-08-23: direct inspection confirmed the shared nested
selection and deterministic latest-run behavior, and the helper plus focused
test file are byte-identical to the other four maintained copies. The
complementary `src/tests/test_repeat_smoke.py` evaluator-to-report workflow
passed in the shared thesis runtime without foundation-model inference. README
and the already-rendered guideline remain current, and the executive summary
correctly remains unchanged because no remote result exists. The selector
entry is resolved with no rerun; environment refresh, TIME preparation,
cluster workflows, artifact inspection, lifecycle observations, and the
publisher check remain pending.

Maintenance 2026-08-24: direct inspection confirmed the isolated evaluator,
external-model, model-loading, pipeline, results, and visualization ownership,
clean obsolete-path and dependency scans, unchanged remote placeholders, and
current README and LaTeX contracts. Importing the relocated packages plus the
evaluation and report fronts passed in the shared thesis runtime without model
inference. The CLI-help boundary was inapplicable because Hydra is absent from
that runtime, so it was not used as a scientific check. The reorganization and
guidance entries are resolved. Environment refresh, selected TIME preparation,
checkpoint-backed test/full workflows, artifact inspection, lifecycle
observations, and the first real publisher run remain pending.

## 2026-08-24 — Five-model foundation parity and shared TIME preparation

- Behavior and affected contracts: moved TIME preparation into the data owner;
  expanded the foundation workflow to Chronos-2, Chronos-Bolt, TS-ICL,
  TiRex-2, and TabPFN-TS; standardized every adapter constructor on lags, dim,
  and horizon; and pinned the official package versions.
- Focused checks and outcomes: Python compilation, adapter protocols, both
  TIME-preparation cases, Slurm contracts, Git Bash syntax, and TOML parsing
  passed. SHA-256 comparison confirmed that TIME and all five adapter files
  are byte-identical to TimeTensors and online adaptation.
- Deferred integration: the shared runtime has no prepared checkpoint stack,
  so real inference was not run. Environment refresh, TIME materialization,
  and checkpoint-backed test/full workflows remain remote work.
- README/LaTeX and reruns: README documents the five-model shared surface.
  Reconcile and render the experiment guideline during maintenance. No
  completed inference result exists; run the foundation test and full profiles
  under the new current contract.

Maintenance 2026-08-25: direct adapter, evaluator, report, workflow, archived
inventory, README, guideline, summary, and handoff inspection confirmed the
five official-model surface and absence of synchronized evaluation results.
The complementary `src/tests/test_repeat_smoke.py` evaluator-to-report workflow
passed without loading a foundation checkpoint. The README and no-results
summary were already current. The guideline now documents Chronos-Bolt, all
five package/checkpoint contracts, and the five-model foundation front; two
pdfLaTeX passes produced a clean four-page PDF and all pages passed visual
inspection. The completed archive-only entry is removed. Real model inference,
selected TIME preparation, environment refresh, test/full workflows, artifact
inspection, lifecycle observations, and the publisher check remain pending.

## 2026-08-25 — Unique foundation aliases and explicit covariate capability

- Behavior and affected contracts: made `chronos2`, `chronos_bolt`, `ts_icl`,
  `tirex2`, and `tabpfn_ts` the sole accepted foundation names, removed every
  historical spelling from model selection and reporting, and changed the
  foundation launcher to `chronos_bolt`. The shared adapters now declare
  covariate capability; Chronos-Bolt rejects structured or named covariates
  instead of allowing named values to disappear through keyword arguments.
- Focused checks completed: direct model-adapter and Slurm workflow tests
  passed without loading publication checkpoints; Python AST parsing, changed
  Bash syntax, exact five-alias parity, and cross-project SHA-256 parity for all
  five basic adapters passed. `pytest` was absent from the prepared runtime, so
  no installation was attempted and the direct test entry points were used.
- Deferred integration: run checkpoint-backed univariate/covariate calls and
  the foundation test profile on the cluster after the user refreshes the
  managed environment. No heavy inference was run locally.
- README/LaTeX and reruns: README, Hydra's model-name comment, and the guideline
  source now specify canonical aliases and explicit unsupported-covariate
  errors. Re-render the guideline during maintenance; the executive summary is
  unchanged. No completed inference result exists, so the already-required
  test/full runs simply use the new aliases.

Maintenance 2026-08-26: direct committed-delta, environment metadata,
adapter/profile, eight-front, synchronization-script, README, guideline,
summary, placeholder, and handoff inspection found no synchronized result.
After removing only scheduler-specific lines, all four DGX/Selena front pairs
were identical; `pyproject.toml` and `uv.lock` metadata parsed consistently.
The focused static/Bash checks were not repeated. The canonical-alias guideline
was newer than its PDF, so two pdfLaTeX passes produced a clean four-page PDF
and every page passed visual inspection. The new host synchronization and
Selena scheduling entries remain open for their first real cross-host/test
exercise. A later same-day update explicitly defined the four-model launch
profile, kept TiRex-2 adapter-supported, and added shared-resource discovery.
The direct Slurm workflow check passed and SHA-256 parity held for all five
sibling adapters. The updated guideline was rebuilt and visually inspected.
Environment mutation, TIME download, foundation inference, lifecycle
execution, and publishing were deliberately not performed.
