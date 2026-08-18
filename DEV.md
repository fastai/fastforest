# Development

`fastforest` is a Rust library with a private PyO3 extension and a small public Python façade.

## Layout

```text
src/forest.rs                 regression tree growth and public forest API
src/classification.rs         multiclass leaves, entropy growth, and public classifier API
src/tree.rs                   shared tree-growth driver and branch representation
src/prediction.rs             shared cache-blocked regression/classification inference
src/ensemble.rs               shared tree assembly, OOB, importance, and forest combination
src/split.rs                  regression split scoring plus shared candidate/histogram layout
src/class_split.rs            classification weighted-entropy scoring and class-count scratch
src/preprocessing.rs          Arrow-to-native fitted mixed-column schema and encoding
src/model.rs                  versioned `.ffm` representation and atomic persistence
src/file.rs                   bounded CSV/Arrow ingestion around one shared Arrow fitting path
src/csv_view.rs               typed compact CSV inspection and sampling
src/cli.rs                    shared argument parsing for the native executables
src/compiled.rs               Cargo-built predictor with embedded model bytes
src/bin/                      four pure Rust command-line executables
src/python.rs                 private PyO3 Arrow boundary and NumPy result bindings
python/fastforest/__init__.py thin public API re-export
python/fastforest/core.py     public estimators and model orchestration
python/fastforest/auto.py     automatic sample and forest sizing
python/fastforest/preprocessing.py Arrow container adapter, display metadata, and aggregation
python/fastforest/sklearn.py       reproducible sklearn mixed-data benchmark preprocessors
python/fastforest/analysis.py NumPy analysis results, perturbations, clustering, and lazy plots
tests/test_fastforest.py      Python API narratives and validation
tools/bench.py                fit, OOB, and prediction timings
tools/accuracy.py             regression and classification accuracy/timing comparisons
tools/meta_benchmark.py       resumable BeyondArena sweep collection with per-dataset timeouts
tools/sweep_advisor.py        group-held-out meta-model fitting from sweep diagnostics
meta/advisor_suite_selection.py inspect sweep coverage and advisor-menu rankings
python/fastforest/tools.py    batched OOB/validation calibration plus benchmark table generation
tools/mk_readme.py            README template renderer
tools/stage_binaries.py       copy release executables into wheel script data
tools/results/                raw measurements used by README benchmark tables
meta/sweeps/                  focused canonical-dataset sweeps (untracked)
meta/meta_benchmark/          large 24-configuration meta-sweep (untracked)
meta/meta_advisor/            held-out advisor artifacts (untracked)
```

The Rust core has no Python dependency. PyO3 and rust-numpy are optional behind the `python` feature; maturin enables `extension-module`, which enables them. The crate therefore remains directly usable as an `rlib` and builds all five native binaries without Python.

## Design

`Config::default()` is the canonical production-regression configuration; `Config::classification()` changes only task-specific defaults. The file API derives `FileFitOptions` from these configurations, the native CLI overlays only explicitly supplied flags, and PyO3 exposes the same values to the Python estimator signatures. Adaptive replacement is likewise resolved by one Rust function. Development tools instantiate an estimator or use an explicit `default` sweep level rather than copying production values.

Arrow `RecordBatch` is the single raw-table boundary. The Python adapter selects the bounded training pool before converting pandas, NumPy, or Arrow-compatible inputs; CSV fitting retains only selected rows in Arrow string arrays; Arrow IPC fitting selects its pool directly from typed batches. The Rust encoder detects date formats from at most 200 sampled pool rows and owns the format catalogue, generated date-part names, parsing, display extraction, numeric and lexical interpretation, missing markers, one-hot groups, schema fitting, and inference transformation. Independent columns run in parallel with Rayon. It emits contiguous `u32` ranks, dummy indicators, and missing indicators for training and `f32` native values for inference. The encoder retains the original-column mapping, type, integral-display flag, fitted median, strict/permissive inference-missing rule, and native cutoff boundaries. Python maps user-facing column selectors to indexes and reconstructs only small display metadata, while numerical model and analysis results remain NumPy arrays. PyO3 detaches from Python while Rust runs.

Split search shares candidate traversal, dense histogram filling, ranked-row construction, child validity, and equality support between tasks. Regression and classification keep their scoring accumulators explicit, because forcing target sums and class histograms through a dynamic abstraction would obscure the hot loop. Tree growth likewise has one generic work-stack/partition driver, while task-specific leaf values and final storage stay concrete. At the Python boundary, common fitted metadata and analysis methods are shared; task-specific fit, prediction, persistence, and PyO3 return types remain explicit where that is easier to read.

Tree construction uses `u32` feature ranks, candidate cutoffs, row indexes, and classification target IDs. Temporary 16-byte training nodes hold rank cutoffs, allowing OOB prediction directly against the encoded training matrix. Once OOB is complete, every split rank becomes the greatest native value on its left side. Final regression trees own flat vectors of 16-byte inference nodes containing an `f32` native cutoff or leaf value plus `u32` feature and child indexes. Classification uses the same 16-byte topology nodes with a leaf index into one contiguous `f32` probability buffer. A native value goes right when it is greater than the stored left boundary, exactly matching insertion-rank behavior for values unseen during training. Sibling children are adjacent, so the right child is `left + 1`.

Regression split scratch tracks target sums and squared sums. Its default three-row prior shrinks each candidate child's mean toward its parent before computing size-weighted sample standard deviation, so weak tiny-child splits are penalized without a hard leaf-size threshold. Classification calculates one normalized weight per class from its exact frequency in the tree sample, proportional to `frequency^-class_weight_power`, then builds lookup tables for weighted `count*log(count)` terms. Moving sampled-node rows across a candidate boundary therefore updates weighted entropy in constant time. The default power is 0.75; zero recovers ordinary entropy. Terminal probabilities remain ordinary unweighted observations. Computing them visits only the rows already sampled for that tree; summed across leaves, that is one capped pass per tree. Regression caps production trees at 40,000 rows. Classification scales this budget by `max(1, classes-1)`, so binary classification matches regression.

Before schema fitting, a Rust planner resolves an estimated tree/sample budget from the original row count and selects `min(rows, ceil(0.63 * trees * rows_per_tree))` uniform rows. Classification estimates its pool multiplier from at most 1,000 targets; the selected pool then supplies the actual classes and final production budget. Schema fitting, target conversion, encoding, and tree construction retain only pool rows. Finite-cap OOB evaluates at most the resolved cap and returns original-row indexes alongside its bounded outputs; `bootstrap_max=None` retains full-pool OOB.

The `.ffm` envelope starts with an explicit magic and format version, records its writer version, and serializes the native encoder and inference forest rather than Python objects or training scratch. It stores portable scalar markers, fitted estimator parameters, typed class labels, grouping, and dates. Loading validates every mapping, dimension, cutoff, node index, and probability buffer before exposing the model. OOB arrays are intentionally omitted to keep files bounded. Python reconstructs only display metadata around the loaded native encoder.

File fitting and prediction are library operations shared by all interfaces. CSV fitting makes a lightweight counting/class-reservoir pass and then builds one Arrow batch from only the uniformly selected pool. Arrow IPC accepts mixed nullable named columns and selects typed rows without first materializing the full file. CSV and Arrow prediction both feed bounded Arrow batches through the same encoder. The standalone converter remains intentionally numeric and emits ordinary nullable `Int64`/`Float64` Arrow batches.

The five Rust binaries use `cli.rs` and are installed directly from each wheel's standard script-data directory. Maturin can build either PyO3 bindings or binary bindings in one invocation, so release builds compile and stage the binaries before Maturin packages the extension and Python facade. Standalone compilation creates a small temporary Cargo project whose program embeds the exact validated `.ffm` bytes and calls the shared embedded-prediction entry point. Installed wheels fall back to the matching crates.io version when their original build source is unavailable.

The uniform `u32` training representation supports arbitrary practical cardinality without the bandwidth cost of `u64`; adaptive `u8`/`u16` columns are intentionally deferred until benchmarks justify their complexity. Candidate de-duplication packs the `u32` feature and cutoff into a `u64` key. `usize` is reserved for Rust indexing boundaries. A seeded RNG first generates one seed per tree, so Rayon can build trees in parallel without changing results.

Regression prediction is parallel over rows. Classification uses two-dimensional cache tiling without shared accumulators: parallel row blocks own their output elements, while an internal tree-batch size is calculated from the fitted forest. Mean bytes per tree include 16-byte topology nodes and `f32` leaf probabilities; as many trees as fit in a conservative 512 KiB budget are processed together. Row blocks target four tasks per Rayon worker and never exceed 4,096 rows. Small trees therefore preserve row-major locality, whereas a tree larger than the cache budget is traversed for every row in a block before moving to the next tree. This reduced full-Covertype 50-tree native probability inference from roughly 96 ms to 56 ms without slowing 10k- or 50k-row fits. Trials at 512 KiB, 1 MiB, and 2 MiB slightly favored the smallest budget at 50k and 100k training rows and were indistinguishable at 10k.

Tree construction accumulates normalized split-gain importance outside the compact inference nodes. Native analysis operations return per-tree predictions and encoded-feature path contributions efficiently without exposing the internal node representation; Python then sums derived features back into their original columns. A path contribution is the change in successive node values assigned to the feature selecting that child, so forest bias plus all contributions exactly reconstructs the prediction.

The higher analysis layer deliberately uses ordinary NumPy result objects and lazy matplotlib plots. Raw inputs still pass through the shared Arrow adapter, and sampling happens first: permutation importance and feature relations use at most 5,000 rows, PDP/ICE uses 500, feature dependence uses 5,000, and drop-column importance uses at most 40,000 training and 5,000 validation rows by default. A shared feature resolver handles indexes, data-frame names, and grouped features; tie-aware Spearman ranks and average linkage remain small local implementations rather than scipy dependencies.

OOB is optional because it predicts each selected evaluation row with every tree for which that row was not sampled. During tree construction a compact pool-sized in-bag mask is retained, then discarded after forest-level OOB sums and counts have been calculated. Regression accumulates scalar predictions; classification accumulates probability vectors and reports OOB accuracy.

`fastforest.auto` contains `AutoForest` and `AutoForestClassifier`; they are deliberately not re-exported from the package root. Above `2 * bootstrap_max * max(1, classes-1)` rows, one parallel screen fits eight trees for the baseline and each available larger one-axis sample setting. Ordinary sizing uses `bootstrap_max=(80k,120k,160k,200k)` and `max_node_samples=(640,960,1280)`; autogrow uses the wider `bootstrap_max=(80k,160k,240k,320k)` and `max_node_samples=(640,1280,1920)`. Resolved duplicates and bootstrap levels above 80% of rows per output are omitted; any resulting vacant autogrow bootstrap slots are filled with eligible ordinary-grid levels. For a trial `j` levels above baseline, `floor(relative improvement / 0.01)` grants the number of justified levels, capped at `j`; the furthest justified level across trials is selected independently for each axis. Thus a noisy intermediate result cannot override a clear cumulative improvement. Ordinary auto fitting then uses the standard adaptive 32–64 tree count without final-model OOB. `autogrow=True` instead enables tracking/OOB and adds 32-tree batches while loss improves by the independently configured growth threshold.

Production then grows in 32-tree batches. Before the first batch it independently chooses at most 40,000 random tracking rows per output. Every batch uses fresh tree seeds but the same tracking rows, and a row contributes only from trees that did not train on it. Native forest combination merges trees, importance, counts, and count-weighted OOB predictions without repeating preprocessing. Cumulative regression MSE or classification Brier loss must improve by 1% at each checkpoint. The first failing batch is discarded by default; `keep_last_batch=True` retains it, and `max_trees` defaults to 512.

Candidate cutoffs are deduplicated as exact `(feature, value)` pairs. The default permits two proposals per requested unique candidate; experiments showed that larger retry bounds disproportionately increase fit time on discrete features. A 20-row floor keeps candidate coverage from collapsing quadratically in small nodes. Reusable per-tree scratch storage keeps de-duplication overhead low.

Candidate children need only be nonempty. Regression's continuous split loss and classification's weighted entropy rank small children directly rather than hiding them behind fixed child-size or gain thresholds. `min_node_size` remains the full-node stopping rule.

`split::find_split` dispatches between the production histogram search and the original random splitter using the public `random_splitter` flag. Numeric value features and their missing indicators are atomic sampling groups. `max_features` defaults to 90% for regression and 60% for classification, and also accepts another fraction or `"sqrt"`.

The histogram search considers every rank boundary observed in its bounded node sample. Regression scores child-size-weighted standard deviation from constant-time target sums and squared sums.

Exact histograms also consider `feature == value` when that value and its complement each have the same support required of any other split. Equality search is skipped once the evaluation window has at most three times that support. Endpoint values remain covered by ordinary ordered splits, and approximate per-tree histograms deliberately retain only their ordered candidates. The split predicate is stored in the high bit of the compact feature index, preserving 16-byte nodes; inference maps known lexical values to their fitted integer ranks and unseen values between ranks so they cannot accidentally match a known category.

Exact dense histograms score supported interior equality predicates during the same ordered pass that accumulates prefix statistics. Binary features therefore perform one ordinary cutoff score and no equality scores. A one-bin binary histogram was tested but removed: deriving the other regression side by `f32` subtraction changed trees slightly, while fit changes across Covertype, Bank, and Walmart were negligible and inconsistent.

Categorical subset splits and learned missing routing belong at a separate boundary: they require encoder metadata and a richer stored split predicate, not another branch inside ordered cutoff scoring.

## Testing

Tests favor a few public-API narratives over private implementation tests. The main Rust integration test covers fitting, compact-tree invariants, prediction quality, OOB, determinism, per-tree predictions, split importance, exact explanation reconstruction, and standalone/batched equivalence. A file integration narrative covers mixed CSV regression/classification, `.ffm`, bounded batches, numeric CSV conversion, and Arrow fit/prediction. The regression and multiclass Python narratives round-trip mixed, date, grouped, and typed-label models and exercise native file prediction. Compile-time assertions keep every training and inference node representation at 16 bytes. A dedicated ignored test builds and invokes a complete standalone release executable; CI runs it on each release platform without slowing ordinary iterations.

```bash
cargo fmt
cargo check
cargo test
cargo build --release --bins
python tools/stage_binaries.py
maturin develop
pytest -q
chkstyle python/fastforest tests tools
```

For performance measurements:

```bash
maturin develop --release
python tools/bench.py --rows 60000 --cols 50 --trees 100
```

For an accuracy and timing comparison using an identical fixed split for each model:

```bash
python tools/accuracy.py
```

README prose lives in `README.tmpl`, while raw displayed benchmark measurements live in `tools/results/`. Run `python tools/mk_readme.py` after changing either. Its `mk_table` helper applies the project rounding rules, determines all winners from displayed values, and generates the summary and complete HTML tables with `fastcore.xml`.

California Housing is the default. The other regression choices are `concrete`, `sgemm`, `diamonds`, `allstate`, `diabetes`, `bluebook`, `bluebook_raw`, `walmart`, `walmart_raw`, and `walmart_nodate`. Blue Book uses the course's final-12,000 validation split and log-price target. Walmart uses its final 12 weeks for validation. Classification choices are `covertype`, `covertype_grouped`, `adult`, and `bank`; binary Covertype is the default-input benchmark while its grouped variant demonstrates `one_hot_groups`. Classification quality is reported as macro-averaged F1 and log loss. Other datasets use one reproducible 80/20 split, stratified for classification. Mixed-data sklearn RF comparisons use `sklearn_preprocessor`: numeric-string detection and median imputation, one-hot encoding through 20 levels, and target encoding above 20. HistGBM uses its documented native categorical support, target-encoding only features beyond its 255-category limit. Neither sklearn path receives fastforest's date detection or expansion.

Each model/dataset combination runs in a spawned process and has a 180-second default timeout. The child signals readiness after process startup and input deserialization; only then does the timeout begin. Reported fit time includes model and preprocessor construction, schema inspection, and `fit`, but excludes process startup and IPC.

The sklearn random forest uses all available cores, as do FastForest and histogram GBM internally.

## Hyperparameter research and calibration

Hyperparameter discovery is interactive, not grid-driven. Start with one small, explicit hypothesis on one or a few informative datasets; inspect validation quality, fit and prediction time, and tree structure immediately; then choose the next experiment from what that result teaches. Stop and discuss material regressions or surprising behavior before adding combinations. A large generic sweep is a poor substitute for this process: it hides interactions and encourages post-hoc explanations rather than understanding why a change worked.

Stored result CSVs retain full precision, but terminal inspection should never dump their raw floats. Use `viewcsv <file>` by default: it reports metrics with four significant figures and timings with three, displays every varying column, and lists constant columns once afterward. `--cols` narrows the view explicitly, `--rows` selects the leading rows, and `--sample N|P% --seed S` selects reproducible random rows before identifying constants. Custom result summaries should apply the same formatting, with relative changes shown to one decimal place.

Classification scoring research is recorded chronologically in `meta/experiments.md`. Fixed child thresholds, plug-in priors, search-cost penalties, and exact/fractional leave-one-out entropy all suppressed weak immediate splits that enable useful descendants. Tree-frequency-weighted entropy with power 0.75 retained or improved macro F1 on every README classification dataset, including gains of 16.4 points on Shuttle and 14.8 on KDDCup99. A separate ten-dataset check found one modest calibration exception (Coil-2000), while probability-smoothing attempts sacrificed the rare-class gains and were removed. `class_weight_power` is intentionally exposed because lower powers can produce smaller trees on some datasets.

Broad scripted sweeps belong later, once the parameter ideas and useful levels are understood. Use them to confirm proposed defaults across many datasets, produce reproducible reporting tables, or collect deliberately designed advisor-training data. Keep the marginal one-parameter calibration sweep separate from randomized joint configurations used to teach the advisor about interactions.

For focused FastForest tuning, add `--ff_only` and vary `--min_node_size`, `--bootstrap_fraction`, `--bootstrap_max`, `--replacement`, `--max_node_samples`, `--cutoff_divisor`, or `--max_features`; regression also exposes `--split_prior_rows`, and classification exposes `--class_weight_power`. Set `--random_splitter` to compare the original split search. The tools use `call_parse`, so CLI names match their underscored function parameters.

Run a reproducible parameter grid (SGEMM by default) and save its metrics and timings with:

```bash
python tools/sweep.py
```

`tools/sweep.py` takes a comma-separated level list for every tree hyperparameter. The first item in each list is the shared baseline; every later item creates one one-axis configuration, and an alternative that cannot produce the requested OOB evaluation is an error. It first fits all configurations as one native parallel batch of eight same-seed trees, recording OOB and training MSE for regression or multiclass Brier loss. It then repeats the identical configurations with each dataset's ordinary resolved tree count and canonical validation split. The calibration pass groups configurations by resolved `(trees, rows_per_tree, pool_rows)`, preprocesses each distinct pool once, and records validation and training loss. The per-dataset CSV joins both views with pool, coverage, and tree-size diagnostics. Separate load, OOB-screen, and full-validation timeouts bound each phase. Rust tests prove that batched regression and classification forests are bit-for-bit identical to standalone fits and independent of configuration order.

The full-validation phase fits each configuration individually against the shared preprocessed matrices. It records fit time with the common measured training-preprocessing cost included, and prediction time with the common validation-transformation cost included. The eight-tree OOB screen remains packed into one parallel batch; its purpose is rapid ranking rather than per-configuration timing.

## Meta-benchmark and sweep advisor

There are nine related result sets; do not confuse them:

1. `meta/sweeps/all.csv` is the focused canonical benchmark sweep assembled from `meta/sweeps/*.csv`. It currently contains 14 datasets: 10 regression and 4 classification.
2. `meta/meta_benchmark/all.csv` is the underlying large sweep. The current snapshot has 4,368 rows: 24 one-axis configurations for each of 182 datasets, comprising 53 regressions and 129 classifications. Its source rows live in `meta/meta_benchmark/results/*.csv`; `all.csv` is their concatenation.
3. `meta/meta_advisor/` contains derived advisor artifacts, not the full sweep. Its current matrices contain eight runtime candidates for 174 datasets after holding out the README dataset groups. Each feature matrix therefore has 1,392 rows. It cannot answer questions about configurations omitted from that eight-candidate menu; use `meta/meta_benchmark/all.csv` for those comparisons.
4. `meta/meta_interpretability/all.csv` contains the newer marginal calibration: 4,806 rows, comprising 27 one-axis configurations for each of 178 datasets (51 regressions and 127 classifications). Its 26-entry `slow.csv` is independent of the historical sweep's 22-entry list.
5. `meta/meta_joint/all.csv` contains 3,560 timed joint configurations: 20 for each of those 178 datasets. Every dataset covers the complete 5×4 product of `bootstrap_max=(20k,40k,80k,160k,240k)` and `max_node_samples=(160,320,640,2560)` exactly once; each row also changes enough randomly chosen forest parameters to contain two to five changes in total. The randomization is deterministically seeded by dataset name. Each row records standalone-equivalent full fit and prediction time, including its shared preprocessing costs.
6. `meta/meta_advisor_joint/input.csv` concatenates the marginal and joint results, giving 8,366 rows or 47 configurations per dataset. `meta/meta_advisor_joint/` contains advisors trained on all those rows, while selection during held-out evaluation remains restricted to the fixed eight configurations that can be screened at runtime. `meta/meta_advisor_marginal/` is the controlled comparison trained on marginal rows alone with the same schemas and selection menus.
7. `meta/meta_interpretability_adaptive/all.csv` is the corrected current marginal set: 4,984 rows, 28 configurations for all 178 datasets. It recalculates plans from discovered multiclass outputs, retains the adaptive replacement baseline, and adds `min_node_size=4`. The otherwise-matching no-replacement diagnostic is in `meta/meta_interpretability_fixed/`; broad comparison showed adaptive replacement remains preferable, especially for regressions below 10,000 rows.
8. `meta/meta_joint_fixed/all.csv` contains 3,500 new joint rows for 175 datasets. Jannis, KDDCup99, and SF Police Incidents exceeded the 60-second joint timeout; their complete corrected marginal rows remain available.
9. `meta/meta_combined_fixed/all.csv` concatenates those corrected marginal and joint rows: 8,484 configurations across 178 datasets. This is the current meta-model training source, prior to README-group filtering.

All of `meta/` is deliberately untracked development state. These paths and reconstruction steps are documented here so ignored artifacts remain understandable. The counts describe the current local snapshot and should be checked from the files after further collection.

### Collecting the large sweep

`tools/meta_benchmark.py` builds a manifest from one canonical repeat/fold of BeyondArena plus locally downloaded, non-overlapping AMLB datasets. It excludes semantic-text tasks unless requested, uses the supplied BeyondArena split where available and one seeded 80/20 split for AMLB, and fits with `allow_new_missing=True` and `allow_unseen_classes=True`. Each dataset runs in a spawned process with a whole-task timeout. Successful datasets are written immediately to `meta/meta_benchmark/results/<dataset>.csv`, so reruns skip completed work. Timed-out datasets are recorded in `slow.csv` and skipped unless `--include-slow` is passed; other failures go to `failures.csv`.

Each dataset result contains:

- an eight-tree batched OOB screen and its training loss, coverage, evaluated rows, and mean tree size;
- a normal-tree-count fit on the canonical validation split, with validation/training loss and tree diagnostics;
- the resolved pool/tree settings and every forest hyperparameter;
- bounded schema and target metadata used by the advisor.
- for newly collected results, full-fit and full-prediction seconds including the appropriate measured preprocessing cost.

The historical 24 configurations came from the then-current standard levels. `STANDARD_ALTERNATIVES` and `standard_levels` in `python/fastforest/tools.py` now construct the one-axis suite from an explicitly configured baseline. The first level for every parameter must equal that baseline, preventing an alternative from silently becoming another parameter's sweep baseline. Historical result files retain their actual parameter values in every row. In particular, the historical large-sweep snapshot used `min_node_size=4` and `max_features=0.75`, not the newer experimental baseline. Comparisons among its `max_features` rows remain controlled because their other parameters are identical.

Run or resume it with:

```bash
python tools/meta_benchmark.py
```

Collect or resume the 20-row joint design separately with:

```bash
python tools/meta_benchmark.py --output_dir meta/meta_joint --suite_kind joint --joint_configs 20
```

Seed the new output directory's `slow.csv` from the current marginal run before starting, rather than pointing `--slow_csv` at an older experiment: a running collector updates its own timeout list. Both modes write each completed dataset immediately and print dataset-level progress, so an interrupted run remains usable and resumes without repeating successful datasets.

Useful controls include `--task_timeout`, `--limit`, `--task_names`, and `--include_slow`. Targeted follow-ups can also replace the marginal suite's baseline `replacement`, node size, absolute child support, and feature fraction without changing package defaults. Do not delete successful per-dataset CSVs merely to rebuild `all.csv`; the script regenerates it from them after each successful task.

Classification planning first estimates output dimensions to bound the initial pool, then both the OOB screen and full validation recompute their tree/sample plans from the classes actually found in that pool. This matters for large multiclass data: a 1,000-target estimate can miss rare classes and would otherwise under-budget rows per tree and OOB evaluation. When comparing historical sweep snapshots, first reproduce an old winning configuration with the current implementation; old validation numbers can reflect implementation changes rather than a parameter regression, especially on tiny validation sets.

### Choosing the runtime comparisons

`meta/advisor_suite_selection.py` reads the full `meta/meta_benchmark/all.csv`. It reports individual candidate wins and relative losses, and greedily measures the additional oracle coverage contributed by each configuration. This is the appropriate source when deciding which few alternatives deserve runtime-menu places. The selected menu is a pragmatic, manually agreed summary of the sweep rather than an AutoML search space.

The broad sweep also answers pairwise questions. After accounting for the lower-feature choices `max_features=0.6` and `sqrt`, adding `0.9` improves the oracle on 14/53 regressions and 21/129 classifications; adding `1.0` improves 13/53 and 19/129. Directly, `0.9` beats `1.0` on 43/53 regressions and 100/129 classifications. The evidence therefore favors `0.9` as the upward feature-fraction trial for both tasks, while `sqrt` remains a complementary low-feature trial. These figures come from the 182-dataset large sweep, not the 14-dataset canonical sweep.

The public runtime sizer is implemented by `fastforest.auto`, not the development advisor machinery. It tries only larger sample settings and independently sizes the two axes as described above. The historical meta-advisor and diversity candidates are not part of runtime selection.

### Training and using the advisor

`tools/sweep_advisor.py` converts sweep rows into a meta-learning problem. `tools/results/datasets.csv:meta_group` identifies README dataset families to exclude from historical advisor evaluation. Related representations share a `source_group` and are held out together. The historical 182-dataset sweep consequently produces 174 advisor-training datasets; the newer 178-dataset collections produce 170. Each output directory's `held_out.csv` records the exclusions.

Each candidate configuration becomes one advisor row. Inputs include raw dataset size and schema counts, target/class statistics, the configuration hyperparameters, and its eight-tree OOB/training diagnostics. Joint training includes every parameter varied by the joint design; its `changed_param` is simply `joint`, since numbered joint labels have different configurations on different datasets. Engineered monotonic transforms, generic products/ratios, duplicate leaf statistics, and fields determined by another input are deliberately excluded: forests can split the raw measurements directly, and speculative single-row feature engineering adds complexity without information. Regression and classification have separate target schemas. Baseline-relative OOB, training, node, and depth values remain because they bring comparison information from another configuration row and normalize dataset-specific loss scales. The target is validation loss relative to the dataset baseline: relative RMSE for regression and relative Brier loss for classification. Continuous- and categorical-hyperparameter representations are both evaluated. Evaluation leaves one `source_group` out at a time, fits on every available training configuration, but chooses only among the fixed eight runtime candidates for the held-out dataset. It then records the selected configuration's actual relative loss, oracle loss within that menu, and regret.

At README benchmark time, `tools/accuracy.py --auto_only` records both an `AutoForest` row with the ordinary adaptive tree count and an `autogrow` row capped at 192 trees. Sample sizing and every attempted tree batch are included in fit time. Ordinary `FastForest.fit` continues to use documented defaults without automatic sizing.

Rebuild the held-out advisor from the large sweep with:

```bash
python tools/sweep_advisor.py --results meta/meta_benchmark/all.csv --output_dir meta/meta_advisor
```

After changing `STANDARD_LEVELS`, defaults, or `_ADVISOR_COMPARISONS`, do not assume existing artifacts describe the new setup. Inspect the resolved parameters stored in the raw rows, collect only newly required configurations where practical, then rebuild the filtered advisor matrices and models. The next full meta-dataset should use the current defaults and add randomly sampled combinations of the supported hyperparameter levels, rather than containing only baseline-plus-one-axis sweeps.

## Versioning and release

The canonical version lives in `Cargo.toml`; `pyproject.toml` gets it through `dynamic = ["version"]`.

Once the repository has been created and added to the workspace, release flow is:

1. Run the Rust and Python tests against a release build.
2. Confirm the release version in `Cargo.toml`.
3. Run `ship-release`.

GitHub Actions builds Linux, macOS, and Windows wheels containing the Python extension and all five native binaries. An sdist is deliberately not published because Maturin cannot build PyO3 and binary bindings in the same source-build invocation. CI also tests and archives the binaries independently on those operating systems; tagged builds publish the wheels and attach the CLI archives to the GitHub release.
