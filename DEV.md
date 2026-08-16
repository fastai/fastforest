# Development

`fastforest` is a Rust library with a private PyO3 extension and a small public Python façade.

## Layout

```text
src/forest.rs                 forest algorithm and Rust tests
src/split.rs                  interchangeable split searches and their reusable scratch
src/classification.rs         probability trees, multiclass forest, OOB, and batched fitting
src/class_split.rs            Gini split searches and reusable class-count scratch
src/preprocessing.rs          Arrow-to-native fitted mixed-column schema and encoding
src/model.rs                  versioned `.ffm` representation and atomic persistence
src/file.rs                   bounded CSV/Arrow fitting, conversion, and batch prediction
src/cli.rs                    shared argument parsing for the native executables
src/compiled.rs               Cargo-built predictor with embedded model bytes
src/bin/                      four pure Rust command-line executables
src/python.rs                 private PyO3 Arrow boundary and NumPy result bindings
python/fastforest/__init__.py public Python API and model orchestration
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

The Rust core has no Python dependency. PyO3 and rust-numpy are optional behind the `python` feature; maturin enables `extension-module`, which enables them. The crate therefore remains directly usable as an `rlib` and builds all four native binaries without Python.

## Design

Arrow `RecordBatch` is the single raw-table boundary. The Python adapter selects the bounded training pool before converting pandas, NumPy, or Arrow-compatible inputs; CSV fitting retains only selected rows in Arrow string arrays; Arrow IPC fitting selects its pool directly from typed batches. The adapter detects date formats from at most 200 sampled pool rows, then one Rust encoder handles numeric and lexical interpretation, missing markers, one-hot groups, full-column date expansion, schema fitting, and inference transformation, parallelizing independent columns with Rayon. It emits contiguous `u32` ranks, dummy indicators, and missing indicators for training and `f32` native values for inference. The encoder retains the original-column mapping, type, integral-display flag, fitted median, strict/permissive inference-missing rule, and native cutoff boundaries. Python reconstructs only the small display metadata, while numerical model and analysis results remain NumPy arrays. PyO3 detaches from Python while Rust runs.

Tree construction uses `u32` feature ranks, candidate cutoffs, row indexes, and classification target IDs. Temporary 16-byte training nodes hold rank cutoffs, allowing OOB prediction directly against the encoded training matrix. Once OOB is complete, every split rank becomes the greatest native value on its left side. Final regression trees own flat vectors of 16-byte inference nodes containing an `f32` native cutoff or leaf value plus `u32` feature and child indexes. Classification uses the same 16-byte topology nodes with a leaf index into one contiguous `f32` probability buffer. A native value goes right when it is greater than the stored left boundary, exactly matching insertion-rank behavior for values unseen during training. Sibling children are adjacent, so the right child is `left + 1`.

Regression split scratch tracks target sums and squared sums. Classification scratch instead tracks class counts and the sum of their squares, so moving a row across a candidate boundary updates Gini in constant time rather than rescanning every class at every cutoff. Computing a terminal probability vector visits only the rows already sampled for that tree; summed across leaves, that is one capped pass per tree. Regression caps production trees at 40,000 rows. Classification scales this budget by `max(1, classes-1)`, so binary classification matches regression.

Before schema fitting, a Rust planner resolves an estimated tree/sample budget from the original row count and selects `min(rows, ceil(0.63 * trees * rows_per_tree))` uniform rows. Classification estimates its pool multiplier from at most 1,000 targets; the selected pool then supplies the actual classes and final production budget. Schema fitting, target conversion, encoding, and tree construction retain only pool rows. Finite-cap OOB evaluates at most the resolved cap and returns original-row indexes alongside its bounded outputs; `bootstrap_max=None` retains full-pool OOB.

The `.ffm` envelope starts with an explicit magic and format version, records its writer version, and serializes the native encoder and inference forest rather than Python objects or training scratch. It stores portable scalar markers, fitted estimator parameters, typed class labels, grouping, and dates. Loading validates every mapping, dimension, cutoff, node index, and probability buffer before exposing the model. OOB arrays are intentionally omitted to keep files bounded. Python reconstructs only display metadata around the loaded native encoder.

File fitting and prediction are library operations shared by all interfaces. CSV fitting makes a lightweight counting/class-reservoir pass and then builds one Arrow batch from only the uniformly selected pool. Arrow IPC accepts mixed nullable named columns and selects typed rows without first materializing the full file. CSV and Arrow prediction both feed bounded Arrow batches through the same encoder. The standalone converter remains intentionally numeric and emits ordinary nullable `Int64`/`Float64` Arrow batches.

The four Rust binaries use `cli.rs` and are installed directly from each wheel's standard script-data directory. Maturin can build either PyO3 bindings or binary bindings in one invocation, so release builds compile and stage the binaries before Maturin packages the extension and Python facade. Standalone compilation creates a small temporary Cargo project whose program embeds the exact validated `.ffm` bytes and calls the shared embedded-prediction entry point. Installed wheels fall back to the matching crates.io version when their original build source is unavailable.

The uniform `u32` training representation supports arbitrary practical cardinality without the bandwidth cost of `u64`; adaptive `u8`/`u16` columns are intentionally deferred until benchmarks justify their complexity. Candidate de-duplication packs the `u32` feature and cutoff into a `u64` key. `usize` is reserved for Rust indexing boundaries. A seeded RNG first generates one seed per tree, so Rayon can build trees in parallel without changing results.

Regression prediction is parallel over rows. Classification uses two-dimensional cache tiling without shared accumulators: parallel row blocks own their output elements, while an internal tree-batch size is calculated from the fitted forest. Mean bytes per tree include 16-byte topology nodes and `f32` leaf probabilities; as many trees as fit in a conservative 512 KiB budget are processed together. Row blocks target four tasks per Rayon worker and never exceed 4,096 rows. Small trees therefore preserve row-major locality, whereas a tree larger than the cache budget is traversed for every row in a block before moving to the next tree. This reduced full-Covertype 50-tree native probability inference from roughly 96 ms to 56 ms without slowing 10k- or 50k-row fits. Trials at 512 KiB, 1 MiB, and 2 MiB slightly favored the smallest budget at 50k and 100k training rows and were indistinguishable at 10k.

Tree construction accumulates normalized split-gain importance outside the compact inference nodes. Native analysis operations return per-tree predictions and encoded-feature path contributions efficiently without exposing the internal node representation; Python then sums derived features back into their original columns. A path contribution is the change in successive node values assigned to the feature selecting that child, so forest bias plus all contributions exactly reconstructs the prediction.

The higher analysis layer deliberately uses ordinary NumPy result objects and lazy matplotlib plots. Raw inputs still pass through the shared Arrow adapter, and sampling happens first: permutation importance and feature relations use at most 5,000 rows, PDP/ICE uses 500, feature dependence uses 5,000, and drop-column importance uses at most 40,000 training and 5,000 validation rows by default. A shared feature resolver handles indexes, data-frame names, and grouped features; tie-aware Spearman ranks and average linkage remain small local implementations rather than scipy dependencies.

OOB is optional because it predicts each selected evaluation row with every tree for which that row was not sampled. During tree construction a compact pool-sized in-bag mask is retained, then discarded after forest-level OOB sums and counts have been calculated. Regression accumulates scalar predictions; classification accumulates probability vectors and reports OOB accuracy.

Candidate cutoffs are deduplicated as exact `(feature, value)` pairs. The default permits two proposals per requested unique candidate; experiments showed that larger retry bounds disproportionately increase fit time on discrete features. A 20-row floor keeps candidate coverage from collapsing quadratically in small nodes. Reusable per-tree scratch storage keeps de-duplication overhead low.

`split::find_split` dispatches between the production histogram search and the original random splitter using the public `random_splitter` flag. Numeric value features and their missing indicators are atomic sampling groups; frequent-value indicators are attached to their ranked parent and one is added as an extra candidate whenever that parent is selected. `max_features` defaults to 60% of feature units and also accepts another fraction or `"sqrt"`.

By default the histogram search considers every observed rank boundary. Setting `tree_cutoff_samples` gives each tree a smaller random grid: each sufficiently high-cardinality feature samples that many unique ranks once, builds a compact rank-to-bin map, and reuses those approximate bins throughout the tree. Low-cardinality features remain exact, and the selected boundary is converted back to its global rank before partitioning and storage. `None` or zero keeps exact histograms. This is an explicit dataset-sensitive speed/regularization option rather than an adaptive default.

Categorical subset splits and learned missing routing belong at a separate boundary: they require encoder metadata and a richer stored split predicate, not another branch inside ordered cutoff scoring.

## Testing

Tests favor a few complete narratives over many single-assertion tests. The main Rust test covers fitting, tree invariants, prediction quality, OOB, determinism, per-tree predictions, split importance, and exact explanation reconstruction. A file narrative covers mixed CSV regression/classification, `.ffm`, bounded batches, numeric CSV conversion, and Arrow fit/prediction. The regression and multiclass Python narratives round-trip mixed, date, grouped, and typed-label models and exercise native file prediction. A dedicated ignored test builds and invokes a complete standalone release executable; CI runs it on each release platform without slowing ordinary iterations.

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
For focused FastForest tuning, add `--ff_only` and vary `--min_node_size`, `--bootstrap_fraction`, `--bootstrap_max`, `--replacement`, `--max_node_samples`, `--cutoff_divisor`, or `--max_features`. Set `--random_splitter` to compare the original split search. The tools use `call_parse`, so CLI names match their underscored function parameters.

Run a reproducible parameter grid (SGEMM by default) and save its metrics and timings with:

```bash
python tools/sweep.py
```

`tools/sweep.py` takes a comma-separated level list for every tree hyperparameter. The first item in each list is the shared baseline; every later item creates one one-axis configuration, and an alternative that cannot produce the requested OOB evaluation is an error. It first fits all configurations as one native parallel batch of eight same-seed trees, recording OOB and training MSE for regression or multiclass Brier loss. It then repeats the identical configurations with each dataset's ordinary resolved tree count and canonical validation split. The calibration pass groups configurations by resolved `(trees, rows_per_tree, pool_rows)`, preprocesses each distinct pool once, and records validation and training loss. The per-dataset CSV joins both views with pool, coverage, and tree-size diagnostics. Separate load, OOB-screen, and full-validation timeouts bound each phase. Rust tests prove that batched regression and classification forests are bit-for-bit identical to standalone fits and independent of configuration order.

## Meta-benchmark and sweep advisor

There are three related result sets; do not confuse them:

1. `meta/sweeps/all.csv` is the focused canonical benchmark sweep assembled from `meta/sweeps/*.csv`. It currently contains 14 datasets: 10 regression and 4 classification.
2. `meta/meta_benchmark/all.csv` is the underlying large sweep. The current snapshot has 4,368 rows: 24 one-axis configurations for each of 182 datasets, comprising 53 regressions and 129 classifications. Its source rows live in `meta/meta_benchmark/results/*.csv`; `all.csv` is their concatenation.
3. `meta/meta_advisor/` contains derived advisor artifacts, not the full sweep. Its current matrices contain eight runtime candidates for 174 datasets after holding out the README dataset groups. Each feature matrix therefore has 1,392 rows. It cannot answer questions about configurations omitted from that eight-candidate menu; use `meta/meta_benchmark/all.csv` for those comparisons.

All of `meta/` is deliberately untracked development state. These paths and reconstruction steps are documented here so ignored artifacts remain understandable. The counts describe the current local snapshot and should be checked from the files after further collection.

### Collecting the large sweep

`tools/meta_benchmark.py` builds a manifest from one canonical repeat/fold of BeyondArena plus locally downloaded, non-overlapping AMLB datasets. It excludes semantic-text tasks unless requested, uses the supplied BeyondArena split where available and one seeded 80/20 split for AMLB, and fits with `allow_new_missing=True` and `allow_unseen_classes=True`. Each dataset runs in a spawned process with a whole-task timeout. Successful datasets are written immediately to `meta/meta_benchmark/results/<dataset>.csv`, so reruns skip completed work. Timed-out datasets are recorded in `slow.csv` and skipped unless `--include-slow` is passed; other failures go to `failures.csv`.

Each dataset result contains:

- an eight-tree batched OOB screen and its training loss, coverage, evaluated rows, and mean tree size;
- a normal-tree-count fit on the canonical validation split, with validation/training loss and tree diagnostics;
- the resolved pool/tree settings and every forest hyperparameter;
- bounded schema and target metadata used by the advisor.

The 24 configurations come from `STANDARD_LEVELS` in `python/fastforest/tools.py`: one shared baseline and every non-baseline level varied one axis at a time. The first level for every parameter must equal the model default, preventing an alternative from silently becoming another parameter's sweep baseline. Historical result files retain their actual parameter values in every row. In particular, the current large-sweep snapshot used the defaults at collection time (`min_node_size=4` and `max_features=0.75`), not the newer defaults. Comparisons among its `max_features` rows remain controlled because their other parameters are identical.

Run or resume it with:

```bash
python tools/meta_benchmark.py
```

Useful controls include `--task_timeout`, `--limit`, `--task_names`, and `--include_slow`. Do not delete successful per-dataset CSVs merely to rebuild `all.csv`; the script regenerates it from them after each successful task.

### Choosing the runtime comparisons

`meta/advisor_suite_selection.py` reads the full `meta/meta_benchmark/all.csv`. It reports individual candidate wins and relative losses, and greedily measures the additional oracle coverage contributed by each configuration. This is the appropriate source when deciding which few alternatives deserve runtime-menu places. The selected menu is a pragmatic, manually agreed summary of the sweep rather than an AutoML search space.

The broad sweep also answers pairwise questions. After accounting for the lower-feature choices `max_features=0.6` and `sqrt`, adding `0.9` improves the oracle on 14/53 regressions and 21/129 classifications; adding `1.0` improves 13/53 and 19/129. Directly, `0.9` beats `1.0` on 43/53 regressions and 100/129 classifications. The evidence therefore favors `0.9` as the upward feature-fraction trial for both tasks, while `sqrt` remains a complementary low-feature trial. These figures come from the 182-dataset large sweep, not the 14-dataset canonical sweep.

The actual runtime candidate order lives in `_ADVISOR_COMPARISONS` in `tools/accuracy.py`. The ordinary path has seven alternatives plus the baseline: regression uses `bootstrap_fraction=0.5`, `tree_cutoff_samples=16`, `max_features=sqrt`, `min_node_size=64`, `max_node_samples=160`, `max_features=0.9`, and `bootstrap_max=160000`; classification uses `max_features=sqrt`, `bootstrap_fraction=0.5`, `tree_cutoff_samples=16`, `min_node_size=64`, `max_features=0.9`, `max_node_samples=2560`, and `min_global_gain=1e-5`. A dataset whose resolved production forest has only 20 trees uses the first three alternatives plus the baseline, giving a 32-tree screen rather than spending 64 trees on tuning. Changing order therefore changes which alternatives the compact path can see.

### Training and using the advisor

`tools/sweep_advisor.py` converts selected sweep rows into a meta-learning problem. `tools/results/datasets.csv:meta_group` identifies README dataset families to exclude, preventing tuned README rows from being evaluated on datasets used to train their advisor. Related representations share a `source_group` and are held out together. The current 182-dataset sweep consequently produces 174 advisor-training datasets; `meta/meta_advisor/held_out.csv` records the exclusions.

Each candidate configuration becomes one advisor row. Inputs include dataset size and schema composition, target/class statistics, the configuration hyperparameters, and its eight-tree OOB/training diagnostics. The target is validation loss relative to the dataset baseline: relative RMSE for regression and relative Brier loss for classification. Continuous- and categorical-hyperparameter representations are both evaluated. Evaluation leaves one `source_group` out at a time, then records the configuration selected by predicted loss, its actual relative loss, oracle loss, and regret. The fitted regression and classification advisor forests and summaries are written to `meta/meta_advisor/`.

At README benchmark time, `tools/accuracy.py` runs the same eight-tree candidate batch on the held-out training data, constructs advisor features, loads the task/encoding model selected by `summary.csv`, predicts every candidate's relative validation loss, and fits the predicted winner normally. Advisor screening, advisor inference, and the final fit are all included in the tuned row's fit time. This advisor is a reporting experiment; ordinary `FastForest.fit` continues to use documented defaults without running a sweep.

Rebuild the held-out advisor from the large sweep with:

```bash
python tools/sweep_advisor.py --results meta/meta_benchmark/all.csv --output_dir meta/meta_advisor
```

After changing `STANDARD_LEVELS`, defaults, or `_ADVISOR_COMPARISONS`, do not assume existing artifacts describe the new setup. Inspect the resolved parameters stored in the raw rows, collect only newly required configurations where practical, then rebuild the filtered advisor matrices and models.

## Versioning and release

The canonical version lives in `Cargo.toml`; `pyproject.toml` gets it through `dynamic = ["version"]`.

Once the repository has been created and added to the workspace, release flow is:

1. Run the Rust and Python tests against a release build.
2. Confirm the release version in `Cargo.toml`.
3. Run `ship-release`.

GitHub Actions builds Linux, macOS, and Windows wheels containing the Python extension and all four native binaries. An sdist is deliberately not published because Maturin cannot build PyO3 and binary bindings in the same source-build invocation. CI also tests and archives the binaries independently on those operating systems; tagged builds publish the wheels and attach the CLI archives to the GitHub release.
