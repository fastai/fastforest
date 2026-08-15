# Development

`fastforest` is a Rust library with a private PyO3 extension and a small public Python façade.

## Layout

```text
src/forest.rs                 forest algorithm and Rust tests
src/split.rs                  interchangeable split searches and their reusable scratch
src/classification.rs         probability trees, multiclass forest, OOB, and adaptive fitting
src/class_split.rs            Gini split searches and reusable class-count scratch
src/workbench.rs              experimental strategy enums and leaf estimation
src/preprocessing.rs          native fitted mixed-column schema and encoding
src/python.rs                 private PyO3/NumPy boundary
python/fastforest/__init__.py public Python API and array conversion
python/fastforest/preprocessing.py container adapter, display metadata, and aggregation
python/fastforest/sklearn.py       reproducible sklearn mixed-data benchmark preprocessors
python/fastforest/analysis.py NumPy analysis results, perturbations, clustering, and lazy plots
tests/test_fastforest.py      Python API narratives and validation
tools/bench.py                fit, OOB, and prediction timings
tools/accuracy.py             regression and classification accuracy/timing comparisons
```

The core depends on `ndarray` and `chrono` but not Python. PyO3 and rust-numpy are optional behind the `python` feature; maturin enables `extension-module`, which enables it. The Rust crate therefore remains directly usable as an `rlib`.

## Design

The Rust encoder fits the input schema and transforms inference data, parallelizing independent columns with Rayon. Every non-missing column is parsed numerically when possible and otherwise lexically ordered. Training features are contiguous `u32` ranks, dummy indicators, and missing indicators. Explicit `one_hot_groups` are checked for exactly one active binary indicator per row and collapsed to one lexical categorical column; explicit `date_columns` are parsed once and expanded to fastai-style calendar, time, boundary, and elapsed features. Both conversions remain inside the native fit and transform paths. The encoder retains the original-column mapping, type, integral-display flag, missing rule, median when required, and native cutoff boundaries. Python only validates container shape and names, selects buffers, and reconstructs small display metadata. Numeric NumPy matrices whose fitted columns require no missing markers bypass `RawColumn` entirely: PyO3 borrows a contiguous `float32` or `float64` matrix, and Rust transforms rows directly into the encoded `f32` output, including one-hot collapse. Other numeric dtypes make one contiguous `float32` conversion. Mixed inputs retain the column path, and Pandas categoricals cross the boundary as integer codes plus one vocabulary rather than per-row Python objects. PyO3 detaches from Python while Rust runs.

Tree construction uses `u32` feature ranks, candidate cutoffs, row indexes, and classification target IDs. Temporary 16-byte training nodes hold rank cutoffs, allowing OOB prediction directly against the encoded training matrix. Once OOB is complete, every split rank becomes the greatest native value on its left side. Final regression trees own flat vectors of 16-byte inference nodes containing an `f32` native cutoff or leaf value plus `u32` feature and child indexes. Classification uses the same 16-byte topology nodes with a leaf index into one contiguous `f32` probability buffer. A native value goes right when it is greater than the stored left boundary, exactly matching insertion-rank behavior for values unseen during training. Sibling children are adjacent, so the right child is `left + 1`.

Regression split scratch tracks target sums and squared sums. Classification scratch instead tracks class counts and the sum of their squares, so moving a row across a candidate boundary updates Gini in constant time rather than rescanning every class at every cutoff. Computing a terminal probability vector visits only the rows already sampled for that tree; summed across leaves, that is one capped pass per tree. The only whole-input work in ordinary fitting is schema transformation and target validation/encoding. Regression caps production trees at 40,000 rows and uses an 8,000-row adaptive pilot. Classification scales both budgets by its number of output classes, giving each output the same nominal 40,000-row production cap and 4,000 in-bag pilot rows. OOB alone intentionally evaluates rows outside each tree's sample.

The uniform `u32` training representation supports arbitrary practical cardinality without the bandwidth cost of `u64`; adaptive `u8`/`u16` columns are intentionally deferred until benchmarks justify their complexity. Candidate de-duplication packs the `u32` feature and cutoff into a `u64` key. `usize` is reserved for Rust indexing boundaries. A seeded RNG first generates one seed per tree, so Rayon can build trees in parallel without changing results.

Regression prediction is parallel over rows. Classification uses two-dimensional cache tiling without shared accumulators: parallel row blocks own their output elements, while an internal tree-batch size is calculated from the fitted forest. Mean bytes per tree include 16-byte topology nodes and `f32` leaf probabilities; as many trees as fit in a conservative 512 KiB budget are processed together. Row blocks target four tasks per Rayon worker and never exceed 4,096 rows. Small trees therefore preserve row-major locality, whereas a tree larger than the cache budget is traversed for every row in a block before moving to the next tree. This reduced full-Covertype 50-tree native probability inference from roughly 96 ms to 56 ms without slowing 10k- or 50k-row fits. Trials at 512 KiB, 1 MiB, and 2 MiB slightly favored the smallest budget at 50k and 100k training rows and were indistinguishable at 10k.

Tree construction accumulates normalized split-gain importance outside the compact inference nodes. Native analysis operations return per-tree predictions and encoded-feature path contributions efficiently without exposing the internal node representation; Python then sums derived features back into their original columns. A path contribution is the change in successive node values assigned to the feature selecting that child, so forest bias plus all contributions exactly reconstructs the prediction.

The higher analysis layer is deliberately NumPy-only. A shared feature resolver handles array indices, data-frame column names, and grouped features. Permutation and drop-column importance share its group representation; PDP/ICE reuse the same named data and prediction interface. Tie-aware Spearman ranks and average-linkage clustering are small local implementations rather than scipy dependencies. Result objects contain ordinary arrays and import matplotlib lazily only for their `plot` methods.

OOB is optional because it predicts every training row with every tree for which that row was not sampled. During tree construction a compact in-bag mask is retained, then discarded after forest-level OOB sums and counts have been calculated. Regression accumulates scalar predictions; classification accumulates probability vectors and reports OOB accuracy. The adaptive classifier compares its pilots with multiclass Brier loss rather than discontinuous accuracy.

Candidate cutoffs are deduplicated as exact `(feature, value)` pairs. The default permits two proposals per requested unique candidate; experiments showed that larger retry bounds disproportionately increase fit time on discrete features. A 20-row floor keeps candidate coverage from collapsing quadratically in small nodes. Reusable per-tree scratch storage keeps de-duplication overhead low.

Interchangeable tree-building choices use enum dispatch rather than trait objects in the node loop. `Workbench` owns orthogonal split-search, feature-selection, and leaf-estimation settings; `split::find_split` dispatches to a self-contained implementation sharing only the split result and scratch storage. Numeric value features and their missing indicators are atomic sampling groups; the broader `feature_sampling="columns"` experiment groups every encoding from each original column. The production histogram splitter evaluates at most 320 sampled ranks and uses 75% of feature units on datasets through 8,000 rows. Above that size, adaptive fitting compares 60% and 90% feature sampling using paired OOB pilots before fitting the production forest. The original random splitter remains available and retains its original RNG call order exactly when selected.

Categorical subset splits and learned missing routing belong at a separate boundary: they require encoder metadata and a richer stored split predicate, not another branch inside ordered cutoff scoring. Those experiments should extend the workbench at that encoder/predicate layer while retaining the same tree builder and leaf strategy.

## Testing

Tests favor a few complete narratives over many single-assertion tests. The main Rust test covers fitting, tree invariants, prediction quality, OOB, determinism, per-tree predictions, split importance, and exact explanation reconstruction. The regression Python narrative demonstrates array conversion, OOB, importance, uncertainty, explanations, PDP/ICE, correlation clustering, and nonlinear dependence. A multiclass narrative covers arbitrary labels, probabilities, OOB, determinism, adaptive selection, and importance. Distinct validation and numerical edge cases share one focused test in each layer.

```bash
cargo fmt
cargo check
cargo test
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

California Housing is the default. The other regression choices are `concrete`, `sgemm`, `diamonds`, `allstate`, `diabetes`, `bluebook`, `bluebook_raw`, `walmart`, `walmart_raw`, and `walmart_nodate`. Blue Book uses the course's final-12,000 validation split and log-price target. Walmart uses its final 12 weeks for validation; its three variants compare native date expansion, the original string, and removing the date. Classification choices are `covertype`, `covertype_grouped`, `adult`, and `bank`; the two Covertype variants compare its supplied 54 columns with native one-hot grouping. Other datasets use one reproducible 80/20 split, stratified for classification. Mixed-data sklearn RF comparisons use `sklearn_preprocessor`: one-hot encoding through 20 levels and target encoding above 20. HistGBM uses its documented native categorical support, target-encoding only features beyond its 255-category limit.

Each model/dataset combination runs in a spawned process and has a 180-second default timeout. The child signals readiness after process startup and input deserialization; only then does the timeout begin. Reported fit time includes model and preprocessor construction, schema inspection, and `fit`, but excludes process startup and IPC.

The sklearn random forest uses all available cores, as do FastForest and histogram GBM internally.
For focused FastForest tuning, add `--ff_only` and vary `--min_node_size`, `--bootstrap_fraction`, `--bootstrap_max`, `--replacement`, `--max_node_samples`, or `--cutoff_divisor`. Pass `--no-adaptive` when directly comparing fixed split-search settings. Experimental comparisons additionally accept `--splitter`, `--max_features`, and `--leaf_regularization`. The tools use `call_parse`, so CLI names match their underscored function parameters.

Run a reproducible parameter grid (SGEMM by default) and save its metrics and timings with:

```bash
python tools/sweep.py
```

`tools/sweep.py` accepts comma-separated `--splitters`, `--max_features`, and `--leaf_regularizations` and records every strategy axis in its CSV. This is the preferred workbench for factorial experiments; selected configurations can then be compared against sklearn with the singular forms accepted by `tools/accuracy.py`.

## Versioning and release

The canonical version lives in `Cargo.toml`; `pyproject.toml` gets it through `dynamic = ["version"]`.

Once the repository has been created and added to the workspace, release flow is:

1. Run the Rust and Python tests against a release build.
2. Confirm the release version in `Cargo.toml`.
3. Run `ship-release`.

GitHub Actions builds Linux and macOS wheels plus an sdist, publishes tagged builds, and creates the GitHub release.
