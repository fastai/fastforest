# Development

`fastforest` is a Rust library with a private PyO3 extension and a small public Python façade.

## Layout

```text
src/forest.rs                 forest algorithm and Rust tests
src/split.rs                  interchangeable split searches and their reusable scratch
src/workbench.rs              experimental strategy enums and leaf estimation
src/preprocessing.rs          native fitted mixed-column schema and encoding
src/python.rs                 private PyO3/NumPy boundary
python/fastforest/__init__.py public Python API and array conversion
python/fastforest/preprocessing.py container adapter, display metadata, and aggregation
python/fastforest/sklearn.py       reproducible sklearn mixed-data benchmark preprocessors
python/fastforest/analysis.py NumPy analysis results, perturbations, clustering, and lazy plots
tests/test_fastforest.py      Python API narratives and validation
tools/bench.py                fit, OOB, and prediction timings
tools/accuracy.py             tabular regression accuracy and timing comparisons
```

The core depends on `ndarray` but not Python. PyO3 and rust-numpy are optional behind the `python` feature; maturin enables `extension-module`, which enables it. The Rust crate therefore remains directly usable as an `rlib`.

## Design

The Rust encoder fits the input schema and transforms inference data, parallelizing independent columns with Rayon. Every non-missing column is parsed numerically when possible and otherwise lexically ordered. Training features are contiguous `u32` ranks, dummy indicators, and missing indicators. The encoder retains the original-column mapping, type, integral-display flag, missing rule, median when required, and native cutoff boundaries. Python only validates container shape and names, selects column buffers, and reconstructs small display metadata. Pandas categoricals cross the boundary as integer codes plus one vocabulary rather than per-row Python objects. Rust-numpy borrows the generated arrays without another copy, and PyO3 detaches from Python while Rust runs.

Tree construction uses `u32` feature ranks, candidate cutoffs, and row indexes. Temporary 16-byte training nodes hold rank cutoffs, allowing OOB prediction directly against the encoded training matrix. Once OOB is complete, every split rank becomes the greatest native value on its left side. Final trees own flat vectors of 16-byte inference nodes containing an `f32` native cutoff or leaf value plus `u32` feature and child indexes. A native value goes right when it is greater than the stored left boundary, exactly matching insertion-rank behavior for values unseen during training. Sibling children are adjacent, so the right child is `left + 1`.

The uniform `u32` training representation supports arbitrary practical cardinality without the bandwidth cost of `u64`; adaptive `u8`/`u16` columns are intentionally deferred until benchmarks justify their complexity. Candidate de-duplication packs the `u32` feature and cutoff into a `u64` key. `usize` is reserved for Rust indexing boundaries. A seeded RNG first generates one seed per tree, so Rayon can build trees in parallel without changing results. Prediction is parallel over rows; each worker evaluates all trees for its rows and writes directly to its own output elements, requiring no shared accumulator or reduction buffer.

Tree construction accumulates normalized split-gain importance outside the compact inference nodes. Native analysis operations return per-tree predictions and encoded-feature path contributions efficiently without exposing the internal node representation; Python then sums derived features back into their original columns. A path contribution is the change in successive node values assigned to the feature selecting that child, so forest bias plus all contributions exactly reconstructs the prediction.

The higher analysis layer is deliberately NumPy-only. A shared feature resolver handles array indices, data-frame column names, and grouped features. Permutation and drop-column importance share its group representation; PDP/ICE reuse the same named data and prediction interface. Tie-aware Spearman ranks and average-linkage clustering are small local implementations rather than scipy dependencies. Result objects contain ordinary arrays and import matplotlib lazily only for their `plot` methods.

OOB is optional because it predicts every training row with every tree for which that row was not sampled. During tree construction a compact in-bag mask is retained, then discarded after forest-level OOB sums and counts have been calculated.

Candidate cutoffs are deduplicated as exact `(feature, value)` pairs. The default permits two proposals per requested unique candidate; experiments showed that larger retry bounds disproportionately increase fit time on discrete features. A 20-row floor keeps candidate coverage from collapsing quadratically in small nodes. Reusable per-tree scratch storage keeps de-duplication overhead low.

Interchangeable tree-building choices use enum dispatch rather than trait objects in the node loop. `Workbench` owns orthogonal split-search, feature-selection, and leaf-estimation settings; `split::find_split` dispatches to a self-contained implementation sharing only the split result and scratch storage. Numeric value features and their missing indicators are atomic sampling groups; the broader `feature_sampling="columns"` experiment groups every encoding from each original column. The production histogram splitter evaluates at most 320 sampled ranks and uses 75% of feature units on datasets through 8,000 rows. Above that size, adaptive fitting compares 60% and 90% feature sampling using paired OOB pilots before fitting the production forest. The original random splitter remains available and retains its original RNG call order exactly when selected.

Categorical subset splits and learned missing routing belong at a separate boundary: they require encoder metadata and a richer stored split predicate, not another branch inside ordered cutoff scoring. Those experiments should extend the workbench at that encoder/predicate layer while retaining the same tree builder and leaf strategy.

## Testing

Tests favor a few complete narratives over many single-assertion tests. The main Rust test covers fitting, tree invariants, prediction quality, OOB, determinism, per-tree predictions, split importance, and exact explanation reconstruction. The main Python test demonstrates the public workflow including array conversion, OOB, importance, uncertainty, explanations, PDP/ICE, correlation clustering, and nonlinear dependence. Distinct validation and numerical edge cases share one focused test in each layer.

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

California Housing is the default. The other choices are `concrete`, `sgemm`, `diamonds`, `allstate`, and `diabetes`. Every dataset uses one reproducible 80/20 split. Mixed-data sklearn RF comparisons use `sklearn_preprocessor`: one-hot encoding through 20 levels and target encoding above 20. HistGBM uses its documented native categorical support, target-encoding only features beyond its 255-category limit.

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
