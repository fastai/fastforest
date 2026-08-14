# Development

`fastforest` is a Rust library with a private PyO3 extension and a small public Python façade.

## Layout

```text
src/forest.rs                 forest algorithm and Rust tests
src/python.rs                 private PyO3/NumPy boundary
python/fastforest/__init__.py public Python API and array conversion
python/fastforest/analysis.py NumPy analysis results, perturbations, clustering, and lazy plots
tests/test_fastforest.py      Python API narratives and validation
tools/bench.py                fit, OOB, and prediction timings
tools/accuracy.py             tabular regression accuracy and timing comparisons
```

The core depends on `ndarray` but not Python. PyO3 and rust-numpy are optional behind the `python` feature; maturin enables `extension-module`, which enables it. The Rust crate therefore remains directly usable as an `rlib`.

## Design

Training and prediction borrow `ndarray` views. The Python façade converts inputs to contiguous `float32`, rust-numpy borrows those arrays without another copy, and PyO3 detaches from Python while Rust runs.

Trees own flat vectors of 16-byte inference nodes. Each node stores a feature-or-leaf sentinel, cutoff or leaf value, and first-child index; sibling children are adjacent, so the right child is `left + 1`. Training rows are indices into the borrowed input matrix; no row arrays or parent pointers require manual allocation. A seeded RNG first generates one seed per tree, so Rayon can build trees in parallel without changing results. Prediction is parallel over rows; each worker evaluates all trees for its rows and writes directly to its own output elements, requiring no shared accumulator or reduction buffer.

Tree construction accumulates normalized split-gain importance outside the compact inference nodes. Native analysis operations return per-tree predictions and path contributions efficiently without exposing the internal node representation. A path contribution is the change in successive node values assigned to the feature selecting that child, so forest bias plus all contributions exactly reconstructs the prediction.

The higher analysis layer is deliberately NumPy-only. A shared feature resolver handles array indices, data-frame column names, and grouped features. Permutation and drop-column importance share its group representation; PDP/ICE reuse the same named data and prediction interface. Tie-aware Spearman ranks and average-linkage clustering are small local implementations rather than scipy dependencies. Result objects contain ordinary arrays and import matplotlib lazily only for their `plot` methods.

OOB is optional because it predicts every training row with every tree for which that row was not sampled. During tree construction a compact in-bag mask is retained, then discarded after forest-level OOB sums and counts have been calculated.

Candidate cutoffs are deliberately still sampled with replacement. Candidate de-duplication belongs in a subsequent isolated change so its fit-time and model-quality effects can be measured with `tools/bench.py`.

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

For an accuracy and timing comparison using identical five-fold dataset splits:

```bash
python tools/accuracy.py
```

California Housing is the default; pass `--dataset concrete` for Concrete Compressive Strength or `--dataset sgemm` for SGEMM GPU Kernel Performance. The small datasets default to five folds, while SGEMM uses one reproducible 80/20 split to keep iteration quick. Pass `--folds` to override this.

The sklearn random forest uses all available cores, as do FastForest and histogram GBM internally; folds run sequentially to avoid nested parallelism.
For focused FastForest tuning, add `--ff_only` and vary `--min_node_size`, `--bootstrap_fraction`, `--bootstrap_max`, `--replacement`, `--max_node_samples`, or `--cutoff_divisor`. The tools use `call_parse`, so CLI names match their underscored function parameters.

Run a reproducible SGEMM parameter grid and save its metrics and timings with:

```bash
python tools/sweep.py
```

## Versioning and release

The canonical version lives in `Cargo.toml`; `pyproject.toml` gets it through `dynamic = ["version"]`.

Once the repository has been created and added to the workspace, release flow is:

1. Run the Rust and Python tests against a release build.
2. Confirm the release version in `Cargo.toml`.
3. Run `ship-release`.

GitHub Actions builds Linux and macOS wheels plus an sdist, publishes tagged builds, and creates the GitHub release.
