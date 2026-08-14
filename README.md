# fastforest

Fast approximate random-forest regression in Rust, with Python bindings.

## Install

```bash
pip install fastforest
```

## Usage

```python
import numpy as np
from fastforest import FastForest

rng = np.random.default_rng(42)
X = rng.random((1_000, 6))
y = 4*X[:, 0] - 2*X[:, 1] + X[:, 5]

model = FastForest(seed=42, oob=True).fit(X, y)
predictions = model.predict(X[:5])
oob_predictions = model.oob_prediction_
oob_counts = model.oob_counts_
```

`X` and `y` may be array-like and are converted to contiguous `float32` arrays. `X` must have shape `(rows, features)`, `y` must have one value per row, and all values must be finite.

## Algorithm

Each tree draws `min(floor(bootstrap_fraction * n_rows), bootstrap_max)` training rows, with replacement when `replacement=True` and otherwise without it. Pass `bootstrap_max=None` to disable the cap. At each node:

1. A node with fewer than `min_node_size` rows, or whose first `max_node_samples` sampled targets are equal, becomes a leaf.
2. The tree draws `max(4, floor(min(node_rows, max_node_samples) * sqrt(n_features) / cutoff_divisor))` candidate `(feature, value)` cutoffs with replacement.
3. Candidates are evaluated on a random contiguous window containing at most `max_node_samples` of the node's shuffled rows.
4. The split that most improves size-weighted negative sample standard deviation is selected, subject to the sampled child-size minimum.
5. A leaf stopped before candidate evaluation predicts its full-node target mean. A node stopped after candidate evaluation predicts the evaluation-window mean.

The defaults are 100 trees, minimum node size 4, bootstrap fraction 0.8 capped at 40,000 rows without replacement, at most 160 evaluated rows per node, and cutoff divisor 3. Trees build in parallel, and batch predictions run in parallel over rows. Supplying `seed` makes the fitted forest deterministic regardless of parallel scheduling.

## Out-of-bag predictions

OOB calculation is opt-in with `oob=True`. After fitting:

- `oob_prediction_` contains each training row's mean prediction from trees that did not sample that row.
- `oob_counts_` contains the number of contributing trees.
- A row with no contributing tree has count zero and prediction `NaN`.
- Sampling without replacement at `bootstrap_fraction=1.0` leaves no OOB rows, so all counts are zero and predictions are `NaN`.

Both attributes are `None` when OOB is disabled.

## Model analysis

FastForest includes NumPy-only analysis tools. Data frames are accepted and supply feature names automatically; arrays use `x0`, `x1`, and so on. Plot methods import matplotlib only when called.

### Importance

Use validation-set permutation importance by default. It measures the drop in model score after shuffling a feature without retraining:

```python
importance = model.feature_importance(X_valid, y_valid)
importance.sorted()
importance.plot()
```

Correlated features can substitute for one another and therefore look individually unimportant. Permute them together to measure their joint importance:

```python
importance = model.feature_importance(X_valid, y_valid,
    features={"location": ["latitude", "longitude"]})
```

`model.drop_column_importance(X_train, y_train, X_valid, y_valid)` performs the slower complementary analysis: it refits the forest without each feature. It accepts the same `features` groups. `model.split_importance()` returns the nearly free, normalized training-time split-gain measure, but permutation or grouped permutation is preferable because split importance is biased by the available cutoffs and correlated predictors.

### Individual predictions and uncertainty

```python
explanation = model.explain(X_valid[:3])
explanation.row(0)   # (feature, observed value, contribution), strongest first
explanation.plot(0)

tree_predictions = model.predict_trees(X_valid)
prediction_std = model.predict_std(X_valid)
```

For every row, `prediction = bias + contributions.sum()`. Contributions telescope through each tree's decision path and are then averaged across trees. They explain this forest's computation, not causality; correlated features can redistribute contributions between themselves.

### Partial dependence and ICE

```python
year = model.partial_dependence(X_train, "year_made")
year.plot()                   # average PDP plus individual conditional-expectation lines
year.plot(centered=True)
year.plot(clusters=5)         # representative centered ICE curves

interaction = model.partial_dependence(X_train, ["year_made", "sale_year"])
interaction.plot()

enclosure = model.partial_dependence(X_train,
    {"enclosure": ["enclosure_ac", "enclosure_erops", "enclosure_orops"]})
```

Partial dependence repeatedly replaces the selected feature values and averages the resulting predictions. ICE retains the individual prediction lines. These plots describe the fitted model rather than a causal intervention, and highly correlated features can produce unrealistic synthetic rows.

### Collinearity and redundancy

```python
from fastforest import feature_dependence,feature_relations

relations = feature_relations(X_train)
relations.groups(threshold=0.2)
relations.plot()
relations.plot_dendrogram()

dependence = feature_dependence(X_train)
dependence.predictability     # validation R² for predicting each feature from the others
dependence.plot()             # which other features provide that predictive information
```

`feature_relations` uses tie-aware Spearman correlation and average linkage implemented directly with NumPy. `feature_dependence` detects nonlinear redundancy by treating each feature in turn as a target, fitting a small forest from the remaining features, and measuring grouped prediction and permutation dependence.

## Development

The project is locally installed with maturin until it joins the aai-ws workspace:

```bash
maturin develop
cargo test
pytest -q
```

For performance work, build the extension in release mode and run the benchmark:

```bash
maturin develop --release
python tools/bench.py
```

Compare five-fold accuracy and timings against sklearn's random forest and histogram GBM on California Housing:

```bash
python tools/accuracy.py
```

Use `--dataset concrete` for the smaller Concrete Compressive Strength regression dataset, or `--dataset sgemm` for the 241,600-row SGEMM GPU Kernel Performance dataset. The small datasets default to five-fold CV; SGEMM defaults to one 80/20 split. Override either with `--folds`.

Use `--ff_only` with `--min_node_size`, `--bootstrap_fraction`, `--bootstrap_max`, `--replacement`, `--max_node_samples`, and `--cutoff_divisor` for focused FastForest experiments. These spellings come directly from the `call_parse` function parameters.
