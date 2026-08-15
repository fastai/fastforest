# fastforest

Fast approximate random-forest regression in Rust, with Python bindings.

Across six numeric and mixed-data regressions, fastforest is fastest to fit and predict in every completed comparison while retaining competitive accuracy. Both forests use 50 trees and otherwise use their default hyperparameters:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Model</th>
      <th align="right">RMSE ↓</th>
      <th align="right">R² ↑</th>
      <th align="right">Fit (s) ↓</th>
      <th align="right">Predict (s) ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong><a href="https://archive.ics.uci.edu/dataset/440/sgemm+gpu+kernel+performance">SGEMM GPU</a></strong><br><sub>241,600 rows · 14 features<br>80/20 split</sub></td>
      <td><strong>fastforest</strong></td>
      <td align="right">0.05</td>
      <td align="right"><strong>1.00</strong></td>
      <td align="right"><strong>0.17</strong></td>
      <td align="right"><strong>0.021</strong></td>
    </tr>
    <tr>
      <td>sklearn RF</td>
      <td align="right"><strong>0.03</strong></td>
      <td align="right"><strong>1.00</strong></td>
      <td align="right">1.03</td>
      <td align="right">0.073</td>
    </tr>
    <tr>
      <td>sklearn HistGBM</td>
      <td align="right">0.20</td>
      <td align="right">0.97</td>
      <td align="right">1.23</td>
      <td align="right">0.022</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset">California Housing</a></strong><br><sub>20,640 rows · 8 features<br>80/20 split</sub></td>
      <td><strong>fastforest</strong></td>
      <td align="right">0.49</td>
      <td align="right">0.81</td>
      <td align="right"><strong>0.08</strong></td>
      <td align="right"><strong>0.003</strong></td>
    </tr>
    <tr>
      <td>sklearn RF</td>
      <td align="right">0.51</td>
      <td align="right">0.80</td>
      <td align="right">0.26</td>
      <td align="right">0.013</td>
    </tr>
    <tr>
      <td>sklearn HistGBM</td>
      <td align="right"><strong>0.47</strong></td>
      <td align="right"><strong>0.83</strong></td>
      <td align="right">0.96</td>
      <td align="right">0.006</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength">Concrete Strength</a></strong><br><sub>1,030 rows · 8 features<br>80/20 split</sub></td>
      <td><strong>fastforest</strong></td>
      <td align="right">5.41</td>
      <td align="right">0.89</td>
      <td align="right"><strong>0.00</strong></td>
      <td align="right"><strong>0.000</strong></td>
    </tr>
    <tr>
      <td>sklearn RF</td>
      <td align="right">5.52</td>
      <td align="right">0.88</td>
      <td align="right">0.05</td>
      <td align="right">0.014</td>
    </tr>
    <tr>
      <td>sklearn HistGBM</td>
      <td align="right"><strong>4.65</strong></td>
      <td align="right"><strong>0.92</strong></td>
      <td align="right">0.78</td>
      <td align="right">0.005</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://www.openml.org/d/42225">Diamonds</a></strong><br><sub>53,940 rows · 9 features<br>80/20 split</sub></td>
      <td><strong>fastforest</strong></td>
      <td align="right">551</td>
      <td align="right">0.98</td>
      <td align="right"><strong>0.17</strong></td>
      <td align="right"><strong>0.008</strong></td>
    </tr>
    <tr>
      <td>sklearn RF</td>
      <td align="right">553</td>
      <td align="right">0.98</td>
      <td align="right">0.53</td>
      <td align="right">0.020</td>
    </tr>
    <tr>
      <td>sklearn HistGBM</td>
      <td align="right"><strong>541</strong></td>
      <td align="right"><strong>0.98</strong></td>
      <td align="right">1.02</td>
      <td align="right">0.018</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://www.openml.org/d/42571">Allstate Claims</a></strong><br><sub>188,318 rows · 130 features<br>80/20 split</sub></td>
      <td><strong>fastforest</strong></td>
      <td align="right">1,937</td>
      <td align="right">0.54</td>
      <td align="right"><strong>1.09</strong></td>
      <td align="right"><strong>0.055</strong></td>
    </tr>
    <tr>
      <td>sklearn RF</td>
      <td colspan="4" align="center">timed out at 180s</td>
    </tr>
    <tr>
      <td>sklearn HistGBM</td>
      <td align="right"><strong>1,861</strong></td>
      <td align="right"><strong>0.58</strong></td>
      <td align="right">3.75</td>
      <td align="right">0.389</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008">Diabetes 130-US Hospitals</a></strong><br><sub>101,766 rows · 46 features<br>80/20 split</sub></td>
      <td><strong>fastforest</strong></td>
      <td align="right">2.22</td>
      <td align="right">0.43</td>
      <td align="right"><strong>0.73</strong></td>
      <td align="right"><strong>0.088</strong></td>
    </tr>
    <tr>
      <td>sklearn RF</td>
      <td align="right">2.21</td>
      <td align="right">0.44</td>
      <td align="right">2.98</td>
      <td align="right">0.132</td>
    </tr>
    <tr>
      <td>sklearn HistGBM</td>
      <td align="right"><strong>2.13</strong></td>
      <td align="right"><strong>0.48</strong></td>
      <td align="right">2.05</td>
      <td align="right">0.141</td>
    </tr>
  </tbody>
</table>

<sub>Bold is best for that dataset and metric. Each dataset uses one fixed 80/20 split on a 16-core Apple M4 Max. Both forests use 50 trees; HistGBM and all other hyperparameters use library defaults. Fastforest's adaptive default selected 90% of features for SGEMM and Diamonds and 60% for California Housing, Allstate, and Diabetes; Concrete retained the 75% fallback because its training split has fewer than 8,000 rows. Fastforest fit time includes this pilot. Mixed-data sklearn RF preprocessing uses median imputation plus missing indicators for numeric columns, one-hot encoding through 20 levels, and target encoding above 20. HistGBM uses native categoricals through its 255-level limit and target encoding above it. Fit includes schema inspection and preprocessing but excludes process startup and IPC. Each model/dataset combination has a 180-second limit; sklearn RF timed out on Allstate. Fixed seeds make comparisons reproducible, and the SGEMM target is log-transformed mean runtime. Reproduce with <a href="tools/accuracy.py"><code>tools/accuracy.py</code></a>.</sub>

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

`X` may contain numeric values, numeric strings, ordinary strings, and configured missing values. `y` is converted to contiguous `float32`; it must have one finite value per row.

## Data preparation

FastForest fits a deterministic schema for every input column:

1. Non-missing values are parsed as `float32` when every value can be parsed and are otherwise treated as strings. Numeric columns sort numerically and other columns sort lexically. Numeric columns whose values are all integral retain that metadata so analysis displays them with no decimal places.
2. A column with more than `max_dummy_cardinality` distinct values is replaced during training by its zero-based rank in that sort order. A column with cardinality `c <= max_dummy_cardinality` becomes `c-1` boolean dummy features; the least-common value is omitted as the all-zero case. Frequency ties are resolved deterministically by sort order. `max_dummy_cardinality` defaults to 4.
3. The default missing value is the empty value. Override it per column with `missing_values`, using column names or indexes. When training contains a missing value, FastForest adds `<column>_missing` and fills the value feature with the observed median. A column containing no training missing values rejects missing values during prediction rather than silently inventing an imputation rule. Entirely missing columns are discarded.

```python
X = np.array([
    ["18", "red",   ""],
    ["42", "blue",  "3.5"],
    ["31", "green", "2.0"],
], dtype=object)

model = FastForest(missing_values={2: ""}).fit(X, [1, 4, 3])
```

Ranking is a compact training representation, not a prediction-time requirement for numeric columns. After fitting, rank cutoffs are converted back to native numeric boundaries, so seen and unseen numeric values are compared directly without a rank lookup. Nonnumeric values are mapped through their fitted lexical ordering. An unseen low-cardinality value naturally receives all-zero dummies; an unseen high-cardinality value receives its insertion rank. Missing checks and median routing are retained only for columns that contained missing training values.

Schema fitting and inference transformation run natively in Rust and parallelize independent columns with Rayon. Python only adapts NumPy and data-frame column buffers and retains display metadata for the analysis API. Pandas categorical columns pass their integer codes and vocabulary directly rather than being expanded into Python object arrays.

Generated ranks, dummies, and missing indicators remain internal. Feature importance, explanations, and partial-dependence results aggregate them back to the original column and display its original values. Fitted interpretations are available in `model.column_info_`.

For reproducible sklearn comparisons on the same raw dataframe, `sklearn_preprocessor` implements the policy used by the benchmark: numeric median imputation with missing indicators, one-hot encoding through 20 categorical levels, target encoding above 20, and removal of empty columns.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from fastforest import sklearn_preprocessor

preprocess = sklearn_preprocessor(X_train, missing_values={"age":"?"})
model = make_pipeline(preprocess, RandomForestRegressor(n_estimators=50, n_jobs=-1))
model.fit(X_train, y_train)
```

Install the optional dependencies with `pip install 'fastforest[sklearn]'`.

## Algorithm

Each tree draws `min(floor(bootstrap_fraction * n_rows), bootstrap_max)` training rows, with replacement when `replacement=True` and otherwise without it. When `bootstrap_fraction=None`, it resolves to 0.8 with OOB enabled and 1 otherwise. Fractions above 1 are supported with replacement; without replacement the maximum is 1. Pass `bootstrap_max=None` to disable the cap. At each node, the default histogram splitter:

1. A node with fewer than `min_node_size` rows, or whose first `max_node_samples` sampled targets are equal, becomes a leaf.
2. A random contiguous window containing at most `max_node_samples` of the node's shuffled rows is selected.
3. The tree randomly selects `floor(max_features * n_features)` feature units, with a minimum of one. Encoded features are independent units except that every numeric value feature and its missingness indicator form one atomic unit.
4. For each selected feature, the sampled rows are sorted by their encoded rank and every distinct observed boundary is evaluated. The split that most improves size-weighted negative sample standard deviation is selected, subject to the sampled child-size minimum.
5. Every terminal leaf predicts the mean target of all training rows that reached it, including leaves where candidate evaluation found no useful split.

The defaults are 50 trees, minimum node size 4, all rows capped at 40,000 without replacement, histogram splitting over 75% of feature units, at most 320 evaluated rows per node, and unregularized leaf means. Enabling OOB changes the default sampling fraction to 0.8 so every row can receive held-out predictions. Preprocessing and trees build in parallel over columns and trees respectively, while batch predictions run in parallel over rows. Supplying `seed` makes the fitted forest deterministic regardless of parallel scheduling.

For more than 8,000 training rows, `adaptive=True` selects a feature fraction of 0.6 or 0.9 while keeping `max_node_samples=320`. It draws one fixed 8,000-row pilot sample and fits `max(2*n_threads, 32)` trees per candidate, sampling 50% of the pilot rows without replacement in each tree. Candidates use matching seeds and in-bag rows and are compared by mean squared OOB error; an exact tie favors 0.6. The selected pair is available as `adaptive_choice_`, and both `(max_features, max_node_samples, oob_mse)` results are in `adaptive_scores_`. Set `adaptive=False` to use `workbench.max_features` directly.

## Tree-building workbench

`Workbench` keeps interchangeable tree-building choices separate from the forest parameters. Its defaults reproduce the histogram algorithm above; the original random-cutoff search remains available for experiments and comparisons:

```python
from fastforest import FastForest,Workbench

alternate = Workbench(
    splitter="random",
    max_features="sqrt",
    leaf_regularization=0,
)
model = FastForest(workbench=alternate, seed=42).fit(X, y)
```

`splitter="histogram"` is the production default. It randomly selects `max_features`, builds sparse target-statistic histograms from the node evaluation window, and checks every observed boundary for those features. `splitter="random"` proposes random `(feature, value)` cutoffs, deduplicates them, and evaluates them on the same kind of node window. Its candidate count is controlled by `min_candidate_rows`, `candidate_attempt_factor`, and `cutoff_divisor`. `max_features` accepts `"sqrt"`, `"all"`, a fraction in `(0, 1]`, or a positive feature count and is ignored by the random splitter.

`leaf_regularization` shrinks each terminal full-node mean towards its parent node mean, treating the value as a number of parent pseudo-rows. Zero selects the unregularized production mean. Split search and leaf regularization are independent, so every splitter, feature selection, and regularization combination can be tested without adding branches to the forest API.

The focused sweep tool takes comma-separated workbench grids. Random splitting ignores the histogram-only `max_features` grid rather than running duplicate configurations:

```bash
python tools/sweep.py --dataset california \
  --splitters random,histogram --max_features sqrt,0.5,all \
  --leaf_regularizations 0,2,8,32
```

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

Compare accuracy and timings against sklearn's random forest and histogram GBM on one fixed California Housing split:

```bash
python tools/accuracy.py
```

Use `--dataset concrete` for the smaller Concrete Compressive Strength regression dataset, or `--dataset sgemm` for the 241,600-row SGEMM GPU Kernel Performance dataset. Every dataset uses one reproducible 80/20 split. Each model/dataset combination runs in an isolated process with a three-minute timeout; process startup and input transfer are excluded from reported timings.

Use `--ff_only` with `--min_node_size`, `--bootstrap_fraction`, `--bootstrap_max`, `--replacement`, `--max_node_samples`, and `--cutoff_divisor` for focused FastForest experiments. These spellings come directly from the `call_parse` function parameters.
