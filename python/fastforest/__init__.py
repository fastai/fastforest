import numpy as np
from dataclasses import dataclass

from ._core import Forest as _Forest
from ._core import __version__
from .preprocessing import ColumnInfo,_Encoder
from .sklearn import sklearn_hist_preprocessor,sklearn_preprocessor

__all__ = ["FastForest", "Workbench", "ColumnInfo", "Importance", "Explanation", "PartialDependence", "FeatureRelations", "FeatureDependence",
    "permutation_importance", "drop_column_importance", "partial_dependence", "feature_relations", "feature_dependence",
    "sklearn_preprocessor", "sklearn_hist_preprocessor", "__version__"]

def _vector(x, name="y"):
    "Convert `x` to a contiguous one-dimensional float32 array."
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1: raise ValueError(f"{name} must be a one-dimensional array")
    return np.ascontiguousarray(x)

@dataclass(frozen=True)
class Workbench:
    "Composable experimental tree-building strategies."
    splitter: str = "histogram"
    max_features: object = 0.75
    leaf_regularization: float = 0.0
    feature_sampling: str = "encoded"

    def __post_init__(self):
        if self.splitter not in ("random", "histogram"): raise ValueError("splitter must be 'random' or 'histogram'")
        value = self.max_features
        valid_named = value in ("sqrt", "all") if isinstance(value, str) else False
        valid_count = isinstance(value, int) and not isinstance(value, bool) and value > 0
        valid_fraction = isinstance(value, float) and np.isfinite(value) and 0 < value <= 1
        if not (valid_named or valid_count or valid_fraction):
            raise ValueError("max_features must be 'sqrt', 'all', a positive integer, or a float in (0, 1]")
        if not np.isfinite(self.leaf_regularization) or self.leaf_regularization < 0:
            raise ValueError("leaf_regularization must be finite and non-negative")
        if self.feature_sampling not in ("encoded", "columns"): raise ValueError("feature_sampling must be 'encoded' or 'columns'")

    def _native(self):
        splitter = {"random":0, "histogram":1}[self.splitter]
        sampling = {"encoded":0, "columns":1}[self.feature_sampling]
        value = self.max_features
        if value == "sqrt": return splitter,0,0.0,float(self.leaf_regularization),sampling
        if value == "all": return splitter,1,0.0,float(self.leaf_regularization),sampling
        if isinstance(value, float): return splitter,2,value,float(self.leaf_regularization),sampling
        return splitter,3,float(value),float(self.leaf_regularization),sampling

class FastForest:
    "A fast approximate random-forest regressor."
    def __init__(self, n_trees=50, min_node_size=4, bootstrap_fraction=None, bootstrap_max=40_000, replacement=False,
        max_node_samples=320, min_candidate_rows=20, candidate_attempt_factor=2, cutoff_divisor=3.0, seed=None, oob=False, adaptive=True,
        missing_values=None, max_dummy_cardinality=4, workbench=None):
        self.n_trees,self.min_node_size,self.bootstrap_fraction = n_trees,min_node_size,bootstrap_fraction
        self.bootstrap_max,self.replacement = bootstrap_max,replacement
        self.max_node_samples,self.min_candidate_rows = max_node_samples,min_candidate_rows
        self.candidate_attempt_factor,self.cutoff_divisor = candidate_attempt_factor,cutoff_divisor
        self.seed,self.oob,self.adaptive,self.missing_values = seed,oob,adaptive,missing_values
        self.max_dummy_cardinality = max_dummy_cardinality
        self.workbench = Workbench() if workbench is None else workbench
        if not isinstance(self.workbench, Workbench): raise TypeError("workbench must be a Workbench")
        self._model = None

    def fit(self, X, y):
        "Fit the forest to feature rows `X` and regression targets `y`."
        y = _vector(y)
        self._encoder = _Encoder(self.missing_values, self.max_dummy_cardinality)
        X = self._encoder.fit_transform(X)
        tree_args = (self.n_trees, self.min_node_size, self.bootstrap_fraction, self.bootstrap_max, self.replacement)
        split_args = (self.max_node_samples, self.min_candidate_rows, self.candidate_attempt_factor, self.cutoff_divisor)
        group_ids = self._encoder.encoded_to_raw if self.workbench.feature_sampling == "columns" else self._encoder.feature_group_ids
        self._model = _Forest.fit(X, y, self._encoder.cutoff_values, self._encoder.cutoff_offsets, group_ids,
            *tree_args, *split_args, self.seed, self.oob, self.adaptive, *self.workbench._native())
        self.n_features_in_ = len(self._encoder.names)
        self.feature_names_in_ = self._encoder.names
        self.column_info_ = self._encoder.column_info
        self.feature_importances_ = self._encoder.aggregate(self._model.feature_importances)
        self.oob_prediction_ = self._model.oob_prediction
        self.oob_counts_ = self._model.oob_counts
        self.adaptive_scores_ = tuple(self._model.adaptive_scores)
        self.adaptive_choice_ = self._model.adaptive_choice
        return self

    def predict(self, X):
        "Predict regression targets for feature rows `X`."
        if self._model is None: raise RuntimeError("FastForest must be fitted before prediction")
        return self._model.predict(self._encoder.transform(X))

    def predict_trees(self, X):
        "Return one prediction per `(row, tree)`."
        if self._model is None: raise RuntimeError("FastForest must be fitted before prediction")
        return self._model.predict_trees(self._encoder.transform(X))

    def predict_std(self, X):
        "Return the standard deviation of the trees' predictions for each row."
        return self.predict_trees(X).std(axis=1)

    def explain(self, X, feature_names=None):
        "Decompose predictions into the forest bias and additive feature contributions."
        if self._model is None: raise RuntimeError("FastForest must be fitted before explanation")
        native = self._encoder.transform(X)
        prediction,bias,contributions = self._model.explain(native)
        contributions = self._encoder.aggregate(contributions)
        values = self._encoder.display(X)
        names = tuple(str(name) for name in feature_names) if feature_names is not None else self.feature_names_in_
        if len(names) != self.n_features_in_: raise ValueError("feature_names must have one name per column")
        return Explanation(prediction, bias, contributions, values, names)

    def split_importance(self, feature_names=None):
        "Return fast split-gain importance; prefer permutation importance for reliable analysis."
        if self._model is None: raise RuntimeError("FastForest must be fitted before importance analysis")
        names = tuple(str(name) for name in feature_names) if feature_names is not None else self.feature_names_in_
        values = np.asarray(self.feature_importances_)
        return Importance(names, values, np.zeros_like(values), np.nan, "split-gain")

    def feature_importance(self, X=None, y=None, method="permutation", **kwargs):
        "Measure feature importance, using reliable validation-set permutation by default."
        if method == "split": return self.split_importance(kwargs.pop("feature_names", None))
        if method != "permutation": raise ValueError("method must be 'permutation' or 'split'")
        if X is None or y is None: raise ValueError("permutation importance requires X and y")
        kwargs.setdefault("feature_names", self.feature_names_in_)
        return permutation_importance(self, X, y, **kwargs)

    def drop_column_importance(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        "Measure importance by refitting without each feature or feature group."
        kwargs.setdefault("feature_names", self.feature_names_in_)
        return drop_column_importance(self, X_train, y_train, X_valid, y_valid, **kwargs)

    def partial_dependence(self, X, features, **kwargs):
        "Compute partial dependence and ICE values for one or two features."
        kwargs.setdefault("feature_names", self.feature_names_in_)
        return partial_dependence(self, X, features, **kwargs)

    def get_params(self):
        "Return constructor parameters."
        names = ("n_trees", "min_node_size", "bootstrap_fraction", "bootstrap_max", "replacement", "max_node_samples",
            "min_candidate_rows", "candidate_attempt_factor", "cutoff_divisor", "seed", "oob", "adaptive", "missing_values", "max_dummy_cardinality")
        names += ("workbench",)
        return {name:getattr(self, name) for name in names}

    def __repr__(self):
        args = f"n_trees={self.n_trees}, min_node_size={self.min_node_size}"
        if self.seed is not None: args += f", seed={self.seed}"
        if self.oob: args += ", oob=True"
        return f"FastForest({args})"

from .analysis import (Explanation, FeatureDependence, FeatureRelations, Importance, PartialDependence,
    drop_column_importance, feature_dependence, feature_relations, partial_dependence, permutation_importance)
