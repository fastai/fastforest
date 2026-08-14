import numpy as np

from ._core import Forest as _Forest
from ._core import __version__

__all__ = ["FastForest", "Importance", "Explanation", "PartialDependence", "FeatureRelations", "FeatureDependence",
    "permutation_importance", "drop_column_importance", "partial_dependence", "feature_relations", "feature_dependence", "__version__"]

def _matrix(x, name="X"):
    "Convert `x` to a contiguous two-dimensional float32 array."
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2: raise ValueError(f"{name} must be a two-dimensional array")
    return np.ascontiguousarray(x)

def _vector(x, name="y"):
    "Convert `x` to a contiguous one-dimensional float32 array."
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1: raise ValueError(f"{name} must be a one-dimensional array")
    return np.ascontiguousarray(x)

class FastForest:
    "A fast approximate random-forest regressor."
    def __init__(self, n_trees=100, min_node_size=4, bootstrap_fraction=0.8, bootstrap_max=40_000, replacement=False,
        max_node_samples=160, cutoff_divisor=3.0, seed=None, oob=False):
        self.n_trees,self.min_node_size,self.bootstrap_fraction = n_trees,min_node_size,bootstrap_fraction
        self.bootstrap_max,self.replacement = bootstrap_max,replacement
        self.max_node_samples,self.cutoff_divisor = max_node_samples,cutoff_divisor
        self.seed,self.oob = seed,oob
        self._model = None

    def fit(self, X, y):
        "Fit the forest to feature rows `X` and regression targets `y`."
        names = getattr(X, "columns", None)
        X,y = _matrix(X),_vector(y)
        self._model = _Forest.fit(X, y, self.n_trees, self.min_node_size, self.bootstrap_fraction, self.bootstrap_max, self.replacement,
            self.max_node_samples, self.cutoff_divisor, self.seed, self.oob)
        self.n_features_in_ = self._model.n_features
        self.feature_names_in_ = tuple(str(name) for name in names) if names is not None else tuple(f"x{i}" for i in range(X.shape[1]))
        self.feature_importances_ = self._model.feature_importances
        self.oob_prediction_ = self._model.oob_prediction
        self.oob_counts_ = self._model.oob_counts
        return self

    def predict(self, X):
        "Predict regression targets for feature rows `X`."
        if self._model is None: raise RuntimeError("FastForest must be fitted before prediction")
        return self._model.predict(_matrix(X))

    def predict_trees(self, X):
        "Return one prediction per `(row, tree)`."
        if self._model is None: raise RuntimeError("FastForest must be fitted before prediction")
        return self._model.predict_trees(_matrix(X))

    def predict_std(self, X):
        "Return the standard deviation of the trees' predictions for each row."
        return self.predict_trees(X).std(axis=1)

    def explain(self, X, feature_names=None):
        "Decompose predictions into the forest bias and additive feature contributions."
        if self._model is None: raise RuntimeError("FastForest must be fitted before explanation")
        values = _matrix(X)
        prediction,bias,contributions = self._model.explain(values)
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
        names = ("n_trees", "min_node_size", "bootstrap_fraction", "bootstrap_max", "replacement",
            "max_node_samples", "cutoff_divisor", "seed", "oob")
        return {name:getattr(self, name) for name in names}

    def __repr__(self):
        args = f"n_trees={self.n_trees}, min_node_size={self.min_node_size}"
        if self.seed is not None: args += f", seed={self.seed}"
        if self.oob: args += ", oob=True"
        return f"FastForest({args})"

from .analysis import (Explanation, FeatureDependence, FeatureRelations, Importance, PartialDependence,
    drop_column_importance, feature_dependence, feature_relations, partial_dependence, permutation_importance)
