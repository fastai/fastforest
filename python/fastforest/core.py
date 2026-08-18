import numpy as np

from ._core import Forest as _Forest
from ._core import ClassifierForest as _ClassifierForest
from ._core import _defaults as _native_defaults
from ._core import _fit_plan,_resolve_replacement as _native_resolve_replacement,_sample_indices
from ._core import _load_model,_save_classification,_save_regression
from ._core import _predict_classification_file,_predict_regression_file
from ._core import _compile_classification,_compile_regression
from ._core import __version__
from .preprocessing import ColumnInfo,_Encoder,_is_nan,_markers,_saved_scalar,_take_rows
from .sklearn import sklearn_hist_preprocessor,sklearn_preprocessor

_REG_DEFAULTS,_CLASS_DEFAULTS = _native_defaults(False),_native_defaults(True)

__all__ = ["FastForest", "FastForestClassifier", "load", "ColumnInfo", "Importance", "Explanation", "PartialDependence", "FeatureRelations", "FeatureDependence",
    "permutation_importance", "drop_column_importance", "partial_dependence", "feature_relations", "feature_dependence",
    "sklearn_preprocessor", "sklearn_hist_preprocessor", "__version__"]

def _vector(x, name="y", indices=None):
    "Convert `x` to a contiguous one-dimensional float32 array."
    x = np.asarray(_take_rows(x, indices))
    if x.ndim != 1: raise ValueError(f"{name} must be a one-dimensional array")
    try: return np.ascontiguousarray(x, dtype=np.float32)
    except (TypeError,ValueError) as error: raise ValueError(f"{name} must be numeric") from error

def _class_vector(y, indices=None):
    "Map arbitrary class labels to contiguous native integer IDs."
    y = np.asarray(_take_rows(y, indices))
    if y.ndim != 1: raise ValueError("y must be a one-dimensional array")
    kind = y.dtype.kind
    missing = np.isnan(y).any() if kind in "fc" else np.isnat(y).any() if kind in "mM" else (
        np.equal(y, None).any() or np.not_equal(y, y).any()) if kind == "O" else False
    if missing: raise ValueError("class labels cannot be missing")
    try: classes,codes = np.unique(y, return_inverse=True)
    except TypeError as error: raise ValueError("class labels must be mutually comparable") from error
    if len(classes) < 2: raise ValueError("classification requires at least two classes")
    return classes,np.ascontiguousarray(codes, dtype=np.uint32)

def _take_rows(values, indices):
    "Select rows without losing dataframe column metadata."
    if indices is None: return values
    if hasattr(values, "iloc"): return values.iloc[indices]
    return np.asarray(values)[indices]

def _estimated_outputs(y, seed):
    "Estimate classification output dimensions from at most 1,000 targets."
    indices = np.asarray(_sample_indices(len(y), 1_000, seed, 1)) if len(y) > 1_000 else None
    sample = np.asarray(_take_rows(y, indices))
    try: return max(1, len(np.unique(sample))-1)
    except TypeError: return 1

def _resolve_replacement(replacement, n_rows, classification=False):
    "Resolve adaptive row sampling or validate an explicit override."
    if replacement is not None and not isinstance(replacement, (bool, np.bool_)):
        raise ValueError("replacement must be True, False, or None")
    return _native_resolve_replacement(n_rows, None if replacement is None else bool(replacement), classification)

def _fit_pool(X, y, n_trees, bootstrap_fraction, bootstrap_max, replacement, oob, seed, classification=False):
    "Select the shared bounded training pool before expensive conversion."
    if getattr(y, "ndim", 1) != 1: raise ValueError("y must be a one-dimensional array")
    try: n_rows = len(X)
    except TypeError: n_rows = 0
    if n_rows != len(y): raise ValueError(f"X has {n_rows} rows but y has {len(y)} values")
    outputs = _estimated_outputs(y, seed) if classification else 1
    _,_,pool_rows = _fit_plan(n_rows, n_trees, bootstrap_fraction, bootstrap_max, replacement, oob, outputs)
    indices = None if pool_rows == n_rows else np.asarray(_sample_indices(n_rows, pool_rows, seed, 2))
    return n_rows,indices,X,y

def _original_indices(pool_indices, local_indices):
    "Map fitted-pool rows back to the caller's original row positions."
    if local_indices is None: return None
    local_indices = np.asarray(local_indices)
    return local_indices if pool_indices is None else pool_indices[local_indices]

def _loaded_scalar(value):
    kind,value = value
    if kind == 0: return None
    if kind == 1: return np.nan
    if kind == 2: return value == "1"
    if kind == 3: return int(value)
    if kind == 4: return float(value)
    if kind == 5: return value
    raise ValueError("saved model contains an unknown scalar type")

def _saved_metadata(model):
    markers = [_saved_scalar(value) for value in _markers(model.missing_values, model._encoder.input_names)]
    dates = [(index,format) for index,format,_ in model._encoder._dates]
    excluded = {"missing_values", "date_columns"}
    params = [(name,_saved_scalar(value)) for name,value in model.get_params().items() if name not in excluded]
    return markers,dates,params

def _restore_fitted(model, encoder, native):
    model._encoder,model._model = encoder,native
    model.n_trees_ = native.n_trees
    model.n_features_in_ = len(encoder.names)
    model.feature_names_in_ = encoder.names
    model.date_columns_ = encoder.date_columns
    model.column_info_ = encoder.column_info
    model.feature_importances_ = encoder.aggregate(native.feature_importances)
    structures = np.asarray(native.tree_structures, dtype=np.int64)
    model.tree_node_counts_,model.tree_leaf_counts_,model.tree_depths_ = structures.T
    model.oob_counts_,model.oob_indices_ = None,None
    return model

def load(path):
    "Load a portable `.ffm` regression or classification model."
    task,encoder,regression,classifier,metadata,classes = _load_model(str(path))
    markers,dates,params = metadata
    markers = [_loaded_scalar(value) for value in markers]
    params = {name:_loaded_scalar(value) for name,value in params}
    names = tuple(encoder.input_names)
    date_spec = {names[index]:format for index,format in dates}
    params.update(missing_values=markers, date_columns=date_spec or None)
    adapted = _Encoder.from_native(encoder, markers, dates)
    if task == 0:
        model = _restore_fitted(FastForest(**params), adapted, regression)
        model.oob_prediction_ = None
        return model
    model = _restore_fitted(FastForestClassifier(**params), adapted, classifier)
    model.classes_ = np.asarray([_loaded_scalar(value) for value in classes])
    model.n_classes_ = len(model.classes_)
    model.prediction_trees_per_batch_ = classifier.prediction_trees_per_batch
    model.oob_decision_function_,model.oob_score_ = None,None
    return model

def _native_max_features(value):
    "Convert the public square-root or fractional feature selection."
    if value == "sqrt": return 1,0.
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and 0 < value <= 1: return 2,float(value)
    raise ValueError("max_features must be 'sqrt' or a float in (0, 1]")

def _finish_common(model, encoder, native, pool_indices):
    "Attach task-independent native results to a fitted estimator."
    model._encoder,model._model = encoder,native
    model.n_trees_ = native.n_trees
    model.n_features_in_ = len(encoder.names)
    model.feature_names_in_ = encoder.names
    model.date_columns_ = encoder.date_columns
    model.column_info_ = encoder.column_info
    model.feature_importances_ = encoder.aggregate(native.feature_importances)
    model.oob_counts_ = native.oob_counts
    model.oob_indices_ = _original_indices(pool_indices, native.oob_indices)
    structures = np.asarray(native.tree_structures, dtype=np.int64)
    model.tree_node_counts_,model.tree_leaf_counts_,model.tree_depths_ = structures.T

def _finish_regression(model, encoder, native, pool_indices):
    "Attach native regression results to a fitted public estimator."
    _finish_common(model, encoder, native, pool_indices)
    model.oob_prediction_ = native.oob_prediction
    return model

def _finish_classifier(model, encoder, native, pool_indices, target):
    "Attach native classification results to a fitted public estimator."
    _finish_common(model, encoder, native, pool_indices)
    model.prediction_trees_per_batch_ = native.prediction_trees_per_batch
    model.oob_decision_function_ = native.oob_decision_function
    local_oob = native.oob_indices
    model.oob_score_ = None
    if model.oob:
        valid = model.oob_counts_ > 0
        if valid.any(): model.oob_score_ = float(np.mean(
            model.classes_[model.oob_decision_function_[valid].argmax(axis=1)] == model.classes_[target[np.asarray(local_oob)[valid]]]))
    return model

class _ForestFacade:
    _analysis_metric = None

    def split_importance(self, feature_names=None):
        "Return fast split-gain importance; prefer permutation importance for reliable analysis."
        if self._model is None: raise RuntimeError(f"{type(self).__name__} must be fitted before importance analysis")
        names = tuple(str(name) for name in feature_names) if feature_names is not None else self.feature_names_in_
        values = np.asarray(self.feature_importances_)
        return Importance(names, values, np.zeros_like(values), np.nan, "split-gain")

    def feature_importance(self, X=None, y=None, method="permutation", **kwargs):
        "Measure feature importance, using reliable validation-set permutation by default."
        if method == "split": return self.split_importance(kwargs.pop("feature_names", None))
        if method != "permutation": raise ValueError("method must be 'permutation' or 'split'")
        if X is None or y is None: raise ValueError("permutation importance requires X and y")
        kwargs.setdefault("feature_names", self._encoder.input_names)
        kwargs.setdefault("features", self._encoder.analysis_groups)
        if self._analysis_metric is not None: kwargs.setdefault("metric", self._analysis_metric)
        return permutation_importance(self, X, y, **kwargs)

    def drop_column_importance(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        "Measure importance by refitting without each feature or feature group."
        kwargs.setdefault("feature_names", self._encoder.input_names)
        kwargs.setdefault("features", self._encoder.analysis_groups)
        if self._analysis_metric is not None: kwargs.setdefault("metric", self._analysis_metric)
        return drop_column_importance(self, X_train, y_train, X_valid, y_valid, **kwargs)

    def get_params(self):
        "Return constructor parameters."
        return {name:getattr(self, name) for name in self._param_names}

    def __repr__(self):
        args = f"n_trees={self.n_trees}, min_node_size={self.min_node_size}"
        if self.seed is not None: args += f", seed={self.seed}"
        if self.oob: args += ", oob=True"
        return f"{type(self).__name__}({args})"

class FastForest(_ForestFacade):
    "A fast approximate-forest regressor."
    _param_names = ("n_trees", "min_node_size", "bootstrap_fraction", "bootstrap_max", "replacement", "max_node_samples", "split_prior_rows",
        "cutoff_divisor", "random_splitter", "max_features", "seed", "oob", "missing_values", "date_columns", "allow_new_missing")
    def __init__(self, n_trees=_REG_DEFAULTS["n_trees"], min_node_size=_REG_DEFAULTS["min_node_size"],
        bootstrap_fraction=_REG_DEFAULTS["bootstrap_fraction"], bootstrap_max=_REG_DEFAULTS["bootstrap_max"], replacement=None,
        max_node_samples=_REG_DEFAULTS["max_node_samples"], split_prior_rows=_REG_DEFAULTS["split_prior_rows"],
        cutoff_divisor=_REG_DEFAULTS["cutoff_divisor"], random_splitter=_REG_DEFAULTS["random_splitter"],
        max_features=_REG_DEFAULTS["max_features"], seed=_REG_DEFAULTS["seed"], oob=_REG_DEFAULTS["oob"], missing_values=None,
        date_columns=None, allow_new_missing=_REG_DEFAULTS["allow_new_missing"]):
        self.n_trees,self.min_node_size,self.bootstrap_fraction = n_trees,min_node_size,bootstrap_fraction
        self.bootstrap_max,self.replacement = bootstrap_max,replacement
        self.max_node_samples,self.split_prior_rows = max_node_samples,split_prior_rows
        self.cutoff_divisor = cutoff_divisor
        self.random_splitter,self.max_features,self.seed,self.oob = random_splitter,max_features,seed,oob
        self.missing_values = missing_values
        self.date_columns = date_columns
        self.allow_new_missing = allow_new_missing
        self._model = None

    def fit(self, X, y):
        "Fit the forest to feature rows `X` and regression targets `y`."
        replacement = _resolve_replacement(self.replacement, len(X))
        n_rows,pool_indices,X,y = _fit_pool(X, y, self.n_trees, self.bootstrap_fraction, self.bootstrap_max,
            replacement, self.oob, self.seed)
        y = _vector(y, indices=pool_indices)
        self._encoder = _Encoder(self.missing_values, self.date_columns, self.allow_new_missing, self.seed)
        X = self._encoder.fit_transform(X, pool_indices)
        self.date_columns_ = self._encoder.date_columns
        self.n_trees_,sample_rows,_ = _fit_plan(n_rows, self.n_trees, self.bootstrap_fraction, self.bootstrap_max,
            replacement, self.oob, 1)
        sample_rows = min(sample_rows, len(y))
        self.replacement_ = replacement
        tree_args = (self.n_trees_, self.min_node_size, self.bootstrap_fraction, self.bootstrap_max, sample_rows, replacement)
        split_args = (self.max_node_samples, self.split_prior_rows, self.cutoff_divisor)
        max_features = _native_max_features(self.max_features)
        native = _Forest.fit(X, y, self._encoder.cutoff_values, self._encoder.cutoff_offsets, self._encoder.missing_ranks,
            *tree_args, *split_args, self.seed, self.oob, self.random_splitter, *max_features, None)
        return _finish_regression(self, self._encoder, native, pool_indices)

    def predict(self, X):
        "Predict regression targets for feature rows `X`."
        if self._model is None: raise RuntimeError("FastForest must be fitted before prediction")
        return self._encoder.predict(self._model, X)

    def save(self, path):
        "Save this fitted model and preprocessing schema to a portable `.ffm` file."
        if self._model is None: raise RuntimeError("FastForest must be fitted before saving")
        _save_regression(str(path), self._encoder._native, self._model, _saved_metadata(self))

    def predict_file(self, input, output, batch_size=65_536):
        "Stream predictions from `input` to `output` with bounded memory."
        if self._model is None: raise RuntimeError("FastForest must be fitted before prediction")
        _predict_regression_file(self._encoder._native, self._model, _saved_metadata(self),
            str(input), str(output), batch_size)

    def save_executable(self, output):
        "Build a standalone native executable containing this fitted model."
        if self._model is None: raise RuntimeError("FastForest must be fitted before compilation")
        _compile_regression(self._encoder._native, self._model, _saved_metadata(self), str(output))

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

    def partial_dependence(self, X, features, **kwargs):
        "Compute partial dependence and ICE values for one or two features."
        kwargs.setdefault("feature_names", self._encoder.input_names)
        return partial_dependence(self, X, features, **kwargs)

class FastForestClassifier(_ForestFacade):
    "A fast multiclass approximate-forest classifier."
    _analysis_metric = "accuracy"
    _param_names = ("n_trees", "min_node_size", "bootstrap_fraction", "bootstrap_max", "replacement", "max_node_samples", "class_weight_power",
        "cutoff_divisor", "random_splitter", "max_features", "seed", "oob", "missing_values", "date_columns", "allow_new_missing")
    def __init__(self, n_trees=_CLASS_DEFAULTS["n_trees"], min_node_size=_CLASS_DEFAULTS["min_node_size"],
        bootstrap_fraction=_CLASS_DEFAULTS["bootstrap_fraction"], bootstrap_max=_CLASS_DEFAULTS["bootstrap_max"], replacement=None,
        max_node_samples=_CLASS_DEFAULTS["max_node_samples"], class_weight_power=_CLASS_DEFAULTS["class_weight_power"],
        cutoff_divisor=_CLASS_DEFAULTS["cutoff_divisor"], random_splitter=_CLASS_DEFAULTS["random_splitter"],
        max_features=_CLASS_DEFAULTS["max_features"], seed=_CLASS_DEFAULTS["seed"], oob=_CLASS_DEFAULTS["oob"], missing_values=None,
        date_columns=None, allow_new_missing=_CLASS_DEFAULTS["allow_new_missing"]):
        self.n_trees,self.min_node_size,self.bootstrap_fraction = n_trees,min_node_size,bootstrap_fraction
        self.bootstrap_max,self.replacement = bootstrap_max,replacement
        self.max_node_samples,self.class_weight_power = max_node_samples,class_weight_power
        self.cutoff_divisor = cutoff_divisor
        self.random_splitter,self.max_features,self.seed,self.oob = random_splitter,max_features,seed,oob
        self.missing_values = missing_values
        self.date_columns = date_columns
        self.allow_new_missing = allow_new_missing
        self._model = None

    def fit(self, X, y):
        "Fit the forest to feature rows `X` and arbitrary class labels `y`."
        replacement = _resolve_replacement(self.replacement, len(X), classification=True)
        n_rows,pool_indices,X,y = _fit_pool(X, y, self.n_trees, self.bootstrap_fraction, self.bootstrap_max,
            replacement, self.oob, self.seed, classification=True)
        self.classes_,y = _class_vector(y, pool_indices)
        self.n_classes_ = len(self.classes_)
        self._encoder = _Encoder(self.missing_values, self.date_columns, self.allow_new_missing, self.seed)
        X = self._encoder.fit_transform(X, pool_indices)
        self.date_columns_ = self._encoder.date_columns
        outputs = max(1, self.n_classes_-1)
        self.n_trees_,sample_rows,_ = _fit_plan(n_rows, self.n_trees, self.bootstrap_fraction, self.bootstrap_max,
            replacement, self.oob, outputs)
        sample_rows = min(sample_rows, len(y))
        self.replacement_ = replacement
        tree_args = (self.n_trees_, self.min_node_size, self.bootstrap_fraction, self.bootstrap_max, sample_rows, replacement)
        split_args = (self.max_node_samples, self.class_weight_power, self.cutoff_divisor)
        max_features = _native_max_features(self.max_features)
        native = _ClassifierForest.fit(X, y, self.n_classes_, self._encoder.cutoff_values, self._encoder.cutoff_offsets,
            self._encoder.missing_ranks,
            *tree_args, *split_args, self.seed, self.oob, self.random_splitter, *max_features, None)
        return _finish_classifier(self, self._encoder, native, pool_indices, y)

    def predict_proba(self, X):
        "Return one probability per `(row, class)`, ordered as `classes_`."
        if self._model is None: raise RuntimeError("FastForestClassifier must be fitted before prediction")
        return self._encoder.predict(self._model, X, proba=True)

    def predict(self, X):
        "Predict class labels for feature rows `X`."
        if self._model is None: raise RuntimeError("FastForestClassifier must be fitted before prediction")
        return self.classes_[self._encoder.predict(self._model, X)]

    def save(self, path):
        "Save this fitted classifier and preprocessing schema to a portable `.ffm` file."
        if self._model is None: raise RuntimeError("FastForestClassifier must be fitted before saving")
        _save_classification(str(path), self._encoder._native, self._model, _saved_metadata(self),
            [_saved_scalar(value) for value in self.classes_])

    def predict_file(self, input, output, batch_size=65_536, proba=False):
        "Stream class predictions or probabilities with bounded memory."
        if self._model is None: raise RuntimeError("FastForestClassifier must be fitted before prediction")
        _predict_classification_file(self._encoder._native, self._model, _saved_metadata(self),
            [_saved_scalar(value) for value in self.classes_], str(input), str(output), batch_size, proba)

    def save_executable(self, output):
        "Build a standalone native executable containing this fitted classifier."
        if self._model is None: raise RuntimeError("FastForestClassifier must be fitted before compilation")
        _compile_classification(self._encoder._native, self._model, _saved_metadata(self),
            [_saved_scalar(value) for value in self.classes_], str(output))

from .analysis import (Explanation, FeatureDependence, FeatureRelations, Importance, PartialDependence,
    drop_column_importance, feature_dependence, feature_relations, partial_dependence, permutation_importance)
