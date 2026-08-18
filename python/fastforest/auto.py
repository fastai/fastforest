"Automatic sample and forest sizing for fastforest."

import numpy as np

from .core import (FastForest,FastForestClassifier,_ClassifierForest,_Encoder,_Forest,_class_vector,
    _estimated_outputs,_finish_classifier,_finish_regression,_fit_plan,_fit_pool,_native_max_features,_resolve_replacement,_sample_indices,_vector)
from .tools import forest_suite,screen

__all__ = ["AutoForest", "AutoForestClassifier"]

_BOOTSTRAP_LEVELS = (80_000,120_000,160_000,200_000)
_NODE_LEVELS = (640,960,1_280)
_GROW_BOOTSTRAP_LEVELS = (80_000,160_000,240_000,320_000)
_GROW_NODE_LEVELS = (640,1_280,1_920)
_SIZING_MIN_IMPROVEMENT = .01

def _seed(value):
    return int(np.random.default_rng().integers(0, 2**63)) if value is None else int(value)

def _capped_levels(primary, fallback, minimum, maximum):
    "Keep the wider grid, filling cap-removed slots from the ordinary grid."
    result = [value for value in primary if minimum<value<=maximum]
    for value in fallback:
        if len(result)>=len(primary): break
        if minimum<value<=maximum and value not in result: result.append(value)
    return tuple(sorted(result))

def _loss(task, forest, target):
    counts,indices = np.asarray(forest.oob_counts),np.asarray(forest.oob_indices)
    valid = counts>0
    if not valid.any(): raise RuntimeError("the forest produced no OOB predictions")
    expected = target[indices[valid]]
    if task == "regression":
        residual = np.asarray(forest.oob_prediction)[valid]-expected
        return float(np.mean(np.square(residual, dtype=np.float64)))
    probabilities = np.asarray(forest.oob_decision_function)[valid]
    rows = np.arange(len(expected))
    return float(np.mean(np.square(probabilities).sum(axis=1)-2*probabilities[rows,expected]+1))

def _chosen_level(results, name, values, baseline_loss, min_improvement):
    "Convert noisy cumulative evidence into the furthest justified level."
    chosen = 0
    losses = {}
    for level,value in enumerate(values[1:], 1):
        result = next(result for result in results if result.changes == {name:value})
        losses[value] = result.oob_loss
        improvement = max(0., (baseline_loss-result.oob_loss)/baseline_loss)
        chosen = max(chosen, min(level, int(np.floor(improvement/min_improvement+1e-12))))
    return values[chosen],losses

class _Auto:
    def __init__(self, autogrow=False, tree_batch_size=32, min_improvement=.01, keep_last_batch=False, max_trees=512, **kwargs):
        if "n_trees" in kwargs or "oob" in kwargs: raise TypeError(f"{type(self).__name__} controls n_trees and oob")
        super().__init__(n_trees=None, oob=autogrow, **kwargs)
        self._auto_init(autogrow, tree_batch_size, min_improvement, keep_last_batch, max_trees)

    def get_params(self):
        params = super().get_params()
        params.pop("n_trees"),params.pop("oob")
        return params | dict(autogrow=self.autogrow, tree_batch_size=self.tree_batch_size, min_improvement=self.min_improvement,
            keep_last_batch=self.keep_last_batch, max_trees=self.max_trees)

    def __repr__(self): return f"{type(self).__name__}(autogrow={self.autogrow})"

    def _auto_init(self, autogrow, tree_batch_size, min_improvement, keep_last_batch, max_trees):
        if tree_batch_size<1: raise ValueError("tree_batch_size must be positive")
        if not 0<min_improvement<1: raise ValueError("min_improvement must be in (0, 1)")
        if max_trees<tree_batch_size or max_trees%tree_batch_size: raise ValueError("max_trees must be a multiple of tree_batch_size")
        self.autogrow,self.tree_batch_size,self.min_improvement = autogrow,tree_batch_size,min_improvement
        self.keep_last_batch,self.max_trees = keep_last_batch,max_trees
        self._requested_bootstrap_max,self._requested_max_node_samples = self.bootstrap_max,self.max_node_samples

    def _size_samples(self, X, y, task, outputs, fit_seed):
        self.bootstrap_max,self.max_node_samples = self._requested_bootstrap_max,self._requested_max_node_samples
        if self.bootstrap_max is None:
            self.sizing_ = dict(active=False, threshold=None, bootstrap_max=None,
                max_node_samples=self.max_node_samples, results=())
            return
        threshold = 2*self.bootstrap_max*outputs
        if len(X)<=threshold:
            self.sizing_ = dict(active=False, threshold=threshold, bootstrap_max=self.bootstrap_max,
                max_node_samples=self.max_node_samples, results=())
            return
        cap = max(self.bootstrap_max, int(.8*len(X)//outputs))
        bootstrap_levels = (_capped_levels(_GROW_BOOTSTRAP_LEVELS, _BOOTSTRAP_LEVELS, self.bootstrap_max, cap)
            if self.autogrow else tuple(value for value in _BOOTSTRAP_LEVELS if self.bootstrap_max<value<=cap))
        node_levels = _GROW_NODE_LEVELS if self.autogrow else _NODE_LEVELS
        bootstrap = (self.bootstrap_max,)+bootstrap_levels
        nodes = (self.max_node_samples,)+tuple(value for value in node_levels if value>self.max_node_samples)
        levels = {"bootstrap_max":bootstrap, "max_node_samples":nodes}
        params = FastForestClassifier.get_params(self) if task=="classification" else FastForest.get_params(self)
        base = FastForestClassifier(**params) if task=="classification" else FastForest(**params)
        report = screen(base, X, y, forest_suite(base, levels), trees=8, seed=fit_seed)
        baseline = report.results[0].oob_loss
        self.bootstrap_max,bootstrap_losses = _chosen_level(report.results, "bootstrap_max", bootstrap, baseline, _SIZING_MIN_IMPROVEMENT)
        self.max_node_samples,node_losses = _chosen_level(report.results, "max_node_samples", nodes, baseline, _SIZING_MIN_IMPROVEMENT)
        self.sizing_ = dict(active=True, threshold=threshold, baseline_loss=baseline,
            bootstrap_max=self.bootstrap_max, max_node_samples=self.max_node_samples,
            bootstrap_losses=bootstrap_losses, node_losses=node_losses, seconds=report.batch_seconds)

    def _fit_args(self, task, n_rows, target_rows, replacement, outputs, trees, seed, tracking_indices=None):
        oob = tracking_indices is not None
        _,sample_rows,_ = _fit_plan(n_rows, trees, self.bootstrap_fraction,
            self.bootstrap_max, replacement, oob, outputs)
        tree = (trees, self.min_node_size, self.bootstrap_fraction, self.bootstrap_max,
            min(sample_rows,target_rows), replacement)
        split = ((self.max_node_samples, self.class_weight_power) if task=="classification"
            else (self.max_node_samples, self.split_prior_rows))
        split += (self.cutoff_divisor,)
        return (*tree,*split,seed,oob,self.random_splitter,*_native_max_features(self.max_features),tracking_indices)

    def _fit_native(self, task, encoded, target, encoder, n_rows, replacement, outputs, classes, trees, seed, tracking_indices=None):
        args = self._fit_args(task, n_rows, len(target), replacement, outputs, trees, seed, tracking_indices)
        common = (encoded,target,encoder.cutoff_values,encoder.cutoff_offsets,encoder.missing_ranks)
        return (_ClassifierForest.fit(encoded,target,len(classes),*common[2:],*args) if task=="classification"
            else _Forest.fit(*common,*args))

    def _fit_once(self, task, encoded, target, encoder, n_rows, replacement, outputs, classes, fit_seed):
        trees,_,_ = _fit_plan(n_rows, None, self.bootstrap_fraction, self.bootstrap_max, replacement, False, outputs)
        self.tree_history_ = ()
        return self._fit_native(task, encoded, target, encoder, n_rows, replacement, outputs, classes, trees, fit_seed)

    def _grow(self, task, encoded, target, encoder, n_rows, replacement, outputs, classes, fit_seed):
        rng = np.random.default_rng(fit_seed)
        tracking_rows = min(len(target),40_000*outputs)
        tracking_indices = np.asarray(_sample_indices(len(target), tracking_rows, fit_seed, 4))
        def fit_batch():
            seed = int(rng.integers(0,2**63))
            return self._fit_native(task, encoded, target, encoder, n_rows, replacement, outputs, classes,
                self.tree_batch_size, seed, tracking_indices)
        forest = fit_batch()
        loss = _loss(task, forest, target)
        self.tree_history_ = [dict(trees=self.tree_batch_size, loss=loss, improvement=None, accepted=True)]
        while forest.n_trees+self.tree_batch_size<=self.max_trees:
            batch = fit_batch()
            combined = forest.combined(batch)
            new_loss = _loss(task, combined, target)
            improvement = (loss-new_loss)/loss
            accepted = improvement>=self.min_improvement
            self.tree_history_.append(dict(trees=combined.n_trees, loss=new_loss, improvement=improvement, accepted=accepted))
            if not accepted:
                if self.keep_last_batch: forest,loss = combined,new_loss
                break
            forest,loss = combined,new_loss
        return forest


class AutoForest(_Auto, FastForest):
    "A regressor that automatically sizes samples and optionally grows the forest."
    def fit(self, X, y):
        "Size samples and fit an approximate-forest regressor."
        fit_seed = _seed(self.seed)
        self.seed_ = fit_seed
        self._size_samples(X, y, "regression", 1, fit_seed)
        replacement = _resolve_replacement(self.replacement, len(X))
        trees = self.tree_batch_size if self.autogrow else None
        n_rows,pool_indices,X,y = _fit_pool(X, y, trees, self.bootstrap_fraction,
            self.bootstrap_max, replacement, self.autogrow, fit_seed)
        target = _vector(y, indices=pool_indices)
        encoder = _Encoder(self.missing_values, self.date_columns, self.allow_new_missing, fit_seed)
        encoded = encoder.fit_transform(X, pool_indices)
        self.replacement_ = replacement
        fit = self._grow if self.autogrow else self._fit_once
        native = fit("regression", encoded, target, encoder, n_rows, replacement, 1, None, fit_seed)
        return _finish_regression(self, encoder, native, pool_indices)

class AutoForestClassifier(_Auto, FastForestClassifier):
    "A classifier that automatically sizes samples and optionally grows the forest."
    def fit(self, X, y):
        "Size samples and fit an approximate-forest classifier."
        fit_seed = _seed(self.seed)
        self.seed_ = fit_seed
        outputs = _estimated_outputs(y, fit_seed)
        self._size_samples(X, y, "classification", outputs, fit_seed)
        replacement = _resolve_replacement(self.replacement, len(X), classification=True)
        trees = self.tree_batch_size if self.autogrow else None
        n_rows,pool_indices,X,y = _fit_pool(X, y, trees, self.bootstrap_fraction,
            self.bootstrap_max, replacement, self.autogrow, fit_seed, classification=True)
        self.classes_,target = _class_vector(y, pool_indices)
        self.n_classes_ = len(self.classes_)
        outputs = max(1,self.n_classes_-1)
        encoder = _Encoder(self.missing_values, self.date_columns, self.allow_new_missing, fit_seed)
        encoded = encoder.fit_transform(X, pool_indices)
        self.replacement_ = replacement
        fit = self._grow if self.autogrow else self._fit_once
        native = fit("classification", encoded, target, encoder, n_rows, replacement, outputs, self.classes_, fit_seed)
        return _finish_classifier(self, encoder, native, pool_indices, target)
