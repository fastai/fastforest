from dataclasses import dataclass

import numpy as np

from .preprocessing import _missing_mask,_table,_take_rows

def _data(X, feature_names=None):
    values,names = _table(X)
    if feature_names is not None: names = tuple(str(name) for name in feature_names)
    if len(names) != values.shape[1]: raise ValueError("feature_names must have one name per column")
    if len(set(names)) != len(names): raise ValueError("feature_names must be unique")
    return np.ascontiguousarray(values),names

def _index(feature, names):
    if isinstance(feature, str):
        if feature not in names: raise ValueError(f"unknown feature {feature!r}")
        return names.index(feature)
    feature = int(feature)
    if feature < 0 or feature >= len(names): raise ValueError(f"feature index {feature} is out of range")
    return feature

def _groups(features, names):
    if features is None: return list(names),[(i,) for i in range(len(names))]
    if isinstance(features, dict): labels,items = list(features),list(features.values())
    else:
        if isinstance(features, (str, int, np.integer)): features = [features]
        labels,items = [],features
    groups = []
    for item in items:
        if isinstance(item, (str, int, np.integer)): item = [item]
        idx = tuple(_index(feature, names) for feature in item)
        if not idx: raise ValueError("feature groups cannot be empty")
        groups.append(idx)
        if not isinstance(features, dict): labels.append(" + ".join(names[i] for i in idx))
    return [str(label) for label in labels],groups

def _sample(X, y=None, n_samples=None, seed=42):
    if n_samples is None or n_samples < 0 or n_samples >= len(X): return (X,y) if y is not None else X
    idx = np.random.default_rng(seed).choice(len(X), n_samples, replace=False)
    return (_take_rows(X, idx),_take_rows(y, idx)) if y is not None else _take_rows(X, idx)

def _r2(y, prediction):
    residual = np.square(y-prediction).sum()
    total = np.square(y-y.mean()).sum()
    return 1-residual/total if total else np.nan

def _metric(metric):
    if callable(metric): return metric
    if metric == "r2": return _r2
    if metric == "neg_rmse": return lambda y,p: -np.sqrt(np.square(y-p).mean())
    if metric == "accuracy": return lambda y,p: np.mean(y == p)
    raise ValueError("metric must be 'r2', 'neg_rmse', 'accuracy', or a callable")

def _plt():
    try: import matplotlib.pyplot as plt
    except ImportError as error: raise ImportError("plotting requires matplotlib") from error
    return plt

@dataclass
class Importance:
    "Feature importance values and their measurement metadata."
    names: tuple
    values: np.ndarray
    std: np.ndarray
    baseline: float
    method: str

    def sorted(self):
        "Return importance ordered from largest to smallest."
        order = np.argsort(-self.values)
        return Importance(tuple(self.names[i] for i in order), self.values[order], self.std[order], self.baseline, self.method)

    def plot(self, top=None, ax=None):
        "Plot a compact horizontal importance chart."
        result = self.sorted()
        if top is not None: result = Importance(result.names[:top], result.values[:top], result.std[:top], result.baseline, result.method)
        order = np.arange(len(result.names))[::-1]
        if ax is None: _,ax = _plt().subplots(figsize=(7, max(2, len(order)*0.32)))
        errors = result.std[order] if np.any(result.std) else None
        ax.barh(order, result.values[order], xerr=errors)
        ax.set(yticks=order, yticklabels=np.asarray(result.names)[order], xlabel=f"{result.method} importance")
        return ax

@dataclass
class Explanation:
    "Additive per-feature explanations for one or more predictions."
    prediction: np.ndarray
    bias: float
    contributions: np.ndarray
    values: np.ndarray
    names: tuple

    def row(self, row=0):
        "Return `(feature, value, contribution)` entries ordered by absolute contribution."
        order = np.argsort(-np.abs(self.contributions[row]))
        return [(self.names[i], self.values[row,i].item() if isinstance(self.values[row,i], np.generic) else self.values[row,i],
            float(self.contributions[row,i])) for i in order]

    def plot(self, row=0, top=12, ax=None):
        "Plot the strongest positive and negative contributions for one row."
        entries = self.row(row)[:top][::-1]
        if ax is None: _,ax = _plt().subplots(figsize=(7, max(2, len(entries)*0.35)))
        labels = [f"{name} = {value:g}" if isinstance(value, (int, float, np.number)) else f"{name} = {value}" for name,value,_ in entries]
        values = np.asarray([contribution for _,_,contribution in entries])
        ax.barh(np.arange(len(entries)), values, color=np.where(values >= 0, "#3a923a", "#c44e52"))
        ax.set(yticks=np.arange(len(entries)), yticklabels=labels, xlabel="contribution",
            title=f"prediction={self.prediction[row]:g}, bias={self.bias:g}")
        ax.axvline(0, color="black", linewidth=.7)
        return ax

@dataclass
class PartialDependence:
    "One- or two-feature partial dependence and optional ICE values."
    names: tuple
    grids: tuple
    average: np.ndarray
    individual: np.ndarray

    def clustered_ice(self, n_clusters=5, centered=True, seed=42):
        "Return representative ICE curves from a small dependency-free k-means."
        if self.individual is None: raise ValueError("ICE clustering requires one-feature partial dependence")
        curves = self.individual-self.individual[:,:1] if centered else self.individual
        if n_clusters < 1 or n_clusters > len(curves): raise ValueError("n_clusters must be between 1 and the number of ICE rows")
        rng = np.random.default_rng(seed)
        centers = curves[rng.choice(len(curves), n_clusters, replace=False)].copy()
        for _ in range(50):
            labels = np.square(curves[:,None]-centers).sum(axis=2).argmin(axis=1)
            updated = np.asarray([curves[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(n_clusters)])
            if np.allclose(updated, centers): break
            centers = updated
        return centers

    def plot(self, ice=True, centered=False, clusters=None, ax=None):
        "Plot a PDP with ICE lines, or a two-feature dependence surface."
        plt = _plt()
        if ax is None: _,ax = plt.subplots(figsize=(7, 4.5))
        if len(self.names) == 1:
            individual,average = self.individual,self.average
            if centered:
                individual = individual-individual[:,:1]
                average = individual.mean(axis=0)
            if clusters is not None: ax.plot(self.grids[0], self.clustered_ice(clusters, centered).T, alpha=.7, linewidth=1)
            elif ice and individual is not None: ax.plot(self.grids[0], individual.T, color="#777777", alpha=.08, linewidth=.7)
            ax.plot(self.grids[0], average, color="#1f77b4", linewidth=2.5)
            ax.set(xlabel=self.names[0], ylabel="partial prediction")
        else:
            contour = ax.contourf(self.grids[0], self.grids[1], self.average.T, levels=20)
            plt.colorbar(contour, ax=ax, label="partial prediction")
            ax.set(xlabel=self.names[0], ylabel=self.names[1])
        return ax

@dataclass
class FeatureRelations:
    "Spearman correlation and average-linkage clustering of features."
    names: tuple
    correlation: np.ndarray
    linkage: np.ndarray
    order: tuple

    def groups(self, threshold=.2):
        "Return average-linkage feature groups merged within `threshold` distance."
        groups = {i:{i} for i in range(len(self.names))}
        selected = []
        for i,(left,right,distance,_) in enumerate(self.linkage):
            groups[len(self.names)+i] = groups[int(left)] | groups[int(right)]
            if distance <= threshold: selected.append(groups[len(self.names)+i])
        selected = [group for group in selected if not any(group < other for other in selected)]
        covered = set().union(*selected) if selected else set()
        selected += [{i} for i in range(len(self.names)) if i not in covered]
        return [tuple(self.names[i] for i in sorted(group)) for group in sorted(selected, key=min)]

    def plot(self, ax=None):
        "Plot the reordered Spearman correlation matrix."
        plt = _plt()
        if ax is None: _,ax = plt.subplots(figsize=(max(5, len(self.names)*.45), max(4, len(self.names)*.4)))
        order = np.asarray(self.order)
        image = ax.imshow(self.correlation[np.ix_(order, order)], vmin=-1, vmax=1, cmap="coolwarm")
        labels = np.asarray(self.names)[order]
        ax.set(xticks=np.arange(len(order)), yticks=np.arange(len(order)), xticklabels=labels, yticklabels=labels)
        ax.tick_params(axis="x", labelrotation=70)
        plt.colorbar(image, ax=ax, label="Spearman correlation")
        return ax

    def plot_dendrogram(self, ax=None):
        "Plot the dependency-free average-linkage dendrogram."
        if ax is None: _,ax = _plt().subplots(figsize=(max(7, len(self.names)*.55), 4))
        x = {feature:i for i,feature in enumerate(self.order)}
        height = {feature:0.0 for feature in range(len(self.names))}
        for i,(left,right,distance,_) in enumerate(self.linkage):
            left,right = int(left),int(right)
            xl,xr,hl,hr = x[left],x[right],height[left],height[right]
            ax.plot([xl,xl,xr,xr], [hl,distance,distance,hr], color="#444444")
            x[len(self.names)+i],height[len(self.names)+i] = (xl+xr)/2,distance
        ax.set(xticks=np.arange(len(self.names)), xticklabels=np.asarray(self.names)[np.asarray(self.order)], ylabel="1 - |correlation|")
        ax.tick_params(axis="x", labelrotation=70)
        return ax

@dataclass
class FeatureDependence:
    "How well each feature can be predicted from, and depends on, the others."
    names: tuple
    predictability: np.ndarray
    importance: np.ndarray

    def plot(self, ax=None):
        "Plot nonlinear feature dependencies as a heatmap."
        plt = _plt()
        if ax is None: _,ax = plt.subplots(figsize=(max(6, len(self.names)*.5), max(4, len(self.names)*.42)))
        image = ax.imshow(self.importance, vmin=0, vmax=max(1, np.nanmax(self.importance)), cmap="Blues")
        labels = [f"{name} ({score:.2f})" for name,score in zip(self.names, self.predictability)]
        ax.set(xticks=np.arange(len(self.names)), yticks=np.arange(len(self.names)), xticklabels=self.names, yticklabels=labels)
        ax.tick_params(axis="x", labelrotation=70)
        plt.colorbar(image, ax=ax, label="permutation dependence")
        return ax

def permutation_importance(model, X, y, features=None, n_repeats=5, n_samples=5000, metric="r2", seed=42, feature_names=None):
    "Measure validation-set importance by permuting individual or grouped features."
    X,y = _sample(X, y, n_samples, seed)
    X,names = _data(X, feature_names)
    y = np.asarray(y)
    if len(y) != len(X): raise ValueError("X and y must contain the same number of rows")
    if n_repeats < 1: raise ValueError("n_repeats must be at least 1")
    labels,groups = _groups(features, names)
    score = _metric(metric)
    baseline = score(y, model.predict(X))
    rng = np.random.default_rng(seed)
    drops = np.empty((len(groups), n_repeats))
    for i,group in enumerate(groups):
        for repeat in range(n_repeats):
            shuffled = X.copy()
            shuffled[:,group] = shuffled[rng.permutation(len(X))][:,group]
            drops[i,repeat] = baseline-score(y, model.predict(shuffled))
    return Importance(tuple(labels), drops.mean(axis=1), drops.std(axis=1), float(baseline), "permutation")

def drop_column_importance(model, X_train, y_train, X_valid=None, y_valid=None, features=None, metric="r2", seed=42,
    feature_names=None, n_train_samples=40_000, n_valid_samples=5000):
    "Measure importance by retraining after dropping each feature or feature group."
    if (X_valid is None) != (y_valid is None): raise ValueError("X_valid and y_valid must be provided together")
    X_train,y_train = _sample(X_train, y_train, n_train_samples, seed)
    X_train,names = _data(X_train, feature_names)
    y_train = np.asarray(y_train)
    if len(y_train) != len(X_train): raise ValueError("X_train and y_train must contain the same number of rows")
    if X_valid is None: X_valid,y_valid = _sample(X_train, y_train, n_valid_samples, seed+1)
    else:
        X_valid,y_valid = _sample(X_valid, y_valid, n_valid_samples, seed+1)
        X_valid,_ = _data(X_valid, names)
    y_valid = np.asarray(y_valid)
    if len(y_valid) != len(X_valid): raise ValueError("X_valid and y_valid must contain the same number of rows")
    labels,groups = _groups(features, names)
    params = model.get_params() | {"seed":seed}
    if "n_trees" in params: params["oob"] = False
    markers = [column.marker for column in model._encoder.columns] if hasattr(model, "_encoder") else None
    if markers is not None: params["missing_values"] = markers
    baseline_model = type(model)(**params).fit(X_train, y_train)
    score = _metric(metric)
    baseline = score(y_valid, baseline_model.predict(X_valid))
    values = []
    for group in groups:
        keep = np.asarray([i for i in range(len(names)) if i not in group])
        if not len(keep): raise ValueError("cannot drop every feature")
        dropped_params = params | ({"missing_values":[markers[i] for i in keep]} if markers is not None else {})
        dropped = type(model)(**dropped_params).fit(X_train[:,keep], y_train)
        values.append(baseline-score(y_valid, dropped.predict(X_valid[:,keep])))
    values = np.asarray(values)
    return Importance(tuple(labels), values, np.zeros_like(values), float(baseline), "drop-column")

def partial_dependence(model, X, features, grid_points=20, n_samples=500, seed=42, feature_names=None):
    "Compute one-feature PDP/ICE data or a two-feature partial-dependence surface."
    X = _sample(X, n_samples=n_samples, seed=seed)
    X,names = _data(X, feature_names)
    if hasattr(model, "_encoder"): X = model._encoder.display(X)
    sample = X
    if isinstance(features, dict):
        if len(features) != 1: raise ValueError("categorical partial dependence requires one named feature group")
        label,selectors = next(iter(features.items()))
        if isinstance(selectors, (str, int, np.integer)): selectors = [selectors]
        idx = tuple(_index(feature, names) for feature in selectors)
        if not idx: raise ValueError("categorical feature groups cannot be empty")
        individual = np.empty((len(sample), len(idx)))
        for j,active in enumerate(idx):
            changed = sample.copy()
            changed[:,idx] = 0
            changed[:,active] = 1
            individual[:,j] = model.predict(changed)
        return PartialDependence((str(label),), (np.asarray(names)[np.asarray(idx)],), individual.mean(axis=0), individual)
    if isinstance(features, (str, int, np.integer)): features = [features]
    idx = tuple(_index(feature, names) for feature in features)
    if len(idx) not in (1, 2): raise ValueError("partial dependence requires one or two features")
    grids = []
    for i in idx:
        column = X[:,i]
        all_int = False
        if hasattr(model, "_encoder"):
            schema = model._encoder.columns[i]
            column = column[~_missing_mask(column, schema.marker)]
            all_int = schema.all_int
        try:
            column = np.asarray(column, dtype=np.float32)
            if not np.isfinite(column).all(): raise ValueError
        except (TypeError, ValueError): column = np.asarray(column, dtype=str)
        unique = np.unique(column)
        if len(unique) <= grid_points: grid = unique
        elif all_int: grid = unique[np.linspace(0, len(unique)-1, grid_points).astype(int)]
        elif np.issubdtype(column.dtype, np.number): grid = np.unique(np.quantile(column, np.linspace(0, 1, grid_points)))
        else: grid = unique[np.linspace(0, len(unique)-1, grid_points).astype(int)]
        if all_int: grid = grid.astype(np.int64)
        grids.append(grid)
    grids = tuple(grids)
    if len(idx) == 1:
        individual = np.empty((len(sample), len(grids[0])))
        for j,value in enumerate(grids[0]):
            changed = sample.copy()
            changed[:,idx[0]] = value
            individual[:,j] = model.predict(changed)
        average = individual.mean(axis=0)
    else:
        individual = None
        average = np.empty((len(grids[0]), len(grids[1])))
        for i,left in enumerate(grids[0]):
            for j,right in enumerate(grids[1]):
                changed = sample.copy()
                changed[:,idx[0]],changed[:,idx[1]] = left,right
                average[i,j] = model.predict(changed).mean()
    return PartialDependence(tuple(names[i] for i in idx), grids, average, individual)

def _ranks(values):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    for start,end in zip(starts, ends): ranks[order[start:end]] = (start+end-1)/2
    return ranks

def feature_relations(X, feature_names=None, n_samples=5000, seed=42):
    "Compute tie-aware Spearman correlation and average-linkage feature clustering."
    X = _sample(X, n_samples=n_samples, seed=seed)
    X,names = _data(X, feature_names)
    if X.shape[1] < 2: raise ValueError("feature relations require at least two features")
    ranked = np.column_stack([_ranks(X[:,i]) for i in range(X.shape[1])])
    correlation = np.nan_to_num(np.corrcoef(ranked, rowvar=False))
    np.fill_diagonal(correlation, 1)
    distance = 1-np.abs(correlation)
    clusters,active,linkage = {i:(i,) for i in range(len(names))},list(range(len(names))),[]
    for step in range(len(names)-1):
        left,right = min(((a,b) for i,a in enumerate(active) for b in active[i+1:]),
            key=lambda pair: distance[np.ix_(clusters[pair[0]], clusters[pair[1]])].mean())
        members = clusters[left]+clusters[right]
        value = distance[np.ix_(clusters[left], clusters[right])].mean()
        linkage.append((left, right, value, len(members)))
        new = len(names)+step
        clusters[new] = members
        active = [cluster for cluster in active if cluster not in (left, right)]+[new]
    order = clusters[active[0]] if active else (0,)
    return FeatureRelations(names, correlation, np.asarray(linkage), order)

def feature_dependence(X, n_samples=5000, n_trees=25, seed=42, feature_names=None):
    "Predict each feature from the others and measure nonlinear permutation dependencies."
    from .core import FastForest
    X = _sample(X, n_samples=n_samples, seed=seed)
    X,names = _data(X, feature_names)
    if X.shape[1] < 2: raise ValueError("feature dependence requires at least two features")
    if len(X) < 5: raise ValueError("feature dependence requires at least five rows")
    order = np.random.default_rng(seed).permutation(len(X))
    split = max(1, len(X)*4//5)
    train,valid = order[:split],order[split:]
    scores = np.empty(len(names))
    matrix = np.zeros((len(names), len(names)))
    for target in range(len(names)):
        keep = np.asarray([i for i in range(len(names)) if i != target])
        try: target_values = np.asarray(X[:,target], dtype=np.float32)
        except (TypeError, ValueError): target_values = _ranks(X[:,target]).astype(np.float32)
        model = FastForest(n_trees=n_trees, seed=seed+target).fit(X[train][:,keep], target_values[train])
        importance = permutation_importance(model, X[valid][:,keep], target_values[valid], n_repeats=1,
            n_samples=None, seed=seed, feature_names=np.asarray(names)[keep])
        scores[target] = importance.baseline
        matrix[target,keep] = np.maximum(0, importance.values)
    np.fill_diagonal(matrix, np.nan)
    return FeatureDependence(names, scores, matrix)
