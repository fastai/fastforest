import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .preprocessing import _take_rows

@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    maximize: bool

@dataclass(frozen=True)
class DatasetView:
    dataset: str
    comparison: str = None
    link: bool = True
    models: tuple = None

REGRESSION = (Metric("rmse", "RMSE ↓", False), Metric("r2", "R² ↑", True),
    Metric("fit", "Fit (s) ↓", False), Metric("predict", "Predict (s) ↓", False))
CLASSIFICATION = (Metric("f1", "F1 acc ↑", True), Metric("log_loss", "Log loss ↓", False),
    Metric("fit", "Fit (s) ↓", False), Metric("proba", "Proba (s) ↓", False))
MODEL_ORDER = ("fastforest", "AutoForest", "autogrow", "sklearn RF", "sklearn HistGBM")

@dataclass(frozen=True)
class ScreenResult:
    label: str
    changes: dict
    oob_loss: float
    train_loss: float
    coverage: float
    evaluated_rows: int
    nodes_mean: float
    leaves_mean: float
    depth_mean: float

@dataclass(frozen=True)
class ScreenReport:
    task: str
    trees: int
    pool_rows: int
    batch_seconds: float
    results: tuple
    feature_metadata: dict

@dataclass(frozen=True)
class ValidationResult:
    label: str
    changes: dict
    validation_loss: float
    train_loss: float
    fit_seconds: float
    predict_seconds: float
    trees: int
    pool_rows: int
    nodes_mean: float
    leaves_mean: float
    depth_mean: float

@dataclass(frozen=True)
class ValidationReport:
    task: str
    batch_seconds: float
    results: tuple

@dataclass(frozen=True)
class _PreparedSweep:
    task: str
    base: dict
    seed: int
    configs: tuple
    target: np.ndarray
    outputs: int
    replacements: tuple
    plans: tuple

FOREST_PARAMS = ("min_node_size", "bootstrap_fraction", "bootstrap_max", "replacement", "max_node_samples",
    "cutoff_divisor", "random_splitter", "max_features")
ADVISOR_PARAMS = ("min_node_size", "bootstrap_fraction", "bootstrap_max", "replacement", "max_node_samples",
    "random_splitter", "max_features")
STANDARD_ALTERNATIVES = {
    "min_node_size":(2,4,16,64), "bootstrap_fraction":(.5,), "bootstrap_max":(80_000,240_000),
    "replacement":(True,False), "max_node_samples":(160,640,2560),
    "split_prior_rows":(1.,3.,5.,8.), "class_weight_power":(.25,.5,1.),
    "cutoff_divisor":(), "random_splitter":(True,),
    "max_features":(.6,.9,1.,"sqrt")}

def forest_params(model):
    "Forest parameters shared by a task's batched screening configurations."
    from .core import FastForestClassifier
    return FOREST_PARAMS+(("class_weight_power",) if isinstance(model, FastForestClassifier) else ("split_prior_rows",))

def standard_levels(model):
    "One-axis levels beginning with this model's task-specific baseline."
    base = model.get_params()
    return {name:(base[name],)+tuple(value for value in STANDARD_ALTERNATIVES[name] if value != base[name]) for name in forest_params(model)}

def _level_text(value):
    if value is None: return "none"
    if isinstance(value, bool): return str(value).lower()
    return f"{value:g}" if isinstance(value, float) else str(value)

def forest_suite(model, levels=None):
    "Return one-axis forest configurations; every level list starts with its shared baseline."
    base = model.get_params()
    names = forest_params(model)
    levels = standard_levels(model) if levels is None else levels
    unknown = set(levels)-set(names)
    if unknown: raise ValueError(f"unknown forest parameters: {sorted(unknown)}")
    for name in names:
        values = tuple(levels.get(name, (base[name],)))
        if not values: raise ValueError(f"{name} must contain at least its baseline")
        if values[0] != base[name]: raise ValueError(f"the first {name} value must equal the model baseline {base[name]!r}")
    candidates = [("defaults", {})]
    candidates += [(f"{name}={_level_text(value)}", {name:value}) for name in names for value in tuple(levels.get(name, ()))[1:]]
    result,seen = [],set()
    for label,changes in candidates:
        params = {name:changes.get(name, base[name]) for name in names}
        key = tuple(params[name] for name in names)
        if key in seen: continue
        seen.add(key)
        result.append((label, changes, params))
    return result

def advisor_features(data, categorical=False, task=None, sweep_labels=None, params=None):
    "Build advisor inputs from one row per screened forest configuration."
    import pandas as pd
    data = data.copy()
    tasks = data.task.unique()
    if task is None:
        if len(tasks) != 1: raise ValueError("task is required when advisor rows contain multiple tasks")
        task = tasks[0]
    if task not in ("regression", "classification") or set(tasks) != {task}: raise ValueError(f"invalid advisor task {task!r}")
    if "n_rows" not in data: data["n_rows"] = data.rows
    for name in ("n_missing_cols", "n_observed_missing_cols", "n_date_cols", "n_grouped_cols"):
        if name not in data: data[name] = 0
    screen_default = data.loc[data.label == "defaults", ["dataset", "screen_oob_loss", "screen_train_loss"]].set_index("dataset")
    data["screen_oob_relative"] = data.screen_oob_loss/data.dataset.map(screen_default.screen_oob_loss)
    data["screen_train_relative"] = data.screen_train_loss/data.dataset.map(screen_default.screen_train_loss)
    if sweep_labels is not None:
        if len(sweep_labels) != 8 or len(set(sweep_labels)) != 8: raise ValueError("sweep_labels must contain eight unique configurations")
        for number,label in enumerate(sweep_labels, 1):
            values = data.loc[data.label == label, ["dataset", "screen_oob_relative"]].set_index("dataset").screen_oob_relative
            if len(values) != data.dataset.nunique(): raise ValueError(f"every dataset must contain {label!r}")
            data[f"hparam_rel_oob{number}"] = data.dataset.map(values)
    data["changed_param"] = data.label.str.partition("=")[0].where(~data.label.str.startswith("joint_"), "joint")
    for name in ("screen_nodes_mean", "screen_depth_mean"):
        baseline = data.loc[data.label == "defaults", ["dataset", name]].set_index("dataset")[name]
        data[f"{name}_relative"] = data[name]/data.dataset.map(baseline)
    inputs = ["n_rows", "n_raw_cols", "n_logical_cols", "n_encoded_cols", "n_numeric_cols", "n_lexical_cols",
        "n_binary_cols", "n_low_card_cols", "n_high_card_cols", "n_constant_cols", "n_observed_missing_cols", "n_date_cols",
        "screen_oob_loss", "screen_train_loss", "screen_oob_relative", "screen_train_relative",
        "screen_coverage", "screen_evaluated_rows", "screen_nodes_mean", "screen_depth_mean",
        "screen_nodes_mean_relative", "screen_depth_mean_relative", "full_trees", "full_pool_rows", "changed_param"]
    if sweep_labels is not None: inputs += [f"hparam_rel_oob{number}" for number in range(1,9)]
    inputs += (["target_mean", "target_std", "target_unique_fraction"] if task == "regression" else
        ["n_classes", "majority_fraction", "target_entropy", "target_unique_fraction"])
    upstream = ["task_type", "domain", "source", "has_datetime", "has_categorical", "has_numerical", "has_binary",
        "has_high_cardinality_categorical", "num_instance_groups", "num_high_cardinality_cats", "missing_value_fraction", "dataset_year"]
    for name in upstream:
        if name not in data: data[name] = np.nan
    if params is None:
        params = (("min_node_size", "bootstrap_fraction", "bootstrap_max", "max_node_samples", "max_features")
            if task == "regression" else
            ("min_node_size", "bootstrap_fraction", "max_node_samples", "class_weight_power", "max_features"))
    inputs += upstream+list(params)
    X = data[inputs].copy()
    for name in X.select_dtypes(include=["object", "str"]):
        if name not in FOREST_PARAMS: X[name] = X[name].fillna("(missing)").astype(str)
    for name in X.columns[X.isna().any()]:
        if name not in FOREST_PARAMS: X[name] = X[name].fillna(-1)
    if categorical:
        for name in params: X[name] = X[name].fillna("none").astype(str)
    else:
        X.bootstrap_fraction = X.bootstrap_fraction.fillna(1.)
        X.replacement = X.replacement.map({True:1.,False:0.}).fillna(-1.)
        X.random_splitter = X.random_splitter.astype(np.uint8)
        X["max_features_sqrt"] = (X.max_features == "sqrt").astype(np.uint8)
        X.max_features = pd.to_numeric(X.max_features, errors="coerce").fillna(0.)
    return X

def _prediction_loss(task, forest, encoded, target, classes=None):
    if task == "classification":
        probabilities = np.asarray(forest.predict_proba(encoded))
        losses = np.square(probabilities).sum(axis=1)+1
        known = target < probabilities.shape[1]
        losses[known] -= 2*probabilities[np.arange(len(target))[known],target[known]]
        return float(np.mean(losses))
    residual = np.asarray(forest.predict(encoded))-target
    return float(np.mean(np.square(residual, dtype=np.float64)))

def _feature_metadata(encoder):
    "Summarize the bounded fitted schema for sweep meta-learning."
    n_dates = len(encoder._dates)
    columns = encoder.column_info[:-16*n_dates] if n_dates else encoder.column_info
    cardinalities = np.asarray([column.cardinality for column in columns])
    return dict(n_raw_cols=len(encoder.input_names), n_logical_cols=len(encoder.column_info),
        n_encoded_cols=len(encoder.encoded_names), n_numeric_cols=sum(column.kind == "numeric" for column in columns),
        n_lexical_cols=sum(column.kind == "lexical" for column in columns), n_binary_cols=int((cardinalities == 2).sum()),
        n_low_card_cols=int(((cardinalities > 2)&(cardinalities <= 4)).sum()),
        n_high_card_cols=int((cardinalities > 4).sum()), n_constant_cols=int((cardinalities <= 1).sum()),
        n_observed_missing_cols=sum(column.had_missing for column in columns), n_date_cols=n_dates,
        n_grouped_cols=len(encoder._bundles))

def _prepare_sweep(model, X, y, configs, seed, trees, oob):
    from .core import FastForest,FastForestClassifier,_estimated_outputs,_fit_plan,_resolve_replacement
    if not isinstance(model, (FastForest,FastForestClassifier)): raise TypeError("model must be FastForest or FastForestClassifier")
    task = "classification" if isinstance(model, FastForestClassifier) else "regression"
    base = model.get_params()
    seed = base["seed"] if seed is None else seed
    if seed is None: seed = 42
    configs = tuple(forest_suite(model) if configs is None else configs)
    if not configs: raise ValueError("configs must contain at least one configuration")
    target = np.asarray(y)
    if target.ndim != 1: raise ValueError("y must be a one-dimensional array")
    if len(X) != len(target): raise ValueError(f"X has {len(X)} rows but y has {len(target)} values")
    outputs = _estimated_outputs(target, seed) if task == "classification" else 1
    replacements = tuple(_resolve_replacement(params["replacement"], len(X), task == "classification") for _,_,params in configs)
    plans = tuple(_fit_plan(len(X), trees, params["bootstrap_fraction"], params["bootstrap_max"], replacement, oob, outputs)
        for (_,_,params),replacement in zip(configs,replacements))
    return _PreparedSweep(task, base, seed, configs, target, outputs, replacements, plans)

def _native_config(params, replacement, plan, target_rows, seed, oob):
    from .core import _native_max_features
    kind,value = _native_max_features(params["max_features"])
    return dict(n_trees=plan[0], min_node_size=params["min_node_size"], bootstrap_fraction=params["bootstrap_fraction"],
        bootstrap_max=params["bootstrap_max"], sample_rows=min(plan[1], target_rows), replacement=replacement,
        max_node_samples=params["max_node_samples"], split_prior_rows=params.get("split_prior_rows", 0.),
        class_weight_power=params.get("class_weight_power", .75), cutoff_divisor=params["cutoff_divisor"], seed=seed, oob=oob,
        random_splitter=params["random_splitter"], max_features_kind=kind, max_features_value=value)

def screen(model, X, y, configs=None, trees=8, seed=None):
    "Fit one encoded, parallel OOB batch and return forest-configuration diagnostics."
    from .core import _ClassifierForest,_Encoder,_Forest,_class_vector,_fit_plan,_sample_indices,_vector
    if trees < 1: raise ValueError("trees must be positive")
    prepared = _prepare_sweep(model, X, y, configs, seed, trees, True)
    task,base,seed,configs,y_array = prepared.task,prepared.base,prepared.seed,prepared.configs,prepared.target
    replacements,plans = prepared.replacements,prepared.plans
    for (_,_,params),replacement,plan in zip(configs, replacements, plans):
        if not replacement and plan[1] >= plan[2]:
            raise ValueError(f"configuration {params} leaves no rows for OOB evaluation")
    pool_rows = max(plan[2] for plan in plans)
    indices = None if pool_rows == len(X) else np.asarray(_sample_indices(len(X), pool_rows, seed, 2))
    encoder = _Encoder(base["missing_values"], base["date_columns"], base["allow_new_missing"])
    encoded = encoder.fit_transform(X, indices)
    training = encoder.transform(_take_rows(X, indices))
    if task == "classification":
        classes,target = _class_vector(y_array, indices)
        fitted_outputs = max(1,len(classes)-1)
    else:
        target = _vector(y_array, indices=indices)
        fitted_outputs = 1
    native_configs = []
    oob_rows = min(len(target), 40_000*fitted_outputs)
    for (_,_,params),replacement,plan in zip(configs, replacements, plans):
        fitted_plan = _fit_plan(len(X), trees, params["bootstrap_fraction"], params["bootstrap_max"],
            replacement, True, fitted_outputs)
        native_configs.append(_native_config(params, replacement, fitted_plan, len(target), seed, True))
    started = perf_counter()
    args = (encoded, target, encoder.cutoff_values, encoder.cutoff_offsets, encoder.feature_group_ids)
    if task == "classification": forests = _ClassifierForest.fit_batch(encoded, target, len(classes), *args[2:], native_configs, oob_rows)
    else: forests = _Forest.fit_batch(*args, native_configs, oob_rows)
    batch_seconds = perf_counter()-started
    results = []
    for (label,changes,_),forest in zip(configs, forests):
        counts,local = np.asarray(forest.oob_counts),np.asarray(forest.oob_indices)
        valid = counts > 0
        if task == "classification":
            probabilities = np.asarray(forest.oob_decision_function)[valid]
            expected = target[local[valid]]
            loss = np.mean(np.square(probabilities).sum(axis=1)-2*probabilities[np.arange(len(expected)),expected]+1)
        else:
            residual = np.asarray(forest.oob_prediction)[valid]-target[local[valid]]
            loss = np.mean(np.square(residual, dtype=np.float64))
        train_loss = _prediction_loss(task, forest, training, target, classes if task == "classification" else None)
        structures = np.asarray(forest.tree_structures)
        results.append(ScreenResult(label, changes, float(loss), train_loss, float(valid.mean()), len(counts),
            float(structures[:,0].mean()), float(structures[:,1].mean()), float(structures[:,2].mean())))
    return ScreenReport(task, trees, pool_rows, batch_seconds, tuple(results), _feature_metadata(encoder))

def validate(model, X_train, y_train, X_valid, y_valid, configs=None, seed=None, allow_unseen_classes=False):
    "Fit the same one-axis configurations with ordinary resolved tree counts and score train/validation data."
    from .core import _ClassifierForest,_Encoder,_Forest,_class_vector,_fit_plan,_sample_indices,_vector
    y_train,y_valid = np.asarray(y_train),np.asarray(y_valid)
    if len(X_valid) != len(y_valid): raise ValueError("feature and target row counts must match")
    prepared = _prepare_sweep(model, X_train, y_train, configs, seed, None, False)
    task,base,seed,configs = prepared.task,prepared.base,prepared.seed,prepared.configs
    replacements,plans = prepared.replacements,prepared.plans
    grouped = {}
    for index,plan in enumerate(plans): grouped.setdefault(tuple(plan), []).append(index)
    results = [None]*len(configs)
    batch_seconds = 0.
    for (trees,rows_per_tree,pool_rows),indices_in_group in grouped.items():
        started = perf_counter()
        indices = None if pool_rows == len(X_train) else np.asarray(_sample_indices(len(X_train), pool_rows, seed, 2))
        encoder = _Encoder(base["missing_values"], base["date_columns"], base["allow_new_missing"])
        encoded = encoder.fit_transform(X_train, indices)
        if task == "classification":
            classes,target = _class_vector(y_train, indices)
            fitted_outputs = max(1,len(classes)-1)
            lookup = {value:index for index,value in enumerate(classes.tolist())}
            if allow_unseen_classes: valid_target = np.asarray([lookup.get(value,len(classes)) for value in y_valid], dtype=np.uint32)
            else:
                try: valid_target = np.asarray([lookup[value] for value in y_valid], dtype=np.uint32)
                except KeyError as error: raise ValueError(f"validation target contains unseen class {error.args[0]!r}") from error
        else:
            classes = None
            fitted_outputs = 1
            target,valid_target = _vector(y_train, indices=indices),_vector(y_valid)
        fit_preprocess_seconds = perf_counter()-started
        training = encoder.transform(_take_rows(X_train, indices))
        started = perf_counter()
        validation = encoder.transform(X_valid)
        predict_preprocess_seconds = perf_counter()-started
        args = (encoded, target, encoder.cutoff_values, encoder.cutoff_offsets, encoder.feature_group_ids)
        for index in indices_in_group:
            params = configs[index][2]
            replacement = replacements[index]
            fitted_plan = _fit_plan(len(X_train), None, params["bootstrap_fraction"], params["bootstrap_max"],
                replacement, False, fitted_outputs)
            native_config = _native_config(params, replacement, fitted_plan, len(target), seed, False)
            started = perf_counter()
            if task == "classification": forest = _ClassifierForest.fit_batch(encoded, target, len(classes), *args[2:], [native_config], 1)[0]
            else: forest = _Forest.fit_batch(*args, [native_config], 1)[0]
            native_fit_seconds = perf_counter()-started
            batch_seconds += native_fit_seconds
            started = perf_counter()
            if task == "classification":
                prediction = np.asarray(forest.predict_proba(validation))
                native_predict_seconds = perf_counter()-started
                losses = np.square(prediction).sum(axis=1)+1
                known = valid_target < prediction.shape[1]
                losses[known] -= 2*prediction[np.arange(len(valid_target))[known],valid_target[known]]
                validation_loss = float(np.mean(losses))
            else:
                prediction = np.asarray(forest.predict(validation))
                native_predict_seconds = perf_counter()-started
                validation_loss = float(np.mean(np.square(prediction-valid_target, dtype=np.float64)))
            label,changes,_ = configs[index]
            structures = np.asarray(forest.tree_structures)
            results[index] = ValidationResult(label, changes, validation_loss,
                _prediction_loss(task, forest, training, target, classes), fit_preprocess_seconds+native_fit_seconds,
                predict_preprocess_seconds+native_predict_seconds, fitted_plan[0], pool_rows,
                float(structures[:,0].mean()), float(structures[:,1].mean()), float(structures[:,2].mean()))
    return ValidationReport(task, batch_seconds, tuple(results))

def _read_csv(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))

def load_datasets(path):
    "Load the shared dataset catalogue keyed by its stable name."
    return {row["dataset"]:row for row in _read_csv(path)}

def load_results(path, datasets):
    "Join benchmark measurements to their shared dataset metadata."
    rows = _read_csv(path)
    missing = {row["dataset"] for row in rows}-datasets.keys()
    if missing: raise ValueError(f"unknown benchmark datasets: {sorted(missing)}")
    return [{**datasets[row["dataset"]], **row} for row in rows]

def _display(metric, value):
    "Return the displayed text and its rounded numeric value."
    value = float(value)
    if metric.key == "rmse":
        text = f"{value:,.0f}" if abs(value) > 100 else f"{value:,.1f}" if abs(value) > 10 else f"{value:.2f}"
    elif metric.key == "fit": text = f"{value:.2f}"
    elif metric.key in ("predict", "proba"): text = f"{value:.3f}"
    else: text = f"{value:.2f}"
    return text,float(text.replace(",", ""))

def _dataset_rows(results, view):
    rows = [row for row in results if row["dataset"] == view.dataset]
    if view.comparison:
        models = {row["model"] for row in rows}
        rows += [row for row in results if row["dataset"] == view.comparison and row["model"] not in models]
    if view.models: rows = [row for row in rows if row["model"] in view.models]
    return sorted(rows, key=lambda row: MODEL_ORDER.index(row["model"]))

def _inline(*children):
    from fastcore.xml import Safe,to_xml
    return Safe(to_xml(children, indent=False))

def _dataset_cell(row, detail, rowspan, link):
    from fastcore.xml import A,Br,Strong,Sub,Td
    label = A(row["label"], href=row["url"]) if link else row["label"]
    children = Strong(label),Br(),Sub(f'{int(row["rows"]):,} rows · {row[f"{detail}_detail"]}')
    return Td(_inline(*children), rowspan=rowspan if rowspan > 1 else None)

def mk_table(results, datasets, metrics, detail="benchmark", dataset_col=True):
    "Render benchmark rows as HTML, bolding every best value after display rounding."
    from fastcore.xml import Strong,Table,Tbody,Td,Th,Thead,Tr,to_xml
    headers = ([Th("Dataset")] if dataset_col else [])+[Th("Model")]+[Th(metric.label, align="right") for metric in metrics]
    body = []
    for view in datasets:
        rows = _dataset_rows(results, view)
        if not rows: raise ValueError(f"no benchmark rows for {view.dataset!r}")
        shown = [{metric.key:_display(metric, row[metric.key]) for metric in metrics if row[metric.key]} for row in rows]
        best = {}
        for metric in metrics:
            values = [display[metric.key][1] for display in shown if metric.key in display]
            best[metric.key] = (max if metric.maximize else min)(values) if values else None
        for index,(row,display) in enumerate(zip(rows, shown)):
            cells = []
            if dataset_col and index == 0: cells.append(_dataset_cell(rows[0], detail, len(rows), view.link))
            model = _inline(Strong(row["model"])) if row["model"].startswith("fastforest") else row["model"]
            cells.append(Td(model))
            if row["status"]:
                cells.append(Td(row["note"], colspan=len(metrics), align="center"))
            else:
                for metric in metrics:
                    text,value = display[metric.key]
                    value = _inline(Strong(text)) if value == best[metric.key] else text
                    cells.append(Td(value, align="right"))
            body.append(Tr(*cells))
    return str(to_xml(Table(Thead(Tr(*headers)), Tbody(*body)))).rstrip()
