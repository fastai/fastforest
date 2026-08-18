import csv,multiprocessing as mp,traceback
from pathlib import Path

import numpy as np
from fastcore.script import call_parse
from sklearn.model_selection import train_test_split

from accuracy import Dataset,_rows,_stop,load_data,split_indices
from fastforest import FastForest,FastForestClassifier
from fastforest.tools import FOREST_PARAMS,forest_suite,screen,validate

def _values(raw, parse, name, none=False):
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens): raise ValueError(f"{name} must be a comma-separated list")
    def convert(token):
        if token.lower() == "default": return "default"
        if none and token.lower() in ("none", "null"): return None
        try: return parse(token)
        except ValueError as error: raise ValueError(f"invalid {name} value {token!r}") from error
    return tuple(convert(token) for token in tokens)

def _boolean(token):
    if token.lower() in ("true", "1", "yes"): return True
    if token.lower() in ("false", "0", "no"): return False
    raise ValueError

def _feature(token): return "sqrt" if token.lower() == "sqrt" else float(token)

def _setup(task, levels):
    cls = FastForestClassifier if task == "classification" else FastForest
    defaults = cls().get_params()
    levels = {name:tuple(defaults[name] if value == "default" else value for value in values) for name,values in levels.items()}
    irrelevant = "split_prior_rows" if task == "classification" else "class_weight_power"
    levels = {name:values for name,values in levels.items() if name != irrelevant}
    baseline = {name:levels[name][0] for name in (*FOREST_PARAMS, "class_weight_power" if task == "classification" else "split_prior_rows")}
    return cls,baseline,levels

def _worker(send, dataset, data_home, max_rows, seed, screen_trees, levels):
    try:
        dataset = Dataset(dataset)
        _,X,y,missing,task = load_data(dataset, data_home)
        y = np.asarray(y, dtype=np.float32 if task == "regression" else None)
        if max_rows is not None and max_rows < len(X):
            selected,_ = train_test_split(np.arange(len(X)), train_size=max_rows, random_state=seed,
                stratify=y if task == "classification" else None)
            X,y = _rows(X, selected),y[selected]
        train_idx,valid_idx,split = split_indices(dataset, X, y if task == "classification" else None)
        if dataset == Dataset.walmart_nodate: X = X.drop(columns="Date")
        X_train,X_valid,y_train,y_valid = _rows(X, train_idx),_rows(X, valid_idx),y[train_idx],y[valid_idx]
        cls,model_args,levels = _setup(task, levels)
        model = cls(**model_args, missing_values=missing, seed=seed)
        suite = forest_suite(model, levels)
        send.send(("ready", (task,len(X),X.shape[1],len(train_idx),len(valid_idx),split,len(suite))))
        send.send(("screen", screen(model, X_train, y_train, suite, screen_trees, seed)))
        send.send(("validation", validate(model, X_train, y_train, X_valid, y_valid, suite, seed)))
    except BaseException: send.send(("error", traceback.format_exc()))
    finally: send.close()

def _receive(receive, process, timeout, phase):
    if not receive.poll(timeout):
        _stop(process)
        raise TimeoutError(f"{phase} exceeded {timeout} seconds")
    status,result = receive.recv()
    if status == "error": raise RuntimeError(result)
    return status,result

def _run(dataset, data_home, max_rows, seed, screen_trees, load_timeout, screen_timeout, validation_timeout, levels):
    context = mp.get_context("spawn")
    receive,send = context.Pipe(False)
    process = context.Process(target=_worker, args=(send,dataset,data_home,max_rows,seed,screen_trees,levels))
    process.start()
    send.close()
    status,meta = _receive(receive, process, load_timeout, "dataset loading")
    if status != "ready": raise RuntimeError(f"expected ready message, got {status!r}")
    status,screen_report = _receive(receive, process, screen_timeout, "eight-tree OOB screen")
    if status != "screen": raise RuntimeError(f"expected screen result, got {status!r}")
    status,validation_report = _receive(receive, process, validation_timeout, "full-tree validation sweep")
    if status != "validation": raise RuntimeError(f"expected validation result, got {status!r}")
    process.join(5)
    if process.is_alive(): _stop(process)
    receive.close()
    return meta,screen_report,validation_report

@call_parse
def main(
    dataset:Dataset=Dataset.sgemm, # Dataset to evaluate
    screen_trees:int=8,            # Trees per batched OOB screening configuration
    min_node_size:str="default,4,16,32,64", # Model default first, followed by one-axis alternatives
    bootstrap_fraction:str="none,0.5", # Baseline first; none uses the model rule
    bootstrap_max:str="default,80000,160000", # Model default first; none disables the cap
    replacement:str="none,true,false", # Baseline first; none uses the adaptive task/row rule
    max_node_samples:str="default,160,640,1280,2560", # Model default first
    split_prior_rows:str="default,1,5,8", # Regression split-score prior rows
    class_weight_power:str="default,0.25,0.5,1", # Classification inverse-frequency weighting
    cutoff_divisor:str="default", # Model default first; relevant to the random splitter
    random_splitter:str="false",  # Baseline first
    max_features:str="default,0.6,0.75,0.9,1,sqrt", # Task-specific model default first
    seed:int=42,                   # Sampling and forest seed
    load_timeout:int=180,          # Maximum dataset-loading seconds
    screen_timeout:int=180,        # Maximum eight-tree OOB batch seconds
    validation_timeout:int=180,    # Maximum full-tree sweep seconds
    max_rows:int=None,             # Optional reproducible dataset row limit
    output:str=None,               # Result CSV; defaults to meta/sweeps/<dataset>.csv
    data_home:str=None,            # Dataset cache directory
):
    "Compare an eight-tree OOB screen with ordinary resolved-tree validation fits."
    if min(load_timeout, screen_timeout, validation_timeout, screen_trees) < 1: raise ValueError("trees and timeouts must be positive")
    root = Path(__file__).parents[1]
    if data_home is None: data_home = root/".data"
    levels = {
        "min_node_size":_values(min_node_size, int, "min_node_size"),
        "bootstrap_fraction":_values(bootstrap_fraction, float, "bootstrap_fraction", True),
        "bootstrap_max":_values(bootstrap_max, int, "bootstrap_max", True),
        "replacement":_values(replacement, _boolean, "replacement", True),
        "max_node_samples":_values(max_node_samples, int, "max_node_samples"),
        "split_prior_rows":_values(split_prior_rows, float, "split_prior_rows"),
        "class_weight_power":_values(class_weight_power, float, "class_weight_power"),
        "cutoff_divisor":_values(cutoff_divisor, float, "cutoff_divisor"),
        "random_splitter":_values(random_splitter, _boolean, "random_splitter"),
        "max_features":_values(max_features, _feature, "max_features"),
    }
    meta,screen_report,validation_report = _run(str(dataset), data_home, max_rows, seed, screen_trees,
        load_timeout, screen_timeout, validation_timeout, levels)
    task,n_rows,n_features,train_rows,valid_rows,split,n_configs = meta
    rows = []
    cls,model_args,levels = _setup(task, levels)
    suite = forest_suite(cls(**model_args), levels)
    for screened,validated,(label,_,params) in zip(screen_report.results, validation_report.results, suite):
        row = dict(dataset=dataset, task=task, loss="brier" if task == "classification" else "mse", rows=n_rows, features=n_features, train_rows=train_rows,
            validation_rows=valid_rows, validation_split=split, label=label, screen_trees=screen_trees,
            screen_oob_loss=screened.oob_loss, screen_train_loss=screened.train_loss, screen_coverage=screened.coverage,
            screen_evaluated_rows=screened.evaluated_rows, screen_nodes_mean=screened.nodes_mean,
            screen_leaves_mean=screened.leaves_mean, screen_depth_mean=screened.depth_mean,
            full_trees=validated.trees, full_validation_loss=validated.validation_loss, full_train_loss=validated.train_loss,
            full_fit_seconds=validated.fit_seconds, full_predict_seconds=validated.predict_seconds,
            full_pool_rows=validated.pool_rows, full_nodes_mean=validated.nodes_mean,
            full_leaves_mean=validated.leaves_mean, full_depth_mean=validated.depth_mean, **params)
        rows.append(row)
        print(f"{label:<32} oob={screened.oob_loss:.6g} screen-train={screened.train_loss:.6g} "
            f"valid={validated.validation_loss:.6g} full-train={validated.train_loss:.6g} fit={validated.fit_seconds:.3f}s "
            f"predict={validated.predict_seconds:.3f}s trees={validated.trees}")
    if len(rows) != n_configs: raise RuntimeError("worker and reporting suites disagree")
    output = Path(output) if output else root/"meta"/"sweeps"/f"{dataset}.csv"
    if not output.is_absolute(): output = root/output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} configurations · OOB batch {screen_report.batch_seconds:.3f}s · "
        f"full batches {validation_report.batch_seconds:.3f}s · saved to {output}")
