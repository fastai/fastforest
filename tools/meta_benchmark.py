import json,multiprocessing as mp
import os,re,signal,traceback,urllib.request
from pathlib import Path

import numpy as np,pandas as pd
from fastcore.script import call_parse
import pyarrow.parquet as pq

from fastforest import FastForest,FastForestClassifier
from fastforest.tools import forest_suite,screen,validate

BEYOND_METADATA = "https://raw.githubusercontent.com/autogluon/tabarena/main/packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/BeyondArena_tasks_metadata.csv"

def beyond_manifest(path, include_text=False):
    "Load one canonical split per BeyondArena dataset."
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BEYOND_METADATA, path)
    source = pd.read_csv(path)
    tasks = source[(source.repeat == 0)&(source.fold == 0)].copy()
    if not include_text: tasks = tasks[~tasks.has_text]
    tasks["dataset"] = tasks.tabarena_task_name
    tasks["source_group"] = tasks.dataset_name
    tasks["task"] = np.where(tasks.is_classification, "classification", "regression")
    tasks["rows"] = tasks.num_instances
    tasks["features"] = tasks.num_features
    tasks["uuid"] = tasks.data_foundry_uri.str.rsplit("/", n=1).str[-1]
    tasks["collection"] = "beyondarena"
    return tasks.reset_index(drop=True)

def _slug(name): return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def amlb_manifest(path, data_home):
    "Load downloaded, non-overlapping AMLB datasets."
    tasks = pd.read_csv(path)
    tasks["dataset"] = "amlb_"+tasks.name.map(_slug)
    tasks["source_group"] = tasks.dataset
    tasks["collection"] = "amlb"
    tasks["data_path"] = tasks.name.map(_slug)
    def downloaded(name):
        path = Path(data_home)/"amlb"/name/"data.pq"
        if not path.exists() or not path.stat().st_size: return False
        try: pq.ParquetFile(path); return True
        except Exception: return False
    available = tasks.data_path.map(downloaded)
    return tasks[available].reset_index(drop=True)

def _date_columns(X):
    dates = {}
    for name in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[name]): dates[name] = "%Y-%m-%d %H:%M:%S"
    if not dates: return X,None
    X = X.copy()
    for name,format in dates.items(): X[name] = X[name].dt.strftime(format)
    return X,dates

def _target_metadata(y, task):
    y = np.asarray(y)
    if task == "classification":
        _,counts = np.unique(y, return_counts=True)
        probabilities = counts/counts.sum()
        return dict(n_classes=len(counts), output_dimensions=max(1,len(counts)-1),
            majority_fraction=float(probabilities.max()), minority_fraction=float(probabilities.min()),
            target_entropy=float(-(probabilities*np.log(probabilities)).sum()), target_mean=0., target_std=0.,
            target_unique_fraction=len(counts)/len(y))
    values = np.asarray(y, dtype=np.float64)
    return dict(n_classes=1, output_dimensions=1, majority_fraction=1., minority_fraction=1., target_entropy=0.,
        target_mean=float(values.mean()), target_std=float(values.std()), target_unique_fraction=len(np.unique(values))/len(values))

def _split(rows, seed):
    indices = np.random.default_rng(seed).permutation(rows)
    cut = max(1,min(rows-1,round(rows*.8)))
    return indices[:cut],indices[cut:]

def _load_task(task, data_home, seed):
    if task["collection"] == "amlb":
        folder = Path(data_home)/"amlb"/task["data_path"]
        frame = pd.read_parquet(folder/"data.pq")
        with open(folder/"metadata.json") as handle: metadata = json.load(handle)["data_set_description"]
        target = task.get("target")
        if not isinstance(target,str) or not target: target = metadata.get("default_target_attribute")
        if not target or target not in frame: raise ValueError(f"invalid target {target!r}")
        y = frame[target]
        X,dates = _date_columns(frame.drop(columns=target))
        train_idx,valid_idx = _split(len(frame), seed)
        return X,y,train_idx,valid_idx,dates,"random 80/20"
    cache = Path(data_home)
    os.environ.setdefault("HF_HOME", str(cache/"huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(cache/"huggingface"/"xet"))
    from data_foundry.collections import BEYOND_ARENA
    container = BEYOND_ARENA.get_dataset(task["uuid"], cache_dir=str(cache/"data_foundry"))
    target = container.task_metadata.target_column_name
    frame = container.dataset
    y = frame[target]
    X,dates = _date_columns(frame.drop(columns=target))
    splits = container.experiment_metadata.splits
    repeat = min(splits)
    fold = min(splits[repeat])
    train_idx,valid_idx = (np.asarray(values, dtype=np.int64) for values in splits[repeat][fold])
    return X,y,train_idx,valid_idx,dates,f"BeyondArena {task['task_type']} r{repeat}f{fold}"

def _worker(send, task, data_home, screen_trees, seed):
    try:
        X,y,train_idx,valid_idx,dates,validation_split = _load_task(task, data_home, seed)
        X_train,X_valid,y_train,y_valid = X.iloc[train_idx],X.iloc[valid_idx],y.iloc[train_idx],y.iloc[valid_idx]
        cls = FastForestClassifier if task["task"] == "classification" else FastForest
        model = cls(seed=seed, date_columns=dates, allow_new_missing=True)
        suite = forest_suite(model)
        screened = screen(model, X_train, y_train, suite, screen_trees, seed)
        validated = validate(model, X_train, y_train, X_valid, y_valid, suite, seed, allow_unseen_classes=True)
        target_meta = _target_metadata(y_train, task["task"])
        base = dict(task, rows=len(X), features=X.shape[1], train_rows=len(train_idx), validation_rows=len(valid_idx),
            validation_split=validation_split, screen_trees=screen_trees,
            **screened.feature_metadata, **target_meta)
        rows = []
        for screen_result,validation_result,(label,_,params) in zip(screened.results, validated.results, suite):
            rows.append(dict(base, loss="brier" if task["task"] == "classification" else "mse", label=label,
                screen_oob_loss=screen_result.oob_loss, screen_train_loss=screen_result.train_loss,
                screen_coverage=screen_result.coverage, screen_evaluated_rows=screen_result.evaluated_rows,
                screen_nodes_mean=screen_result.nodes_mean, screen_leaves_mean=screen_result.leaves_mean,
                screen_depth_mean=screen_result.depth_mean, full_trees=validation_result.trees,
                full_validation_loss=validation_result.validation_loss, full_train_loss=validation_result.train_loss,
                full_pool_rows=validation_result.pool_rows, full_nodes_mean=validation_result.nodes_mean,
                full_leaves_mean=validation_result.leaves_mean, full_depth_mean=validation_result.depth_mean, **params))
        send.send((True,rows))
    except Exception: send.send((False,traceback.format_exc()))
    finally: send.close()

def _stop(process):
    try: os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError,ProcessLookupError): process.terminate()
    process.join(5)
    if process.is_alive(): process.kill()

def run_task(task, data_home, screen_trees, seed, timeout):
    context = mp.get_context("spawn")
    receive,send = context.Pipe(False)
    process = context.Process(target=_worker, args=(send,task,data_home,screen_trees,seed))
    process.start()
    send.close()
    if not receive.poll(timeout):
        _stop(process)
        receive.close()
        raise TimeoutError(f"task exceeded {timeout} seconds")
    success,result = receive.recv()
    process.join(5)
    if process.is_alive(): _stop(process)
    receive.close()
    if not success: raise RuntimeError(result)
    return result

def _combined(results_dir, output):
    files = sorted(Path(results_dir).glob("*.csv"))
    if not files: return 0
    rows = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    rows.to_csv(output, index=False)
    return len(rows)

def _read_slow(path):
    if not path.exists(): return set()
    return set(pd.read_csv(path).dataset)

def _write_slow(path, datasets):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"dataset":sorted(datasets)}).to_csv(path, index=False)

@call_parse
def main(
    metadata_csv:str="meta/meta_benchmark/beyondarena_metadata.csv", # Cached upstream task metadata
    amlb_csv:str="meta/amlb/datasets.csv",       # Downloaded non-overlapping AMLB manifest
    output_dir:str="meta/meta_benchmark",       # Manifest, task results, failures, and combined output
    data_home:str=".data/meta_benchmark",       # Download cache
    slow_csv:str="meta/meta_benchmark/slow.csv", # Persistent datasets skipped after timing out
    task_timeout:int=60,                         # Maximum seconds including loading and both sweep stages
    screen_trees:int=8,                          # Trees per cheap OOB configuration
    seed:int=42,                                 # Sampling and forest seed
    limit:int=None,                              # Optional number of pending tasks to run
    include_text:bool=False,                     # Include datasets containing semantic text
    include_slow:bool=False,                     # Retry datasets recorded as slow
    task_names:str=None,                         # Optional comma-separated dataset names
):
    "Run resumable FastForest sweeps over one canonical split per BeyondArena dataset."
    root = Path(__file__).parents[1]
    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else root/path
    output,data_home,slow_path = resolve(output_dir),resolve(data_home),resolve(slow_csv)
    output.mkdir(parents=True, exist_ok=True)
    results_dir = output/"results"
    results_dir.mkdir(exist_ok=True)
    tasks = pd.concat([beyond_manifest(resolve(metadata_csv), include_text),amlb_manifest(resolve(amlb_csv), data_home)], ignore_index=True, sort=False)
    if task_names:
        selected = {name.strip() for name in task_names.split(",") if name.strip()}
        tasks = tasks[tasks.dataset.isin(selected)]
    tasks.to_csv(output/"manifest.csv", index=False)
    slow = _read_slow(slow_path)
    pending = [row for _,row in tasks.iterrows() if not (results_dir/f"{row.dataset}.csv").exists() and (include_slow or row.dataset not in slow)]
    if limit is not None: pending = pending[:limit]
    failures = []
    print(f"{len(tasks)} tasks · {len(pending)} pending · {len(slow)} slow", flush=True)
    for number,row in enumerate(pending, 1):
        name = row.dataset
        print(f"[{number}/{len(pending)}] {name} ({row.task}, {int(row.rows)}×{int(row.features)})", flush=True)
        task = row.to_dict()
        try:
            result = run_task(task, data_home, screen_trees, seed, task_timeout)
            pd.DataFrame(result).to_csv(results_dir/f"{name}.csv", index=False)
            total = _combined(results_dir, output/"all.csv")
            print(f"  complete · {total//24} datasets in combined output", flush=True)
        except Exception as error:
            failures.append(dict(dataset=name, error=str(error)))
            if isinstance(error,TimeoutError):
                slow.add(name)
                _write_slow(slow_path, slow)
            print(f"  failed · {error}", flush=True)
    pd.DataFrame(failures, columns=["dataset","error"]).to_csv(output/"failures.csv", index=False)
    total = _combined(results_dir, output/"all.csv")
    print(f"finished · {total//24} complete · {len(failures)} failed", flush=True)
