import multiprocessing as mp
import os,signal,time,traceback,urllib.request,zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np,pandas as pd
from fastcore.script import call_parse
from fastcore.utils import str_enum
from sklearn.datasets import fetch_california_housing,fetch_covtype,fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from fastforest import FastForest,FastForestClassifier,Workbench
from fastforest.sklearn import sklearn_hist_preprocessor,sklearn_preprocessor

Dataset = str_enum("Dataset", "california", "concrete", "sgemm", "diamonds", "allstate", "diabetes", "covertype", "covertype_grouped",
    "adult", "bank", "bluebook", "bluebook_raw", "walmart", "walmart_raw", "walmart_nodate")

_sgemm_url = "https://archive.ics.uci.edu/static/public/440/sgemm%2Bgpu%2Bkernel%2Bperformance.zip"
_diabetes_url = "https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"

def _cached_zip(url, data_home, name):
    "Download and cache a zip archive."
    path = Path(data_home)/name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path

def _zip_member(archive, suffix):
    "Find a zip member by its case-insensitive filename suffix."
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix.lower())]
    if len(matches) != 1: raise ValueError(f"expected one {suffix!r} in archive, found {matches}")
    return matches[0]

def load_sgemm(data_home):
    "Load and cache the UCI SGEMM GPU kernel performance dataset."
    cache_dir = Path(data_home)/"sgemm_gpu"
    csv_path = cache_dir/"sgemm_product.csv"
    if not csv_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir/"sgemm.zip"
        urllib.request.urlretrieve(_sgemm_url, zip_path)
        with zipfile.ZipFile(zip_path) as archive,archive.open("sgemm_product.csv") as src,open(csv_path, "wb") as dst:
            dst.write(src.read())
        zip_path.unlink()
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:,:14],np.log1p(data[:,14:].mean(axis=1))

def load_diabetes(data_home):
    "Load the UCI Diabetes 130-US Hospitals data as mixed raw columns."
    with zipfile.ZipFile(_cached_zip(_diabetes_url, data_home, "diabetes-130.zip")) as archive:
        with archive.open(_zip_member(archive, "diabetic_data.csv")) as src: data = pd.read_csv(src, dtype=str, keep_default_na=False)
    y = data.pop("time_in_hospital").astype(np.float32).to_numpy()
    X = data.drop(columns=["encounter_id", "patient_nbr", "readmitted"])
    return X,y,{name:"?" for name in X.columns}

def load_data(dataset, data_home):
    "Load a dataset and return its name, features, target, and missing-value rules."
    if dataset == Dataset.california:
        X,y = fetch_california_housing(return_X_y=True, data_home=data_home)
        return "California Housing",X,y,None,"regression",None
    if dataset == Dataset.concrete:
        X,y = fetch_openml(data_id=44959, return_X_y=True, as_frame=False, data_home=data_home)
        return "Concrete Compressive Strength",X,y,None,"regression",None
    if dataset == Dataset.sgemm:
        X,y = load_sgemm(data_home)
        return "SGEMM GPU Kernel Performance",X,y,None,"regression",None
    if dataset == Dataset.diamonds:
        X,y = fetch_openml(data_id=42225, return_X_y=True, as_frame=True, data_home=data_home)
        return "Diamonds",X,y,None,"regression",None
    if dataset == Dataset.allstate:
        X,y = fetch_openml(data_id=42571, return_X_y=True, as_frame=True, data_home=data_home)
        return "Allstate Claims Severity",X,y,None,"regression",None
    if dataset == Dataset.diabetes:
        X,y,missing = load_diabetes(data_home)
        return "Diabetes 130-US Hospitals",X,y,missing,"regression",None
    if dataset == Dataset.covertype:
        X,y = fetch_covtype(return_X_y=True, data_home=data_home)
        return "Covertype",X,y,None,"classification",None
    if dataset == Dataset.covertype_grouped:
        X,y = fetch_covtype(return_X_y=True, data_home=data_home)
        groups = {"wilderness_area":list(range(10, 14)), "soil_type":list(range(14, 54))}
        return "Covertype (one-hot groups)",X,y,None,"classification",groups
    if dataset == Dataset.adult:
        X,y = fetch_openml(data_id=1590, return_X_y=True, as_frame=True, data_home=data_home)
        return "Adult Census Income",X,y,None,"classification",None
    if dataset == Dataset.bank:
        X,y = fetch_openml(data_id=1461, return_X_y=True, as_frame=True, data_home=data_home)
        return "Bank Marketing",X,y,None,"classification",None
    if dataset in (Dataset.bluebook, Dataset.bluebook_raw):
        path = Path(data_home)/"bluebook"/"TrainAndValid.csv"
        if not path.exists(): raise FileNotFoundError(f"download the Kaggle Blue Book TrainAndValid.csv to {path}")
        X = pd.read_csv(path, low_memory=False, keep_default_na=False)
        y = np.log(X.pop("SalePrice").to_numpy(dtype=np.float32))
        name = "Blue Book for Bulldozers"+(" (raw date)" if dataset == Dataset.bluebook_raw else "")
        return name,X,y,None,"regression",None
    if dataset in (Dataset.walmart, Dataset.walmart_raw, Dataset.walmart_nodate):
        folder = Path(data_home)/"walmart"
        paths = [folder/name for name in ("train.csv", "features.csv", "stores.csv")]
        if not all(path.exists() for path in paths): raise FileNotFoundError(f"download the Walmart train, features, and stores CSVs to {folder}")
        train,features,stores = (pd.read_csv(path, keep_default_na=False) for path in paths)
        X = train.merge(features, on=["Store", "Date", "IsHoliday"], validate="many_to_one").merge(stores, on="Store", validate="many_to_one")
        y = X.pop("Weekly_Sales").to_numpy(dtype=np.float32)
        suffix = " (raw date)" if dataset == Dataset.walmart_raw else " (date removed)" if dataset == Dataset.walmart_nodate else ""
        missing = {name:"NA" for name in ("MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5")}
        return "Walmart Store Sales"+suffix,X,y,missing,"regression",None
    raise ValueError(f"unknown dataset: {dataset}")

def _rows(X, indexes): return X.iloc[indexes] if hasattr(X, "iloc") else X[indexes]

def _max_features(value):
    "Parse a workbench max-features CLI value."
    if value in ("sqrt", "all"): return value
    try: return int(value)
    except ValueError: return float(value)

def _make_model(name, task, X_train, missing_values, seed, rf_trees, ff_kwargs, max_dummy_cardinality):
    if name == "FastForest":
        params = dict(ff_kwargs, seed=seed, missing_values=missing_values, max_dummy_cardinality=max_dummy_cardinality)
        return (FastForestClassifier if task == "classification" else FastForest)(**params)
    if name == "RandomForest":
        forest = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = forest(n_estimators=rf_trees, n_jobs=-1, random_state=seed)
        target_type = "auto" if task == "classification" else "continuous"
        return make_pipeline(sklearn_preprocessor(X_train, missing_values, target_type=target_type), model) if hasattr(X_train, "columns") else model
    hist = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
    if not hasattr(X_train, "columns"): return hist(random_state=seed)
    preprocessor,categories = sklearn_hist_preprocessor(X_train, missing_values, target_type="auto" if task == "classification" else "continuous")
    return make_pipeline(preprocessor, hist(random_state=seed, categorical_features=categories))

def _evaluate(send, name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality):
    "Fit and score one model in an isolated process."
    try:
        try: os.setsid()
        except (AttributeError, PermissionError): pass
        send.send(("ready", None))
        result = {}
        start = time.perf_counter()
        model = _make_model(name, task, X_train, missing_values, 42, rf_trees, ff_kwargs, max_dummy_cardinality)
        model.fit(X_train, y_train)
        result["fit"] = time.perf_counter()-start
        if name == "FastForest":
            result["n_trees"] = model.n_trees_
            result["adaptive_choice"] = model.adaptive_choice_
            result["adaptive_scores"] = model.adaptive_scores_
            if task == "classification": result["prediction_trees_per_batch"] = model.prediction_trees_per_batch_
        if name == "HistGBM": result["iterations"] = model.n_iter_ if hasattr(model, "n_iter_") else model[-1].n_iter_
        start = time.perf_counter()
        if task == "classification":
            if name == "FastForest":
                native_X = model._encoder.transform(X_test)
                result["preprocess_predict"] = time.perf_counter()-start
                start = time.perf_counter()
                probabilities = model._model.predict_proba(native_X)
                result["native_proba"] = time.perf_counter()-start
                start = time.perf_counter()
                model._model.predict(native_X)
                result["native_labels"] = time.perf_counter()-start
                result["predict"] = result["preprocess_predict"]+result["native_proba"]
            else: probabilities = model.predict_proba(X_test)
            predictions = model.classes_[probabilities.argmax(axis=1)]
        else: predictions = model.predict(X_test)
        if "predict" not in result: result["predict"] = time.perf_counter()-start
        if task == "classification":
            result["accuracy"] = accuracy_score(y_test, predictions)
            result["balanced_accuracy"] = balanced_accuracy_score(y_test, predictions)
            result["log_loss"] = log_loss(y_test, probabilities, labels=model.classes_)
        else:
            result["rmse"] = mean_squared_error(y_test, predictions)**0.5
            result["r2"] = r2_score(y_test, predictions)
        send.send((True, result))
    except BaseException: send.send((False, traceback.format_exc()))
    finally: send.close()

def _stop(process):
    "Stop a timed-out process and any workers it created."
    try: os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError): process.terminate()
    process.join(5)
    if process.is_alive():
        try: os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError): process.kill()
        process.join()

def _run_timed(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality, timeout):
    "Evaluate a model in a fresh process, returning `None` on timeout."
    context = mp.get_context("spawn")
    receive,send = context.Pipe(False)
    args = (send, name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality)
    process = context.Process(target=_evaluate, args=args)
    process.start()
    send.close()
    if not receive.poll(30):
        _stop(process)
        receive.close()
        raise RuntimeError("model process did not start within 30 seconds")
    status,result = receive.recv()
    if status != "ready": raise RuntimeError(result)
    process.join(timeout)
    if process.is_alive():
        _stop(process)
        receive.close()
        return None
    if not receive.poll(): raise RuntimeError(f"model process exited with code {process.exitcode} without a result")
    succeeded,result = receive.recv()
    receive.close()
    if not succeeded: raise RuntimeError(result)
    return result

@call_parse
def main(
    dataset:Dataset=Dataset.california, # Regression or classification dataset
    rf_trees:int=100,            # sklearn RF trees (its default)
    ff_trees:int=None,           # FastForest trees; defaults to its sampled-row rule
    min_node_size:int=4,         # FastForest minimum node size
    bootstrap_fraction:float=None,# Defaults to 1, or 0.8 with OOB
    bootstrap_max:int=40_000,    # Maximum sampled rows per output; multiplied by classes
    replacement:bool=False,      # Sample tree rows with replacement
    max_node_samples:int=320,    # Maximum rows evaluated per node
    min_candidate_rows:int=20,   # Candidate-count row floor
    candidate_attempt_factor:int=2, # Maximum proposals per requested unique candidate
    cutoff_divisor:float=3.0,    # Divisor controlling candidate cutoff count
    splitter:str="histogram",    # Split search: random or histogram
    max_features:str="0.75",     # Fixed histogram features when adaptive=False
    feature_sampling:str="encoded", # Histogram sampling unit: encoded or columns
    leaf_regularization:float=0, # Parent-mean leaf pseudo-row count
    adaptive:bool=True,          # Select 60% or 90% of features on datasets over 8k rows
    max_dummy_cardinality:int=4, # Largest cardinality expanded to c-1 dummies
    timeout:int=180,             # Maximum seconds for each model/dataset combination
    ff_only:bool=False,          # Run only FastForest
    rf_only:bool=False,          # Run only sklearn RandomForest
    max_rows:int=None,           # Optional reproducible dataset row limit
    data_home:str=None,          # sklearn dataset cache directory
):
    "Compare accuracy and timing on one fixed dataset split."
    if timeout < 1: raise ValueError("timeout must be positive")
    if ff_only and rf_only: raise ValueError("ff_only and rf_only cannot both be enabled")
    if data_home is None: data_home = Path(__file__).parents[1]/".data"
    dataset_name,X,y,missing_values,task,one_hot_groups = load_data(dataset, data_home)
    date_columns = {"saledate":"%m/%d/%Y %H:%M"} if dataset == Dataset.bluebook else {"Date":"%Y-%m-%d"} if dataset == Dataset.walmart else None
    y = np.asarray(y, dtype=np.float32 if task == "regression" else None)
    if max_rows is not None and max_rows < len(X):
        if max_rows < 2: raise ValueError("max_rows must be at least 2")
        selected,_ = train_test_split(np.arange(len(X)), train_size=max_rows, random_state=42,
            stratify=y if task == "classification" else None)
        X,y = _rows(X, selected),y[selected]
    idx = np.arange(len(X))
    if dataset in (Dataset.bluebook, Dataset.bluebook_raw): train_idx,test_idx,split_name = idx[:-12_000],idx[-12_000:],"final 12,000 rows"
    elif dataset in (Dataset.walmart, Dataset.walmart_raw, Dataset.walmart_nodate):
        dates = pd.to_datetime(X.Date)
        cutoff = np.sort(dates.unique())[-12]
        train_idx,test_idx,split_name = idx[dates < cutoff],idx[dates >= cutoff],"final 12 weeks"
    else:
        train_idx,test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y if task == "classification" else None)
        split_name = "one 80/20 split"
    if dataset == Dataset.walmart_nodate: X = X.drop(columns="Date")
    workbench = Workbench(splitter, _max_features(max_features), leaf_regularization, feature_sampling)
    ff_kwargs = dict(n_trees=ff_trees, min_node_size=min_node_size, bootstrap_fraction=bootstrap_fraction, bootstrap_max=bootstrap_max,
        replacement=replacement, max_node_samples=max_node_samples, min_candidate_rows=min_candidate_rows,
        candidate_attempt_factor=candidate_attempt_factor, cutoff_divisor=cutoff_divisor, adaptive=adaptive,
        one_hot_groups=one_hot_groups, date_columns=date_columns, workbench=workbench)
    X_train,X_test,y_train,y_test = _rows(X, train_idx),_rows(X, test_idx),y[train_idx],y[test_idx]
    models = ["FastForest", "RandomForest", "HistGBM"]
    if ff_only: models = models[:1]
    if rf_only: models = models[1:2]
    results = defaultdict(dict)
    for name in models:
        print(f"Running {name}...", flush=True)
        results[name] = _run_timed(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs,
            max_dummy_cardinality, timeout)

    print(f"{dataset_name}: {len(X):,} rows, {X.shape[1]} features, {split_name}")
    if "FastForest" in models:
        resolved_trees = results["FastForest"]["n_trees"] if results["FastForest"] is not None else ff_trees
        print(f"FastForest: trees={resolved_trees}, min_node_size={min_node_size}, bootstrap_fraction={bootstrap_fraction}, "
            f"bootstrap_max={bootstrap_max}, replacement={replacement}, max_node_samples={max_node_samples}, "
            f"min_candidate_rows={min_candidate_rows}, candidate_attempt_factor={candidate_attempt_factor}, cutoff_divisor={cutoff_divisor}, "
            f"max_dummy_cardinality={max_dummy_cardinality}, adaptive={adaptive}, workbench={ff_kwargs['workbench']}")
    if "RandomForest" in models: print(f"sklearn RF trees={rf_trees}, one-hot through 20 levels then target encoding")
    print(f"Timeout: {timeout}s per model/dataset combination")
    if task == "classification": print(f"{'model':<15} {'accuracy':>14} {'balanced':>14} {'log loss':>14} {'fit total':>12} {'predict total':>14}")
    else: print(f"{'model':<15} {'RMSE':>14} {'R²':>14} {'fit total':>12} {'predict total':>14}")
    for name in models:
        result = results[name]
        if result is None:
            print(f"{name:<15} {'timeout':>14}")
            continue
        if task == "classification": scores = f"{result['accuracy']:>14.4f} {result['balanced_accuracy']:>14.4f} {result['log_loss']:>14.4f}"
        else:
            r2_digits = 3 if result["r2"] > 0.995 else 2
            scores = f"{result['rmse']:>14.2f} {result['r2']:>14.{r2_digits}f}"
        print(f"{name:<15} {scores} {result['fit']:>10.3f}s {result['predict']:>12.3f}s")
        if name == "FastForest" and result["adaptive_choice"] is not None:
            print(f"{'pilot':<15} selected features={result['adaptive_choice'][0]:g}, rows={result['adaptive_choice'][1]}; "
                f"scores={result['adaptive_scores']}")
        if name == "FastForest" and task == "classification":
            print(f"{'prediction':<15} preprocess={result['preprocess_predict']:.4f}s, native proba={result['native_proba']:.4f}s, "
                f"native labels={result['native_labels']:.4f}s, trees/batch={result['prediction_trees_per_batch']}")
    if "HistGBM" in models and results["HistGBM"] is not None: print(f"HistGBM iterations: {results['HistGBM']['iterations']}")
