import multiprocessing as mp
import os,signal,time,traceback,urllib.request,zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np,pandas as pd
from fastcore.script import call_parse
from fastcore.utils import str_enum
from sklearn.datasets import fetch_california_housing,fetch_covtype,fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import f1_score,log_loss,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from fastforest import FastForest,FastForestClassifier,_resolve_replacement,load
from fastforest.sklearn import sklearn_hist_preprocessor,sklearn_preprocessor
from fastforest.tools import advisor_features,forest_suite,screen

Dataset = str_enum("Dataset", "california", "concrete", "sgemm", "diamonds", "allstate", "diabetes", "covertype", "covertype_grouped",
    "adult", "bank", "bluebook", "bluebook_raw", "walmart", "walmart_raw", "walmart_nodate", "ashrae", "rossmann")

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

def load_ashrae(data_home):
    "Load the supplied meter, building, and weather tables without feature engineering."
    folder = Path(data_home)/"ashrae"
    paths = {name:folder/f"{name}.csv" for name in ("train", "building_metadata", "weather_train")}
    if not all(path.exists() for path in paths.values()): raise FileNotFoundError(f"download the ASHRAE competition files to {folder}")
    train = pd.read_csv(paths["train"], dtype={"building_id":"uint16", "meter":"uint8", "meter_reading":"float32"})
    buildings = pd.read_csv(paths["building_metadata"], dtype={"site_id":"uint8", "building_id":"uint16", "primary_use":"category",
        "square_feet":"float32", "year_built":"float32", "floor_count":"float32"})
    weather = pd.read_csv(paths["weather_train"], dtype={"site_id":"uint8", "air_temperature":"float32", "cloud_coverage":"float32",
        "dew_temperature":"float32", "precip_depth_1_hr":"float32", "sea_level_pressure":"float32", "wind_direction":"float32", "wind_speed":"float32"})
    y = np.log1p(train.pop("meter_reading").to_numpy(dtype=np.float32))
    X = train.merge(buildings, on="building_id", how="left", validate="many_to_one")
    X = X.merge(weather, on=["site_id", "timestamp"], how="left", validate="many_to_one")
    X["timestamp"] = pd.Categorical(X.timestamp, ordered=True)
    missing = {name:np.nan for name in ("year_built", "floor_count", "air_temperature", "cloud_coverage", "dew_temperature",
        "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed")}
    return X,y,missing

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
    if dataset == Dataset.ashrae:
        X,y,missing = load_ashrae(data_home)
        return "ASHRAE Great Energy Predictor III",X,y,missing,"regression",None
    if dataset == Dataset.rossmann:
        folder = Path(data_home)/"rossmann"
        paths = [folder/name for name in ("train.csv", "store.csv")]
        if not all(path.exists() for path in paths): raise FileNotFoundError(f"download the Rossmann train and store CSVs to {folder}")
        train = pd.read_csv(paths[0], dtype={"StateHoliday":str})
        stores = pd.read_csv(paths[1])
        train = train[train.Sales > 0].copy()
        y = np.log1p(train.pop("Sales").to_numpy(dtype=np.float32))
        X = train.drop(columns="Customers").merge(stores, on="Store", how="left", validate="many_to_one")
        missing = {name:np.nan for name in ("CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval")}
        return "Rossmann Store Sales",X,y,missing,"regression",None
    raise ValueError(f"unknown dataset: {dataset}")

def _rows(X, indexes): return X.iloc[indexes] if hasattr(X, "iloc") else X[indexes]

def split_indices(dataset, X, y, seed=42):
    "Return the dataset's canonical training and validation row indexes."
    idx = np.arange(len(X))
    if dataset in (Dataset.bluebook, Dataset.bluebook_raw): return idx[:-12_000],idx[-12_000:],"final 12,000 rows"
    if dataset in (Dataset.walmart, Dataset.walmart_raw, Dataset.walmart_nodate):
        dates = pd.to_datetime(X.Date)
        cutoff = np.sort(dates.unique())[-12]
        return idx[dates < cutoff],idx[dates >= cutoff],"final 12 weeks"
    if dataset == Dataset.rossmann:
        dates = pd.to_datetime(X.Date)
        cutoff = dates.max()-pd.Timedelta(weeks=6)
        return idx[dates < cutoff],idx[dates >= cutoff],"final 6 weeks"
    if dataset == Dataset.ashrae:
        test = np.asarray(X.timestamp >= "2016-12-01 00:00:00")
        return idx[~test],idx[test],"December 2016"
    train,test = train_test_split(idx, test_size=.2, random_state=seed, stratify=y)
    return train,test,"one 80/20 split"

def _max_features(value):
    "Parse square-root or fractional feature selection."
    if value == "sqrt": return value
    return float(value)

def _replacement(value):
    value = value.lower()
    if value == "none": return None
    if value == "true": return True
    if value == "false": return False
    raise ValueError("replacement must be none, true, or false")

def _make_model(name, task, X_train, missing_values, seed, rf_trees, ff_kwargs, max_dummy_cardinality):
    if name == "FastForest":
        params = dict(ff_kwargs, seed=seed, missing_values=missing_values, max_dummy_cardinality=max_dummy_cardinality)
        return (FastForestClassifier if task == "classification" else FastForest)(**params)
    if name == "RandomForest":
        forest = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = forest(n_estimators=rf_trees, n_jobs=-1, random_state=seed)
        if not hasattr(X_train, "columns"): return model
        target_type = "auto" if task == "classification" else "continuous"
        preprocessor = sklearn_preprocessor(X_train, missing_values, target_type=target_type)
        return make_pipeline(preprocessor, model)
    hist = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
    if not hasattr(X_train, "columns"): return hist(random_state=seed)
    preprocessor,categories = sklearn_hist_preprocessor(X_train, missing_values,
        target_type="auto" if task == "classification" else "continuous")
    return make_pipeline(preprocessor, hist(random_state=seed, categorical_features=categories))

def _advisor_target_metadata(y, task):
    y = np.asarray(y)
    if task == "classification":
        _,counts = np.unique(y, return_counts=True)
        probabilities = counts/counts.sum()
        return dict(n_classes=len(counts), output_dimensions=max(1,len(counts)-1), majority_fraction=float(probabilities.max()),
            target_entropy=float(-(probabilities*np.log(probabilities)).sum()), target_mean=0., target_std=0.,
            target_unique_fraction=len(counts)/len(y))
    values = np.asarray(y, dtype=np.float64)
    return dict(n_classes=1, output_dimensions=1, majority_fraction=1., target_entropy=0., target_mean=float(values.mean()),
        target_std=float(values.std()), target_unique_fraction=len(np.unique(values))/len(values))

_ADVISOR_COMPARISONS = {
    "regression":(("bootstrap_fraction",.5), ("tree_cutoff_samples",16), ("max_features","sqrt"), ("min_node_size",64),
        ("max_node_samples",160), ("max_features",.9), ("bootstrap_max",160_000)),
    "classification":(("max_features","sqrt"), ("bootstrap_fraction",.5), ("tree_cutoff_samples",16),
        ("min_node_size",64), ("max_features",.9), ("max_node_samples",2560), ("min_global_gain",1e-5))}

def _advisor_configs(model, task, compact=False):
    "Return the task-specific useful-gain-ranked advisor configurations."
    base = model.get_params()
    comparisons = _ADVISOR_COMPARISONS[task][:3 if compact else 7]
    levels = {}
    for name,value in comparisons: levels.setdefault(name, [base[name]]).append(value)
    return forest_suite(model, {name:tuple(values) for name,values in levels.items()})

def _advisor_select(model, task, X_train, y_train, total_rows, advisor_dir, seed=42):
    from fastforest import _estimated_outputs,_fit_plan
    outputs = _estimated_outputs(y_train, seed) if task == "classification" else 1
    params = model.get_params()
    replacement = _resolve_replacement(params["replacement"], len(X_train), task == "classification")
    final_trees,_,_ = _fit_plan(len(X_train), None, params["bootstrap_fraction"], params["bootstrap_max"],
        replacement, False, outputs)
    configs = _advisor_configs(model, task, compact=final_trees == 20)
    report = screen(model, X_train, y_train, configs, trees=8, seed=seed)
    rows = []
    base = dict(dataset="readme", task=task, rows=total_rows, screen_trees=8, **report.feature_metadata,
        **_advisor_target_metadata(y_train, task))
    for result,(_,_,params) in zip(report.results, configs):
        replacement = _resolve_replacement(params["replacement"], len(X_train), task == "classification")
        trees,_,pool_rows = _fit_plan(len(X_train), None, params["bootstrap_fraction"], params["bootstrap_max"],
            replacement, False, outputs)
        rows.append(dict(base, label=result.label, screen_oob_loss=result.oob_loss, screen_train_loss=result.train_loss,
            screen_coverage=result.coverage, screen_evaluated_rows=result.evaluated_rows, screen_nodes_mean=result.nodes_mean,
            screen_leaves_mean=result.leaves_mean, screen_depth_mean=result.depth_mean, full_trees=trees,
            full_pool_rows=pool_rows, **{**params,"replacement":replacement}))
    summary = pd.read_csv(Path(advisor_dir)/"summary.csv")
    encoding = summary[summary.task == task].sort_values("mean_selected").iloc[0].encoding
    features = advisor_features(pd.DataFrame(rows), categorical=encoding == "categorical")
    advisor = load(Path(advisor_dir)/f"{task}_{encoding}.ffm")
    prediction = advisor.predict(features)
    prediction[[result.label == "defaults" for result in report.results]] = 1.
    selected = int(np.argmin(prediction))
    return configs[selected],encoding,float(prediction[selected])

def _fit_score_tuned(task, X_train, y_train, X_test, y_test, missing_values, ff_kwargs, max_dummy_cardinality, advisor_dir, total_rows):
    start = time.perf_counter()
    base = (FastForestClassifier if task == "classification" else FastForest)(seed=42, missing_values=missing_values,
        max_dummy_cardinality=max_dummy_cardinality, **ff_kwargs)
    (label,_,params),encoding,predicted = _advisor_select(base, task, X_train, y_train, total_rows, advisor_dir)
    model = (FastForestClassifier if task == "classification" else FastForest)(seed=42, missing_values=missing_values,
        max_dummy_cardinality=max_dummy_cardinality, **{**ff_kwargs,**params})
    model.fit(X_train, y_train)
    result = dict(fit=time.perf_counter()-start, selected=label, advisor_encoding=encoding, predicted_relative=predicted,
        n_trees=model.n_trees_, tree_nodes=(float(np.mean(model.tree_node_counts_)),float(np.median(model.tree_node_counts_))),
        tree_leaves=(float(np.mean(model.tree_leaf_counts_)),float(np.median(model.tree_leaf_counts_))),
        tree_depth=(float(np.mean(model.tree_depths_)),float(np.median(model.tree_depths_))))
    start = time.perf_counter()
    if task == "classification":
        probabilities = model.predict_proba(X_test)
        predictions = model.classes_[probabilities.argmax(axis=1)]
        result.update(predict=time.perf_counter()-start, f1=f1_score(y_test, predictions, average="macro"),
            log_loss=log_loss(y_test, probabilities, labels=model.classes_), prediction_trees_per_batch=model.prediction_trees_per_batch_)
    else:
        predictions = model.predict(X_test)
        result.update(predict=time.perf_counter()-start, rmse=mean_squared_error(y_test, predictions)**.5, r2=r2_score(y_test, predictions))
    return result

def _fit_score(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality,
    advisor_dir=None, total_rows=None):
    "Fit, time, and score one model."
    if name == "FastForest tuned":
        return _fit_score_tuned(task, X_train, y_train, X_test, y_test, missing_values, ff_kwargs,
            max_dummy_cardinality, advisor_dir, total_rows)
    result = {}
    start = time.perf_counter()
    model = _make_model(name, task, X_train, missing_values, 42, rf_trees, ff_kwargs, max_dummy_cardinality)
    model.fit(X_train, y_train)
    result["fit"] = time.perf_counter()-start
    if name == "FastForest":
        result["n_trees"] = model.n_trees_
        result["tree_nodes"] = float(np.mean(model.tree_node_counts_)),float(np.median(model.tree_node_counts_))
        result["tree_leaves"] = float(np.mean(model.tree_leaf_counts_)),float(np.median(model.tree_leaf_counts_))
        result["tree_depth"] = float(np.mean(model.tree_depths_)),float(np.median(model.tree_depths_))
        if task == "classification": result["prediction_trees_per_batch"] = model.prediction_trees_per_batch_
    if name == "HistGBM": result["iterations"] = model.n_iter_ if hasattr(model, "n_iter_") else model[-1].n_iter_
    start = time.perf_counter()
    if task == "classification":
        probabilities = model.predict_proba(X_test)
        predictions = model.classes_[probabilities.argmax(axis=1)]
    else: predictions = model.predict(X_test)
    if "predict" not in result: result["predict"] = time.perf_counter()-start
    if task == "classification":
        result["f1"] = f1_score(y_test, predictions, average="macro")
        result["log_loss"] = log_loss(y_test, probabilities, labels=model.classes_)
    else:
        result["rmse"] = mean_squared_error(y_test, predictions)**0.5
        result["r2"] = r2_score(y_test, predictions)
    return result

def _evaluate(send, name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality,
    advisor_dir, total_rows):
    "Fit and score one model in an isolated process."
    try:
        try: os.setsid()
        except (AttributeError, PermissionError): pass
        send.send(("ready", None))
        result = _fit_score(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs,
            max_dummy_cardinality, advisor_dir, total_rows)
        send.send((True, result))
    except BaseException: send.send((False, traceback.format_exc()))
    finally: send.close()

def _evaluate_ashrae(send, name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality, advisor_dir):
    "Load, chronologically split, and evaluate ASHRAE inside the timed process."
    try:
        try: os.setsid()
        except (AttributeError, PermissionError): pass
        send.send(("ready", None))
        dataset_name,X,y,missing_values,task,_ = load_data(Dataset.ashrae, data_home)
        if max_rows is not None and max_rows < len(X):
            if max_rows < 2: raise ValueError("max_rows must be at least 2")
            selected,_ = train_test_split(np.arange(len(X)), train_size=max_rows, random_state=42)
            X,y = _rows(X, selected),y[selected]
        train_idx,test_idx,_ = split_indices(Dataset.ashrae, X, y)
        result = _fit_score(name, task, _rows(X, train_idx), y[train_idx], _rows(X, test_idx), y[test_idx], missing_values,
            rf_trees, ff_kwargs, max_dummy_cardinality, advisor_dir, len(X))
        result["dataset_meta"] = dataset_name,len(X),X.shape[1],"December 2016"
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

def _run_timed(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality,
    timeout, advisor_dir=None, total_rows=None):
    "Evaluate a model in a fresh process, returning `None` on timeout."
    context = mp.get_context("spawn")
    receive,send = context.Pipe(False)
    args = (send, name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality,
        advisor_dir, total_rows)
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

def _run_ashrae_timed(name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality, timeout, advisor_dir=None):
    "Load and evaluate ASHRAE in a fresh process, returning `None` on timeout."
    context = mp.get_context("spawn")
    receive,send = context.Pipe(False)
    args = (send, name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality, advisor_dir)
    process = context.Process(target=_evaluate_ashrae, args=args)
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

def _save_results(dataset, task, results):
    ids = {Dataset.covertype:"covertype_bin", Dataset.covertype_grouped:"covertype_group", Dataset.walmart_raw:"walmart"}
    dataset_id = ids.get(dataset, str(dataset))
    path = Path(__file__).parent/"results"/("classification.csv" if task == "classification" else "regression.csv")
    table = pd.read_csv(path)
    names = {"FastForest":"fastforest", "FastForest tuned":"fastforest tuned", "RandomForest":"sklearn RF", "HistGBM":"sklearn HistGBM"}
    for name,result in results.items():
        model = names[name]
        table = table[~((table.dataset == dataset_id)&(table.model == model))]
        row = dict(dataset=dataset_id, model=model, status="", note="")
        if result is None: row.update(status="timeout", note="timed out")
        elif task == "classification": row.update(f1=result["f1"], log_loss=result["log_loss"], fit=result["fit"], proba=result["predict"])
        else: row.update(rmse=result["rmse"], r2=result["r2"], fit=result["fit"], predict=result["predict"])
        table = pd.concat([table,pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)

@call_parse
def main(
    dataset:Dataset=Dataset.california, # Regression or classification dataset
    rf_trees:int=100,            # sklearn RF trees (its default)
    ff_trees:int=None,           # FastForest trees; defaults to its sampled-row rule
    min_node_size:int=8,         # FastForest minimum node size
    bootstrap_fraction:float=None,# Defaults to 1, or 0.8 with OOB
    bootstrap_max:int=40_000,    # Maximum sampled rows per output; multiplied by classes
    replacement:str="none",      # none is adaptive; true or false overrides it
    max_node_samples:int=320,    # Maximum rows evaluated per node
    tree_cutoff_samples:int=None, # Random unique cutoff values sampled per tree and feature
    min_local_gain:float=0.,     # Minimum node-normalized split gain; 0 disables
    min_global_gain:float=0.,    # Minimum root-normalized row-weighted split gain; 0 disables
    cutoff_divisor:float=10.,    # No-sort splitter candidate-count divisor
    random_splitter:bool=False,   # Use the original random split search
    max_features:str="0.6",       # sqrt or a feature fraction
    max_dummy_cardinality:int=4, # Largest cardinality expanded to binary features
    frequent_value_fraction:float=.08, # Minimum row fraction for an ordered value to also receive a binary feature
    dates:bool=True,             # Let fastforest auto-detect dates
    timeout:int=180,             # Maximum seconds for each model/dataset combination
    ff_only:bool=False,          # Run only FastForest
    tuned_only:bool=False,       # Run only the held-out meta-forest-selected FastForest
    rf_only:bool=False,          # Run only sklearn RandomForest
    hist_only:bool=False,        # Run only sklearn HistGradientBoosting
    max_rows:int=None,           # Optional reproducible dataset row limit
    data_home:str=None,          # sklearn dataset cache directory
    advisor_dir:str="meta/meta_advisor", # Held-out sweep-advisor models
    save:bool=False,              # Update the README benchmark result CSV
):
    "Compare accuracy and timing on one fixed dataset split."
    if timeout < 1: raise ValueError("timeout must be positive")
    if sum((ff_only,tuned_only,rf_only,hist_only)) > 1: raise ValueError("model-only options are mutually exclusive")
    if data_home is None: data_home = Path(__file__).parents[1]/".data"
    task = "regression" if dataset == Dataset.ashrae else None
    one_hot_groups = None
    ff_kwargs = dict(n_trees=ff_trees, min_node_size=min_node_size, bootstrap_fraction=bootstrap_fraction, bootstrap_max=bootstrap_max,
        replacement=_replacement(replacement), max_node_samples=max_node_samples, tree_cutoff_samples=tree_cutoff_samples,
        min_local_gain=min_local_gain, min_global_gain=min_global_gain, cutoff_divisor=cutoff_divisor, random_splitter=random_splitter,
        max_features=_max_features(max_features),
        frequent_value_fraction=frequent_value_fraction,
        one_hot_groups=one_hot_groups, date_columns=None if dates else {})
    models = ["FastForest", "RandomForest", "HistGBM"]
    if ff_only: models = models[:1]
    if tuned_only: models = ["FastForest tuned"]
    if rf_only: models = models[1:2]
    if hist_only: models = models[2:]
    results = defaultdict(dict)
    if dataset == Dataset.ashrae:
        for name in models:
            print(f"Running {name}...", flush=True)
            results[name] = _run_ashrae_timed(name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality, timeout, advisor_dir)
        completed = next((result for result in results.values() if result is not None), None)
        dataset_name,n_rows,n_features,split_name = (completed["dataset_meta"] if completed else
            ("ASHRAE Great Energy Predictor III", 20_216_100, 15, "December 2016"))
    else:
        dataset_name,X,y,missing_values,task,one_hot_groups = load_data(dataset, data_home)
        y = np.asarray(y, dtype=np.float32 if task == "regression" else None)
        if max_rows is not None and max_rows < len(X):
            if max_rows < 2: raise ValueError("max_rows must be at least 2")
            selected,_ = train_test_split(np.arange(len(X)), train_size=max_rows, random_state=42,
                stratify=y if task == "classification" else None)
            X,y = _rows(X, selected),y[selected]
        train_idx,test_idx,split_name = split_indices(dataset, X, y if task == "classification" else None)
        if dataset == Dataset.walmart_nodate: X = X.drop(columns="Date")
        ff_kwargs["one_hot_groups"] = one_hot_groups
        X_train,X_test,y_train,y_test = _rows(X, train_idx),_rows(X, test_idx),y[train_idx],y[test_idx]
        for name in models:
            print(f"Running {name}...", flush=True)
            results[name] = _run_timed(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs,
                max_dummy_cardinality, timeout, advisor_dir, len(X))
        n_rows,n_features = len(X),X.shape[1]

    print(f"{dataset_name}: {n_rows:,} rows, {n_features} features, {split_name}")
    fastforest_name = next((name for name in models if name.startswith("FastForest")), None)
    if fastforest_name:
        resolved_trees = results[fastforest_name]["n_trees"] if results[fastforest_name] is not None else ff_trees
        print(f"FastForest: trees={resolved_trees}, min_node_size={min_node_size}, bootstrap_fraction={bootstrap_fraction}, "
            f"bootstrap_max={bootstrap_max}, replacement={replacement}, max_node_samples={max_node_samples}, "
            f"tree_cutoff_samples={tree_cutoff_samples}, "
            f"min_local_gain={min_local_gain}, min_global_gain={min_global_gain}, "
            f"cutoff_divisor={cutoff_divisor}, "
            f"max_dummy_cardinality={max_dummy_cardinality}, frequent_value_fraction={frequent_value_fraction}, "
            f"random_splitter={random_splitter}, max_features={max_features}")
    if "RandomForest" in models: print(f"sklearn RF trees={rf_trees}, one-hot through 20 levels then target encoding")
    print(f"Timeout: {timeout}s per model/dataset combination")
    if task == "classification": print(f"{'model':<15} {'F1 acc':>14} {'log loss':>14} {'fit total':>12} {'predict total':>14}")
    else: print(f"{'model':<15} {'RMSE':>14} {'R²':>14} {'fit total':>12} {'predict total':>14}")
    for name in models:
        result = results[name]
        if result is None:
            print(f"{name:<15} {'timeout':>14}")
            continue
        if task == "classification": scores = f"{result['f1']:>14.4f} {result['log_loss']:>14.4f}"
        else:
            r2_digits = 3 if result["r2"] > 0.995 else 2
            scores = f"{result['rmse']:>14.2f} {result['r2']:>14.{r2_digits}f}"
        print(f"{name:<15} {scores} {result['fit']:>10.3f}s {result['predict']:>12.3f}s")
        if name.startswith("FastForest"):
            print(f"{'trees':<15} nodes mean/median={result['tree_nodes'][0]:.1f}/{result['tree_nodes'][1]:.0f}, "
                f"leaves={result['tree_leaves'][0]:.1f}/{result['tree_leaves'][1]:.0f}, "
                f"depth={result['tree_depth'][0]:.1f}/{result['tree_depth'][1]:.0f}")
        if name.startswith("FastForest") and task == "classification":
            print(f"{'prediction':<15} trees/batch={result['prediction_trees_per_batch']}")
        if name == "FastForest tuned":
            print(f"{'advisor':<15} selected={result['selected']}, encoding={result['advisor_encoding']}, "
                f"predicted relative={result['predicted_relative']:.3f}")
    if "HistGBM" in models and results["HistGBM"] is not None: print(f"HistGBM iterations: {results['HistGBM']['iterations']}")
    if save: _save_results(dataset, task, results)
