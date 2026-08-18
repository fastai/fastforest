import json,multiprocessing as mp
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

from fastforest import FastForest,FastForestClassifier
from fastforest.auto import AutoForest,AutoForestClassifier
from fastforest.sklearn import sklearn_hist_preprocessor,sklearn_preprocessor

Dataset = str_enum("Dataset", "california", "concrete", "sgemm", "diamonds", "allstate", "diabetes", "covertype", "covertype_grouped",
    "adult", "bank", "click", "shuttle", "airlines", "higgs", "kddcup99", "sf_police",
    "bluebook", "bluebook_raw", "walmart", "walmart_raw", "walmart_nodate", "ashrae", "rossmann")

_amlb = {
    Dataset.click:("click_prediction_small", "Click Prediction Small"),
    Dataset.shuttle:("shuttle", "Statlog Shuttle"),
    Dataset.airlines:("airlines", "Airlines Delay"),
    Dataset.higgs:("higgs", "HIGGS"),
    Dataset.kddcup99:("kddcup99", "KDD Cup 1999"),
    Dataset.sf_police:("sf_police_incidents", "San Francisco Police Incidents"),
}

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

def load_amlb(dataset, data_home):
    "Load one locally cached AMLB table with its OpenML target metadata."
    folder = Path(data_home)/"meta_benchmark"/"amlb"/_amlb[dataset][0]
    frame = pd.read_parquet(folder/"data.pq")
    with open(folder/"metadata.json") as handle: target = json.load(handle)["data_set_description"]["default_target_attribute"]
    return frame.drop(columns=target),frame[target]

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
    if dataset in _amlb:
        X,y = load_amlb(dataset, data_home)
        return _amlb[dataset][1],X,y,None,"classification",None
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
        params.pop("split_prior_rows" if task == "classification" else "class_weight_power", None)
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

def _fit_score_auto(task, X_train, y_train, X_test, y_test, missing_values, ff_kwargs, max_dummy_cardinality, autogrow):
    start = time.perf_counter()
    params = {name:value for name,value in ff_kwargs.items() if name != "n_trees"}
    params.pop("split_prior_rows" if task == "classification" else "class_weight_power", None)
    model = (AutoForestClassifier if task == "classification" else AutoForest)(seed=42, autogrow=autogrow, max_trees=192,
        missing_values=missing_values, max_dummy_cardinality=max_dummy_cardinality, **params)
    model.fit(X_train, y_train)
    result = dict(fit=time.perf_counter()-start, sizing_active=model.sizing_["active"], sizing=model.sizing_,
        selected=f"bootstrap_max={model.bootstrap_max}, max_node_samples={model.max_node_samples}, trees={model.n_trees_}",
        selection_method="autogrow" if autogrow else "AutoForest", selection_scores=model.tree_history_,
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

def _fit_score(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality):
    "Fit, time, and score one model."
    if name in ("AutoForest","Autogrow"):
        return _fit_score_auto(task, X_train, y_train, X_test, y_test, missing_values, ff_kwargs,
            max_dummy_cardinality, name=="Autogrow")
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
        result["feature_importances"] = np.asarray(model.feature_importances_)
        result["feature_names"] = tuple(map(str, model.feature_names_in_))
        result["split_diagnostics"] = np.zeros((len(model.feature_importances_), 4), dtype=np.int64)
        if hasattr(model._model, "split_counts_by_depth"):
            depth_counts = np.asarray(model._model.split_counts_by_depth)
            encoded_diagnostics = np.column_stack((depth_counts[:,:,0].sum(axis=1), depth_counts[:,:,1].sum(axis=1),
                depth_counts[:,:4,0].sum(axis=1), depth_counts[:,4:8,0].sum(axis=1)))
            raw_ids = np.asarray(model._encoder._native.encoded_to_raw)
            np.add.at(result["split_diagnostics"], raw_ids, encoded_diagnostics)
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

def _evaluate(send, name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality):
    "Fit and score one model in an isolated process."
    try:
        try: os.setsid()
        except (AttributeError, PermissionError): pass
        send.send(("ready", None))
        result = _fit_score(name, task, X_train, y_train, X_test, y_test, missing_values, rf_trees, ff_kwargs, max_dummy_cardinality)
        send.send((True, result))
    except BaseException: send.send((False, traceback.format_exc()))
    finally: send.close()

def _evaluate_ashrae(send, name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality):
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
            rf_trees, ff_kwargs, max_dummy_cardinality)
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

def _run_ashrae_timed(name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality, timeout):
    "Load and evaluate ASHRAE in a fresh process, returning `None` on timeout."
    context = mp.get_context("spawn")
    receive,send = context.Pipe(False)
    args = (send, name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality)
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
    names = {"FastForest":"fastforest", "AutoForest":"AutoForest", "Autogrow":"autogrow",
        "RandomForest":"sklearn RF", "HistGBM":"sklearn HistGBM"}
    for name,result in results.items():
        model = names[name]
        table = table[~((table.dataset == dataset_id)&(table.model == model))]
        if name in ("AutoForest","Autogrow") and result is not None and not result["sizing_active"]: continue
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
    min_node_size:int=None,      # FastForest minimum node size; model default when omitted
    bootstrap_fraction:float=None,# Defaults to 1, or 0.8 with OOB
    bootstrap_max:int=None,      # Maximum sampled rows per output; model default when omitted
    replacement:str="none",      # none is adaptive; true or false overrides it
    max_node_samples:int=None,   # Maximum rows evaluated per node; model default when omitted
    split_prior_rows:float=None, # Prior rows used by the split score; model default when omitted
    class_weight_power:float=None,# Classification inverse-frequency weighting; model default when omitted
    cutoff_divisor:float=None,   # No-sort splitter candidate-count divisor; model default when omitted
    random_splitter:bool=False,   # Use the original random split search
    max_features:str=None,       # sqrt or a feature fraction; task default when omitted
    max_dummy_cardinality:int=None,# Largest cardinality expanded to binary features; model default when omitted
    dates:bool=True,             # Let fastforest auto-detect dates
    diagnostics:bool=False,      # Print FastForest feature/depth split diagnostics
    timeout:int=180,             # Maximum seconds for each model/dataset combination
    ff_only:bool=False,          # Run only FastForest
    auto_only:bool=False,        # Run AutoForest with and without autogrow
    sizer_only:bool=False,       # Run AutoForest sample sizing without autogrow
    autogrow_only:bool=False,    # Run only AutoForest with autogrow
    rf_only:bool=False,          # Run only sklearn RandomForest
    hist_only:bool=False,        # Run only sklearn HistGradientBoosting
    max_rows:int=None,           # Optional reproducible dataset row limit
    data_home:str=None,          # sklearn dataset cache directory
    save:bool=False,              # Update the README benchmark result CSV
):
    "Compare accuracy and timing on one fixed dataset split."
    if timeout < 1: raise ValueError("timeout must be positive")
    if sum((ff_only,auto_only,sizer_only,autogrow_only,rf_only,hist_only)) > 1: raise ValueError("model-only options are mutually exclusive")
    if data_home is None: data_home = Path(__file__).parents[1]/".data"
    task = "regression" if dataset == Dataset.ashrae else None
    one_hot_groups = None
    ff_kwargs = dict(n_trees=ff_trees, bootstrap_fraction=bootstrap_fraction, replacement=_replacement(replacement),
        random_splitter=random_splitter, one_hot_groups=one_hot_groups, date_columns=None if dates else {})
    for name,value in dict(min_node_size=min_node_size, bootstrap_max=bootstrap_max, max_node_samples=max_node_samples,
        split_prior_rows=split_prior_rows, class_weight_power=class_weight_power, cutoff_divisor=cutoff_divisor).items():
        if value is not None: ff_kwargs[name] = value
    if max_dummy_cardinality is None: max_dummy_cardinality = FastForest().max_dummy_cardinality
    models = ["FastForest", "RandomForest", "HistGBM"]
    if ff_only: models = models[:1]
    if auto_only: models = ["AutoForest","Autogrow"]
    if sizer_only: models = ["AutoForest"]
    if autogrow_only: models = ["Autogrow"]
    if rf_only: models = models[1:2]
    if hist_only: models = models[2:]
    results = defaultdict(dict)
    feature_names = ()
    if dataset == Dataset.ashrae:
        if max_features is not None: ff_kwargs["max_features"] = _max_features(max_features)
        for name in models:
            print(f"Running {name}...", flush=True)
            results[name] = _run_ashrae_timed(name, data_home, max_rows, rf_trees, ff_kwargs, max_dummy_cardinality, timeout)
        completed = next((result for result in results.values() if result is not None), None)
        dataset_name,n_rows,n_features,split_name = (completed["dataset_meta"] if completed else
            ("ASHRAE Great Energy Predictor III", 20_216_100, 15, "December 2016"))
    else:
        dataset_name,X,y,missing_values,task,one_hot_groups = load_data(dataset, data_home)
        feature_names = tuple(map(str, X.columns)) if hasattr(X, "columns") else tuple(f"x{i}" for i in range(X.shape[1]))
        if max_features is not None: ff_kwargs["max_features"] = _max_features(max_features)
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
                max_dummy_cardinality, timeout)
        n_rows,n_features = len(X),X.shape[1]

    print(f"{dataset_name}: {n_rows:,} rows, {n_features} features, {split_name}")
    fastforest_name = next((name for name in models if name in ("FastForest","AutoForest","Autogrow")), None)
    if fastforest_name:
        resolved_trees = results[fastforest_name]["n_trees"] if results[fastforest_name] is not None else ff_trees
        shown = (FastForestClassifier if task == "classification" else FastForest)(**{
            name:value for name,value in ff_kwargs.items() if name not in ("one_hot_groups","date_columns") and
            name != ("split_prior_rows" if task == "classification" else "class_weight_power")})
        print(f"{fastforest_name}: trees={resolved_trees}, min_node_size={shown.min_node_size}, bootstrap_fraction={shown.bootstrap_fraction}, "
            f"bootstrap_max={shown.bootstrap_max}, replacement={replacement}, max_node_samples={shown.max_node_samples}, "
            f"{f'class_weight_power={shown.class_weight_power}' if task == 'classification' else f'split_prior_rows={shown.split_prior_rows}'}, "
            f"cutoff_divisor={shown.cutoff_divisor}, "
            f"max_dummy_cardinality={max_dummy_cardinality}, "
            f"random_splitter={random_splitter}, max_features={shown.max_features}")
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
            scores = f"{result['rmse']:>14.5f} {result['r2']:>14.5f}"
        print(f"{name:<15} {scores} {result['fit']:>10.3f}s {result['predict']:>12.3f}s")
        if name in ("FastForest","AutoForest","Autogrow"):
            print(f"{'trees':<15} nodes mean/median={result['tree_nodes'][0]:.1f}/{result['tree_nodes'][1]:.0f}, "
                f"leaves={result['tree_leaves'][0]:.1f}/{result['tree_leaves'][1]:.0f}, "
                f"depth={result['tree_depth'][0]:.1f}/{result['tree_depth'][1]:.0f}")
        if diagnostics and name == "FastForest":
            counts = result["split_diagnostics"]
            names = result["feature_names"]
            for index in np.argsort(result["feature_importances"])[::-1]:
                total,equality,early,middle = map(int, counts[index])
                late = total-early-middle
                print(f"{'split detail':<15} {names[index]:<12} importance={result['feature_importances'][index]:.4f}, "
                    f"splits={total}, equality={equality}, depth 0-3/4-7/8+={early}/{middle}/{late}")
        if name in ("FastForest","AutoForest","Autogrow") and task == "classification":
            print(f"{'prediction':<15} trees/batch={result['prediction_trees_per_batch']}")
        if name in ("AutoForest","Autogrow"):
            if result["selection_scores"]:
                history = ", ".join(f"{row['trees']}={row['loss']:.3g}" for row in result["selection_scores"])
                print(f"{'selection':<15} selected={result['selected']}, tracked losses: {history}")
            sizing = result["sizing"]
            if sizing["active"]:
                bootstrap = ", ".join(f"{key}={value/sizing['baseline_loss']:.3f}" for key,value in sizing["bootstrap_losses"].items())
                nodes = ", ".join(f"{key}={value/sizing['baseline_loss']:.3f}" for key,value in sizing["node_losses"].items())
                print(f"{'sizing':<15} native={sizing['seconds']:.3f}s; bootstrap: {bootstrap}; nodes: {nodes}")
    if "HistGBM" in models and results["HistGBM"] is not None: print(f"HistGBM iterations: {results['HistGBM']['iterations']}")
    if save: _save_results(dataset, task, results)
