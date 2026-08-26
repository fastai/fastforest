"Benchmark dataset loading and canonical train/validation splits."
import io,json,urllib.request,zipfile
from pathlib import Path

import numpy as np,pandas as pd
from fastcore.utils import str_enum
from sklearn.datasets import fetch_california_housing,fetch_covtype,fetch_openml
from sklearn.model_selection import train_test_split

Dataset = str_enum("Dataset", "california", "concrete", "sgemm", "diamonds", "allstate", "diabetes", "covertype",
    "adult", "bank", "click", "shuttle", "airlines", "higgs", "kddcup99", "sf_police",
    "bluebook", "bluebook_raw", "walmart", "walmart_raw", "ashrae", "rossmann")

_amlb = {
    Dataset.click:("click_prediction_small", "Click Prediction Small"),
    Dataset.shuttle:("shuttle", "Statlog Shuttle"),
    Dataset.airlines:("airlines", "Airlines Delay"),
    Dataset.higgs:("higgs", "HIGGS"),
    Dataset.kddcup99:("kddcup99", "KDD Cup 1999"),
}

_classification = frozenset((Dataset.covertype, Dataset.adult, Dataset.bank, Dataset.sf_police, *_amlb))
_orders = {Dataset.bluebook:"saledate", Dataset.bluebook_raw:"saledate", Dataset.walmart:"Date", Dataset.walmart_raw:"Date",
    Dataset.rossmann:"Date", Dataset.ashrae:"timestamp", Dataset.sf_police:"Dates"}

def dataset_task(dataset):
    "Return the dataset's modelling task without loading it."
    return "classification" if Dataset(dataset) in _classification else "regression"

def order_column(dataset):
    "Return the chronological feature used by a canonical time split, or None."
    return _orders.get(Dataset(dataset))

_sgemm_url = "https://archive.ics.uci.edu/static/public/440/sgemm%2Bgpu%2Bkernel%2Bperformance.zip"
_diabetes_url = "https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
_beyond_metadata = "https://raw.githubusercontent.com/autogluon/tabarena/main/packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/BeyondArena_tasks_metadata.csv"

def beyond_manifest(path, include_text=False):
    "Load one canonical split per BeyondArena dataset."
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_beyond_metadata, path)
    tasks = pd.read_csv(path)
    tasks = tasks[(tasks.repeat == 0)&(tasks.fold == 0)].copy()
    if not include_text: tasks = tasks[~tasks.has_text]
    tasks["dataset"] = tasks.tabarena_task_name
    tasks["source_group"] = tasks.dataset_name
    tasks["task"] = np.where(tasks.is_classification, "classification", "regression")
    tasks["rows"],tasks["features"] = tasks.num_instances,tasks.num_features
    tasks["uuid"] = tasks.data_foundry_uri.str.rsplit("/", n=1).str[-1]
    tasks["collection"] = "beyondarena"
    return tasks.reset_index(drop=True)

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
    return X,y

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
    return X,y

def load_data(dataset, data_home):
    "Load a dataset and return its name, features, target, and missing-value rules."
    if dataset == Dataset.california:
        X,y = fetch_california_housing(return_X_y=True, data_home=data_home)
        return "California Housing",X,y,None,"regression"
    if dataset == Dataset.concrete:
        X,y = fetch_openml(data_id=44959, return_X_y=True, as_frame=False, data_home=data_home)
        return "Concrete Compressive Strength",X,y,None,"regression"
    if dataset == Dataset.sgemm:
        X,y = load_sgemm(data_home)
        return "SGEMM GPU Kernel Performance",X,y,None,"regression"
    if dataset == Dataset.diamonds:
        X,y = fetch_openml(data_id=42225, return_X_y=True, as_frame=True, data_home=data_home)
        return "Diamonds",X,y,None,"regression"
    if dataset == Dataset.allstate:
        X,y = fetch_openml(data_id=42571, return_X_y=True, as_frame=True, data_home=data_home)
        return "Allstate Claims Severity",X,y,None,"regression"
    if dataset == Dataset.diabetes:
        X,y = load_diabetes(data_home)
        return "Diabetes 130-US Hospitals",X,y,None,"regression"
    if dataset == Dataset.covertype:
        X,y = fetch_covtype(return_X_y=True, data_home=data_home)
        return "Covertype",X,y,None,"classification"
    if dataset == Dataset.adult:
        X,y = fetch_openml(data_id=1590, return_X_y=True, as_frame=True, data_home=data_home)
        return "Adult Census Income",X,y,None,"classification"
    if dataset == Dataset.bank:
        X,y = fetch_openml(data_id=1461, return_X_y=True, as_frame=True, data_home=data_home)
        return "Bank Marketing",X,y,None,"classification"
    if dataset in _amlb:
        X,y = load_amlb(dataset, data_home)
        return _amlb[dataset][1],X,y,None,"classification"
    if dataset in (Dataset.bluebook, Dataset.bluebook_raw):
        path = Path(data_home)/"bluebook"/"TrainAndValid.csv"
        if not path.exists(): raise FileNotFoundError(f"download the Kaggle Blue Book TrainAndValid.csv to {path}")
        X = pd.read_csv(path, low_memory=False, keep_default_na=False)
        y = np.log(X.pop("SalePrice").to_numpy(dtype=np.float32))
        name = "Blue Book for Bulldozers"+(" (raw date)" if dataset == Dataset.bluebook_raw else "")
        return name,X,y,None,"regression"
    if dataset in (Dataset.walmart, Dataset.walmart_raw):
        folder = Path(data_home)/"walmart"
        paths = [folder/name for name in ("train.csv", "features.csv", "stores.csv")]
        if not all(path.exists() for path in paths): raise FileNotFoundError(f"download the Walmart train, features, and stores CSVs to {folder}")
        train,features,stores = (pd.read_csv(path) for path in paths)
        X = train.merge(features, on=["Store", "Date", "IsHoliday"], validate="many_to_one").merge(stores, on="Store", validate="many_to_one")
        y = X.pop("Weekly_Sales").to_numpy(dtype=np.float32)
        suffix = " (raw date)" if dataset == Dataset.walmart_raw else ""
        return "Walmart Store Sales"+suffix,X,y,None,"regression"
    if dataset == Dataset.ashrae:
        X,y = load_ashrae(data_home)
        return "ASHRAE Great Energy Predictor III",X,y,None,"regression"
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
        return "Rossmann Store Sales",X,y,missing,"regression"
    if dataset == Dataset.sf_police:
        path = Path(data_home)/"sf_police"/"sf-crime.zip"
        if not path.exists(): raise FileNotFoundError(f"download the Kaggle sf-crime competition zip to {path}")
        with zipfile.ZipFile(path) as archive:
            X = pd.read_csv(io.BytesIO(archive.read("train.csv.zip")), compression="zip")
        y = X.pop("Category").to_numpy()
        X = X.drop(columns=["Descript", "Resolution"])
        return "SF Crime",X,y,None,"classification"
    raise ValueError(f"unknown dataset: {dataset}")

def _rows(X, indexes): return X.iloc[indexes] if hasattr(X, "iloc") else X[indexes]

def split_indices(
    dataset, # `Dataset` member
    X, # Features, used by date-based splits
    y, # Classification labels for stratification; None for regression
    seed=42, # Random state for the default 80/20 split
):
    "Return the dataset's canonical training and validation row indexes."
    idx = np.arange(len(X))
    if dataset in (Dataset.bluebook, Dataset.bluebook_raw): return idx[:-12_000],idx[-12_000:],"final 12,000 rows"
    if dataset in (Dataset.walmart, Dataset.walmart_raw):
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
    if dataset == Dataset.sf_police:
        dates = pd.to_datetime(X.Dates)
        cutoff = dates.quantile(.9)
        return idx[dates < cutoff],idx[dates >= cutoff],"final 10%"
    train,test = train_test_split(idx, test_size=.2, random_state=seed, stratify=y)
    return train,test,"one 80/20 split"
