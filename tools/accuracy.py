import time, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastcore.script import call_parse
from fastcore.utils import str_enum
from sklearn.datasets import fetch_california_housing,fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold,train_test_split

from fastforest import FastForest

Dataset = str_enum("Dataset", "california", "concrete", "sgemm")

_sgemm_url = "https://archive.ics.uci.edu/static/public/440/sgemm%2Bgpu%2Bkernel%2Bperformance.zip"

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

def load_data(dataset, data_home):
    "Load a numeric regression dataset and return its display name, features, and target."
    if dataset == Dataset.california:
        X,y = fetch_california_housing(return_X_y=True, data_home=data_home)
        return "California Housing",X,y
    if dataset == Dataset.concrete:
        X,y = fetch_openml(data_id=44959, return_X_y=True, as_frame=False, data_home=data_home)
        return "Concrete Compressive Strength",X,y
    X,y = load_sgemm(data_home)
    return "SGEMM GPU Kernel Performance",X,y

@call_parse
def main(
    dataset:Dataset=Dataset.california, # Regression dataset
    folds:int=None,              # CV folds; defaults to 5, or one 80/20 split for SGEMM
    rf_trees:int=100,            # Trees for sklearn RF
    ff_trees:int=None,           # FastForest trees; defaults to `rf_trees`
    min_node_size:int=4,         # FastForest minimum node size
    bootstrap_fraction:float=0.8,# Fraction of training rows sampled per tree
    bootstrap_max:int=40_000,    # Maximum training rows sampled per tree
    replacement:bool=False,      # Sample tree rows with replacement
    max_node_samples:int=160,    # Maximum rows evaluated per node
    cutoff_divisor:float=3.0,    # Divisor controlling candidate cutoff count
    ff_only:bool=False,          # Run only FastForest
    data_home:str=None,          # sklearn dataset cache directory
):
    "Compare regression accuracy and timing on identical dataset splits."
    if ff_trees is None: ff_trees = rf_trees
    if data_home is None: data_home = Path(__file__).parents[1]/".data"
    dataset_name,X,y = load_data(dataset, data_home)
    X,y = np.asarray(X, dtype=np.float32),np.asarray(y, dtype=np.float32)
    if folds is None: folds = 1 if dataset == Dataset.sgemm else 5
    if folds == 1:
        idx = np.arange(len(X))
        train_idx,test_idx = train_test_split(idx, test_size=0.2, random_state=42)
        splits = [(train_idx,test_idx)]
    else: splits = list(KFold(folds, shuffle=True, random_state=42).split(X))
    ff_kwargs = dict(n_trees=ff_trees, min_node_size=min_node_size, bootstrap_fraction=bootstrap_fraction, bootstrap_max=bootstrap_max,
        replacement=replacement, max_node_samples=max_node_samples, cutoff_divisor=cutoff_divisor)
    fastforest = lambda seed: FastForest(seed=seed, **ff_kwargs)
    random_forest = lambda seed: RandomForestRegressor(n_estimators=rf_trees, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    hist_gbm = lambda seed: HistGradientBoostingRegressor(random_state=seed)
    factories = [("FastForest", fastforest), ("RandomForest", random_forest), ("HistGBM", hist_gbm)]
    if ff_only: factories = factories[:1]
    results = defaultdict(lambda: defaultdict(list))
    for fold,(train_idx,test_idx) in enumerate(splits):
        X_train,X_test,y_train,y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx]
        for name,factory in factories:
            model = factory(42+fold)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            results[name]["fit"].append(time.perf_counter()-start)
            if name == "HistGBM": results[name]["iterations"].append(model.n_iter_)
            start = time.perf_counter()
            predictions = model.predict(X_test)
            results[name]["predict"].append(time.perf_counter()-start)
            results[name]["rmse"].append(mean_squared_error(y_test, predictions)**0.5)
            results[name]["r2"].append(r2_score(y_test, predictions))

    evaluation = "one 80/20 split" if folds == 1 else f"{folds}-fold CV"
    print(f"{dataset_name}: {len(X):,} rows, {X.shape[1]} features, {evaluation}")
    print(f"FastForest: trees={ff_trees}, min_node_size={min_node_size}, bootstrap_fraction={bootstrap_fraction}, "
        f"bootstrap_max={bootstrap_max}, replacement={replacement}, max_node_samples={max_node_samples}, cutoff_divisor={cutoff_divisor}")
    if not ff_only: print(f"sklearn RF trees={rf_trees}, HistGBM sklearn defaults")
    print(f"{'model':<15} {'RMSE':>14} {'R²':>14} {'fit total':>12} {'predict total':>14}")
    for name,_ in factories:
        result = results[name]
        rmse,r2 = np.asarray(result["rmse"]),np.asarray(result["r2"])
        if folds == 1: scores = f"{rmse[0]:>14.3f} {r2[0]:>14.3f}"
        else: scores = f"{rmse.mean():>6.3f} ± {rmse.std():<5.3f} {r2.mean():>6.3f} ± {r2.std():<5.3f}"
        print(f"{name:<15} {scores} {sum(result['fit']):>10.3f}s {sum(result['predict']):>12.3f}s")
    if not ff_only: print(f"HistGBM iterations by split: {results['HistGBM']['iterations']}")
