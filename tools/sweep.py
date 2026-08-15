import csv,time
from itertools import product
from pathlib import Path

import numpy as np
from fastcore.script import call_parse
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

from accuracy import Dataset,_max_features,_rows,load_data
from fastforest import FastForest,Workbench

def ints(values):
    "Parse comma-separated integers."
    return [int(value) for value in values.split(",")]

def floats(values):
    "Parse comma-separated floats."
    return [float(value) for value in values.split(",")]

@call_parse
def main(
    dataset:Dataset=Dataset.sgemm,        # Regression dataset
    bootstrap_maxes:str="40000,80000", # Per-tree row caps
    trees:str="25,50,100",             # Tree counts
    min_node_sizes:str="2,4,8",        # Minimum node sizes
    max_node_samples:str="320",        # Rows evaluated per node
    cutoff_divisors:str="2,3",         # Candidate-count divisors
    max_dummy_cardinalities:str="4",    # Largest cardinalities expanded to c-1 dummies
    min_candidate_rows:int=20,          # Candidate-count row floor
    candidate_attempt_factor:int=2,     # Maximum proposals per requested unique candidate
    splitters:str="histogram",          # Split searches
    max_features:str="0.75",            # Histogram feature counts/fractions
    feature_samplings:str="encoded",    # Histogram sampling units: encoded or columns
    leaf_regularizations:str="0",       # Parent-mean leaf pseudo-row counts
    seed:int=42,                         # Validation split and forest seed
    output:str="meta/sgemm_sweep.csv", # Result CSV
    data_home:str=None,                  # Dataset cache directory
):
    "Sweep FastForest hyperparameters on one validation split."
    root = Path(__file__).parents[1]
    if data_home is None: data_home = root/".data"
    _,X,y,missing_values = load_data(dataset, data_home)
    y = np.asarray(y, dtype=np.float32)
    train_idx,valid_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
    X_train,X_valid,y_train,y_valid = _rows(X, train_idx),_rows(X, valid_idx),y[train_idx],y[valid_idx]
    feature_values = [_max_features(value) for value in max_features.split(",")]
    workbenches = []
    for splitter in splitters.split(","):
        features = ["sqrt"] if splitter == "random" else feature_values
        samplings = ["encoded"] if splitter == "random" else feature_samplings.split(",")
        workbenches += [Workbench(splitter, feature, regularization, sampling)
            for feature,regularization,sampling in product(features, floats(leaf_regularizations), samplings)]
    configs = product(ints(bootstrap_maxes), ints(trees), ints(min_node_sizes), ints(max_node_samples), floats(cutoff_divisors),
        ints(max_dummy_cardinalities), workbenches)
    candidate_args = dict(min_candidate_rows=min_candidate_rows, candidate_attempt_factor=candidate_attempt_factor)
    rows = []
    for i,(cap,n_trees,min_size,node_samples,divisor,max_dummy,workbench) in enumerate(configs, 1):
        params = dict(candidate_args)
        params.update(n_trees=n_trees, min_node_size=min_size, bootstrap_max=cap, max_node_samples=node_samples)
        params.update(cutoff_divisor=divisor, max_dummy_cardinality=max_dummy, missing_values=missing_values, seed=seed, adaptive=False, workbench=workbench)
        model = FastForest(**params)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit = time.perf_counter()-start
        start = time.perf_counter()
        predictions = model.predict(X_valid)
        predict = time.perf_counter()-start
        rmse,r2 = mean_squared_error(y_valid, predictions)**0.5,r2_score(y_valid, predictions)
        row = dict(dataset=dataset, bootstrap_max=cap, trees=n_trees, min_node_size=min_size, max_node_samples=node_samples,
            cutoff_divisor=divisor, max_dummy_cardinality=max_dummy, min_candidate_rows=min_candidate_rows, candidate_attempt_factor=candidate_attempt_factor)
        row.update(splitter=workbench.splitter, max_features=workbench.max_features, feature_sampling=workbench.feature_sampling,
            leaf_regularization=workbench.leaf_regularization, rmse=rmse, r2=r2, fit=fit, predict=predict)
        rows.append(row)
        print(f"{i:>3}: cap={cap:>5} trees={n_trees:>3} min={min_size} samples={node_samples:>3} div={divisor:g} "
            f"dummy={max_dummy:>2} split={workbench.splitter:<9} features={str(workbench.max_features):<4} "
            f"sampling={workbench.feature_sampling:<7} reg={workbench.leaf_regularization:g} RMSE={row['rmse']:.4f} fit={fit:.3f}s")

    output = root/output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    print("\nBest accuracy:")
    for row in sorted(rows, key=lambda o: o["rmse"])[:10]: print(f"RMSE={row['rmse']:.4f} fit={row['fit']:.3f}s {row}")
    print(f"\nSaved {len(rows)} runs to {output}")
