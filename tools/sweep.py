import csv,time
from itertools import product
from pathlib import Path

import numpy as np
from fastcore.script import call_parse
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

from accuracy import Dataset,load_data
from fastforest import FastForest

def ints(values):
    "Parse comma-separated integers."
    return [int(value) for value in values.split(",")]

def floats(values):
    "Parse comma-separated floats."
    return [float(value) for value in values.split(",")]

@call_parse
def main(
    bootstrap_maxes:str="40000,80000", # Per-tree row caps
    trees:str="25,50,100",             # Tree counts
    min_node_sizes:str="2,4,8",        # Minimum node sizes
    max_node_samples:str="80,160",     # Rows evaluated per node
    cutoff_divisors:str="2,3",         # Candidate-count divisors
    output:str="meta/sgemm_sweep.csv", # Result CSV
    data_home:str=None,                  # Dataset cache directory
):
    "Sweep FastForest hyperparameters on one SGEMM validation split."
    root = Path(__file__).parents[1]
    if data_home is None: data_home = root/".data"
    _,X,y = load_data(Dataset.sgemm, data_home)
    X,y = np.asarray(X, dtype=np.float32),np.asarray(y, dtype=np.float32)
    train_idx,valid_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_train,X_valid,y_train,y_valid = X[train_idx],X[valid_idx],y[train_idx],y[valid_idx]
    configs = product(ints(bootstrap_maxes), ints(trees), ints(min_node_sizes), ints(max_node_samples), floats(cutoff_divisors))
    rows = []
    for i,(cap,n_trees,min_size,node_samples,divisor) in enumerate(configs, 1):
        model = FastForest(n_trees=n_trees, min_node_size=min_size, bootstrap_max=cap,
            max_node_samples=node_samples, cutoff_divisor=divisor, seed=42)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit = time.perf_counter()-start
        start = time.perf_counter()
        predictions = model.predict(X_valid)
        predict = time.perf_counter()-start
        rmse,r2 = mean_squared_error(y_valid, predictions)**0.5,r2_score(y_valid, predictions)
        row = dict(bootstrap_max=cap, trees=n_trees, min_node_size=min_size, max_node_samples=node_samples,
            cutoff_divisor=divisor, rmse=rmse, r2=r2, fit=fit, predict=predict)
        rows.append(row)
        print(f"{i:>3}: cap={cap:>5} trees={n_trees:>3} min={min_size} samples={node_samples:>3} div={divisor:g} "
            f"RMSE={row['rmse']:.4f} fit={fit:.3f}s")

    output = root/output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    print("\nBest accuracy:")
    for row in sorted(rows, key=lambda o: o["rmse"])[:10]: print(f"RMSE={row['rmse']:.4f} fit={row['fit']:.3f}s {row}")
    print(f"\nSaved {len(rows)} runs to {output}")
