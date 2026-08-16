import statistics,subprocess,tempfile,time
from pathlib import Path

import numpy as np
from fastcore.script import call_parse
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def _run(*args): subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

def _csv(path, values, names): np.savetxt(path, values, delimiter=",", header=",".join(names), comments="")

def _bench(command, repeats):
    _run(*command)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        _run(*command)
        elapsed.append(time.perf_counter()-start)
    return statistics.median(elapsed)

@call_parse
def main(
    repeats:int=20, # Timed CLI invocations; the median is reported
    data_home:str="data", # sklearn download/cache directory
):
    "Benchmark end-to-end one-row Arrow CLI prediction on Concrete Strength."
    X,y = fetch_openml(data_id=44959, return_X_y=True, as_frame=False, data_home=data_home)
    X,y = np.asarray(X, dtype=np.float64),np.asarray(y, dtype=np.float64)
    X_train,X_valid,y_train,_ = train_test_split(X, y, test_size=.2, random_state=42)
    names = [f"x{i}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        train_csv,one_csv,valid_csv = directory/"train.csv",directory/"one.csv",directory/"valid.csv"
        train_arrow,one_arrow,valid_arrow = directory/"train.arrow",directory/"one.arrow",directory/"valid.arrow"
        model,output = directory/"concrete.ffm",directory/"prediction.arrow"
        _csv(train_csv, np.column_stack((X_train, y_train)), names+["strength"])
        _csv(one_csv, X_valid[:1], names)
        _csv(valid_csv, X_valid, names)
        _run("fastforest-convert", train_csv, "--output", train_arrow)
        _run("fastforest-convert", one_csv, "--output", one_arrow)
        _run("fastforest-convert", valid_csv, "--output", valid_arrow)
        _run("fastforest-fit", train_arrow, "--target", "strength", "--task", "regression", "--output", model)
        one = _bench(("fastforest-predict", model, one_arrow, "--output", output), repeats)
        valid = _bench(("fastforest-predict", model, valid_arrow, "--output", output), repeats)
    print(f"Concrete Strength, Arrow IPC: 1 validation row {one*1e3:.1f} ms; all {len(X_valid)} rows {valid*1e3:.1f} ms")
