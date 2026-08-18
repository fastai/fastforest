import time, numpy as np
from fastcore.script import call_parse

from fastforest import FastForest

def best_time(f, repeats):
    "Run `f` repeatedly and return the best elapsed time and final result."
    best,result = float("inf"),None
    for _ in range(repeats):
        start = time.perf_counter()
        result = f()
        best = min(best, time.perf_counter()-start)
    return best,result

@call_parse
def main(
    rows:int=60_000, # Number of prediction rows
    cols:int=50,     # Number of features
    trees:int=50,    # Number of trees
    bootstrap_max:int=None, # Maximum training rows sampled per tree; model default when omitted
    cutoff_divisor:float=None, # No-sort splitter candidate-count divisor; model default when omitted
    max_features:float=None, # Fraction of feature units considered per node; model default when omitted
    repeats:int=5,   # Timed repetitions; the best is reported
):
    "Benchmark ordinary fit, OOB fit, and batch prediction on synthetic data."
    rng = np.random.default_rng(42)
    X = rng.random((rows, cols), dtype=np.float32)
    y = (10 + 4*X[:, 0] - 2*X[:, 1] + X[:, -1]).astype(np.float32)
    kwargs = dict(n_trees=trees, seed=42)
    for name,value in dict(bootstrap_max=bootstrap_max, cutoff_divisor=cutoff_divisor, max_features=max_features).items():
        if value is not None: kwargs[name] = value

    fit_time,model = best_time(lambda: FastForest(**kwargs).fit(X, y), repeats)
    oob_time,_ = best_time(lambda: FastForest(**kwargs, oob=True).fit(X, y), repeats)
    predict_time,_ = best_time(lambda: model.predict(X), repeats)
    print(f"rows={rows:,} cols={cols} trees={trees} repeats={repeats}")
    print(f"fit      {fit_time*1e3:9.2f} ms")
    print(f"fit+oob  {oob_time*1e3:9.2f} ms")
    print(f"predict  {predict_time*1e3:9.2f} ms")
