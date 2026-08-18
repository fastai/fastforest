import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from accuracy import Dataset,_rows,load_data,split_indices
from fastforest import FastForest


def rmse(y, prediction): return float(np.mean((np.asarray(y)-prediction)**2, dtype=np.float64)**.5)

def main():
    _,X,y,missing_values,_,_ = load_data(Dataset.walmart, Path(__file__).parents[1]/".data")
    train_idx,valid_idx,_ = split_indices(Dataset.walmart,X,None)
    X_train,X_valid = _rows(X,train_idx),_rows(X,valid_idx)
    y_train,y_valid = y[train_idx],y[valid_idx]
    encoder = FastForest(bootstrap_max=40_000,max_features=.9,missing_values=missing_values,seed=42).fit(X_train,y_train)._encoder
    encoded_train,encoded_valid = encoder.transform(X_train),encoder.transform(X_valid)
    configs = [
        ("default",{}),
        ("sample40k",dict(max_samples=40_000)),
        ("sample160k",dict(max_samples=160_000)),
        ("leaf8",dict(min_samples_leaf=8)),
        ("sample160k/leaf8",dict(max_samples=160_000,min_samples_leaf=8)),
    ]
    for label,params in configs:
        model = RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=42,**params)
        started = time.perf_counter()
        model.fit(encoded_train,y_train)
        fit = time.perf_counter()-started
        started = time.perf_counter()
        prediction = model.predict(encoded_valid)
        predict = time.perf_counter()-started
        nodes = np.asarray([tree.tree_.node_count for tree in model.estimators_])
        depths = np.asarray([tree.tree_.max_depth for tree in model.estimators_])
        print(f"{label:<20} rmse={rmse(y_valid,prediction):.2f} fit={fit:.2f}s predict={predict:.3f}s "
            f"nodes={nodes.mean():.0f} depth={depths.mean():.1f}",flush=True)
        if label == "default":
            ranked = sorted(zip(model.feature_importances_,encoder.encoded_names),reverse=True)
            print("Top default RF impurity importance:",flush=True)
            for importance,name in ranked[:15]: print(f"  {name:<24} {importance:.4f}",flush=True)

if __name__ == "__main__": main()
