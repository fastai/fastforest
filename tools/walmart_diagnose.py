import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from accuracy import Dataset,_rows,load_data,split_indices
from fastforest import FastForest


def rmse(y, prediction): return float(np.mean((np.asarray(y)-prediction)**2, dtype=np.float64)**.5)

def fit_hist(X, y, **kwargs):
    model = HistGradientBoostingRegressor(random_state=42, **kwargs)
    started = time.perf_counter()
    model.fit(X,y)
    return model,time.perf_counter()-started

def split_usage(model, names):
    gains,counts = np.zeros(len(names)),np.zeros(len(names),dtype=int)
    for iteration in model._predictors:
        for predictor in iteration:
            nodes = predictor.nodes
            for node in nodes[nodes["is_leaf"] == 0]:
                feature = int(node["feature_idx"])
                gains[feature] += float(node["gain"])
                counts[feature] += 1
    return sorted(zip(gains,counts,names), reverse=True)

def main():
    _,X,y,missing_values,_,_ = load_data(Dataset.walmart, Path(__file__).parents[1]/".data")
    train_idx,valid_idx,_ = split_indices(Dataset.walmart,X,None)
    X_train,X_valid = _rows(X,train_idx),_rows(X,valid_idx)
    y_train,y_valid = y[train_idx],y[valid_idx]

    started = time.perf_counter()
    forest = FastForest(bootstrap_max=40_000,max_features=.9,missing_values=missing_values,seed=42).fit(X_train,y_train)
    forest_fit = time.perf_counter()-started
    print(f"FF 40k/.9: rmse={rmse(y_valid,forest.predict(X_valid)):.2f} fit={forest_fit:.3f}s",flush=True)
    started = time.perf_counter()
    encoded_train,encoded_valid = forest._encoder.transform(X_train),forest._encoder.transform(X_valid)
    print(f"FF encoding: {encoded_train.shape[1]} columns in {time.perf_counter()-started:.3f}s",flush=True)
    print("Encoded columns: "+", ".join(forest._encoder.encoded_names),flush=True)

    deep,deep_fit = fit_hist(encoded_train,y_train,max_iter=300,max_leaf_nodes=255)
    print(f"Deep HistGBM: rmse={rmse(y_valid,deep.predict(encoded_valid)):.2f} fit={deep_fit:.3f}s iterations={deep.n_iter_}",flush=True)
    categorical = [name in ("Store","Dept") for name in forest._encoder.encoded_names]
    deep_cat,deep_cat_fit = fit_hist(encoded_train,y_train,max_iter=300,max_leaf_nodes=255,categorical_features=categorical)
    print(f"Deep HistGBM + categorical Store/Dept: rmse={rmse(y_valid,deep_cat.predict(encoded_valid)):.2f} "
        f"fit={deep_cat_fit:.3f}s iterations={deep_cat.n_iter_}",flush=True)

    inspected = min((deep,deep_cat),key=lambda model:rmse(y_valid,model.predict(encoded_valid)))
    print(f"Inspecting best HistGBM RMSE={rmse(y_valid,inspected.predict(encoded_valid)):.2f}",flush=True)

    print("Top HistGBM split gains:",flush=True)
    for gain,count,name in split_usage(inspected,forest._encoder.encoded_names)[:20]: print(f"  {name:<24} gain={gain:14.0f} splits={count}",flush=True)
    rng = np.random.default_rng(42)
    selected = rng.choice(len(encoded_valid),min(5_000,len(encoded_valid)),replace=False)
    importance = permutation_importance(inspected,encoded_valid[selected],y_valid[selected],scoring="neg_root_mean_squared_error",n_repeats=1,random_state=42)
    ranked = sorted(zip(importance.importances_mean,forest._encoder.encoded_names),reverse=True)
    print("Top validation permutation RMSE increases:",flush=True)
    for increase,name in ranked[:20]: print(f"  {name:<24} {increase:10.2f}",flush=True)

if __name__ == "__main__": main()
