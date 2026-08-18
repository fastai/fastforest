from pathlib import Path

import numpy as np,pandas as pd
from fastcore.script import call_parse

from accuracy import Dataset,_rows,load_data,split_indices
from fastforest import FastForest,FastForestClassifier
from fastforest.preprocessing import _Encoder
from fastforest.tools import ADVISOR_PARAMS,advisor_features

def _sample(X, y, n=40_000, seed=42):
    if len(X) <= n: return X,np.asarray(y)
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return _rows(X, idx),np.asarray(y)[idx]

def _ashrae_sample(data_home, n=40_000):
    folder = Path(data_home)/"ashrae"
    train = pd.read_csv(folder/"train.csv", nrows=n)
    buildings = pd.read_csv(folder/"building_metadata.csv")
    weather = pd.read_csv(folder/"weather_train.csv")
    y = np.log1p(train.pop("meter_reading").to_numpy(dtype=np.float32))
    X = train.merge(buildings, on="building_id", how="left", validate="many_to_one")
    X = X.merge(weather, on=["site_id", "timestamp"], how="left", validate="many_to_one")
    X["timestamp"] = pd.Categorical(X.timestamp, ordered=True)
    missing = {name:np.nan for name in ("year_built", "floor_count", "air_temperature", "cloud_coverage", "dew_temperature",
        "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed")}
    return X,y,missing

def _dataset_metadata(dataset, rows, data_home):
    dataset = Dataset(dataset)
    if dataset == Dataset.ashrae: X,y,missing = _ashrae_sample(data_home)
    else:
        _,X,y,missing,task = load_data(dataset, data_home)
        train,_,_ = split_indices(dataset, X, y if task == "classification" else None)
        X,y = _sample(_rows(X, train), np.asarray(y)[train])
    encoder = _Encoder(missing)
    encoder.fit_transform(X)
    n_dates = len(encoder.date_columns)
    base = encoder.column_info[:-16*n_dates] if n_dates else encoder.column_info
    cardinalities = np.asarray([col.cardinality for col in base])
    y = np.asarray(y)
    if rows.task == "classification":
        _,counts = np.unique(y, return_counts=True)
        probabilities = counts/counts.sum()
        n_classes = len(counts)
        majority_fraction = float(probabilities.max())
        target_entropy = float(-(probabilities*np.log(probabilities)).sum())
        target_mean,target_std,target_unique_fraction = 0.,0.,n_classes/max(1,len(y))
    else:
        n_classes,majority_fraction,target_entropy = 1,1.,0.
        values = np.asarray(y, dtype=np.float64)
        target_mean,target_std = float(values.mean()),float(values.std())
        target_unique_fraction = len(np.unique(values))/len(values)
    return dict(dataset=str(dataset), n_rows=int(rows.rows), n_raw_cols=int(rows.features),
        n_logical_cols=len(encoder.column_info), n_encoded_cols=len(encoder.encoded_names),
        n_numeric_cols=sum(col.kind == "numeric" for col in base), n_lexical_cols=sum(col.kind == "lexical" for col in base),
        n_binary_cols=int((cardinalities == 2).sum()), n_low_card_cols=int(((cardinalities > 2)&(cardinalities <= 4)).sum()),
        n_high_card_cols=int((cardinalities > 4).sum()), n_constant_cols=int((cardinalities <= 1).sum()),
        n_missing_cols=len(missing or {}), n_observed_missing_cols=sum(col.had_missing for col in base),
        n_date_cols=n_dates, n_grouped_cols=len(encoder._bundles), n_classes=n_classes, output_dimensions=max(1,n_classes-1),
        majority_fraction=majority_fraction, target_entropy=target_entropy, target_mean=target_mean,
        target_std=target_std, target_unique_fraction=target_unique_fraction)

def metadata(results, path, data_home, refresh=False):
    "Build or load the bounded schema summary used by the advisor."
    path = Path(path)
    if path.exists() and not refresh: return pd.read_csv(path)
    rows = []
    for dataset,group in results.groupby("dataset", sort=False):
        print(f"schema {dataset}")
        rows.append(_dataset_metadata(dataset, group.iloc[0], data_home))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)

def _relative_targets(rows):
    default = rows.loc[rows.label == "defaults", ["dataset", "full_validation_loss"]].set_index("dataset").full_validation_loss
    baseline = rows.dataset.map(default)
    result = rows.full_validation_loss/baseline
    regression = rows.task == "regression"
    result.loc[regression] = np.sqrt(result.loc[regression])
    return result

def _matrix(rows, meta=None, categorical=False, task=None, sweep_labels=None):
    data = rows.copy() if meta is None else rows.merge(meta, on="dataset", validate="many_to_one")
    data["relative_quality"] = _relative_targets(data)
    return data,advisor_features(data, categorical, task, sweep_labels, ADVISOR_PARAMS)

def _evaluate(data, X, categorical, selection_labels, seed=42):
    predictions = np.empty(len(data), dtype=np.float32)
    evaluation_group = data.source_group if "source_group" in data else data.dataset.str.removesuffix("_grouped")
    for task in data.task.unique():
        task_rows = data.task == task
        for group in evaluation_group[task_rows].unique():
            valid = np.asarray(task_rows & (evaluation_group == group))
            train = np.asarray(task_rows & (evaluation_group != group))
            model = FastForest(seed=seed)
            model.fit(X.loc[train], data.loc[train, "relative_quality"])
            predictions[valid] = model.predict(X.loc[valid])
    columns = ["dataset", "task", "label", "relative_quality"]
    if "source_group" in data: columns.append("source_group")
    data = data[columns].copy()
    data["evaluation_group"] = evaluation_group
    data["predicted_quality"] = predictions
    selected = []
    for dataset,group in data.groupby("dataset", sort=False):
        group = group[group.label.isin(selection_labels[data.loc[group.index[0], "task"]])]
        chosen = group.loc[group.predicted_quality.idxmin()]
        oracle = group.loc[group.relative_quality.idxmin()]
        selected.append(dict(dataset=dataset, task=chosen.task, encoding="categorical" if categorical else "continuous",
            evaluation_group=chosen.evaluation_group,
            selected=chosen.label, selected_relative=chosen.relative_quality, predicted_relative=chosen.predicted_quality,
            oracle=oracle.label, oracle_relative=oracle.relative_quality, regret=chosen.relative_quality-oracle.relative_quality,
            exact=chosen.label == oracle.label))
    return data,pd.DataFrame(selected)

@call_parse
def main(
    results:str="meta/sweeps/all.csv",              # Consolidated sweep results
    metadata_csv:str="meta/sweep_advisor/datasets.csv", # Cached bounded dataset/schema summary
    output_dir:str="meta/sweep_advisor",            # Advisor matrices, evaluations, and models
    exclude_csv:str="tools/results/datasets.csv",   # README dataset groups held out from advisor training
    data_home:str=".data",                          # Dataset cache directory
    seed:int=42,                                     # Forest seed
    trees:int=None,                                  # Advisor trees; defaults to FastForest's rule
    refresh_metadata:bool=False,                     # Rebuild dataset/schema summaries from training rows
    quick:bool=False,                                # Train continuous advisors without held-out evaluation
):
    "Fit and group-evaluate sweep advisors using continuous and categorical hyperparameter representations."
    root = Path(__file__).parents[1]
    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else root/path
    rows = pd.read_csv(resolve(results), low_memory=False)
    exclusions = pd.read_csv(resolve(exclude_csv)).meta_group.dropna().unique()
    rows = rows[~rows.source_group.isin(exclusions)].reset_index(drop=True)
    sweep_labels = {
        "regression":["defaults", "split_prior_rows=3", "bootstrap_max=240000", "min_node_size=64",
            "split_prior_rows=8", "max_features=0.6", "max_node_samples=160", "replacement=false"],
        "classification":["defaults", "max_node_samples=2560", "replacement=false", "max_features=sqrt",
            "class_weight_power=0.5", "min_node_size=64", "max_features=1", "random_splitter=true"]}
    meta = None if "n_raw_cols" in rows else metadata(rows, resolve(metadata_csv), resolve(data_home), refresh_metadata)
    output = resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source_group":sorted(exclusions)}).to_csv(output/"held_out.csv", index=False)
    if quick:
        matrices,summaries = [],[]
        for task in ("regression", "classification"):
            task_rows = rows[rows.task == task].reset_index(drop=True)
            data,X = _matrix(task_rows, meta, False, task, None)
            model = FastForest(seed=seed, n_trees=trees).fit(X, data.relative_quality)
            model.save(output/f"{task}_continuous.ffm")
            matrices.append(pd.concat([data[["dataset", "task", "label", "relative_quality"]].reset_index(drop=True),
                X.reset_index(drop=True)], axis=1))
            summaries.append(dict(task=task, encoding="continuous", context_vectors=False, datasets=data.dataset.nunique()))
        pd.concat(matrices, ignore_index=True).to_csv(output/"features_continuous.csv", index=False)
        summary = pd.DataFrame(summaries)
        summary.to_csv(output/"summary.csv", index=False)
        print(summary.to_string(index=False))
        print(f"saved advisor artifacts to {output}")
        return
    summaries = []
    for categorical in (False,True):
        encoding = "categorical" if categorical else "continuous"
        matrices,predictions,selections = [],[],[]
        for task in ("regression", "classification"):
            task_rows = rows[rows.task == task].reset_index(drop=True)
            data,X = _matrix(task_rows, meta, categorical, task, sweep_labels[task])
            matrices.append(pd.concat([data[["dataset", "task", "label", "relative_quality"]].reset_index(drop=True),
                X.reset_index(drop=True)], axis=1))
            predicted,selected = _evaluate(data, X, categorical, sweep_labels, seed)
            predictions.append(predicted)
            selections.append(selected)
            group = selected
            task_predictions = predicted
            correlations = [candidate.relative_quality.corr(candidate.predicted_quality, method="spearman")
                for _,candidate in task_predictions.groupby("dataset")]
            summaries.append(dict(task=task, encoding=encoding, datasets=len(group),
                evaluation_groups=group.evaluation_group.nunique(), exact=int(group.exact.sum()),
                mean_selected=group.selected_relative.mean(), mean_oracle=group.oracle_relative.mean(),
                mean_regret=group.regret.mean(), improved=int((group.selected_relative < 1).sum()),
                prediction_mae=np.mean(np.abs(task_predictions.relative_quality-task_predictions.predicted_quality)),
                mean_rank_correlation=np.nanmean(correlations)))
            model = FastForest(seed=seed)
            model.fit(X, data.relative_quality)
            model.save(output/f"{task}_{encoding}.ffm")
        pd.concat(matrices, ignore_index=True).to_csv(output/f"features_{encoding}.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(output/f"predictions_{encoding}.csv", index=False)
        pd.concat(selections, ignore_index=True).to_csv(output/f"selection_{encoding}.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/"summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value:f"{value:.4f}"))
    print(f"saved advisor artifacts to {output}")
