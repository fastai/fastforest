from pathlib import Path

from fastcore.script import call_parse,is_cli

from fastforest.bench import benchmark_datasets
from fastforest.datasets import Dataset

README_DATASETS = tuple(map(Dataset, (
    "sgemm", "california", "concrete", "diamonds", "allstate", "diabetes", "bluebook", "walmart", "rossmann", "ashrae",
    "covertype", "adult", "bank", "click", "shuttle", "airlines", "higgs", "kddcup99", "sf_police")))


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


@call_parse
def main(
    dataset:Dataset=Dataset.california, # Regression or classification dataset
    datasets:str=None,             # Comma-separated datasets, or readme for all displayed datasets
    rf_trees:int=100,              # sklearn RF trees (its default)
    ff_trees:int=None,             # FastForest trees; defaults to its sampled-row rule
    min_node_size:int=None,        # FastForest minimum node size; model default when omitted
    bootstrap_fraction:float=None, # Defaults to 1, or 0.8 with OOB
    bootstrap_max:int=None,        # Maximum sampled rows per output; model default when omitted
    replacement:str="none",       # none is adaptive; true or false overrides it
    max_node_samples:int=None,     # Maximum rows evaluated per node; model default when omitted
    split_prior_rows:float=None,   # Prior rows used by the split score; model default when omitted
    class_weight_power:float=None, # Classification inverse-frequency weighting; model default when omitted
    cutoff_divisor:float=None,     # No-sort splitter candidate-count divisor; model default when omitted
    random_splitter:bool=False,    # Use the original random split search
    max_features:str=None,         # sqrt or a feature fraction; task default when omitted
    dates:bool=True,               # Let fastforest auto-detect dates
    target_statistics:bool=False, # Add per-level target statistics where eligible
    min_stat_cardinality:int=None, # Minimum eligible level cardinality; model default when omitted
    min_rows_per_level:int=None,   # Minimum training rows supporting a derived level
    stat_prior_rows:float=None,    # Prior strength for target statistics
    stat_permutations:int=None,    # Ordered statistic variants; zero uses plain means
    frequency:bool=False,          # Add exact per-level frequency features
    keep_rank:bool=True,           # Keep the original rank beside derived features
    order_buckets:int=None,        # Quantile buckets for target statistics
    natural_sort:bool=False,       # Sort text digit runs numerically
    timeout:int=180,               # Maximum seconds for each model/dataset combination, including loading
    ff_only:bool=False,            # Run only FastForest
    auto_only:bool=False,          # Run AutoForest with and without autogrow
    sizer_only:bool=False,         # Run AutoForest sample sizing without autogrow
    autogrow_only:bool=False,      # Run only AutoForest with autogrow
    rf_only:bool=False,            # Run only sklearn RandomForest
    hist_only:bool=False,          # Run only sklearn HistGradientBoosting
    max_rows:int=None,             # Optional reproducible dataset row limit
    data_home:str=None,            # Dataset cache directory
    save:bool=False,               # Update the README benchmark result CSV
    output:str=None,               # Resumable CSV collecting this run's full-precision results
    resume:bool=False,             # Skip dataset/model combinations already present in output
):
    "Compare accuracy and timing on canonical dataset splits."
    if timeout < 1: raise ValueError("timeout must be positive")
    if sum((ff_only,auto_only,sizer_only,autogrow_only,rf_only,hist_only)) > 1: raise ValueError("model-only options are mutually exclusive")
    if data_home is None: data_home = Path(__file__).parents[1]/".data"
    ff_kwargs = dict(n_trees=ff_trees, bootstrap_fraction=bootstrap_fraction, replacement=_replacement(replacement),
        random_splitter=random_splitter, date_columns=None if dates else {}, target_statistics=target_statistics,
        frequency=frequency, keep_rank=keep_rank, natural_sort=natural_sort)
    if max_features is not None: ff_kwargs["max_features"] = _max_features(max_features)
    for name,value in dict(min_node_size=min_node_size, bootstrap_max=bootstrap_max, max_node_samples=max_node_samples,
        split_prior_rows=split_prior_rows, class_weight_power=class_weight_power, cutoff_divisor=cutoff_divisor,
        min_stat_cardinality=min_stat_cardinality, min_rows_per_level=min_rows_per_level, stat_prior_rows=stat_prior_rows,
        stat_permutations=stat_permutations, order_buckets=order_buckets).items():
        if value is not None: ff_kwargs[name] = value
    models = ["FastForest", "RandomForest", "HistGBM"]
    if ff_only: models = models[:1]
    if auto_only: models = ["AutoForest", "Autogrow"]
    if sizer_only: models = ["AutoForest"]
    if autogrow_only: models = ["Autogrow"]
    if rf_only: models = models[1:2]
    if hist_only: models = models[2:]
    selected = README_DATASETS if datasets == "readme" else tuple(map(Dataset, datasets.split(","))) if datasets else (dataset,)
    results = benchmark_datasets(selected, models=models, output=output, resume=resume, timeout=timeout, rf_trees=rf_trees,
        ff_kwargs=ff_kwargs, max_rows=max_rows, data_home=data_home)
    if save:
        for current in selected:
            subset = results.for_dataset(current)
            if subset: subset.save(Path(__file__).parent/"results")
    if is_cli(): print(results)
    else: return results
