import shutil,subprocess
from pathlib import Path

import numpy as np,pandas as pd,pyarrow as pa,pytest

from fastforest import FastForest,FastForestClassifier,feature_dependence,feature_relations,load,sklearn_preprocessor
from fastforest.auto import AutoForest,AutoForestClassifier
from fastforest._core import _sample_indices
from fastforest.preprocessing import _Encoder
from fastforest.tools import forest_suite,screen,validate

def test_signed_zero_and_new_missing_preprocessing(tmp_path):
    X = np.array([[0.], [-0.], [1.], [-1.]], dtype=np.float32)
    tiny = FastForest(n_trees=2, seed=42).fit(X, X[:,0])
    assert tiny.replacement_ and np.isfinite(tiny.predict(X)).all()
    train = pd.DataFrame({"number":[1.,2.,3.,4.], "label":["a","b","b","c"]})
    missing = pd.DataFrame({"number":[np.nan,2.], "label":[None,"a"]})
    with pytest.raises(ValueError, match="had none during training"):
        FastForest(n_trees=2, seed=42).fit(train, X[:,0]).predict(missing)
    permissive = FastForest(n_trees=2, seed=42, allow_new_missing=True).fit(train, X[:,0])
    assert np.isfinite(permissive.predict(missing)).all()
    config = [("defaults", {}, permissive.get_params())]
    assert np.isfinite(validate(permissive, train, X[:,0], missing, X[:2,0], config).results[0].validation_loss)
    permissive.save(tmp_path/"permissive.ffm")
    restored = load(tmp_path/"permissive.ffm")
    assert restored.allow_new_missing and np.array_equal(restored.predict(missing), permissive.predict(missing))

def test_fit_predict_oob_story(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.random((300, 5))
    y = 5*X[:, 0] - 3*X[:, 1] + X[:, 4]
    model = FastForest(n_trees=24, min_node_size=8, replacement=False, max_node_samples=80, seed=99, oob=True)

    assert model.fit(X[:, ::-1][:, ::-1], y) is model # non-contiguous float64 input is accepted
    predictions = model.predict(X)
    baseline_mse = np.mean((y-y.mean())**2)
    assert predictions.dtype == np.float32
    assert predictions.shape == y.shape
    assert np.mean((predictions-y)**2) < baseline_mse*0.3
    assert model.n_features_in_ == X.shape[1]
    assert model.get_params()["cutoff_divisor"] == 10
    assert model.get_params()["max_dummy_cardinality"] == 1
    assert model.get_params()["max_features"] == .9
    assert model.get_params()["split_prior_rows"] == 3
    assert FastForest().max_node_samples == 320
    assert FastForestClassifier().class_weight_power == .75
    assert FastForest().min_node_size == 8 and FastForest().max_features == .9
    assert model.oob_prediction_.shape == y.shape
    assert model.oob_counts_.shape == y.shape
    assert np.array_equal(model.oob_indices_, np.arange(len(y)))
    assert np.mean(model.oob_counts_ > 0) >= 0.99
    assert np.array_equal(predictions, FastForest(n_trees=24, min_node_size=8,
        replacement=False, max_node_samples=80, seed=99, oob=True).fit(X, y).predict(X))

    alternate = FastForest(n_trees=12, min_node_size=8, max_node_samples=80, seed=99,
        random_splitter=True, max_features="sqrt").fit(X, y)
    alternate_predictions = alternate.predict(X)
    assert np.mean((alternate_predictions-y)**2) < baseline_mse*0.3
    assert alternate.get_params()["random_splitter"] is True

    tree_predictions = model.predict_trees(X)
    assert tree_predictions.shape == (len(X), model.n_trees)
    assert np.allclose(tree_predictions.mean(axis=1), predictions, atol=1e-5)
    assert np.array_equal(model.predict_std(X), tree_predictions.std(axis=1))

    explanation = model.explain(X[:4])
    assert np.allclose(explanation.prediction, explanation.bias+explanation.contributions.sum(axis=1), atol=1e-4)
    assert explanation.row()[0][0] == "x0"
    split_importance = model.split_importance()
    assert np.isclose(split_importance.values.sum(), 1)
    importance = model.feature_importance(X, y, n_repeats=2, n_samples=None)
    assert importance.sorted().names[0] == "x0"
    assert model.drop_column_importance(X, y, features=["x0"]).values[0] > 0

    pdp = model.partial_dependence(X, "x0", grid_points=8, n_samples=100)
    assert pdp.average.shape == (8,) and pdp.individual.shape == (100, 8)
    assert pdp.average[-1] > pdp.average[0]
    assert pdp.clustered_ice(3).shape == (3, 8)
    categorical = model.partial_dependence(X, {"choice":["x0", "x1"]}, n_samples=50)
    assert categorical.grids[0].tolist() == ["x0", "x1"] and categorical.individual.shape == (50, 2)
    surface = model.partial_dependence(X, ["x0", "x1"], grid_points=5, n_samples=50)
    assert surface.average.shape == (5, 5) and surface.individual is None

    related = np.column_stack([X[:200, :3], X[:200, 0]])
    relations = feature_relations(related, ["signal", "other", "weak", "signal_copy"])
    assert ("signal", "signal_copy") in relations.groups(threshold=0.01)
    dependence = feature_dependence(related, n_samples=None, n_trees=5, feature_names=['signal', 'other', 'weak', 'signal_copy'])
    assert dependence.predictability[0] > 0.8 and dependence.predictability[3] > 0.8

    mixed = np.empty((300, 5), dtype=object)
    mixed[:,0] = ["NA" if i%37 == 0 else str(i) for i in range(len(mixed))]
    mixed[:,1] = ["rare" if i%29 == 0 else "middle" if i%3 == 0 else "common" for i in range(len(mixed))]
    mixed[:,2] = [f"group-{i%9}" for i in range(len(mixed))]
    mixed[:,3] = [str(i%12) for i in range(len(mixed))]
    mixed[:,4] = ""
    mixed_y = np.asarray([0 if value == "NA" else float(value) for value in mixed[:,0]]) + (mixed[:,1] == "common")*20
    mixed_frame = pd.DataFrame(mixed, columns=[f"x{i}" for i in range(mixed.shape[1])])
    mixed_frame[["x1", "x2"]] = mixed_frame[["x1", "x2"]].astype("category")
    with pytest.warns(UserWarning, match="Skipping features"): sklearn_X = sklearn_preprocessor(
        mixed_frame, {0:"NA"}, onehot_max=4).fit_transform(mixed_frame, mixed_y)
    assert sklearn_X.shape == (len(mixed_frame), 6)
    mixed_model = FastForest(n_trees=20, seed=42, missing_values={0:"NA"}, max_dummy_cardinality=4).fit(mixed_frame, mixed_y)
    arrow_model = FastForest(n_trees=20, seed=42, missing_values={0:"NA"}, max_dummy_cardinality=4).fit(
        pa.Table.from_pandas(mixed_frame, preserve_index=False), mixed_y)
    assert np.array_equal(arrow_model.predict(pa.Table.from_pandas(mixed_frame.iloc[:20], preserve_index=False)), mixed_model.predict(mixed_frame.iloc[:20]))
    assert [info.kind for info in mixed_model.column_info_] == ["numeric", "lexical", "lexical", "numeric", "discarded"]
    assert mixed_model.column_info_[3].all_int and mixed_model.column_info_[3].encoded_features == ("x3",)
    assert mixed_model.column_info_[4].encoded_features == ()
    assert mixed_model.column_info_[2].encoded_features == ("x2",)
    assert mixed_model._encoder.feature_group_ids[0] == mixed_model._encoder.feature_group_ids[1]
    assert len(np.unique(mixed_model._encoder.feature_group_ids)) == len(mixed_model._encoder.feature_group_ids)-1
    assert set(mixed_model.column_info_[1].encoded_features) == {"x1=common", "x1=middle", "x1=rare"}
    assert mixed_model.feature_importances_.shape == (mixed.shape[1],) and np.isclose(mixed_model.feature_importances_.sum(), 1)
    assert np.isfinite(mixed_model.predict(mixed_frame.iloc[:4])).all()
    novel = mixed[:4].copy()
    novel[:,0] = ["10.5", "301", "NA", "-2"]
    novel[:,1] = ["unseen", "common", "middle", "rare"]
    novel[:,2] = ["group-new", "group-2", "group-8", "aaa"]
    assert np.isfinite(mixed_model.predict(novel)).all()
    mixed_model.save(tmp_path/"mixed.ffm")
    restored = load(tmp_path/"mixed.ffm")
    assert np.array_equal(restored.predict(novel), mixed_model.predict(novel))
    assert restored.feature_names_in_ == mixed_model.feature_names_in_ and restored.oob_prediction_ is None
    mixed_frame.iloc[:25].to_csv(tmp_path/"mixed.csv", index=False)
    restored.predict_file(tmp_path/"mixed.csv", tmp_path/"mixed-predictions.csv", batch_size=7)
    assert np.allclose(pd.read_csv(tmp_path/"mixed-predictions.csv").prediction, mixed_model.predict(mixed_frame.iloc[:25]))
    mixed_explanation = mixed_model.explain(novel)
    assert mixed_explanation.values[0,1] == "unseen" and isinstance(mixed_explanation.values[0,3], int)
    assert np.allclose(mixed_explanation.prediction,
        mixed_explanation.bias+mixed_explanation.contributions.sum(axis=1), atol=1e-4)
    assert len(mixed_model.tree_node_counts_) == mixed_model.n_trees_
    assert np.array_equal(mixed_model.tree_node_counts_, 2*mixed_model.tree_leaf_counts_-1)
    assert (mixed_model.tree_depths_ > 0).all()
    mixed_importance = mixed_model.feature_importance(mixed, mixed_y, n_repeats=1, n_samples=100)
    assert mixed_importance.values.shape == (mixed.shape[1],)
    numeric_pdp = mixed_model.partial_dependence(mixed, "x0", grid_points=6, n_samples=30)
    category_pdp = mixed_model.partial_dependence(mixed, "x1", grid_points=6, n_samples=30)
    assert np.issubdtype(numeric_pdp.grids[0].dtype, np.integer)
    assert category_pdp.grids[0].tolist() == ["common", "middle", "rare"]
    bad_missing = novel.copy()
    bad_missing[0,1] = ""
    with pytest.raises(ValueError, match="had none during training"): mixed_model.predict(bad_missing)
    bad_numeric = novel.copy()
    bad_numeric[0,0] = "not-a-number"
    with pytest.raises(ValueError, match="was numeric during training"): mixed_model.predict(bad_numeric)

    pool = np.asarray(_sample_indices(len(mixed_frame), 117, 42, 2))
    borrowed,owned = (_Encoder({0:"NA"}, 4),_Encoder({0:"NA"}, 4))
    borrowed_values = borrowed.fit_transform(mixed_frame, pool)
    owned_values = owned.fit_transform(mixed_frame.iloc[pool])
    assert np.array_equal(borrowed_values, owned_values)
    assert np.array_equal(borrowed.cutoff_values, owned.cutoff_values)
    assert borrowed.column_info == owned.column_info and borrowed.encoded_names == owned.encoded_names

    dated = pd.DataFrame({"eventDate":["2023-12-31 23:30:15", "2024-01-01 00:00:00", "2024-02-29 12:05:09", "2024-04-01 08:15:30"]*20,
        "signal":np.arange(80)})
    dated_y = dated.signal+pd.to_datetime(dated.eventDate).dt.month
    dated_model = FastForest(n_trees=12, seed=42, max_features=.75).fit(dated, dated_y)
    assert dated_model.date_columns is None and dated_model.date_columns_ == {'eventDate': '%Y-%m-%d %H:%M:%S'}
    assert dated_model.feature_names_in_[:3] == ("signal", "eventYear", "eventMonth")
    assert dated_model.feature_names_in_[-4:] == ("eventHour", "eventMinute", "eventSecond", "eventElapsed")
    assert len(dated_model.feature_names_in_) == 17 and np.isfinite(dated_model.predict(dated.iloc[:4])).all()
    displayed = dated_model._encoder.display(dated.iloc[:1])
    assert displayed[0,1:5].tolist() == [2023, 12, 52, 31]
    malformed = dated.iloc[:2].copy()
    malformed.loc[malformed.index[0], "eventDate"] = "not-a-date"
    assert np.isfinite(dated_model.predict(malformed)).all()
    categorical_dated = dated.copy()
    categorical_dated.loc[categorical_dated.index[0], "eventDate"] = "not-a-date"
    categorical_dated = categorical_dated.astype({"eventDate":"category"})
    categorical_model = FastForest(n_trees=12, seed=42, max_features=.75,
        date_columns={'eventDate': '%Y-%m-%d %H:%M:%S'}).fit(categorical_dated, dated_y)
    assert np.isfinite(categorical_model.predict(categorical_dated.iloc[:4])).all()
    dated_model.save(tmp_path/"dated.ffm")
    loaded_dated = load(tmp_path/"dated.ffm")
    assert loaded_dated.date_columns_ == dated_model.date_columns_
    assert np.array_equal(loaded_dated.predict(dated.iloc[:8]), dated_model.predict(dated.iloc[:8]))
    compact = pd.DataFrame({"compactDate":[20240113,20240229,20241231,20250816]*20,
        "dayDate":["13/01/2024","29/02/2024","31/12/2024","16/08/2025"]*20, "signal":np.arange(80)})
    compact_model = FastForest(n_trees=4, seed=42).fit(compact, compact.signal)
    assert compact_model.date_columns_ == {"compactDate":"%Y%m%d", "dayDate":"%d/%m/%Y"}

def test_bounded_pool_defaults_story():
    rng = np.random.default_rng(42)
    X = rng.random((8_000, 6), dtype=np.float32)
    y = 5*X[:,0] - 3*X[:,1] + X[:,5]
    model = FastForest(n_trees=8, min_node_size=16, seed=42).fit(X, y)
    assert model.max_features == .9
    assert np.isfinite(model.predict(X[:10])).all()

    fixed = FastForest(n_trees=8, min_node_size=16, max_features="sqrt", seed=42).fit(X, y)
    assert fixed.get_params()["max_features"] == "sqrt"

    suite = forest_suite(FastForest(seed=42))
    report = screen(FastForest(seed=42), X[:600], y[:600], suite, trees=8)
    assert report.task == "regression" and report.trees == 8 and len(report.results) == len(suite)
    assert report.results[0].label == "defaults" and report.results[0].oob_loss > report.results[0].train_loss > 0
    assert all(0 < result.coverage <= 1 and result.nodes_mean >= result.leaves_mean for result in report.results)
    calibrated = validate(FastForest(seed=42), X[:500], y[:500], X[500:600], y[500:600], suite[:2])
    assert len(calibrated.results) == 2 and all(result.trees >= 32 for result in calibrated.results)
    assert all(result.validation_loss > 0 and result.train_loss > 0 for result in calibrated.results)
    assert all(result.fit_seconds > 0 and result.predict_seconds > 0 for result in calibrated.results)

    classifier = FastForestClassifier(n_trees=2, seed=42)
    classification_suite = [("defaults", {}, classifier.get_params())]
    with pytest.raises(ValueError, match="unseen class.*new"):
        validate(classifier, X[:20], np.where(y[:20] > 0,"high","low"), X[20:22], ["high","new"], classification_suite)
    unseen = validate(classifier, X[:20], np.where(y[:20] > 0,"high","low"), X[20:22], ["high","new"],
        classification_suite, allow_unseen_classes=True)
    assert np.isfinite(unseen.results[0].validation_loss)

    large_X = rng.random((10_000, 2), dtype=np.float32)
    bounded = FastForest(n_trees=2, bootstrap_max=100, seed=42, oob=True).fit(large_X, large_X[:,0])
    assert bounded.column_info_[0].cardinality == 126 # ceil(.63 * 2 trees * 100 rows)
    assert len(bounded.oob_indices_) == len(bounded.oob_counts_) == 100
    assert len(np.unique(bounded.oob_indices_)) == 100 and bounded.oob_indices_.max() < len(large_X)
    untracked = FastForest(n_trees=2, bootstrap_max=100, seed=42).fit(large_X, large_X[:,0])
    assert np.array_equal(untracked.predict(large_X[:20]), bounded.predict(large_X[:20]))

    sized = AutoForest(bootstrap_max=100, seed=42).fit(X[:600], y[:600])
    assert 32 <= sized.n_trees_ <= 64 and sized.tree_history_ == () and sized.oob_prediction_ is None and sized.min_improvement == .01
    automatic = AutoForest(autogrow=True, tree_batch_size=8, max_trees=16, min_improvement=.99, bootstrap_max=100, seed=42).fit(X[:600], y[:600])
    assert automatic.n_trees_ == 8 and len(automatic.tree_history_) == 2
    assert automatic.tree_history_[-1]["accepted"] is False and automatic.sizing_["active"] is True
    labels = np.where(y[:600]>np.median(y[:600]), "high", "low")
    auto_classifier = AutoForestClassifier(autogrow=True, tree_batch_size=8, max_trees=16, min_improvement=.99, seed=42).fit(X[:600], labels)
    assert auto_classifier.n_trees_ == 8 and auto_classifier.predict_proba(X[:4]).shape == (4,2)

def test_multiclass_prediction_oob_and_analysis_story(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.random((900, 6), dtype=np.float32)
    class_id = np.select([X[:,0]+X[:,1] > 1.25, X[:,2]-X[:,3] > .15], [0, 1], default=2)
    y = np.asarray(["canopy", "soil", "water"])[class_id]
    model = FastForestClassifier(n_trees=32, min_node_size=6, max_node_samples=120, seed=42, oob=True, max_features=.75).fit(X, y)

    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    assert model.classes_.tolist() == ["canopy", "soil", "water"]
    assert 1 <= model.prediction_trees_per_batch_ <= model.n_trees
    assert prediction.dtype.kind == "U" and probabilities.dtype == np.float32
    assert prediction.shape == y.shape and probabilities.shape == (len(y), 3)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    assert np.array_equal(prediction, model.classes_[probabilities.argmax(axis=1)])
    assert np.mean(prediction == y) > .95
    assert model.oob_decision_function_.shape == probabilities.shape
    assert np.array_equal(model.oob_indices_, np.arange(len(y)))
    valid = model.oob_counts_ > 0
    assert valid.mean() > .99 and np.allclose(model.oob_decision_function_[valid].sum(axis=1), 1, atol=1e-5)
    assert model.oob_score_ > .85
    model.save(tmp_path/"classes.ffm")
    restored = load(tmp_path/"classes.ffm")
    assert np.array_equal(restored.classes_, model.classes_)
    assert np.array_equal(restored.predict_proba(X[:20]), probabilities[:20])
    assert np.array_equal(restored.tree_node_counts_, model.tree_node_counts_)
    pd.DataFrame(X[:25], columns=[f"x{i}" for i in range(X.shape[1])]).to_csv(tmp_path/"classes.csv", index=False)
    restored.predict_file(tmp_path/"classes.csv", tmp_path/"class-probabilities.csv", batch_size=9, proba=True)
    assert np.allclose(pd.read_csv(tmp_path/"class-probabilities.csv"), probabilities[:25], atol=1e-6)
    assert np.isclose(model.feature_importances_.sum(), 1)
    assert model.feature_importance(X, y, n_repeats=1, n_samples=300).values[:4].max() > 0
    assert model.drop_column_importance(X, y, features=["x0"]).values[0] > 0
    again = FastForestClassifier(n_trees=32, min_node_size=6, max_node_samples=120, seed=42, oob=True, max_features=.75).fit(X, y)
    assert np.array_equal(probabilities, again.predict_proba(X))

    category = np.arange(len(X))%3
    grouped_X = np.column_stack([X[:,:2], np.eye(3, dtype=np.float32)[category]])
    grouped_y = np.asarray(["red", "green", "blue"])[category]
    grouped = FastForestClassifier(n_trees=16, seed=42, max_features=.75,
        one_hot_groups={"color":["x2", "x3", "x4"]}).fit(grouped_X, grouped_y)
    assert grouped.feature_names_in_ == ("x0", "x1", "color") and grouped.n_features_in_ == 3
    assert grouped.column_info_[-1].kind == "lexical" and grouped.column_info_[-1].cardinality == 3
    assert grouped.feature_importances_.shape == (3,) and np.mean(grouped.predict(grouped_X) == grouped_y) > .99
    assert grouped._encoder.display(grouped_X[:3])[:,-1].tolist() == ["x2", "x3", "x4"]
    assert grouped.feature_importance(grouped_X, grouped_y, n_repeats=1, n_samples=100).values.shape == (3,)
    grouped.save(tmp_path/"grouped.ffm")
    assert np.array_equal(load(tmp_path/"grouped.ffm").predict(grouped_X[:20]), grouped.predict(grouped_X[:20]))
    invalid_group = grouped_X[:1].copy()
    invalid_group[:,2:] = 0
    with pytest.raises(ValueError, match="has no active category"): grouped.predict(invalid_group)

    large_X = rng.random((8_000, 4), dtype=np.float32)
    large_y = np.where(large_X[:,0]+large_X[:,1] > 1, "high", "low")
    defaulted = FastForestClassifier(n_trees=8, min_node_size=12, seed=42).fit(large_X, large_y)
    assert defaulted.max_features == .6 and defaulted.replacement_ and np.isfinite(defaulted.predict_proba(large_X[:10])).all()

def test_validation_errors():
    for command in ("fastforest-fit", "fastforest-predict", "fastforest-convert", "fastforest-compile", "viewcsv"):
        executable = Path(shutil.which(command))
        assert b"\0" in executable.read_bytes()[:1024]
        result = subprocess.run([command, "--help"], capture_output=True, text=True)
        assert result.returncode == 0 and "Usage:" in result.stdout

    X,y = np.ones((4, 2)),np.ones(4)
    def check(call, message):
        with pytest.raises((TypeError, ValueError, RuntimeError), match=message): call()
    check(lambda: FastForest().fit(X[:, 0], y), "X must be a two-dimensional array")
    check(lambda: FastForest().fit(X, y[:, None]), "y must be a one-dimensional array")
    check(lambda: FastForestClassifier().fit(X, ["one"]*4), "at least two classes")
    check(lambda: FastForestClassifier().fit(X, ["one", "two", None, "two"]), "cannot be missing")
    check(lambda: FastForestClassifier().fit(X, [1, 2, np.nan, 2]), "cannot be missing")
    check(lambda: FastForest().fit(X, y[:-1]), "X has 4 rows but y has 3 values")
    check(lambda: FastForest().fit(np.array([[1, np.nan]]), [1]), "non-finite numeric value")
    check(lambda: FastForest(bootstrap_max=0).fit(X, y), "bootstrap_max must be greater than zero")
    check(lambda: FastForest(bootstrap_fraction=1.1, replacement=False).fit(X, y), "cannot exceed 1 without replacement")
    assert FastForest(bootstrap_fraction=1.1, replacement=True).fit(X, y).predict(X).shape == y.shape
    check(lambda: FastForest(max_features=0).fit(X, y), "max_features must be")
    check(lambda: FastForest(max_features="all").fit(X, y), "max_features must be")
    check(lambda: FastForest(max_dummy_cardinality=0).fit(X, y), "max_dummy_cardinality must be a positive integer")
    check(lambda: FastForest(cutoff_divisor=np.nan).fit(X, y), "cutoff_divisor must be finite and greater than zero")
    check(lambda: FastForest().predict(X), "must be fitted")
    check(lambda: FastForest().explain(X), "must be fitted")
