import numpy as np,pandas as pd,pytest

from fastforest import FastForest,Workbench,feature_dependence,feature_relations,sklearn_preprocessor

def test_fit_predict_oob_story():
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
    assert model.get_params()["min_candidate_rows"] == 20
    assert model.get_params()["candidate_attempt_factor"] == 2
    assert model.get_params()["max_dummy_cardinality"] == 4
    assert model.get_params()["adaptive"] is True
    assert FastForest().max_node_samples == 320
    assert Workbench() == Workbench("histogram", 0.75, 0)
    assert model.oob_prediction_.shape == y.shape
    assert model.oob_counts_.shape == y.shape
    assert np.mean(model.oob_counts_ > 0) >= 0.99
    assert np.array_equal(predictions, FastForest(n_trees=24, min_node_size=8,
        replacement=False, max_node_samples=80, seed=99, oob=True).fit(X, y).predict(X))

    workbench = Workbench(splitter="random", max_features="sqrt")
    alternate = FastForest(n_trees=12, min_node_size=8, max_node_samples=80, seed=99, workbench=workbench).fit(X, y)
    alternate_predictions = alternate.predict(X)
    assert np.mean((alternate_predictions-y)**2) < baseline_mse*0.3
    assert alternate.get_params()["workbench"] == workbench

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
    assert dependence.predictability[0] > 0.9 and dependence.predictability[3] > 0.9

    mixed = np.empty((300, 5), dtype=object)
    mixed[:,0] = ["NA" if i%37 == 0 else str(i) for i in range(len(mixed))]
    mixed[:,1] = ["rare" if i%29 == 0 else "middle" if i%3 == 0 else "common" for i in range(len(mixed))]
    mixed[:,2] = [f"group-{i%9}" for i in range(len(mixed))]
    mixed[:,3] = [str(i%12) for i in range(len(mixed))]
    mixed[:,4] = ""
    mixed_y = np.asarray([0 if value == "NA" else float(value) for value in mixed[:,0]]) + (mixed[:,1] == "common")*20
    mixed_frame = pd.DataFrame(mixed, columns=[f"x{i}" for i in range(mixed.shape[1])])
    mixed_frame[["x1", "x2"]] = mixed_frame[["x1", "x2"]].astype("category")
    sklearn_X = sklearn_preprocessor(mixed_frame, {0:"NA"}, onehot_max=4).fit_transform(mixed_frame, mixed_y)
    assert sklearn_X.shape == (len(mixed_frame), 7)
    grouped = Workbench(feature_sampling="columns")
    mixed_model = FastForest(n_trees=20, seed=42, missing_values={0:"NA"}, max_dummy_cardinality=10, workbench=grouped).fit(mixed_frame, mixed_y)
    assert [info.kind for info in mixed_model.column_info_] == ["numeric", "lexical", "lexical", "numeric", "discarded"]
    assert mixed_model.column_info_[3].all_int and mixed_model.column_info_[4].encoded_features == ()
    assert len(mixed_model.column_info_[2].encoded_features) == 8
    assert mixed_model._encoder.feature_group_ids[0] == mixed_model._encoder.feature_group_ids[1]
    assert len(np.unique(mixed_model._encoder.feature_group_ids)) == len(mixed_model._encoder.feature_group_ids)-1
    assert not any("rare" in name for name in mixed_model.column_info_[1].encoded_features)
    assert mixed_model.feature_importances_.shape == (mixed.shape[1],) and np.isclose(mixed_model.feature_importances_.sum(), 1)
    assert np.isfinite(mixed_model.predict(mixed_frame.iloc[:4])).all()
    novel = mixed[:4].copy()
    novel[:,0] = ["10.5", "301", "NA", "-2"]
    novel[:,1] = ["unseen", "common", "middle", "rare"]
    novel[:,2] = ["group-new", "group-2", "group-8", "aaa"]
    assert np.isfinite(mixed_model.predict(novel)).all()
    mixed_explanation = mixed_model.explain(novel)
    assert mixed_explanation.values[0,1] == "unseen" and isinstance(mixed_explanation.values[0,3], int)
    assert np.allclose(mixed_explanation.prediction,
        mixed_explanation.bias+mixed_explanation.contributions.sum(axis=1), atol=1e-4)
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

def test_adaptive_defaults_story():
    rng = np.random.default_rng(42)
    X = rng.random((8_100, 6), dtype=np.float32)
    y = 5*X[:,0] - 3*X[:,1] + X[:,5]
    model = FastForest(n_trees=8, min_node_size=16, seed=42).fit(X, y)
    assert [(score[0], score[1]) for score in model.adaptive_scores_] == [(0.6, 320), (0.9, 320)]
    assert model.adaptive_choice_ == min(model.adaptive_scores_, key=lambda score:score[2])[:2]
    assert np.isfinite(model.predict(X[:10])).all()

    fixed = FastForest(n_trees=8, min_node_size=16, adaptive=False, seed=42).fit(X, y)
    assert fixed.adaptive_scores_ == () and fixed.adaptive_choice_ is None

def test_validation_errors():
    X,y = np.ones((4, 2)),np.ones(4)
    def check(call, message):
        with pytest.raises((TypeError, ValueError, RuntimeError), match=message): call()
    check(lambda: FastForest().fit(X[:, 0], y), "X must be a two-dimensional array")
    check(lambda: FastForest().fit(X, y[:, None]), "y must be a one-dimensional array")
    check(lambda: FastForest().fit(X, y[:-1]), "X has 4 rows but y has 3 values")
    check(lambda: FastForest().fit(np.array([[1, np.nan]]), [1]), "non-finite numeric value")
    check(lambda: FastForest(bootstrap_max=0).fit(X, y), "bootstrap_max must be greater than zero")
    check(lambda: FastForest(bootstrap_fraction=1.1).fit(X, y), "cannot exceed 1 without replacement")
    assert FastForest(bootstrap_fraction=1.1, replacement=True).fit(X, y).predict(X).shape == y.shape
    check(lambda: FastForest(candidate_attempt_factor=0).fit(X, y), "candidate_attempt_factor must be greater than zero")
    check(lambda: Workbench(splitter="nope"), "splitter must be")
    check(lambda: Workbench(max_features=0), "max_features must be")
    check(lambda: Workbench(leaf_regularization=-1), "leaf_regularization must be")
    check(lambda: Workbench(feature_sampling="nope"), "feature_sampling must be")
    check(lambda: FastForest(workbench={}), "workbench must be")
    check(lambda: FastForest(max_dummy_cardinality=0).fit(X, y), "max_dummy_cardinality must be a positive integer")
    check(lambda: FastForest(cutoff_divisor=np.nan).fit(X, y), "cutoff_divisor must be finite and greater than zero")
    check(lambda: FastForest().predict(X), "must be fitted")
    check(lambda: FastForest().explain(X), "must be fitted")
