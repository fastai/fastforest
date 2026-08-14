import numpy as np, pytest

from fastforest import FastForest,feature_dependence,feature_relations

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
    assert model.oob_prediction_.shape == y.shape
    assert model.oob_counts_.shape == y.shape
    assert np.mean(model.oob_counts_ > 0) >= 0.99
    assert np.array_equal(predictions, FastForest(n_trees=24, min_node_size=8,
        replacement=False, max_node_samples=80, seed=99).fit(X, y).predict(X))

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

def test_validation_errors():
    X,y = np.ones((4, 2)),np.ones(4)
    def check(call, message):
        with pytest.raises((ValueError, RuntimeError), match=message): call()
    check(lambda: FastForest().fit(X[:, 0], y), "X must be a two-dimensional array")
    check(lambda: FastForest().fit(X, y[:, None]), "y must be a one-dimensional array")
    check(lambda: FastForest().fit(X, y[:-1]), "X has 4 rows but y has 3 values")
    check(lambda: FastForest().fit(np.array([[1, np.nan]]), [1]), "features must all be finite")
    check(lambda: FastForest(bootstrap_max=0).fit(X, y), "bootstrap_max must be greater than zero")
    check(lambda: FastForest(cutoff_divisor=np.nan).fit(X, y), "cutoff_divisor must be finite and greater than zero")
    check(lambda: FastForest().predict(X), "must be fitted")
    check(lambda: FastForest().explain(X), "must be fitted")
