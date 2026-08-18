use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fastforest::{ClassifierForest, Config, Encoder, FitPlan, Forest, MaxFeatures, SavedValue, plan_fit};
use ndarray::Array1;

fn numeric_batch(rows: usize, cols: usize) -> RecordBatch {
    let fields = (0..cols).map(|col| Field::new(format!("x{col}"), DataType::Float32, false)).collect::<Vec<_>>();
    let arrays = (0..cols)
        .map(|col| {
            Arc::new(Float32Array::from_iter_values((0..rows).map(|row| {
                let value = ((row * 17 + col * 31) % 101) as f32 / 100.;
                if col == cols - 1 { row as f32 / rows as f32 } else { value }
            }))) as ArrayRef
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn encoded(batch: &RecordBatch) -> (Encoder, ndarray::Array2<u32>, ndarray::Array2<f32>, Vec<SavedValue>) {
    let markers = (0..batch.num_columns()).map(|_| SavedValue { kind: 5, value: String::new() }).collect::<Vec<_>>();
    let (encoder, ranked) = Encoder::fit_arrow(batch, &markers, false, vec![], Some(42)).unwrap();
    let native = encoder.transform_arrow(batch, &markers).unwrap();
    (encoder, ranked, native, markers)
}

fn same_floats(left: &[f32], right: &[f32]) -> bool {
    left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[test]
fn regression_and_classification_behaviour_story() {
    let batch = numeric_batch(240, 5);
    let (encoder, x, native, _) = encoded(&batch);
    let y = Array1::from_iter((0..x.nrows()).map(|row| 4. * native[[row, 0]] - 2. * native[[row, 1]] + native[[row, 4]]));
    let config = Config { n_trees: 24, min_node_size: 8, max_node_samples: 80, seed: Some(42), oob: true, ..Config::default() };
    let fit = |config: &Config| {
        Forest::fit(x.view(), y.view(), encoder.cutoff_values(), encoder.cutoff_offsets(), Some(encoder.feature_group_ids()), config)
            .unwrap()
    };
    let forest = fit(&config);
    let predictions = forest.predict(native.view()).unwrap();
    let baseline = y.iter().sum::<f32>() / y.len() as f32;
    let mse = predictions.iter().zip(&y).map(|(prediction, target)| (prediction - target).powi(2)).sum::<f32>() / y.len() as f32;
    let baseline_mse = y.iter().map(|target| (target - baseline).powi(2)).sum::<f32>() / y.len() as f32;
    assert!(mse < baseline_mse * 0.3 && (forest.feature_importances().iter().sum::<f32>() - 1.).abs() < 1e-5);
    assert!(same_floats(&predictions, &fit(&config).predict(native.view()).unwrap()));
    assert!(forest.oob_counts().unwrap().iter().filter(|&&count| count > 0).count() > x.nrows() * 99 / 100);

    let trees = forest.predict_trees(native.view()).unwrap();
    for (row, prediction) in trees.chunks_exact(config.n_trees).zip(&predictions) {
        assert!((row.iter().sum::<f32>() / config.n_trees as f32 - prediction).abs() < 1e-5);
    }
    let (explained, bias, contributions) = forest.explain(native.view()).unwrap();
    for ((prediction, explained), parts) in predictions.iter().zip(explained).zip(contributions.chunks_exact(x.ncols())) {
        assert!((prediction - explained).abs() < 1e-5 && (explained - bias - parts.iter().sum::<f32>()).abs() < 1e-4);
    }

    let configs = [Config { n_trees: 8, ..config.clone() }, Config { n_trees: 8, min_node_size: 16, ..config.clone() }];
    let standalone = configs.iter().map(|config| fit(config)).collect::<Vec<_>>();
    let batched = Forest::fit_batch(
        x.view(),
        y.view(),
        encoder.cutoff_values(),
        encoder.cutoff_offsets(),
        Some(encoder.feature_group_ids()),
        &configs,
        None,
    )
    .unwrap();
    for (standalone, batched) in standalone.iter().zip(&batched) {
        assert!(same_floats(&standalone.predict(native.view()).unwrap(), &batched.predict(native.view()).unwrap()));
        assert_eq!(standalone.oob_counts(), batched.oob_counts());
        assert!(same_floats(standalone.oob_prediction().unwrap(), batched.oob_prediction().unwrap()));
    }
    let reversed = Forest::fit_batch(
        x.view(),
        y.view(),
        encoder.cutoff_values(),
        encoder.cutoff_offsets(),
        Some(encoder.feature_group_ids()),
        &[configs[1].clone(), configs[0].clone()],
        None,
    )
    .unwrap();
    assert!(same_floats(&batched[0].predict(native.view()).unwrap(), &reversed[1].predict(native.view()).unwrap()));

    let classes = Array1::from_iter((0..x.nrows()).map(|row| ((x[[row, 0]] + 2 * x[[row, 1]]) % 3) as u32));
    let classifier = ClassifierForest::fit(
        x.view(),
        classes.view(),
        3,
        encoder.cutoff_values(),
        encoder.cutoff_offsets(),
        Some(encoder.feature_group_ids()),
        &config,
    )
    .unwrap();
    let predicted = classifier.predict(native.view()).unwrap();
    let probabilities = classifier.predict_proba(native.view()).unwrap();
    assert!(predicted.iter().zip(&classes).filter(|(a, b)| a == b).count() > x.nrows() * 9 / 10);
    assert!(probabilities.chunks_exact(3).all(|row| (row.iter().sum::<f32>() - 1.).abs() < 1e-5));
    assert!(
        classifier
            .oob_decision()
            .unwrap()
            .chunks_exact(3)
            .zip(classifier.oob_counts().unwrap())
            .all(|(row, &count)| count == 0 || (row.iter().sum::<f32>() - 1.).abs() < 1e-5)
    );

    let categorical = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("category", DataType::Float32, false)])),
        vec![Arc::new(Float32Array::from_iter_values((0..120).map(|row| (row % 4) as f32))) as ArrayRef],
    )
    .unwrap();
    let (encoder, categorical_x, native, _) = encoded(&categorical);
    let categorical_y = Array1::from_iter((0..120).map(|row| if row % 4 == 2 { 10. } else { 0. }));
    let equality = Forest::fit(
        categorical_x.view(),
        categorical_y.view(),
        encoder.cutoff_values(),
        encoder.cutoff_offsets(),
        Some(encoder.feature_group_ids()),
        &Config {
            n_trees: 1,
            bootstrap_fraction: Some(1.),
            bootstrap_max: None,
            max_node_samples: 120,
            min_node_size: 2,
            max_features: MaxFeatures::Fraction(1.),
            seed: Some(42),
            ..Config::default()
        },
    )
    .unwrap();
    assert_eq!(equality.predict(native.view()).unwrap(), categorical_y.to_vec());
    assert_eq!(equality.tree_structures(), vec![(3, 2, 1)]);

    assert_eq!(
        plan_fit(1_000_000, Some(20), None, Some(20_000), false, false, 1).unwrap(),
        FitPlan { n_trees: 20, rows_per_tree: 20_000, pool_rows: 252_000 }
    );
    assert_eq!(plan_fit(1_000_000, None, None, Some(40_000), false, false, 1).unwrap().n_trees, 50);
    assert_eq!(plan_fit(1_000_000, None, None, Some(240_000), false, false, 1).unwrap().n_trees, 32);
    assert_eq!(plan_fit(10_000, None, None, Some(40_000), false, false, 1).unwrap().n_trees, 64);
}
