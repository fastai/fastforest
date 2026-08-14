use ndarray::{ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::fmt;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_trees: usize,
    pub min_node_size: usize,
    pub bootstrap_fraction: f32,
    pub bootstrap_max: Option<usize>,
    pub replacement: bool,
    pub max_node_samples: usize,
    pub cutoff_divisor: f32,
    pub seed: Option<u64>,
    pub oob: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_trees: 100,
            min_node_size: 4,
            bootstrap_fraction: 0.8,
            bootstrap_max: Some(40_000),
            replacement: false,
            max_node_samples: 160,
            cutoff_divisor: 3.0,
            seed: None,
            oob: false,
        }
    }
}

impl Config {
    fn validate(&self) -> Result<(), ForestError> {
        if self.n_trees == 0 {
            return Err(ForestError::new("n_trees must be greater than zero"));
        }
        if self.min_node_size < 2 {
            return Err(ForestError::new("min_node_size must be at least 2"));
        }
        if !(0.0 < self.bootstrap_fraction && self.bootstrap_fraction <= 1.0) {
            return Err(ForestError::new("bootstrap_fraction must be in (0, 1]"));
        }
        if self.bootstrap_max == Some(0) {
            return Err(ForestError::new("bootstrap_max must be greater than zero"));
        }
        if self.max_node_samples < 2 {
            return Err(ForestError::new("max_node_samples must be at least 2"));
        }
        if !self.cutoff_divisor.is_finite() || self.cutoff_divisor <= 0.0 {
            return Err(ForestError::new(
                "cutoff_divisor must be finite and greater than zero",
            ));
        }
        Ok(())
    }
}

const LEAF_COL: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    cut_val: f32,
    value: f32,
    child: u32,
    cut_col: u32,
}

impl Node {
    fn new() -> Self {
        Self {
            cut_val: 0.0,
            value: 0.0,
            child: 0,
            cut_col: LEAF_COL,
        }
    }

    fn is_leaf(&self) -> bool {
        self.cut_col == LEAF_COL
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict_by(&self, value: impl Fn(usize) -> f32) -> f32 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.value;
            }
            let go_right = usize::from(value(node.cut_col as usize) >= node.cut_val);
            node_idx = node.child as usize + go_right;
        }
    }

    fn explain_by(&self, value: impl Fn(usize) -> f32, contributions: &mut [f32]) -> f32 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.value;
            }
            let go_right = usize::from(value(node.cut_col as usize) >= node.cut_val);
            let child = &self.nodes[node.child as usize + go_right];
            contributions[node.cut_col as usize] += child.value - node.value;
            node_idx = node.child as usize + go_right;
        }
    }

    fn build(
        x: ArrayView2<'_, f32>,
        y: ArrayView1<'_, f32>,
        config: &Config,
        seed: u64,
    ) -> (Self, Option<Vec<bool>>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sample_size = ((x.nrows() as f32 * config.bootstrap_fraction) as usize).max(1);
        if let Some(max) = config.bootstrap_max {
            sample_size = sample_size.min(max);
        }
        let mut rows = if config.replacement {
            (0..sample_size)
                .map(|_| rng.random_range(0..x.nrows()))
                .collect()
        } else {
            rand::seq::index::sample(&mut rng, x.nrows(), sample_size).into_vec()
        };
        let in_bag = config.oob.then(|| {
            let mut mask = vec![false; x.nrows()];
            rows.iter().for_each(|&row| mask[row] = true);
            mask
        });

        let mut nodes = vec![Node::new()];
        let mut importance = vec![0.0; x.ncols()];
        let mut work = vec![(0, 0, rows.len())];
        while let Some((node_idx, start, n_rows)) = work.pop() {
            let split = best_cutoff(x, y, &rows, start, n_rows, config, &mut rng);
            nodes[node_idx].value = split.value;
            let (Some(cut_col), cut_val) = (split.cut_col, split.cut_val) else {
                continue;
            };
            importance[cut_col] += split.gain * n_rows as f32;

            let left_n = partition(x, &mut rows, start, n_rows, cut_col, cut_val);
            debug_assert!(left_n > 0 && left_n < n_rows);
            let left_idx = nodes.len();
            let right_idx = left_idx + 1;
            nodes.push(Node::new());
            nodes.push(Node::new());
            nodes[node_idx].child = u32::try_from(left_idx).expect("tree has too many nodes");
            nodes[node_idx].cut_col = u32::try_from(cut_col).expect("matrix has too many columns");
            nodes[node_idx].cut_val = cut_val;
            work.push((right_idx, start + left_n, n_rows - left_n));
            work.push((left_idx, start, left_n));
        }

        (Self { nodes }, in_bag, importance)
    }
}

#[derive(Clone, Debug)]
pub struct Forest {
    trees: Vec<Tree>,
    n_features: usize,
    feature_importances: Vec<f32>,
    oob_prediction: Option<Vec<f32>>,
    oob_counts: Option<Vec<u32>>,
}

impl Forest {
    pub fn fit(
        x: ArrayView2<'_, f32>,
        y: ArrayView1<'_, f32>,
        config: &Config,
    ) -> Result<Self, ForestError> {
        validate_training_data(x, y)?;
        config.validate()?;

        let mut seed_rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
        let seeds: Vec<u64> = (0..config.n_trees).map(|_| seed_rng.random()).collect();
        let built: Vec<_> = seeds
            .into_par_iter()
            .map(|seed| Tree::build(x, y, config, seed))
            .collect();

        let mut trees = Vec::with_capacity(config.n_trees);
        let (mut oob_prediction, mut oob_counts) = if config.oob {
            (Some(vec![0.0; x.nrows()]), Some(vec![0; x.nrows()]))
        } else {
            (None, None)
        };
        let mut feature_importances = vec![0.0; x.ncols()];
        for (tree, in_bag, importance) in built {
            feature_importances
                .iter_mut()
                .zip(importance)
                .for_each(|(total, value)| *total += value);
            if let Some(mask) = in_bag {
                let sums = oob_prediction.as_mut().unwrap();
                let counts = oob_counts.as_mut().unwrap();
                let data = x.as_slice();
                for row_idx in 0..x.nrows() {
                    if mask[row_idx] {
                        continue;
                    }
                    sums[row_idx] += if let Some(data) = data {
                        let start = row_idx * x.ncols();
                        tree.predict_by(|col| data[start + col])
                    } else {
                        tree.predict_by(|col| x[[row_idx, col]])
                    };
                    counts[row_idx] += 1;
                }
            }
            trees.push(tree);
        }
        let importance_sum = feature_importances.iter().sum::<f32>();
        if importance_sum > 0.0 {
            feature_importances
                .iter_mut()
                .for_each(|value| *value /= importance_sum);
        }
        if let (Some(predictions), Some(counts)) = (&mut oob_prediction, &oob_counts) {
            for (prediction, &count) in predictions.iter_mut().zip(counts) {
                *prediction = if count == 0 {
                    f32::NAN
                } else {
                    *prediction / count as f32
                };
            }
        }

        Ok(Self {
            trees,
            n_features: x.ncols(),
            feature_importances,
            oob_prediction,
            oob_counts,
        })
    }

    pub fn predict(&self, x: ArrayView2<'_, f32>) -> Result<Vec<f32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        let mut predictions = vec![0.0; x.nrows()];
        let n_trees = self.trees.len() as f32;
        if let Some(data) = x.as_slice() {
            predictions
                .par_iter_mut()
                .zip(data.par_chunks_exact(self.n_features))
                .for_each(|(prediction, row)| {
                    *prediction = self
                        .trees
                        .iter()
                        .map(|tree| tree.predict_by(|col| row[col]))
                        .sum::<f32>()
                        / n_trees;
                });
        } else {
            predictions
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_idx, prediction)| {
                    *prediction = self
                        .trees
                        .iter()
                        .map(|tree| tree.predict_by(|col| x[[row_idx, col]]))
                        .sum::<f32>()
                        / n_trees;
                });
        }
        Ok(predictions)
    }

    pub fn predict_trees(&self, x: ArrayView2<'_, f32>) -> Result<Vec<f32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        let mut predictions = vec![0.0; x.nrows() * self.trees.len()];
        if let Some(data) = x.as_slice() {
            predictions
                .par_chunks_mut(self.trees.len())
                .zip(data.par_chunks_exact(self.n_features))
                .for_each(|(predictions, row)| {
                    predictions
                        .iter_mut()
                        .zip(&self.trees)
                        .for_each(|(prediction, tree)| {
                            *prediction = tree.predict_by(|col| row[col]);
                        });
                });
        } else {
            predictions
                .par_chunks_mut(self.trees.len())
                .enumerate()
                .for_each(|(row_idx, predictions)| {
                    predictions
                        .iter_mut()
                        .zip(&self.trees)
                        .for_each(|(prediction, tree)| {
                            *prediction = tree.predict_by(|col| x[[row_idx, col]]);
                        });
                });
        }
        Ok(predictions)
    }

    pub fn explain(
        &self,
        x: ArrayView2<'_, f32>,
    ) -> Result<(Vec<f32>, f32, Vec<f32>), ForestError> {
        validate_prediction_data(x, self.n_features)?;
        let n_trees = self.trees.len() as f32;
        let bias = self
            .trees
            .iter()
            .map(|tree| tree.nodes[0].value)
            .sum::<f32>()
            / n_trees;
        let mut predictions = vec![0.0; x.nrows()];
        let mut contributions = vec![0.0; x.nrows() * self.n_features];
        predictions
            .par_iter_mut()
            .zip(contributions.par_chunks_mut(self.n_features))
            .enumerate()
            .for_each(|(row_idx, (prediction, contributions))| {
                for tree in &self.trees {
                    *prediction += tree.explain_by(|col| x[[row_idx, col]], contributions);
                }
                *prediction /= n_trees;
                contributions.iter_mut().for_each(|value| *value /= n_trees);
            });
        Ok((predictions, bias, contributions))
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn feature_importances(&self) -> &[f32] {
        &self.feature_importances
    }

    pub fn oob_prediction(&self) -> Option<&[f32]> {
        self.oob_prediction.as_deref()
    }

    pub fn oob_counts(&self) -> Option<&[u32]> {
        self.oob_counts.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForestError {
    message: String,
}

impl ForestError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ForestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ForestError {}

#[derive(Clone, Copy, Debug)]
struct Split {
    cut_col: Option<usize>,
    cut_val: f32,
    value: f32,
    gain: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Candidate {
    left_target: f32,
    left_sqr_target: f32,
    left_count: usize,
    cut_col: usize,
    cut_val: f32,
}

fn best_cutoff(
    x: ArrayView2<'_, f32>,
    y: ArrayView1<'_, f32>,
    rows: &[usize],
    start: usize,
    n_rows: usize,
    config: &Config,
    rng: &mut StdRng,
) -> Split {
    if n_rows < config.min_node_size || all_same(y, rows, start, n_rows, config.max_node_samples) {
        let sum = rows[start..start + n_rows]
            .iter()
            .map(|&row| y[row])
            .sum::<f32>();
        return Split {
            cut_col: None,
            cut_val: 0.0,
            value: sum / n_rows as f32,
            gain: 0.0,
        };
    }

    let used_n = n_rows.min(config.max_node_samples);
    let n_candidates =
        ((used_n as f64 * (x.ncols() as f64).sqrt()) / config.cutoff_divisor as f64) as usize;
    let mut candidates = vec![Candidate::default(); n_candidates.max(4)];
    for candidate in &mut candidates {
        candidate.cut_col = rng.random_range(0..x.ncols());
        let split_pos = rng.random_range(start..start + n_rows);
        candidate.cut_val = x[[rows[split_pos], candidate.cut_col]];
    }

    let used_start = if n_rows == used_n {
        start
    } else {
        start + rng.random_range(0..=n_rows - used_n)
    };
    let mut sum_target = 0.0;
    let mut sum_sqr_target = 0.0;
    for &row in &rows[used_start..used_start + used_n] {
        let target = y[row];
        sum_target += target;
        sum_sqr_target += target * target;
        for candidate in &mut candidates {
            if x[[row, candidate.cut_col]] >= candidate.cut_val {
                continue;
            }
            candidate.left_target += target;
            candidate.left_sqr_target += target * target;
            candidate.left_count += 1;
        }
    }

    let impurity = weighted_loss(0.0, 0.0, 0, sum_target, sum_sqr_target, used_n);
    let mut criterion = impurity;
    let mut best = None;
    let min_size = ((used_n as f32 * 0.05) as usize)
        .max(config.min_node_size / 3)
        .max(1);
    for candidate in candidates {
        let right_count = used_n - candidate.left_count;
        if candidate.left_count < min_size || right_count < min_size {
            continue;
        }
        let candidate_loss = weighted_loss(
            candidate.left_target,
            candidate.left_sqr_target,
            candidate.left_count,
            sum_target,
            sum_sqr_target,
            used_n,
        );
        if candidate_loss > criterion {
            criterion = candidate_loss;
            best = Some((candidate.cut_col, candidate.cut_val));
        }
    }

    Split {
        cut_col: best.map(|v| v.0),
        cut_val: best.map_or(0.0, |v| v.1),
        value: sum_target / used_n as f32,
        gain: (criterion - impurity).max(0.0),
    }
}

fn all_same(
    y: ArrayView1<'_, f32>,
    rows: &[usize],
    start: usize,
    n_rows: usize,
    max_samples: usize,
) -> bool {
    let end = start + n_rows.min(max_samples);
    let first = y[rows[start]];
    rows[start + 1..end]
        .iter()
        .all(|&row| (y[row] - first).abs() <= 1.0e-8 + 1.0e-5 * first.abs())
}

fn partition(
    x: ArrayView2<'_, f32>,
    rows: &mut [usize],
    start: usize,
    n_rows: usize,
    cut_col: usize,
    cut_val: f32,
) -> usize {
    let mut left = start;
    let mut right = start + n_rows;
    while left < right {
        if x[[rows[left], cut_col]] < cut_val {
            left += 1;
        } else {
            right -= 1;
            rows.swap(left, right);
        }
    }
    left - start
}

fn weighted_loss(
    left_target: f32,
    left_sqr_target: f32,
    left_count: usize,
    sum_target: f32,
    sum_sqr_target: f32,
    total_count: usize,
) -> f32 {
    let mut result = 0.0;
    if left_count > 0 {
        result +=
            loss(left_target, left_sqr_target, left_count) * left_count as f32 / total_count as f32;
    }
    let right_count = total_count - left_count;
    if right_count > 0 {
        result += loss(
            sum_target - left_target,
            sum_sqr_target - left_sqr_target,
            right_count,
        ) * right_count as f32
            / total_count as f32;
    }
    result
}

fn loss(sum_target: f32, sum_sqr_target: f32, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let variance =
        ((sum_sqr_target - sum_target * sum_target / n as f32) / (n - 1) as f32).max(0.0);
    -variance.sqrt()
}

fn validate_training_data(
    x: ArrayView2<'_, f32>,
    y: ArrayView1<'_, f32>,
) -> Result<(), ForestError> {
    if x.nrows() == 0 {
        return Err(ForestError::new(
            "training data must contain at least one row",
        ));
    }
    if x.ncols() == 0 {
        return Err(ForestError::new(
            "training data must contain at least one feature",
        ));
    }
    if x.nrows() != y.len() {
        return Err(ForestError::new(format!(
            "X has {} rows but y has {} values",
            x.nrows(),
            y.len()
        )));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ForestError::new("features must all be finite"));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ForestError::new("targets must all be finite"));
    }
    Ok(())
}

fn validate_prediction_data(x: ArrayView2<'_, f32>, n_features: usize) -> Result<(), ForestError> {
    if x.ncols() != n_features {
        return Err(ForestError::new(format!(
            "expected {n_features} features, got {}",
            x.ncols()
        )));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ForestError::new("features must all be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn forest_training_prediction_and_oob_story() {
        let x = Array2::from_shape_fn((240, 4), |(r, c)| {
            let v = ((r * 17 + c * 31) % 101) as f32 / 100.0;
            if c == 3 { r as f32 / 240.0 } else { v }
        });
        let y = Array1::from_iter(x.rows().into_iter().map(|r| 4.0 * r[0] - 2.0 * r[1] + r[3]));
        let config = Config {
            n_trees: 24,
            min_node_size: 8,
            max_node_samples: 80,
            seed: Some(42),
            oob: true,
            ..Config::default()
        };

        let forest = Forest::fit(x.view(), y.view(), &config).unwrap();
        let predictions = forest.predict(x.view()).unwrap();
        let baseline = y.iter().sum::<f32>() / y.len() as f32;
        let model_mse = predictions
            .iter()
            .zip(&y)
            .map(|(p, y)| (p - y).powi(2))
            .sum::<f32>()
            / y.len() as f32;
        let baseline_mse = y.iter().map(|y| (baseline - y).powi(2)).sum::<f32>() / y.len() as f32;

        assert_eq!(forest.n_trees(), config.n_trees);
        assert_eq!(forest.n_features(), x.ncols());
        assert!(model_mse < baseline_mse * 0.3);
        assert!((forest.feature_importances().iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
        assert_eq!(
            forest
                .feature_importances()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            0
        );
        assert_eq!(std::mem::size_of::<Node>(), 16);
        assert!(forest.trees.iter().all(|tree| {
            tree.nodes.iter().enumerate().all(|(idx, node)| {
                node.is_leaf()
                    || (node.child as usize > idx && node.child as usize + 1 < tree.nodes.len())
            })
        }));

        let counts = forest.oob_counts().unwrap();
        let oob = forest.oob_prediction().unwrap();
        assert_eq!(counts.len(), x.nrows());
        assert!(counts.iter().all(|&count| count <= config.n_trees as u32));
        assert!(counts.iter().filter(|&&count| count > 0).count() > x.nrows() * 99 / 100);
        assert!(
            oob.iter()
                .zip(counts)
                .all(|(prediction, &count)| prediction.is_finite() == (count > 0))
        );

        let again = Forest::fit(x.view(), y.view(), &config).unwrap();
        assert_eq!(forest.trees, again.trees);
        assert_eq!(forest.oob_counts, again.oob_counts);
        assert!(
            forest
                .oob_prediction
                .as_ref()
                .unwrap()
                .iter()
                .zip(again.oob_prediction.as_ref().unwrap())
                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
        );

        let tree_predictions = forest.predict_trees(x.view()).unwrap();
        assert_eq!(tree_predictions.len(), x.nrows() * config.n_trees);
        for (row, prediction) in tree_predictions
            .chunks_exact(config.n_trees)
            .zip(&predictions)
        {
            assert!((row.iter().sum::<f32>() / config.n_trees as f32 - prediction).abs() < 1.0e-5);
        }
        let (explained, bias, contributions) = forest.explain(x.view()).unwrap();
        for ((prediction, explained), contribution) in predictions
            .iter()
            .zip(explained)
            .zip(contributions.chunks_exact(x.ncols()))
        {
            assert!((prediction - explained).abs() < 1.0e-5);
            assert!((explained - bias - contribution.iter().sum::<f32>()).abs() < 1.0e-4);
        }

        let no_replacement = Forest::fit(
            x.view(),
            y.view(),
            &Config {
                bootstrap_fraction: 1.0,
                replacement: false,
                ..config
            },
        )
        .unwrap();
        assert!(
            no_replacement
                .oob_counts()
                .unwrap()
                .iter()
                .all(|&count| count == 0)
        );
        assert!(
            no_replacement
                .oob_prediction()
                .unwrap()
                .iter()
                .all(|prediction| prediction.is_nan())
        );

        let capped = Forest::fit(
            x.view(),
            y.view(),
            &Config {
                bootstrap_fraction: 1.0,
                bootstrap_max: Some(40),
                replacement: false,
                ..config
            },
        )
        .unwrap();
        let total_oob: u32 = capped.oob_counts().unwrap().iter().sum();
        assert_eq!(
            total_oob,
            (x.nrows() * config.n_trees - 40 * config.n_trees) as u32
        );
    }

    #[test]
    fn validation_and_numerical_edges() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0];
        assert_eq!(
            Forest::fit(x.view(), y.view(), &Config::default())
                .unwrap_err()
                .to_string(),
            "X has 2 rows but y has 1 values"
        );

        let bad = array![[1.0, f32::NAN]];
        assert_eq!(
            Forest::fit(bad.view(), array![1.0].view(), &Config::default())
                .unwrap_err()
                .to_string(),
            "features must all be finite"
        );

        let invalid = Config {
            bootstrap_max: Some(0),
            ..Config::default()
        };
        assert_eq!(
            Forest::fit(x.view(), array![1.0, 2.0].view(), &invalid)
                .unwrap_err()
                .to_string(),
            "bootstrap_max must be greater than zero"
        );

        assert_eq!(weighted_loss(0.0, 0.0, 0, 3.0, 5.0, 2), loss(3.0, 5.0, 2));
        assert_eq!(loss(2.0, 1.999_999_9, 2), 0.0);
    }
}
