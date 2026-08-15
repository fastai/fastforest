use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;

use crate::Workbench;
use crate::split::{NodeRows, SplitScratch, find_split, partition};

#[derive(Clone, Debug)]
pub struct Config {
    pub n_trees: usize,
    pub min_node_size: usize,
    pub bootstrap_fraction: Option<f32>,
    pub bootstrap_max: Option<usize>,
    pub replacement: bool,
    pub max_node_samples: usize,
    pub min_candidate_rows: usize,
    pub candidate_attempt_factor: usize,
    pub cutoff_divisor: f32,
    pub seed: Option<u64>,
    pub oob: bool,
    pub adaptive: bool,
    pub workbench: Workbench,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_trees: 50,
            min_node_size: 4,
            bootstrap_fraction: None,
            bootstrap_max: Some(40_000),
            replacement: false,
            max_node_samples: 320,
            min_candidate_rows: 20,
            candidate_attempt_factor: 2,
            cutoff_divisor: 3.0,
            seed: None,
            oob: false,
            adaptive: true,
            workbench: Workbench::default(),
        }
    }
}

impl Config {
    pub(crate) fn resolved_bootstrap_fraction(&self) -> f32 {
        self.bootstrap_fraction
            .unwrap_or(if self.oob { 0.8 } else { 1.0 })
    }

    pub(crate) fn validate(&self) -> Result<(), ForestError> {
        if self.n_trees == 0 {
            return Err(ForestError::new("n_trees must be greater than zero"));
        }
        if self.min_node_size < 2 {
            return Err(ForestError::new("min_node_size must be at least 2"));
        }
        let bootstrap_fraction = self.resolved_bootstrap_fraction();
        if !bootstrap_fraction.is_finite() || bootstrap_fraction <= 0.0 {
            return Err(ForestError::new(
                "bootstrap_fraction must be finite and greater than zero",
            ));
        }
        if !self.replacement && bootstrap_fraction > 1.0 {
            return Err(ForestError::new(
                "bootstrap_fraction cannot exceed 1 without replacement",
            ));
        }
        if self.bootstrap_max == Some(0) {
            return Err(ForestError::new("bootstrap_max must be greater than zero"));
        }
        if self.max_node_samples < 2 {
            return Err(ForestError::new("max_node_samples must be at least 2"));
        }
        if self.candidate_attempt_factor == 0 {
            return Err(ForestError::new(
                "candidate_attempt_factor must be greater than zero",
            ));
        }
        if !self.cutoff_divisor.is_finite() || self.cutoff_divisor <= 0.0 {
            return Err(ForestError::new(
                "cutoff_divisor must be finite and greater than zero",
            ));
        }
        self.workbench.validate()?;
        Ok(())
    }
}

pub(crate) const LEAF_COL: u32 = u32::MAX;
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    cut_val: f32,
    value: f32,
    child: u32,
    cut_col: u32,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.cut_col == LEAF_COL
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrainingNode {
    cut_val: u32,
    value: f32,
    child: u32,
    cut_col: u32,
}

impl TrainingNode {
    fn new() -> Self {
        Self {
            cut_val: 0,
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
            let go_right = usize::from(value(node.cut_col as usize) > node.cut_val);
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
            let go_right = usize::from(value(node.cut_col as usize) > node.cut_val);
            let child = &self.nodes[node.child as usize + go_right];
            contributions[node.cut_col as usize] += child.value - node.value;
            node_idx = node.child as usize + go_right;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TrainingTree {
    nodes: Vec<TrainingNode>,
}

impl TrainingTree {
    fn predict_by(&self, value: impl Fn(usize) -> u32) -> f32 {
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

    fn into_native(self, cutoff_values: &[f32], cutoff_offsets: &[usize]) -> Tree {
        let nodes = self
            .nodes
            .into_iter()
            .map(|node| Node {
                cut_val: if node.is_leaf() {
                    0.0
                } else {
                    cutoff_values[cutoff_offsets[node.cut_col as usize] + node.cut_val as usize]
                },
                value: node.value,
                child: node.child,
                cut_col: node.cut_col,
            })
            .collect();
        Tree { nodes }
    }

    fn build(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, f32>,
        config: &Config,
        feature_groups: Option<&[Vec<usize>]>,
        seed: u64,
    ) -> (Self, Option<Vec<bool>>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = sample_rows(x.nrows(), config, &mut rng);
        let in_bag = config.oob.then(|| {
            let mut mask = vec![false; x.nrows()];
            rows.iter().for_each(|&row| mask[row as usize] = true);
            mask
        });

        let mut nodes = vec![TrainingNode::new()];
        let mut importance = vec![0.0; x.ncols()];
        let root_mean = rows.iter().map(|&row| y[row as usize]).sum::<f32>() / rows.len() as f32;
        let mut split_scratch = SplitScratch::default();
        let mut work = vec![(0, 0, rows.len(), root_mean)];
        while let Some((node_idx, start, n_rows, parent_mean)) = work.pop() {
            let split = find_split(
                x,
                y,
                NodeRows {
                    rows: &rows,
                    start,
                    n_rows,
                },
                config,
                feature_groups,
                &mut rng,
                &mut split_scratch,
            );
            nodes[node_idx].value = if split.cut_col.is_none() {
                config
                    .workbench
                    .leaf_value(split.value, n_rows, parent_mean)
            } else {
                split.value
            };
            let (Some(cut_col), cut_val) = (split.cut_col, split.cut_val) else {
                continue;
            };
            importance[cut_col] += split.gain * n_rows as f32;

            let left_n = partition(x, &mut rows, start, n_rows, cut_col, cut_val);
            debug_assert!(left_n > 0 && left_n < n_rows);
            let left_idx = nodes.len();
            let right_idx = left_idx + 1;
            nodes.push(TrainingNode::new());
            nodes.push(TrainingNode::new());
            nodes[node_idx].child = u32::try_from(left_idx).expect("tree has too many nodes");
            nodes[node_idx].cut_col = u32::try_from(cut_col).expect("matrix has too many columns");
            nodes[node_idx].cut_val = cut_val;
            work.push((right_idx, start + left_n, n_rows - left_n, split.value));
            work.push((left_idx, start, left_n, split.value));
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
    adaptive_scores: Vec<AdaptiveScore>,
    adaptive_choice: Option<AdaptiveScore>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveScore {
    pub max_features: f32,
    pub max_node_samples: usize,
    pub oob_mse: f64,
}

impl Forest {
    pub fn fit(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, f32>,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>,
        config: &Config,
    ) -> Result<Self, ForestError> {
        validate_training_data(x, y, cutoff_values, cutoff_offsets)?;
        config.validate()?;
        let feature_groups = match feature_group_ids {
            Some(ids) => Some(group_features(x.ncols(), ids)?),
            None if config.workbench.feature_sampling == crate::FeatureSampling::Columns => {
                return Err(ForestError::new(
                    "column feature sampling requires feature group ids",
                ));
            }
            None => None,
        };

        if config.adaptive
            && config.workbench.splitter == crate::Splitter::Histogram
            && x.nrows() > 8_000
            && x.ncols() > 0
        {
            return Self::fit_adaptive(
                x,
                y,
                cutoff_values,
                cutoff_offsets,
                feature_groups.as_deref(),
                config,
            );
        }
        Self::fit_fixed(
            x,
            y,
            cutoff_values,
            cutoff_offsets,
            feature_groups.as_deref(),
            config,
        )
    }

    fn fit_fixed(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, f32>,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_groups: Option<&[Vec<usize>]>,
        config: &Config,
    ) -> Result<Self, ForestError> {
        let mut seed_rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
        let seeds: Vec<u64> = (0..config.n_trees).map(|_| seed_rng.random()).collect();
        let built: Vec<_> = seeds
            .into_par_iter()
            .map(|seed| TrainingTree::build(x, y, config, feature_groups.as_deref(), seed))
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
            trees.push(tree.into_native(cutoff_values, cutoff_offsets));
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
            adaptive_scores: Vec::new(),
            adaptive_choice: None,
        })
    }

    fn fit_adaptive(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, f32>,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_groups: Option<&[Vec<usize>]>,
        config: &Config,
    ) -> Result<Self, ForestError> {
        const PILOT_ROWS: usize = 8_000;
        const CANDIDATES: [f32; 2] = [0.6, 0.9];

        let seed = config.seed.unwrap_or_else(rand::random);
        let mut sample_rng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
        let pilot_indexes =
            rand::seq::index::sample(&mut sample_rng, x.nrows(), PILOT_ROWS).into_vec();
        let pilot_x = Array2::from_shape_fn((PILOT_ROWS, x.ncols()), |(row, col)| {
            x[[pilot_indexes[row], col]]
        });
        let pilot_y = Array1::from_shape_fn(PILOT_ROWS, |row| y[pilot_indexes[row]]);
        let pilot_trees = rayon::current_num_threads().saturating_mul(2).max(32);
        let mut scores = Vec::with_capacity(CANDIDATES.len());
        for max_features in CANDIDATES {
            let mut pilot_config = config.clone();
            pilot_config.n_trees = pilot_trees;
            pilot_config.bootstrap_fraction = Some(0.5);
            pilot_config.bootstrap_max = None;
            pilot_config.replacement = false;
            pilot_config.seed = Some(seed);
            pilot_config.oob = true;
            pilot_config.adaptive = false;
            pilot_config.workbench.max_features = crate::MaxFeatures::Fraction(max_features);
            let forest = Self::fit_fixed(
                pilot_x.view(),
                pilot_y.view(),
                cutoff_values,
                cutoff_offsets,
                feature_groups,
                &pilot_config,
            )?;
            let predictions = forest.oob_prediction.as_ref().unwrap();
            let counts = forest.oob_counts.as_ref().unwrap();
            let (squared_error, rows) = predictions
                .iter()
                .zip(counts)
                .zip(&pilot_y)
                .filter(|((_, count), _)| **count > 0)
                .fold(
                    (0.0, 0_usize),
                    |(error, rows), ((prediction, _), target)| {
                        let residual = f64::from(*prediction) - f64::from(*target);
                        (error + residual * residual, rows + 1)
                    },
                );
            scores.push(AdaptiveScore {
                max_features,
                max_node_samples: config.max_node_samples,
                oob_mse: squared_error / rows as f64,
            });
        }
        let choice = scores
            .iter()
            .copied()
            .min_by(|left, right| left.oob_mse.total_cmp(&right.oob_mse))
            .unwrap();
        let mut selected = config.clone();
        selected.seed = Some(seed);
        selected.adaptive = false;
        selected.workbench.max_features = crate::MaxFeatures::Fraction(choice.max_features);
        let mut forest = Self::fit_fixed(
            x,
            y,
            cutoff_values,
            cutoff_offsets,
            feature_groups,
            &selected,
        )?;
        forest.adaptive_scores = scores;
        forest.adaptive_choice = Some(choice);
        Ok(forest)
    }

    pub fn predict(&self, x: ArrayView2<'_, f32>) -> Result<Vec<f32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        if self.n_features == 0 {
            let prediction = self
                .trees
                .iter()
                .map(|tree| tree.nodes[0].value)
                .sum::<f32>()
                / self.trees.len() as f32;
            return Ok(vec![prediction; x.nrows()]);
        }
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
        if self.n_features == 0 {
            return Ok((0..x.nrows())
                .flat_map(|_| self.trees.iter().map(|tree| tree.nodes[0].value))
                .collect());
        }
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
        if self.n_features == 0 {
            return Ok((vec![bias; x.nrows()], bias, Vec::new()));
        }
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

    pub fn adaptive_scores(&self) -> &[AdaptiveScore] {
        &self.adaptive_scores
    }

    pub fn adaptive_choice(&self) -> Option<AdaptiveScore> {
        self.adaptive_choice
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForestError {
    message: String,
}

impl ForestError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
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

pub(crate) fn group_features(
    n_features: usize,
    feature_group_ids: &[usize],
) -> Result<Vec<Vec<usize>>, ForestError> {
    if feature_group_ids.len() != n_features {
        return Err(ForestError::new(format!(
            "expected {n_features} feature group ids, got {}",
            feature_group_ids.len()
        )));
    }
    let mut indexes = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (feature, &id) in feature_group_ids.iter().enumerate() {
        let next = indexes.len();
        let group = *indexes.entry(id).or_insert(next);
        if group == groups.len() {
            groups.push(Vec::new());
        }
        groups[group].push(feature);
    }
    Ok(groups)
}

pub(crate) fn validate_training_data(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, f32>,
    cutoff_values: &[f32],
    cutoff_offsets: &[usize],
) -> Result<(), ForestError> {
    validate_encoded_data(x, y.len(), cutoff_values, cutoff_offsets)?;
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ForestError::new("targets must all be finite"));
    }
    Ok(())
}

pub(crate) fn validate_encoded_data(
    x: ArrayView2<'_, u32>,
    y_len: usize,
    cutoff_values: &[f32],
    cutoff_offsets: &[usize],
) -> Result<(), ForestError> {
    if x.nrows() == 0 {
        return Err(ForestError::new(
            "training data must contain at least one row",
        ));
    }
    if x.nrows() > u32::MAX as usize {
        return Err(ForestError::new("training data cannot exceed 2^32-1 rows"));
    }
    if x.nrows() != y_len {
        return Err(ForestError::new(format!(
            "X has {} rows but y has {} values",
            x.nrows(),
            y_len
        )));
    }
    if cutoff_offsets.len() != x.ncols() + 1
        || cutoff_offsets.first() != Some(&0)
        || cutoff_offsets.last() != Some(&cutoff_values.len())
        || cutoff_offsets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(ForestError::new("invalid native cutoff offsets"));
    }
    if cutoff_values.iter().any(|value| !value.is_finite()) {
        return Err(ForestError::new("native cutoff values must all be finite"));
    }
    for col in 0..x.ncols() {
        let cardinality = cutoff_offsets[col + 1] - cutoff_offsets[col];
        if x.column(col)
            .iter()
            .any(|&value| value as usize >= cardinality)
        {
            return Err(ForestError::new(format!(
                "encoded feature {col} contains a value outside its cutoff mapping"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_prediction_data(
    x: ArrayView2<'_, f32>,
    n_features: usize,
) -> Result<(), ForestError> {
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

pub(crate) fn sample_rows(n_rows: usize, config: &Config, rng: &mut StdRng) -> Vec<u32> {
    let mut sample_size = ((n_rows as f32 * config.resolved_bootstrap_fraction()) as usize).max(1);
    if let Some(max) = config.bootstrap_max {
        sample_size = sample_size.min(max);
    }
    if config.replacement {
        (0..sample_size)
            .map(|_| u32::try_from(rng.random_range(0..n_rows)).unwrap())
            .collect()
    } else {
        rand::seq::index::sample(rng, n_rows, sample_size)
            .into_vec()
            .into_iter()
            .map(|row| u32::try_from(row).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::split::{test_loss as loss, test_weighted_loss as weighted_loss};
    use ndarray::{Array1, Array2, array};

    fn encode(x: &Array2<f32>) -> (Array2<u32>, Vec<f32>, Vec<usize>) {
        let mut ranked = Array2::zeros(x.raw_dim());
        let mut cutoffs = Vec::new();
        let mut offsets = vec![0];
        for col in 0..x.ncols() {
            let mut values = x.column(col).to_vec();
            values.sort_by(f32::total_cmp);
            values.dedup();
            for row in 0..x.nrows() {
                ranked[[row, col]] = values
                    .binary_search_by(|value| value.total_cmp(&x[[row, col]]))
                    .unwrap() as u32;
            }
            cutoffs.push(values[0]);
            cutoffs.extend_from_slice(&values[..values.len() - 1]);
            offsets.push(cutoffs.len());
        }
        (ranked, cutoffs, offsets)
    }

    fn fit(x: &Array2<f32>, y: &Array1<f32>, config: &Config) -> Result<Forest, ForestError> {
        let (ranked, cutoffs, offsets) = encode(x);
        Forest::fit(ranked.view(), y.view(), &cutoffs, &offsets, None, config)
    }

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

        let forest = fit(&x, &y, &config).unwrap();
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
        let most_important = forest
            .feature_importances()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert!([0, 1].contains(&most_important));
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

        let again = fit(&x, &y, &config).unwrap();
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

        let alternate = fit(
            &x,
            &y,
            &Config {
                oob: false,
                workbench: Workbench {
                    splitter: crate::Splitter::Random,
                    max_features: crate::MaxFeatures::Sqrt,
                    leaf_regularization: 0.0,
                    ..Workbench::default()
                },
                ..config.clone()
            },
        )
        .unwrap();
        let alternate_predictions = alternate.predict(x.view()).unwrap();
        let alternate_mse = alternate_predictions
            .iter()
            .zip(&y)
            .map(|(prediction, target)| (prediction - target).powi(2))
            .sum::<f32>()
            / y.len() as f32;
        assert!(alternate_mse < baseline_mse * 0.3);
        assert_ne!(alternate_predictions, predictions);

        let no_replacement = fit(
            &x,
            &y,
            &Config {
                bootstrap_fraction: Some(1.0),
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

        let capped = fit(
            &x,
            &y,
            &Config {
                bootstrap_fraction: Some(1.0),
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
        assert_eq!(
            group_features(5, &[3, 3, 8, 3, 9]).unwrap(),
            vec![vec![0, 1, 3], vec![2], vec![4]]
        );
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0];
        assert_eq!(
            fit(&x, &y, &Config::default()).unwrap_err().to_string(),
            "X has 2 rows but y has 1 values"
        );

        let bad = array![[0_u32, 1]];
        assert_eq!(
            Forest::fit(
                bad.view(),
                array![1.0].view(),
                &[0.0, 0.0],
                &[0, 1, 2],
                None,
                &Config::default()
            )
            .unwrap_err()
            .to_string(),
            "encoded feature 1 contains a value outside its cutoff mapping"
        );

        let invalid = Config {
            bootstrap_max: Some(0),
            ..Config::default()
        };
        assert_eq!(
            fit(&x, &array![1.0, 2.0], &invalid)
                .unwrap_err()
                .to_string(),
            "bootstrap_max must be greater than zero"
        );

        let invalid = Config {
            workbench: Workbench {
                leaf_regularization: f32::NAN,
                ..Workbench::default()
            },
            ..Config::default()
        };
        assert_eq!(
            fit(&x, &array![1.0, 2.0], &invalid)
                .unwrap_err()
                .to_string(),
            "leaf_regularization must be finite and non-negative"
        );

        let invalid = Config {
            bootstrap_fraction: Some(1.1),
            ..Config::default()
        };
        assert_eq!(
            fit(&x, &array![1.0, 2.0], &invalid)
                .unwrap_err()
                .to_string(),
            "bootstrap_fraction cannot exceed 1 without replacement"
        );
        fit(
            &x,
            &array![1.0, 2.0],
            &Config {
                bootstrap_fraction: Some(1.1),
                replacement: true,
                ..Config::default()
            },
        )
        .unwrap();

        let ranked_tree = TrainingTree {
            nodes: vec![
                TrainingNode {
                    cut_val: 1,
                    value: 0.0,
                    child: 1,
                    cut_col: 0,
                },
                TrainingNode {
                    cut_val: 0,
                    value: -1.0,
                    child: 0,
                    cut_col: LEAF_COL,
                },
                TrainingNode {
                    cut_val: 0,
                    value: 1.0,
                    child: 0,
                    cut_col: LEAF_COL,
                },
            ],
        };
        let native_tree = ranked_tree.into_native(&[10.0, 10.0], &[0, 2]);
        assert_eq!(native_tree.predict_by(|_| 10.0), -1.0);
        assert_eq!(native_tree.predict_by(|_| 15.0), 1.0);

        assert_eq!(weighted_loss(0.0, 0.0, 0, 3.0, 5.0, 2), loss(3.0, 5.0, 2));
        assert_eq!(loss(2.0, 1.999_999_9, 2), 0.0);

        let x = Array2::zeros((10, 1));
        let y = Array1::from_iter((0..10).map(|value| value as f32));
        let rows = (0..10).collect::<Vec<u32>>();
        let mut rng = StdRng::seed_from_u64(42);
        let split = find_split(
            x.view(),
            y.view(),
            NodeRows {
                rows: &rows,
                start: 0,
                n_rows: rows.len(),
            },
            &Config {
                max_node_samples: 2,
                ..Config::default()
            },
            None,
            &mut rng,
            &mut SplitScratch::default(),
        );
        assert!(split.cut_col.is_none());
        assert_eq!(split.value, 4.5);
    }
}
