use ndarray::{ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::ensemble::{assemble_forest, combined_importance, combined_oob, tree_seeds};
use crate::prediction::{PredictionTree, predict_outputs};
use crate::split::{FeatureGroup, SplitScratch, find_split};
use crate::tree::{Branch, TreeNode, grow_tree, leaf_index, native_node, structure};

pub const DEFAULT_MAX_DUMMY_CARDINALITY: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxFeatures {
    Sqrt,
    Fraction(f32),
}

impl MaxFeatures {
    pub(crate) fn resolve(self, total: usize) -> usize {
        let selected = match self {
            Self::Sqrt => (total as f64).sqrt() as usize,
            Self::Fraction(fraction) => (total as f32 * fraction) as usize,
        };
        selected.clamp(1, total)
    }

    fn validate(self) -> Result<(), ForestError> {
        if let Self::Fraction(fraction) = self
            && !(fraction.is_finite() && 0.0 < fraction && fraction <= 1.0)
        {
            return Err(ForestError::new("max_features fraction must be finite and in (0, 1]"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub n_trees: usize,
    pub min_node_size: usize,
    pub bootstrap_fraction: Option<f32>,
    pub bootstrap_max: Option<usize>,
    pub sample_rows: Option<usize>,
    pub replacement: bool,
    pub max_node_samples: usize,
    pub split_prior_rows: f32,
    pub class_weight_power: f32,
    pub cutoff_divisor: f32,
    pub seed: Option<u64>,
    pub oob: bool,
    pub random_splitter: bool,
    pub max_features: MaxFeatures,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_trees: 50,
            min_node_size: 8,
            bootstrap_fraction: None,
            bootstrap_max: Some(40_000),
            sample_rows: None,
            replacement: false,
            max_node_samples: 320,
            split_prior_rows: 3.0,
            class_weight_power: 0.75,
            cutoff_divisor: 10.0,
            seed: None,
            oob: false,
            random_splitter: false,
            max_features: MaxFeatures::Fraction(0.9),
        }
    }
}

impl Config {
    pub fn classification() -> Self {
        Self { max_features: MaxFeatures::Fraction(0.6), ..Self::default() }
    }

    pub(crate) fn resolved_bootstrap_fraction(&self) -> f32 {
        self.bootstrap_fraction.unwrap_or(if self.oob { 0.8 } else { 1.0 })
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
            return Err(ForestError::new("bootstrap_fraction must be finite and greater than zero"));
        }
        if !self.replacement && bootstrap_fraction > 1.0 {
            return Err(ForestError::new("bootstrap_fraction cannot exceed 1 without replacement"));
        }
        if self.bootstrap_max == Some(0) {
            return Err(ForestError::new("bootstrap_max must be greater than zero"));
        }
        if self.sample_rows == Some(0) {
            return Err(ForestError::new("sample_rows must be greater than zero"));
        }
        if self.max_node_samples < 2 {
            return Err(ForestError::new("max_node_samples must be at least 2"));
        }
        if !self.split_prior_rows.is_finite() || self.split_prior_rows < 0.0 {
            return Err(ForestError::new("split_prior_rows must be finite and non-negative"));
        }
        if !self.class_weight_power.is_finite() || !(0.0..=1.0).contains(&self.class_weight_power) {
            return Err(ForestError::new("class_weight_power must be between zero and one"));
        }
        if !self.cutoff_divisor.is_finite() || self.cutoff_divisor <= 0.0 {
            return Err(ForestError::new("cutoff_divisor must be finite and greater than zero"));
        }
        self.max_features.validate()?;
        Ok(())
    }
}

pub fn resolve_replacement(n_rows: usize, replacement: Option<bool>, classification: bool) -> bool {
    replacement.unwrap_or(n_rows < if classification { 40_000 } else { 10_000 })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitPlan {
    pub n_trees: usize,
    pub rows_per_tree: usize,
    pub pool_rows: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn plan_fit(
    n_rows: usize, n_trees: Option<usize>, bootstrap_fraction: Option<f32>, bootstrap_max: Option<usize>, replacement: bool, oob: bool,
    output_dimensions: usize,
) -> Result<FitPlan, ForestError> {
    if n_rows == 0 {
        return Err(ForestError::new("X must contain at least one row"));
    }
    if n_trees == Some(0) {
        return Err(ForestError::new("n_trees must be greater than zero"));
    }
    if output_dimensions == 0 {
        return Err(ForestError::new("output_dimensions must be greater than zero"));
    }
    let fraction = bootstrap_fraction.unwrap_or(if oob { 0.8 } else { 1.0 });
    if !fraction.is_finite() || fraction <= 0.0 {
        return Err(ForestError::new("bootstrap_fraction must be finite and greater than zero"));
    }
    if !replacement && fraction > 1.0 {
        return Err(ForestError::new("bootstrap_fraction cannot exceed 1 without replacement"));
    }
    if bootstrap_max == Some(0) {
        return Err(ForestError::new("bootstrap_max must be greater than zero"));
    }
    let mut rows_per_tree = ((n_rows as f32 * fraction) as usize).max(1);
    if let Some(max) = bootstrap_max {
        rows_per_tree = rows_per_tree.min(max.saturating_mul(output_dimensions));
    }
    let n_trees = n_trees.unwrap_or_else(|| 2_000_000_usize.div_ceil(rows_per_tree).clamp(32, 64));
    let pool_rows = n_rows.min(n_trees.saturating_mul(rows_per_tree).saturating_mul(63).div_ceil(100));
    Ok(FitPlan { n_trees, rows_per_tree, pool_rows })
}

pub(crate) fn uniform_sample_indices(n_rows: usize, sample_rows: usize, seed: Option<u64>, stream: u64) -> Vec<usize> {
    if sample_rows >= n_rows {
        return (0..n_rows).collect();
    }
    let seed = seed.unwrap_or_else(rand::random) ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut rng = StdRng::seed_from_u64(seed);
    rand::seq::index::sample(&mut rng, n_rows, sample_rows).into_vec()
}

pub(crate) fn oob_rows(n_rows: usize, config: &Config, rows: Option<usize>) -> Option<Vec<usize>> {
    config.oob.then(|| {
        let sample_rows = rows.or(config.bootstrap_max).unwrap_or(n_rows).min(n_rows);
        uniform_sample_indices(n_rows, sample_rows, config.seed, 3)
    })
}

type Node = TreeNode<f32, f32>;
type TrainingNode = TreeNode<u32, f32>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn root_value(&self) -> f32 {
        self.nodes[0].value
    }

    fn predict_by(&self, value: impl Fn(usize) -> f32) -> f32 {
        self.predict_dyn(&value)
    }

    fn predict_dyn(&self, value: &dyn Fn(usize) -> f32) -> f32 {
        self.nodes[leaf_index(&self.nodes, value, |observed, cutoff| observed > cutoff)].value
    }

    fn structure(&self) -> (usize, usize, usize) {
        structure(&self.nodes)
    }

    fn explain_by(&self, value: impl Fn(usize) -> f32, contributions: &mut [f32]) -> f32 {
        self.explain_dyn(&value, contributions)
    }

    fn explain_dyn(&self, value: &dyn Fn(usize) -> f32, contributions: &mut [f32]) -> f32 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.value;
            }
            let value = value(node.feature());
            let go_right = usize::from(if node.equality() { value != node.cut_val } else { value > node.cut_val });
            let child = &self.nodes[node.child as usize + go_right];
            contributions[node.feature()] += child.value - node.value;
            node_idx = node.child as usize + go_right;
        }
    }
}

impl PredictionTree for Tree {
    fn prediction_bytes(&self) -> usize {
        std::mem::size_of_val(self.nodes.as_slice())
    }
    fn add_prediction_by(&self, value: impl Fn(usize) -> f32, output: &mut [f32]) {
        output[0] += self.predict_by(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TrainingTree {
    nodes: Vec<TrainingNode>,
}

impl TrainingTree {
    fn predict_by(&self, value: impl Fn(usize) -> u32) -> f32 {
        self.nodes[leaf_index(&self.nodes, value, |observed, cutoff| observed >= cutoff)].value
    }

    fn into_native(self, cutoff_values: &[f32], cutoff_offsets: &[usize]) -> Tree {
        let nodes = self.nodes.into_iter().map(|node| native_node(node, cutoff_values, cutoff_offsets)).collect();
        Tree { nodes }
    }

    fn build(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, cutoff_offsets: &[usize], config: &Config, feature_groups: Option<&[FeatureGroup]>,
        seed: u64, track_in_bag: bool,
    ) -> (Self, Option<Vec<bool>>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut rows, in_bag) = sampled_rows_with_mask(x.nrows(), config, &mut rng, track_in_bag);

        let mut nodes = vec![TrainingNode::new()];
        let mut importance = vec![0.0; x.ncols()];
        let mut split_scratch = SplitScratch::default();
        grow_tree(x, &mut rows, &mut nodes, &mut importance, |node, tree_node| {
            let split = find_split(x, y, node, config, cutoff_offsets, feature_groups, &mut rng, &mut split_scratch);
            tree_node.value = split.value;
            split.cut_col.map(|cut_col| Branch { cut_col, cut_val: split.cut_val, equality: split.equality, gain: split.gain })
        });

        (Self { nodes }, in_bag, importance)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Forest {
    trees: Vec<Tree>,
    n_features: usize,
    feature_importances: Vec<f32>,
    #[serde(skip)]
    oob_prediction: Option<Vec<f32>>,
    #[serde(skip)]
    oob_counts: Option<Vec<u32>>,
    #[serde(skip)]
    oob_indices: Option<Vec<usize>>,
}

impl Forest {
    pub(crate) fn validate_loaded(&self, encoded_features: usize) -> Result<(), ForestError> {
        if self.trees.is_empty() {
            return Err(ForestError::new("saved forest contains no trees"));
        }
        if self.n_features != encoded_features || self.feature_importances.len() != encoded_features {
            return Err(ForestError::new("saved forest feature dimensions are inconsistent"));
        }
        for tree in &self.trees {
            if tree.nodes.is_empty() {
                return Err(ForestError::new("saved forest contains an empty tree"));
            }
            for node in &tree.nodes {
                if !node.cut_val.is_finite() || !node.value.is_finite() {
                    return Err(ForestError::new("saved forest contains a non-finite value"));
                }
                if !node.is_leaf() && (node.feature() >= encoded_features || node.child as usize + 1 >= tree.nodes.len()) {
                    return Err(ForestError::new("saved forest contains an invalid node index"));
                }
            }
        }
        Ok(())
    }

    pub fn fit(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>, config: &Config,
    ) -> Result<Self, ForestError> {
        validate_training_data(x, y, cutoff_values, cutoff_offsets)?;
        let feature_groups = fit_groups(x.ncols(), feature_group_ids, config)?;

        Self::fit_fixed(x, y, cutoff_values, cutoff_offsets, feature_groups.as_deref(), config, None, None)
    }

    pub fn fit_on_tracking(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>, config: &Config, tracking_indices: &[usize],
    ) -> Result<Self, ForestError> {
        validate_training_data(x, y, cutoff_values, cutoff_offsets)?;
        let feature_groups = fit_groups(x.ncols(), feature_group_ids, config)?;
        validate_tracking(config, tracking_indices, x.nrows())?;
        Self::fit_fixed(x, y, cutoff_values, cutoff_offsets, feature_groups.as_deref(), config, None, Some(tracking_indices))
    }

    pub fn fit_batch(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>, configs: &[Config], oob_rows: Option<usize>,
    ) -> Result<Vec<Self>, ForestError> {
        validate_training_data(x, y, cutoff_values, cutoff_offsets)?;
        validate_batch(configs, oob_rows)?;
        let feature_groups = feature_group_ids.map(|ids| group_features(x.ncols(), ids)).transpose()?;
        configs
            .par_iter()
            .map(|config| Self::fit_fixed(x, y, cutoff_values, cutoff_offsets, feature_groups.as_deref(), config, oob_rows, None))
            .collect()
    }

    fn fit_fixed(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_groups: Option<&[FeatureGroup]>, config: &Config, oob_row_override: Option<usize>, tracking_rows: Option<&[usize]>,
    ) -> Result<Self, ForestError> {
        let built: Vec<_> = tree_seeds(config)
            .into_par_iter()
            .map(|seed| TrainingTree::build(x, y, cutoff_offsets, config, feature_groups, seed, config.oob))
            .collect();

        let oob_indices = tracking_rows.map(<[usize]>::to_vec).or_else(|| oob_rows(x.nrows(), config, oob_row_override));
        let data = x.as_slice();
        let (trees, feature_importances, oob_prediction, oob_counts, oob_indices) = assemble_forest(
            built,
            x.ncols(),
            1,
            oob_indices,
            config.oob,
            |tree, row, output| {
                output[0] += if let Some(data) = data {
                    let start = row * x.ncols();
                    tree.predict_by(|col| data[start + col])
                } else {
                    tree.predict_by(|col| x[[row, col]])
                };
            },
            |tree| tree.into_native(cutoff_values, cutoff_offsets),
        );

        Ok(Self { trees, n_features: x.ncols(), feature_importances, oob_prediction, oob_counts, oob_indices })
    }

    pub fn combined(&self, other: &Self) -> Result<Self, ForestError> {
        if self.n_features != other.n_features || self.oob_indices != other.oob_indices {
            return Err(ForestError::new("forests have incompatible feature or OOB dimensions"));
        }
        let left_trees = self.trees.len();
        let right_trees = other.trees.len();
        let mut trees = self.trees.clone();
        trees.extend(other.trees.iter().cloned());
        let feature_importances = combined_importance(&self.feature_importances, &other.feature_importances, left_trees, right_trees);
        let left = self.oob_prediction.as_deref().zip(self.oob_counts.as_deref());
        let right = other.oob_prediction.as_deref().zip(other.oob_counts.as_deref());
        let (oob_prediction, oob_counts) = combined_oob(left, right, 1)?;
        Ok(Self {
            trees,
            n_features: self.n_features,
            feature_importances,
            oob_prediction,
            oob_counts,
            oob_indices: self.oob_indices.clone(),
        })
    }

    pub fn predict(&self, x: ArrayView2<'_, f32>) -> Result<Vec<f32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        if self.n_features == 0 {
            let prediction = self.trees.iter().map(Tree::root_value).sum::<f32>() / self.trees.len() as f32;
            return Ok(vec![prediction; x.nrows()]);
        }
        Ok(predict_outputs(&self.trees, self.n_features, 1, x))
    }

    pub fn predict_trees(&self, x: ArrayView2<'_, f32>) -> Result<Vec<f32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        if self.n_features == 0 {
            return Ok((0..x.nrows()).flat_map(|_| self.trees.iter().map(Tree::root_value)).collect());
        }
        let mut predictions = vec![0.0; x.nrows() * self.trees.len()];
        if let Some(data) = x.as_slice() {
            predictions.par_chunks_mut(self.trees.len()).zip(data.par_chunks_exact(self.n_features)).for_each(|(predictions, row)| {
                predictions.iter_mut().zip(&self.trees).for_each(|(prediction, tree)| {
                    *prediction = tree.predict_by(|col| row[col]);
                });
            });
        } else {
            predictions.par_chunks_mut(self.trees.len()).enumerate().for_each(|(row_idx, predictions)| {
                predictions.iter_mut().zip(&self.trees).for_each(|(prediction, tree)| {
                    *prediction = tree.predict_by(|col| x[[row_idx, col]]);
                });
            });
        }
        Ok(predictions)
    }

    pub fn explain(&self, x: ArrayView2<'_, f32>) -> Result<(Vec<f32>, f32, Vec<f32>), ForestError> {
        validate_prediction_data(x, self.n_features)?;
        let n_trees = self.trees.len() as f32;
        let bias = self.trees.iter().map(Tree::root_value).sum::<f32>() / n_trees;
        if self.n_features == 0 {
            return Ok((vec![bias; x.nrows()], bias, Vec::new()));
        }
        let mut predictions = vec![0.0; x.nrows()];
        let mut contributions = vec![0.0; x.nrows() * self.n_features];
        predictions.par_iter_mut().zip(contributions.par_chunks_mut(self.n_features)).enumerate().for_each(
            |(row_idx, (prediction, contributions))| {
                for tree in &self.trees {
                    *prediction += tree.explain_by(|col| x[[row_idx, col]], contributions);
                }
                *prediction /= n_trees;
                contributions.iter_mut().for_each(|value| *value /= n_trees);
            },
        );
        Ok((predictions, bias, contributions))
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn tree_structures(&self) -> Vec<(usize, usize, usize)> {
        self.trees.iter().map(Tree::structure).collect()
    }

    pub fn split_counts_by_depth(&self) -> Vec<Vec<(usize, usize)>> {
        let max_depth = self.trees.iter().map(|tree| tree.structure().2).max().unwrap_or(0);
        let mut counts = vec![vec![(0, 0); max_depth + 1]; self.n_features];
        for tree in &self.trees {
            let mut stack = vec![(0, 0)];
            while let Some((index, depth)) = stack.pop() {
                let node = &tree.nodes[index];
                if node.is_leaf() {
                    continue;
                }
                let count = &mut counts[node.feature()][depth];
                count.0 += 1;
                count.1 += usize::from(node.equality());
                stack.push((node.child as usize, depth + 1));
                stack.push((node.child as usize + 1, depth + 1));
            }
        }
        counts
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

    pub fn oob_indices(&self) -> Option<&[usize]> {
        self.oob_indices.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForestError {
    message: String,
}

impl ForestError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for ForestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ForestError {}

pub(crate) fn group_features(n_features: usize, feature_group_ids: &[usize]) -> Result<Vec<FeatureGroup>, ForestError> {
    if feature_group_ids.len() != n_features {
        return Err(ForestError::new(format!("expected {n_features} feature group ids, got {}", feature_group_ids.len())));
    }
    let mut indexes = HashMap::new();
    let mut groups: Vec<FeatureGroup> = Vec::new();
    let mut feature_groups = vec![usize::MAX; n_features];
    for (feature, &id) in feature_group_ids.iter().enumerate() {
        let next = indexes.len();
        let group = *indexes.entry(id).or_insert(next);
        if group == groups.len() {
            groups.push(FeatureGroup::default());
        }
        groups[group].base.push(feature);
        feature_groups[feature] = group;
    }
    Ok(groups)
}

pub(crate) fn fit_groups(
    n_features: usize, feature_group_ids: Option<&[usize]>, config: &Config,
) -> Result<Option<Vec<FeatureGroup>>, ForestError> {
    config.validate()?;
    feature_group_ids.map(|ids| group_features(n_features, ids)).transpose()
}

pub(crate) fn validate_tracking(config: &Config, rows: &[usize], n_rows: usize) -> Result<(), ForestError> {
    if !config.oob {
        return Err(ForestError::new("tracking rows require oob=true"));
    }
    if rows.is_empty() || rows.iter().any(|&row| row >= n_rows) {
        return Err(ForestError::new("tracking rows are empty or out of range"));
    }
    Ok(())
}

pub(crate) fn validate_batch(configs: &[Config], oob_rows: Option<usize>) -> Result<(), ForestError> {
    if configs.is_empty() {
        return Err(ForestError::new("batch must contain at least one configuration"));
    }
    configs.iter().try_for_each(Config::validate)?;
    if oob_rows == Some(0) {
        return Err(ForestError::new("OOB evaluation rows must be greater than zero"));
    }
    Ok(())
}

pub(crate) fn validate_training_data(
    x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, cutoff_values: &[f32], cutoff_offsets: &[usize],
) -> Result<(), ForestError> {
    validate_encoded_data(x, y.len(), cutoff_values, cutoff_offsets)?;
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ForestError::new("targets must all be finite"));
    }
    Ok(())
}

pub(crate) fn validate_encoded_data(
    x: ArrayView2<'_, u32>, y_len: usize, cutoff_values: &[f32], cutoff_offsets: &[usize],
) -> Result<(), ForestError> {
    if x.nrows() == 0 {
        return Err(ForestError::new("training data must contain at least one row"));
    }
    if x.nrows() > u32::MAX as usize {
        return Err(ForestError::new("training data cannot exceed 2^32-1 rows"));
    }
    if x.nrows() != y_len {
        return Err(ForestError::new(format!("X has {} rows but y has {} values", x.nrows(), y_len)));
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
        if x.column(col).iter().any(|&value| value as usize >= cardinality) {
            return Err(ForestError::new(format!("encoded feature {col} contains a value outside its cutoff mapping")));
        }
    }
    Ok(())
}

pub(crate) fn validate_prediction_data(x: ArrayView2<'_, f32>, n_features: usize) -> Result<(), ForestError> {
    if x.ncols() != n_features {
        return Err(ForestError::new(format!("expected {n_features} features, got {}", x.ncols())));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ForestError::new("features must all be finite"));
    }
    Ok(())
}

pub(crate) fn sample_rows(n_rows: usize, config: &Config, rng: &mut StdRng) -> Vec<u32> {
    let mut sample_size = config.sample_rows.unwrap_or_else(|| {
        let mut size = ((n_rows as f32 * config.resolved_bootstrap_fraction()) as usize).max(1);
        if let Some(max) = config.bootstrap_max {
            size = size.min(max);
        }
        size
    });
    sample_size = sample_size.min(n_rows);
    if config.replacement {
        (0..sample_size).map(|_| u32::try_from(rng.random_range(0..n_rows)).unwrap()).collect()
    } else {
        rand::seq::index::sample(rng, n_rows, sample_size).into_vec().into_iter().map(|row| u32::try_from(row).unwrap()).collect()
    }
}

pub(crate) fn sampled_rows_with_mask(
    n_rows: usize, config: &Config, rng: &mut StdRng, track_in_bag: bool,
) -> (Vec<u32>, Option<Vec<bool>>) {
    let rows = sample_rows(n_rows, config, rng);
    let in_bag = track_in_bag.then(|| {
        let mut mask = vec![false; n_rows];
        rows.iter().for_each(|&row| mask[row as usize] = true);
        mask
    });
    (rows, in_bag)
}
