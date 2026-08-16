use ndarray::{ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::class_split::{ClassSplitScratch, find_class_split, root_impurity};
use crate::forest::{
    LEAF_COL, group_features, oob_rows, sample_rows, validate_encoded_data,
    validate_prediction_data,
};
use crate::split::{FeatureGroup, NodeRows, TreeCutoffs, partition};
use crate::{Config, ForestError};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct ClassNode {
    cut_val: f32,
    leaf: u32,
    child: u32,
    cut_col: u32,
}

impl ClassNode {
    fn is_leaf(&self) -> bool {
        self.cut_col == LEAF_COL
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrainingClassNode {
    cut_val: u32,
    leaf: u32,
    child: u32,
    cut_col: u32,
}

impl TrainingClassNode {
    fn new() -> Self {
        Self {
            cut_val: 0,
            leaf: 0,
            child: 0,
            cut_col: LEAF_COL,
        }
    }
    fn is_leaf(&self) -> bool {
        self.cut_col == LEAF_COL
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ClassTree {
    nodes: Vec<ClassNode>,
    probabilities: Vec<f32>,
    n_classes: usize,
}

impl ClassTree {
    fn leaf_by(&self, value: impl Fn(usize) -> f32) -> usize {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.leaf as usize;
            }
            let go_right = usize::from(value(node.cut_col as usize) > node.cut_val);
            node_idx = node.child as usize + go_right;
        }
    }

    fn structure(&self) -> (usize, usize, usize) {
        let mut leaves = 0;
        let mut depth = 0;
        let mut stack = vec![(0, 0)];
        while let Some((index, node_depth)) = stack.pop() {
            let node = &self.nodes[index];
            depth = depth.max(node_depth);
            if node.is_leaf() { leaves += 1 }
            else {
                stack.push((node.child as usize, node_depth+1));
                stack.push((node.child as usize+1, node_depth+1));
            }
        }
        (self.nodes.len(), leaves, depth)
    }

    fn add_probabilities_by(&self, value: impl Fn(usize) -> f32, output: &mut [f32]) {
        let leaf = self.leaf_by(value);
        output
            .iter_mut()
            .zip(&self.probabilities[leaf * self.n_classes..(leaf + 1) * self.n_classes])
            .for_each(|(total, value)| *total += value);
    }
}

struct TrainingClassTree {
    nodes: Vec<TrainingClassNode>,
    probabilities: Vec<f32>,
    n_classes: usize,
}

impl TrainingClassTree {
    fn leaf_by(&self, value: impl Fn(usize) -> u32) -> usize {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.leaf as usize;
            }
            let go_right = usize::from(value(node.cut_col as usize) >= node.cut_val);
            node_idx = node.child as usize + go_right;
        }
    }

    fn add_probabilities_by(&self, value: impl Fn(usize) -> u32, output: &mut [f32]) {
        let leaf = self.leaf_by(value);
        output
            .iter_mut()
            .zip(&self.probabilities[leaf * self.n_classes..(leaf + 1) * self.n_classes])
            .for_each(|(total, value)| *total += value);
    }

    fn into_native(self, cutoff_values: &[f32], cutoff_offsets: &[usize]) -> ClassTree {
        let nodes = self
            .nodes
            .into_iter()
            .map(|node| ClassNode {
                cut_val: if node.is_leaf() {
                    0.0
                } else {
                    cutoff_values[cutoff_offsets[node.cut_col as usize] + node.cut_val as usize]
                },
                leaf: node.leaf,
                child: node.child,
                cut_col: node.cut_col,
            })
            .collect();
        ClassTree {
            nodes,
            probabilities: self.probabilities,
            n_classes: self.n_classes,
        }
    }

    fn build(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, u32>,
        n_classes: usize,
        cutoff_offsets: &[usize],
        config: &Config,
        feature_groups: Option<&[FeatureGroup]>,
        seed: u64,
    ) -> (Self, Option<Vec<bool>>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = sample_rows(x.nrows(), config, &mut rng);
        let mut cutoff_rng = StdRng::seed_from_u64(seed ^ 0x9e3779b97f4a7c15);
        let tree_cutoffs = TreeCutoffs::fit(x, &rows, cutoff_offsets, config.tree_cutoff_samples, &mut cutoff_rng);
        let root_rows = rows.len();
        let root_impurity = if config.min_local_gain > 0.0 || config.min_global_gain > 0.0 {
            root_impurity(y, &rows, n_classes)
        } else { 1.0 };
        let in_bag = config.oob.then(|| {
            let mut mask = vec![false; x.nrows()];
            rows.iter().for_each(|&row| mask[row as usize] = true);
            mask
        });
        let mut nodes = vec![TrainingClassNode::new()];
        let mut probabilities = Vec::new();
        let mut importance = vec![0.0; x.ncols()];
        let mut scratch = ClassSplitScratch::default();
        let mut work = vec![(0, 0, rows.len())];
        while let Some((node_idx, start, n_rows)) = work.pop() {
            let split = find_class_split(
                x,
                y,
                NodeRows {
                    rows: &rows,
                    start,
                    n_rows,
                },
                n_classes,
                config,
                cutoff_offsets,
                tree_cutoffs.as_ref(),
                feature_groups,
                &mut rng,
                &mut scratch,
                root_impurity,
                root_rows,
            );
            let (Some(cut_col), cut_val) = (split.cut_col, split.cut_val) else {
                nodes[node_idx].leaf = u32::try_from(probabilities.len() / n_classes)
                    .expect("tree has too many leaves");
                let offset = probabilities.len();
                probabilities.resize(offset + n_classes, 0.0);
                for &row in &rows[start..start + n_rows] {
                    probabilities[offset + y[row as usize] as usize] += 1.0;
                }
                probabilities[offset..offset + n_classes]
                    .iter_mut()
                    .for_each(|value| *value /= n_rows as f32);
                continue;
            };
            importance[cut_col] += split.gain * n_rows as f32;
            let left_n = partition(x, &mut rows, start, n_rows, cut_col, cut_val);
            debug_assert!(left_n > 0 && left_n < n_rows);
            let left_idx = nodes.len();
            let right_idx = left_idx + 1;
            nodes.push(TrainingClassNode::new());
            nodes.push(TrainingClassNode::new());
            nodes[node_idx].child = u32::try_from(left_idx).expect("tree has too many nodes");
            nodes[node_idx].cut_col = u32::try_from(cut_col).expect("matrix has too many columns");
            nodes[node_idx].cut_val = cut_val;
            work.push((right_idx, start + left_n, n_rows - left_n));
            work.push((left_idx, start, left_n));
        }
        (
            Self {
                nodes,
                probabilities,
                n_classes,
            },
            in_bag,
            importance,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifierForest {
    trees: Vec<ClassTree>,
    n_features: usize,
    n_classes: usize,
    feature_importances: Vec<f32>,
    #[serde(skip)]
    oob_decision: Option<Vec<f32>>,
    #[serde(skip)]
    oob_counts: Option<Vec<u32>>,
    #[serde(skip)]
    oob_indices: Option<Vec<usize>>,
}

impl ClassifierForest {
    const PREDICTION_CACHE_BYTES: usize = 1 << 19;

    pub(crate) fn validate_loaded(&self, encoded_features: usize) -> Result<(), ForestError> {
        if self.trees.is_empty() || self.n_classes < 2 {
            return Err(ForestError::new("saved classifier dimensions are invalid"));
        }
        if self.n_features != encoded_features || self.feature_importances.len() != encoded_features {
            return Err(ForestError::new("saved classifier feature dimensions are inconsistent"));
        }
        for tree in &self.trees {
            if tree.nodes.is_empty()
                || tree.n_classes != self.n_classes
                || tree.probabilities.len() % self.n_classes != 0
            {
                return Err(ForestError::new("saved classifier tree dimensions are invalid"));
            }
            let leaves = tree.probabilities.len() / self.n_classes;
            if tree.probabilities.iter().any(|value| !value.is_finite()) {
                return Err(ForestError::new("saved classifier contains a non-finite probability"));
            }
            for node in &tree.nodes {
                if !node.cut_val.is_finite()
                    || node.is_leaf() && node.leaf as usize >= leaves
                    || !node.is_leaf()
                        && (node.cut_col as usize >= encoded_features
                            || node.child as usize + 1 >= tree.nodes.len())
                {
                    return Err(ForestError::new("saved classifier contains an invalid node index"));
                }
            }
        }
        Ok(())
    }


    fn trees_per_batch(&self) -> usize {
        let bytes: usize = self
            .trees
            .iter()
            .map(|tree| {
                std::mem::size_of_val(tree.nodes.as_slice())
                    + std::mem::size_of_val(tree.probabilities.as_slice())
            })
            .sum();
        let mean_bytes = bytes.div_ceil(self.trees.len()).max(1);
        (Self::PREDICTION_CACHE_BYTES / mean_bytes).clamp(1, self.trees.len())
    }

    fn row_block_size(n_rows: usize) -> usize {
        n_rows
            .div_ceil(rayon::current_num_threads().saturating_mul(4).max(1))
            .clamp(1, 4_096)
    }

    #[inline]
    fn add_block_by(
        &self,
        n_rows: usize,
        probabilities: &mut [f32],
        trees_per_batch: usize,
        value: impl Fn(usize, usize) -> f32,
    ) {
        probabilities.fill(0.0);
        for trees in self.trees.chunks(trees_per_batch) {
            for row in 0..n_rows {
                let output = &mut probabilities[row * self.n_classes..(row + 1) * self.n_classes];
                trees
                    .iter()
                    .for_each(|tree| tree.add_probabilities_by(|col| value(row, col), output));
            }
        }
    }

    fn class_index(probabilities: &[f32]) -> u32 {
        probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0 as u32
    }

    pub fn fit(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, u32>,
        n_classes: usize,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>,
        frequent_parents: Option<&[usize]>,
        config: &Config,
    ) -> Result<Self, ForestError> {
        validate_encoded_data(x, y.len(), cutoff_values, cutoff_offsets)?;
        config.validate()?;
        if n_classes < 2 {
            return Err(ForestError::new(
                "classification requires at least two classes",
            ));
        }
        if y.iter().any(|&class| class as usize >= n_classes) {
            return Err(ForestError::new(
                "targets contain a class outside 0..n_classes",
            ));
        }
        let feature_groups = feature_group_ids
            .map(|ids| group_features(x.ncols(), ids, frequent_parents))
            .transpose()?;
        let output_dimensions = n_classes.saturating_sub(1).max(1);
        let mut class_config = config.clone();
        class_config.bootstrap_max = config
            .bootstrap_max
            .map(|max| max.saturating_mul(output_dimensions));
        Self::fit_fixed(
            x,
            y,
            n_classes,
            cutoff_values,
            cutoff_offsets,
            feature_groups.as_deref(),
            &class_config,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_batch(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, u32>,
        n_classes: usize,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>,
        frequent_parents: Option<&[usize]>,
        configs: &[Config],
        oob_rows: Option<usize>,
    ) -> Result<Vec<Self>, ForestError> {
        validate_encoded_data(x, y.len(), cutoff_values, cutoff_offsets)?;
        if configs.is_empty() { return Err(ForestError::new("batch must contain at least one configuration")) }
        configs.iter().try_for_each(Config::validate)?;
        if oob_rows == Some(0) { return Err(ForestError::new("OOB evaluation rows must be greater than zero")) }
        if n_classes < 2 || y.iter().any(|&class| class as usize >= n_classes) {
            return Err(ForestError::new("classification targets must use at least two classes in 0..n_classes"));
        }
        let feature_groups = feature_group_ids
            .map(|ids| group_features(x.ncols(), ids, frequent_parents))
            .transpose()?;
        let output_dimensions = n_classes.saturating_sub(1).max(1);
        let configs: Vec<_> = configs.iter().map(|config| {
            let mut config = config.clone();
            config.bootstrap_max = config.bootstrap_max.map(|max| max.saturating_mul(output_dimensions));
            config
        }).collect();
        configs.par_iter().map(|config| Self::fit_fixed(
            x, y, n_classes, cutoff_values, cutoff_offsets, feature_groups.as_deref(), config, oob_rows)).collect()
    }

    fn fit_fixed(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, u32>,
        n_classes: usize,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_groups: Option<&[FeatureGroup]>,
        config: &Config,
        oob_row_override: Option<usize>,
    ) -> Result<Self, ForestError> {
        let mut seed_rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
        let seeds: Vec<u64> = (0..config.n_trees).map(|_| seed_rng.random()).collect();
        let built: Vec<_> = seeds
            .into_par_iter()
            .map(|seed| TrainingClassTree::build(x, y, n_classes, cutoff_offsets, config, feature_groups, seed))
            .collect();
        let mut trees = Vec::with_capacity(config.n_trees);
        let oob_indices = oob_rows(x.nrows(), config, oob_row_override);
        let mut oob_decision = oob_indices
            .as_ref()
            .map(|indices| vec![0.0; indices.len() * n_classes]);
        let mut oob_counts = oob_indices.as_ref().map(|indices| vec![0; indices.len()]);
        let mut feature_importances = vec![0.0; x.ncols()];
        for (tree, in_bag, importance) in built {
            feature_importances
                .iter_mut()
                .zip(importance)
                .for_each(|(total, value)| *total += value);
            if let Some(mask) = in_bag {
                let sums = oob_decision.as_mut().unwrap();
                let counts = oob_counts.as_mut().unwrap();
                let data = x.as_slice();
                for (output_idx, &row_idx) in oob_indices.as_ref().unwrap().iter().enumerate() {
                    if mask[row_idx] {
                        continue;
                    }
                    let output =
                        &mut sums[output_idx * n_classes..(output_idx + 1) * n_classes];
                    if let Some(data) = data {
                        let start = row_idx * x.ncols();
                        tree.add_probabilities_by(|col| data[start + col], output);
                    } else {
                        tree.add_probabilities_by(|col| x[[row_idx, col]], output);
                    }
                    counts[output_idx] += 1;
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
        if let (Some(decision), Some(counts)) = (&mut oob_decision, &oob_counts) {
            for (row, &count) in decision.chunks_exact_mut(n_classes).zip(counts) {
                if count == 0 {
                    row.fill(f32::NAN);
                } else {
                    row.iter_mut().for_each(|value| *value /= count as f32);
                }
            }
        }
        Ok(Self {
            trees,
            n_features: x.ncols(),
            n_classes,
            feature_importances,
            oob_decision,
            oob_counts,
            oob_indices,
        })
    }

    pub fn predict_proba(&self, x: ArrayView2<'_, f32>) -> Result<Vec<f32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        let mut probabilities = vec![0.0; x.nrows() * self.n_classes];
        if x.nrows() == 0 {
            return Ok(probabilities);
        }
        let n_trees = self.trees.len() as f32;
        let block_rows = Self::row_block_size(x.nrows());
        let output_block = block_rows * self.n_classes;
        let trees_per_batch = self.trees_per_batch();
        if let Some(data) = x.as_slice() {
            probabilities
                .par_chunks_mut(output_block)
                .enumerate()
                .for_each(|(block, output)| {
                    let row_start = block * block_rows;
                    let n_rows = output.len() / self.n_classes;
                    self.add_block_by(n_rows, output, trees_per_batch, |row, col| {
                        data[(row_start + row) * self.n_features + col]
                    });
                    output.iter_mut().for_each(|value| *value /= n_trees);
                });
        } else {
            probabilities
                .par_chunks_mut(output_block)
                .enumerate()
                .for_each(|(block, output)| {
                    let row_start = block * block_rows;
                    let n_rows = output.len() / self.n_classes;
                    self.add_block_by(n_rows, output, trees_per_batch, |row, col| {
                        x[[row_start + row, col]]
                    });
                    output.iter_mut().for_each(|value| *value /= n_trees);
                });
        }
        Ok(probabilities)
    }

    pub fn predict(&self, x: ArrayView2<'_, f32>) -> Result<Vec<u32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        let mut predictions = vec![0; x.nrows()];
        if x.nrows() == 0 {
            return Ok(predictions);
        }
        let block_rows = Self::row_block_size(x.nrows());
        let trees_per_batch = self.trees_per_batch();
        if let Some(data) = x.as_slice() {
            predictions
                .par_chunks_mut(block_rows)
                .enumerate()
                .for_each_init(
                    || vec![0.0; block_rows * self.n_classes],
                    |probabilities, (block, predictions)| {
                        let row_start = block * block_rows;
                        let output = &mut probabilities[..predictions.len() * self.n_classes];
                        self.add_block_by(
                            predictions.len(),
                            output,
                            trees_per_batch,
                            |row, col| data[(row_start + row) * self.n_features + col],
                        );
                        predictions
                            .iter_mut()
                            .zip(output.chunks_exact(self.n_classes))
                            .for_each(|(prediction, probabilities)| {
                                *prediction = Self::class_index(probabilities)
                            });
                    },
                );
        } else {
            predictions
                .par_chunks_mut(block_rows)
                .enumerate()
                .for_each_init(
                    || vec![0.0; block_rows * self.n_classes],
                    |probabilities, (block, predictions)| {
                        let row_start = block * block_rows;
                        let output = &mut probabilities[..predictions.len() * self.n_classes];
                        self.add_block_by(
                            predictions.len(),
                            output,
                            trees_per_batch,
                            |row, col| x[[row_start + row, col]],
                        );
                        predictions
                            .iter_mut()
                            .zip(output.chunks_exact(self.n_classes))
                            .for_each(|(prediction, probabilities)| {
                                *prediction = Self::class_index(probabilities)
                            });
                    },
                );
        }
        Ok(predictions)
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
    pub fn tree_structures(&self) -> Vec<(usize, usize, usize)> {
        self.trees.iter().map(ClassTree::structure).collect()
    }
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
    pub fn prediction_trees_per_batch(&self) -> usize {
        self.trees_per_batch()
    }
    pub fn feature_importances(&self) -> &[f32] {
        &self.feature_importances
    }
    pub fn oob_decision(&self) -> Option<&[f32]> {
        self.oob_decision.as_deref()
    }
    pub fn oob_counts(&self) -> Option<&[u32]> {
        self.oob_counts.as_deref()
    }
    pub fn oob_indices(&self) -> Option<&[usize]> {
        self.oob_indices.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn batch_matches_standalone_forests_exactly() {
        let x = Array2::from_shape_fn((240, 5), |(row, col)| ((row*7 + col*11) % 17) as u32);
        let y = Array1::from_shape_fn(240, |row| ((x[[row, 0]] + 2*x[[row, 1]]) % 3) as u32);
        let mut cutoffs = Vec::new();
        let mut offsets = vec![0];
        for _ in 0..x.ncols() {
            cutoffs.extend((0..17).map(|value| value as f32));
            offsets.push(cutoffs.len());
        }
        let base = Config { n_trees: 8, bootstrap_max: None, min_node_size: 6, max_node_samples: 60,
            seed: Some(77), oob: true, ..Config::default() };
        let configs = [base.clone(), Config { min_node_size: 12, tree_cutoff_samples: Some(8),
            max_features: crate::MaxFeatures::Fraction(0.6), ..base }];
        let standalone: Vec<_> = configs.iter().map(|config| ClassifierForest::fit(x.view(), y.view(), 3,
            &cutoffs, &offsets, None, None, config).unwrap()).collect();
        let batch = ClassifierForest::fit_batch(x.view(), y.view(), 3, &cutoffs, &offsets, None, None, &configs, None).unwrap();
        for (standalone, batched) in standalone.iter().zip(&batch) {
            assert_eq!(standalone.trees, batched.trees);
            assert_eq!(standalone.feature_importances, batched.feature_importances);
            assert_eq!(standalone.oob_counts, batched.oob_counts);
            assert_eq!(standalone.oob_indices, batched.oob_indices);
            assert_eq!(standalone.oob_decision.as_ref().unwrap().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                batched.oob_decision.as_ref().unwrap().iter().map(|value| value.to_bits()).collect::<Vec<_>>());
        }
        let reversed = ClassifierForest::fit_batch(x.view(), y.view(), 3, &cutoffs, &offsets, None, None,
            &[configs[1].clone(), configs[0].clone()], None).unwrap();
        assert_eq!(batch[0].trees, reversed[1].trees);
        assert_eq!(batch[1].trees, reversed[0].trees);
    }
}
