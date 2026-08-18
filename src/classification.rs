use ndarray::{ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::class_split::{ClassSplitScratch, find_class_split};
use crate::ensemble::{assemble_forest, combined_importance, combined_oob, tree_seeds};
use crate::forest::{
    fit_groups, oob_rows, sampled_rows_with_mask, validate_batch, validate_encoded_data, validate_prediction_data, validate_tracking,
};
use crate::prediction::{PredictionTree, add_block_by, predict_outputs, row_block_size, trees_per_batch};
use crate::split::FeatureGroup;
use crate::tree::{Branch, TreeNode, grow_tree, leaf_index, native_node, structure};
use crate::{Config, ForestError};

type ClassNode = TreeNode<f32, u32>;
type TrainingClassNode = TreeNode<u32, u32>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ClassTree {
    nodes: Vec<ClassNode>,
    probabilities: Vec<f32>,
    n_classes: usize,
}

impl ClassTree {
    fn leaf_by(&self, value: impl Fn(usize) -> f32) -> usize {
        self.nodes[leaf_index(&self.nodes, value, |observed, cutoff| observed > cutoff)].value as usize
    }

    fn structure(&self) -> (usize, usize, usize) {
        structure(&self.nodes)
    }

    fn add_probabilities_by(&self, value: impl Fn(usize) -> f32, output: &mut [f32]) {
        let leaf = self.leaf_by(|col| value(col));
        let base = &self.probabilities[leaf * self.n_classes..(leaf + 1) * self.n_classes];
        output.iter_mut().zip(base).for_each(|(total, value)| *total += value);
    }
}

impl PredictionTree for ClassTree {
    fn prediction_bytes(&self) -> usize {
        std::mem::size_of_val(self.nodes.as_slice()) + std::mem::size_of_val(self.probabilities.as_slice())
    }
    fn add_prediction_by(&self, value: impl Fn(usize) -> f32, output: &mut [f32]) {
        self.add_probabilities_by(value, output)
    }
}

struct TrainingClassTree {
    nodes: Vec<TrainingClassNode>,
    probabilities: Vec<f32>,
    n_classes: usize,
}

impl TrainingClassTree {
    fn leaf_by(&self, value: impl Fn(usize) -> u32) -> usize {
        self.nodes[leaf_index(&self.nodes, value, |observed, cutoff| observed >= cutoff)].value as usize
    }

    fn add_probabilities_by(&self, value: impl Fn(usize) -> u32, output: &mut [f32]) {
        let leaf = self.leaf_by(value);
        output
            .iter_mut()
            .zip(&self.probabilities[leaf * self.n_classes..(leaf + 1) * self.n_classes])
            .for_each(|(total, value)| *total += value);
    }

    fn into_native(self, cutoff_values: &[f32], cutoff_offsets: &[usize]) -> ClassTree {
        let nodes = self.nodes.into_iter().map(|node| native_node(node, cutoff_values, cutoff_offsets)).collect();
        ClassTree { nodes, probabilities: self.probabilities, n_classes: self.n_classes }
    }

    fn build(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, n_classes: usize, cutoff_offsets: &[usize], config: &Config,
        feature_groups: Option<&[FeatureGroup]>, seed: u64, track_in_bag: bool,
    ) -> (Self, Option<Vec<bool>>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut rows, in_bag) = sampled_rows_with_mask(x.nrows(), config, &mut rng, track_in_bag);
        let mut tree_classes = vec![0_u32; n_classes];
        rows.iter().for_each(|&row| tree_classes[y[row as usize] as usize] += 1);
        let mut nodes = vec![TrainingClassNode::new()];
        let mut probabilities = Vec::new();
        let mut importance = vec![0.0; x.ncols()];
        let mut scratch = ClassSplitScratch::new(&tree_classes, config.max_node_samples, config.class_weight_power);
        grow_tree(x, &mut rows, &mut nodes, &mut importance, |node, tree_node| {
            let split = find_class_split(x, y, node, n_classes, config, cutoff_offsets, feature_groups, &mut rng, &mut scratch);
            let Some(cut_col) = split.cut_col else {
                tree_node.value = u32::try_from(probabilities.len() / n_classes).expect("tree has too many leaves");
                let offset = probabilities.len();
                probabilities.resize(offset + n_classes, 0.0);
                for &row in &node.rows[node.start..node.start + node.n_rows] {
                    probabilities[offset + y[row as usize] as usize] += 1.0;
                }
                probabilities[offset..offset + n_classes].iter_mut().for_each(|value| *value /= node.n_rows as f32);
                return None;
            };
            Some(Branch { cut_col, cut_val: split.cut_val, equality: split.equality, gain: split.gain })
        });
        (Self { nodes, probabilities, n_classes }, in_bag, importance)
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
    pub(crate) fn validate_loaded(&self, encoded_features: usize) -> Result<(), ForestError> {
        if self.trees.is_empty() || self.n_classes < 2 {
            return Err(ForestError::new("saved classifier dimensions are invalid"));
        }
        if self.n_features != encoded_features || self.feature_importances.len() != encoded_features {
            return Err(ForestError::new("saved classifier feature dimensions are inconsistent"));
        }
        for tree in &self.trees {
            if tree.nodes.is_empty() || tree.n_classes != self.n_classes || tree.probabilities.len() % self.n_classes != 0 {
                return Err(ForestError::new("saved classifier tree dimensions are invalid"));
            }
            let leaves = tree.probabilities.len() / self.n_classes;
            if tree.probabilities.iter().any(|value| !value.is_finite()) {
                return Err(ForestError::new("saved classifier contains a non-finite value"));
            }
            for node in &tree.nodes {
                if !node.cut_val.is_finite()
                    || node.is_leaf() && node.value as usize >= leaves
                    || !node.is_leaf() && (node.feature() >= encoded_features || node.child as usize + 1 >= tree.nodes.len())
                {
                    return Err(ForestError::new("saved classifier contains an invalid node index"));
                }
            }
        }
        Ok(())
    }

    fn trees_per_batch(&self) -> usize {
        trees_per_batch(&self.trees)
    }

    fn class_index(probabilities: &[f32]) -> u32 {
        probabilities.iter().enumerate().reduce(|best, item| if item.1 > best.1 { item } else { best }).unwrap().0 as u32
    }

    pub fn fit(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, n_classes: usize, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>, config: &Config,
    ) -> Result<Self, ForestError> {
        validate_encoded_data(x, y.len(), cutoff_values, cutoff_offsets)?;
        if n_classes < 2 {
            return Err(ForestError::new("classification requires at least two classes"));
        }
        if y.iter().any(|&class| class as usize >= n_classes) {
            return Err(ForestError::new("targets contain a class outside 0..n_classes"));
        }
        let feature_groups = fit_groups(x.ncols(), feature_group_ids, config)?;
        let output_dimensions = n_classes.saturating_sub(1).max(1);
        let mut class_config = config.clone();
        class_config.bootstrap_max = config.bootstrap_max.map(|max| max.saturating_mul(output_dimensions));
        Self::fit_fixed(x, y, n_classes, cutoff_values, cutoff_offsets, feature_groups.as_deref(), &class_config, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_on_tracking(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, n_classes: usize, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>, config: &Config, tracking_indices: &[usize],
    ) -> Result<Self, ForestError> {
        validate_encoded_data(x, y.len(), cutoff_values, cutoff_offsets)?;
        if n_classes < 2 || y.iter().any(|&class| class as usize >= n_classes) {
            return Err(ForestError::new("classification targets must use at least two classes in 0..n_classes"));
        }
        let feature_groups = fit_groups(x.ncols(), feature_group_ids, config)?;
        validate_tracking(config, tracking_indices, x.nrows())?;
        let mut config = config.clone();
        config.bootstrap_max = config.bootstrap_max.map(|max| max.saturating_mul(n_classes.saturating_sub(1).max(1)));
        Self::fit_fixed(x, y, n_classes, cutoff_values, cutoff_offsets, feature_groups.as_deref(), &config, None, Some(tracking_indices))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_batch(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, n_classes: usize, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_group_ids: Option<&[usize]>, configs: &[Config], oob_rows: Option<usize>,
    ) -> Result<Vec<Self>, ForestError> {
        validate_encoded_data(x, y.len(), cutoff_values, cutoff_offsets)?;
        validate_batch(configs, oob_rows)?;
        if n_classes < 2 || y.iter().any(|&class| class as usize >= n_classes) {
            return Err(ForestError::new("classification targets must use at least two classes in 0..n_classes"));
        }
        let feature_groups = feature_group_ids.map(|ids| crate::forest::group_features(x.ncols(), ids)).transpose()?;
        let output_dimensions = n_classes.saturating_sub(1).max(1);
        let configs: Vec<_> = configs
            .iter()
            .map(|config| {
                let mut config = config.clone();
                config.bootstrap_max = config.bootstrap_max.map(|max| max.saturating_mul(output_dimensions));
                config
            })
            .collect();
        configs
            .par_iter()
            .map(|config| {
                Self::fit_fixed(x, y, n_classes, cutoff_values, cutoff_offsets, feature_groups.as_deref(), config, oob_rows, None)
            })
            .collect()
    }

    fn fit_fixed(
        x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, n_classes: usize, cutoff_values: &[f32], cutoff_offsets: &[usize],
        feature_groups: Option<&[FeatureGroup]>, config: &Config, oob_row_override: Option<usize>, tracking_rows: Option<&[usize]>,
    ) -> Result<Self, ForestError> {
        let built: Vec<_> = tree_seeds(config)
            .into_par_iter()
            .map(|seed| TrainingClassTree::build(x, y, n_classes, cutoff_offsets, config, feature_groups, seed, config.oob))
            .collect();
        let oob_indices = tracking_rows.map(<[usize]>::to_vec).or_else(|| oob_rows(x.nrows(), config, oob_row_override));
        let data = x.as_slice();
        let (trees, feature_importances, oob_decision, oob_counts, oob_indices) = assemble_forest(
            built,
            x.ncols(),
            n_classes,
            oob_indices,
            config.oob,
            |tree, row, output| {
                if let Some(data) = data {
                    let start = row * x.ncols();
                    tree.add_probabilities_by(|col| data[start + col], output)
                } else {
                    tree.add_probabilities_by(|col| x[[row, col]], output)
                }
            },
            |tree| tree.into_native(cutoff_values, cutoff_offsets),
        );
        Ok(Self { trees, n_features: x.ncols(), n_classes, feature_importances, oob_decision, oob_counts, oob_indices })
    }

    pub fn combined(&self, other: &Self) -> Result<Self, ForestError> {
        if self.n_features != other.n_features || self.n_classes != other.n_classes || self.oob_indices != other.oob_indices {
            return Err(ForestError::new("forests have incompatible class, feature, or OOB dimensions"));
        }
        let left_trees = self.trees.len();
        let right_trees = other.trees.len();
        let mut trees = self.trees.clone();
        trees.extend(other.trees.iter().cloned());
        let feature_importances = combined_importance(&self.feature_importances, &other.feature_importances, left_trees, right_trees);
        let left = self.oob_decision.as_deref().zip(self.oob_counts.as_deref());
        let right = other.oob_decision.as_deref().zip(other.oob_counts.as_deref());
        let (oob_decision, oob_counts) = combined_oob(left, right, self.n_classes)?;
        Ok(Self {
            trees,
            n_features: self.n_features,
            n_classes: self.n_classes,
            feature_importances,
            oob_decision,
            oob_counts,
            oob_indices: self.oob_indices.clone(),
        })
    }

    pub fn predict_proba(&self, x: ArrayView2<'_, f32>) -> Result<Vec<f32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        Ok(predict_outputs(&self.trees, self.n_features, self.n_classes, x))
    }

    pub fn predict(&self, x: ArrayView2<'_, f32>) -> Result<Vec<u32>, ForestError> {
        validate_prediction_data(x, self.n_features)?;
        let mut predictions = vec![0; x.nrows()];
        if x.nrows() == 0 {
            return Ok(predictions);
        }
        let block_rows = row_block_size(x.nrows());
        let trees_per_batch = self.trees_per_batch();
        if let Some(data) = x.as_slice() {
            predictions.par_chunks_mut(block_rows).enumerate().for_each_init(
                || vec![0.0; block_rows * self.n_classes],
                |probabilities, (block, predictions)| {
                    let row_start = block * block_rows;
                    let output = &mut probabilities[..predictions.len() * self.n_classes];
                    add_block_by(&self.trees, predictions.len(), self.n_classes, output, trees_per_batch, |row, col| {
                        data[(row_start + row) * self.n_features + col]
                    });
                    predictions
                        .iter_mut()
                        .zip(output.chunks_exact(self.n_classes))
                        .for_each(|(prediction, probabilities)| *prediction = Self::class_index(probabilities));
                },
            );
        } else {
            predictions.par_chunks_mut(block_rows).enumerate().for_each_init(
                || vec![0.0; block_rows * self.n_classes],
                |probabilities, (block, predictions)| {
                    let row_start = block * block_rows;
                    let output = &mut probabilities[..predictions.len() * self.n_classes];
                    add_block_by(&self.trees, predictions.len(), self.n_classes, output, trees_per_batch, |row, col| {
                        x[[row_start + row, col]]
                    });
                    predictions
                        .iter_mut()
                        .zip(output.chunks_exact(self.n_classes))
                        .for_each(|(prediction, probabilities)| *prediction = Self::class_index(probabilities));
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
