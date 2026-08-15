use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::class_split::{ClassSplitScratch, find_class_split};
use crate::forest::{
    LEAF_COL, group_features, sample_rows, validate_encoded_data, validate_prediction_data,
};
use crate::split::{NodeRows, partition};
use crate::{Config, FeatureSampling, ForestError, MaxFeatures, Splitter};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
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
                feature_groups,
                &mut rng,
                &mut scratch,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassificationAdaptiveScore {
    pub max_features: f32,
    pub max_node_samples: usize,
    pub oob_brier: f64,
}

#[derive(Clone, Debug)]
pub struct ClassifierForest {
    trees: Vec<ClassTree>,
    n_features: usize,
    n_classes: usize,
    feature_importances: Vec<f32>,
    oob_decision: Option<Vec<f32>>,
    oob_counts: Option<Vec<u32>>,
    adaptive_scores: Vec<ClassificationAdaptiveScore>,
    adaptive_choice: Option<ClassificationAdaptiveScore>,
}

impl ClassifierForest {
    const PILOT_ROWS_PER_CLASS: usize = 8_000;
    const PREDICTION_CACHE_BYTES: usize = 1 << 19;

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
        if config.workbench.leaf_regularization != 0.0 {
            return Err(ForestError::new(
                "leaf_regularization is not supported for classification",
            ));
        }
        let feature_groups = match feature_group_ids {
            Some(ids) => Some(group_features(x.ncols(), ids)?),
            None if config.workbench.feature_sampling == FeatureSampling::Columns => {
                return Err(ForestError::new(
                    "column feature sampling requires feature group ids",
                ));
            }
            None => None,
        };
        let mut class_config = config.clone();
        class_config.bootstrap_max = config
            .bootstrap_max
            .map(|max| max.saturating_mul(n_classes));
        let pilot_rows = Self::PILOT_ROWS_PER_CLASS.saturating_mul(n_classes);
        if config.adaptive
            && config.workbench.splitter == Splitter::Histogram
            && x.nrows() > pilot_rows
            && x.ncols() > 0
        {
            Self::fit_adaptive(
                x,
                y,
                n_classes,
                cutoff_values,
                cutoff_offsets,
                feature_groups.as_deref(),
                &class_config,
            )
        } else {
            Self::fit_fixed(
                x,
                y,
                n_classes,
                cutoff_values,
                cutoff_offsets,
                feature_groups.as_deref(),
                &class_config,
            )
        }
    }

    fn fit_fixed(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, u32>,
        n_classes: usize,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_groups: Option<&[Vec<usize>]>,
        config: &Config,
    ) -> Result<Self, ForestError> {
        let mut seed_rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
        let seeds: Vec<u64> = (0..config.n_trees).map(|_| seed_rng.random()).collect();
        let built: Vec<_> = seeds
            .into_par_iter()
            .map(|seed| TrainingClassTree::build(x, y, n_classes, config, feature_groups, seed))
            .collect();
        let mut trees = Vec::with_capacity(config.n_trees);
        let mut oob_decision = config.oob.then(|| vec![0.0; x.nrows() * n_classes]);
        let mut oob_counts = config.oob.then(|| vec![0; x.nrows()]);
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
                for row_idx in 0..x.nrows() {
                    if mask[row_idx] {
                        continue;
                    }
                    let output = &mut sums[row_idx * n_classes..(row_idx + 1) * n_classes];
                    if let Some(data) = data {
                        let start = row_idx * x.ncols();
                        tree.add_probabilities_by(|col| data[start + col], output);
                    } else {
                        tree.add_probabilities_by(|col| x[[row_idx, col]], output);
                    }
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
            adaptive_scores: Vec::new(),
            adaptive_choice: None,
        })
    }

    fn fit_adaptive(
        x: ArrayView2<'_, u32>,
        y: ArrayView1<'_, u32>,
        n_classes: usize,
        cutoff_values: &[f32],
        cutoff_offsets: &[usize],
        feature_groups: Option<&[Vec<usize>]>,
        config: &Config,
    ) -> Result<Self, ForestError> {
        const CANDIDATES: [f32; 2] = [0.6, 0.9];
        let pilot_rows = Self::PILOT_ROWS_PER_CLASS * n_classes;
        let seed = config.seed.unwrap_or_else(rand::random);
        let mut sample_rng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
        let pilot_indexes =
            rand::seq::index::sample(&mut sample_rng, x.nrows(), pilot_rows).into_vec();
        let pilot_x = Array2::from_shape_fn((pilot_rows, x.ncols()), |(row, col)| {
            x[[pilot_indexes[row], col]]
        });
        let pilot_y = Array1::from_shape_fn(pilot_rows, |row| y[pilot_indexes[row]]);
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
            pilot_config.workbench.max_features = MaxFeatures::Fraction(max_features);
            let forest = Self::fit_fixed(
                pilot_x.view(),
                pilot_y.view(),
                n_classes,
                cutoff_values,
                cutoff_offsets,
                feature_groups,
                &pilot_config,
            )?;
            let decision = forest.oob_decision.as_ref().unwrap();
            let counts = forest.oob_counts.as_ref().unwrap();
            let (loss, rows) = decision
                .chunks_exact(n_classes)
                .zip(counts)
                .zip(&pilot_y)
                .filter(|((_, count), _)| **count > 0)
                .fold(
                    (0.0, 0_usize),
                    |(loss, rows), ((probabilities, _), &target)| {
                        let row_loss = probabilities
                            .iter()
                            .enumerate()
                            .map(|(class, &probability)| {
                                let expected = f64::from(u8::from(class == target as usize));
                                (f64::from(probability) - expected).powi(2)
                            })
                            .sum::<f64>();
                        (loss + row_loss, rows + 1)
                    },
                );
            scores.push(ClassificationAdaptiveScore {
                max_features,
                max_node_samples: config.max_node_samples,
                oob_brier: loss / rows as f64,
            });
        }
        let choice = scores
            .iter()
            .copied()
            .min_by(|left, right| left.oob_brier.total_cmp(&right.oob_brier))
            .unwrap();
        let mut selected = config.clone();
        selected.seed = Some(seed);
        selected.adaptive = false;
        selected.workbench.max_features = MaxFeatures::Fraction(choice.max_features);
        let mut forest = Self::fit_fixed(
            x,
            y,
            n_classes,
            cutoff_values,
            cutoff_offsets,
            feature_groups,
            &selected,
        )?;
        forest.adaptive_scores = scores;
        forest.adaptive_choice = Some(choice);
        Ok(forest)
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
    pub fn adaptive_scores(&self) -> &[ClassificationAdaptiveScore] {
        &self.adaptive_scores
    }
    pub fn adaptive_choice(&self) -> Option<ClassificationAdaptiveScore> {
        self.adaptive_choice
    }
}
