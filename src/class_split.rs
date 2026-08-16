use ndarray::{ArrayView1, ArrayView2};
use rand::RngExt;
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::Config;
use crate::split::{FeatureGroup, NodeRows, TreeCutoffs, evaluation_rows, sample_features, sampled_min_size};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ClassSplit {
    pub cut_col: Option<usize>,
    pub cut_val: u32,
    pub gain: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Candidate {
    left_count: usize,
    left_squares: u64,
    right_squares: u64,
    cut_col: usize,
    cut_val: u32,
}

#[derive(Clone, Copy, Debug)]
struct Bin {
    value: u32,
    class: u32,
}

#[derive(Clone, Copy, Debug)]
struct DenseFeature {
    cut_col: usize,
    cardinality: usize,
    offset: usize,
    mapped: bool,
}

#[derive(Default)]
pub(crate) struct ClassSplitScratch {
    candidates: Vec<Candidate>,
    candidate_classes: Vec<u32>,
    total_classes: Vec<u32>,
    left_classes: Vec<u32>,
    keys: HashSet<u64>,
    bins: Vec<Bin>,
    bin_classes: Vec<u32>,
    dense_features: Vec<DenseFeature>,
    dense_lookup: Vec<usize>,
}

struct EvaluationWindow {
    start: usize,
    n_rows: usize,
    total_squares: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn find_class_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, u32>,
    node: NodeRows<'_>,
    n_classes: usize,
    config: &Config,
    cutoff_offsets: &[usize],
    tree_cutoffs: Option<&TreeCutoffs>,
    feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
    root_impurity: f32,
    root_rows: usize,
) -> ClassSplit {
    let max_samples = evaluation_rows(node.n_rows, config);
    if x.ncols() == 0
        || node.n_rows < config.min_node_size
        || all_same(y, node, max_samples)
    {
        return leaf();
    }
    if config.random_splitter {
        random_split(x, y, node, n_classes, config, feature_groups, rng, scratch, root_impurity, root_rows)
    } else {
        histogram_split(x, y, node, n_classes, config, cutoff_offsets, tree_cutoffs, feature_groups, rng, scratch, root_impurity, root_rows)
    }
}

fn leaf() -> ClassSplit {
    ClassSplit {
        cut_col: None,
        cut_val: 0,
        gain: 0.0,
    }
}

fn evaluation_window(
    y: ArrayView1<'_, u32>,
    node: NodeRows<'_>,
    n_classes: usize,
    max_samples: usize,
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
) -> EvaluationWindow {
    let n_rows = node.n_rows.min(max_samples);
    let start = if node.n_rows == n_rows {
        node.start
    } else {
        node.start + rng.random_range(0..=node.n_rows - n_rows)
    };
    scratch.total_classes.clear();
    scratch.total_classes.resize(n_classes, 0);
    for &row in &node.rows[start..start + n_rows] {
        scratch.total_classes[y[row as usize] as usize] += 1;
    }
    let total_squares = scratch
        .total_classes
        .iter()
        .map(|&count| u64::from(count) * u64::from(count))
        .sum();
    EvaluationWindow {
        start,
        n_rows,
        total_squares,
    }
}

#[allow(clippy::too_many_arguments)]
fn random_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, u32>,
    node: NodeRows<'_>,
    n_classes: usize,
    config: &Config,
    feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
    root_impurity: f32,
    root_rows: usize,
) -> ClassSplit {
    let used_n = evaluation_rows(node.n_rows, config);
    let features = sample_features(x.ncols(), config, feature_groups, rng);
    propose_candidates(x, node, used_n, &features, config.cutoff_divisor, rng, scratch);
    let window = evaluation_window(y, node, n_classes, used_n, rng, scratch);
    scratch.candidate_classes.clear();
    scratch
        .candidate_classes
        .resize(scratch.candidates.len() * n_classes, 0);
    if let Some(data) = x.as_slice() {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let class = y[row] as usize;
            let offset = row * x.ncols();
            for (candidate_idx, candidate) in scratch.candidates.iter_mut().enumerate() {
                if data[offset + candidate.cut_col] < candidate.cut_val {
                    let count = &mut scratch.candidate_classes[candidate_idx * n_classes + class];
                    candidate.left_squares += 2 * u64::from(*count) + 1;
                    *count += 1;
                    candidate.left_count += 1;
                }
            }
        }
    } else {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let class = y[row] as usize;
            for (candidate_idx, candidate) in scratch.candidates.iter_mut().enumerate() {
                if x[[row, candidate.cut_col]] < candidate.cut_val {
                    let count = &mut scratch.candidate_classes[candidate_idx * n_classes + class];
                    candidate.left_squares += 2 * u64::from(*count) + 1;
                    *count += 1;
                    candidate.left_count += 1;
                }
            }
        }
    }
    for (candidate_idx, candidate) in scratch.candidates.iter_mut().enumerate() {
        candidate.right_squares = scratch
            .total_classes
            .iter()
            .enumerate()
            .map(|(class, &total)| {
                let left = scratch.candidate_classes[candidate_idx * n_classes + class];
                let right = total - left;
                u64::from(right) * u64::from(right)
            })
            .sum();
    }
    select_best(
        scratch.candidates.iter().copied(),
        &window,
        config.min_node_size,
        node,
        config,
        root_impurity,
        root_rows,
    )
}

fn propose_candidates(
    x: ArrayView2<'_, u32>,
    node: NodeRows<'_>,
    used_n: usize,
    features: &[usize],
    divisor: f32,
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
) {
    scratch.candidates.clear();
    scratch.keys.clear();
    if features.is_empty() { return }
    let attempts = ((used_n as f32 * (features.len() as f32).sqrt() / divisor) as usize)
        .max(4)
        .min(node.n_rows.saturating_mul(features.len()));
    for _ in 0..attempts {
        let cut_col = features[rng.random_range(0..features.len())];
        let position = rng.random_range(node.start..node.start + node.n_rows);
        let cut_val = x[[node.rows[position] as usize, cut_col]];
        let key = (cut_col as u64) << 32 | cut_val as u64;
        if scratch.keys.insert(key) {
            scratch.candidates.push(Candidate { cut_col, cut_val, ..Candidate::default() });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn histogram_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, u32>,
    node: NodeRows<'_>,
    n_classes: usize,
    config: &Config,
    cutoff_offsets: &[usize],
    tree_cutoffs: Option<&TreeCutoffs>,
    feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
    root_impurity: f32,
    root_rows: usize,
) -> ClassSplit {
    let features = sample_features(x.ncols(), config, feature_groups, rng);
    let window = evaluation_window(y, node, n_classes, evaluation_rows(node.n_rows, config), rng, scratch);
    let impurity = weighted_gini(0, 0, window.n_rows, window.total_squares);
    let min_size = sampled_min_size(window.n_rows, config.min_node_size);
    let mut criterion = impurity;
    let mut best = None;
    scratch.dense_features.clear();
    scratch.dense_lookup.clear();
    scratch.dense_lookup.resize(x.ncols(), usize::MAX);
    let sort_work = window.n_rows * (window.n_rows.ilog2() as usize + 1);
    let mut total_bins = 0;
    for &cut_col in &features {
        let mapped = tree_cutoffs.and_then(|cutoffs| cutoffs.feature(cut_col));
        let cardinality = mapped.map_or(cutoff_offsets[cut_col + 1] - cutoff_offsets[cut_col], |_| tree_cutoffs.unwrap().cardinality(cut_col));
        if mapped.is_some() || cardinality <= window.n_rows && cardinality * n_classes <= sort_work {
            scratch.dense_lookup[cut_col] = scratch.dense_features.len();
            scratch.dense_features.push(DenseFeature { cut_col, cardinality, offset: total_bins, mapped: mapped.is_some() });
            total_bins += cardinality;
        }
    }
    scratch.bin_classes.clear();
    scratch.bin_classes.resize(total_bins * n_classes, 0);
    if let Some(data) = x.as_slice() {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let class = y[row] as usize;
            let row_offset = row * x.ncols();
            for feature in &scratch.dense_features {
                let rank = data[row_offset + feature.cut_col];
                let local = if feature.mapped { tree_cutoffs.unwrap().bin(feature.cut_col, rank) } else { rank as usize };
                let bin = feature.offset + local;
                scratch.bin_classes[bin * n_classes + class] += 1;
            }
        }
    } else {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let class = y[row] as usize;
            for feature in &scratch.dense_features {
                let rank = x[[row, feature.cut_col]];
                let local = if feature.mapped { tree_cutoffs.unwrap().bin(feature.cut_col, rank) } else { rank as usize };
                let bin = feature.offset + local;
                scratch.bin_classes[bin * n_classes + class] += 1;
            }
        }
    }
    for cut_col in features {
        let dense = scratch.dense_lookup[cut_col];
        if dense != usize::MAX {
            let DenseFeature { cardinality, offset, mapped, .. } = scratch.dense_features[dense];
            scratch.left_classes.clear();
            scratch.left_classes.resize(n_classes, 0);
            let (mut left_count, mut left_squares, mut right_squares) = (0, 0_u64, window.total_squares);
            for cut_val in 1..cardinality {
                let start = (offset + cut_val - 1) * n_classes;
                let counts = &scratch.bin_classes[start..start + n_classes];
                for (class, &added) in counts.iter().enumerate() {
                    let added = u64::from(added);
                    let left = u64::from(scratch.left_classes[class]);
                    let right = u64::from(scratch.total_classes[class]) - left;
                    left_squares += 2 * left * added + added * added;
                    right_squares -= 2 * right * added - added * added;
                    scratch.left_classes[class] += added as u32;
                    left_count += added as usize;
                }
                let next = (offset + cut_val) * n_classes;
                if scratch.bin_classes[next..next + n_classes].iter().all(|&count| count == 0) { continue }
                if left_count < min_size || window.n_rows - left_count < min_size { continue }
                let score = weighted_gini(left_count, left_squares, window.n_rows, right_squares);
                if score > criterion {
                    criterion = score;
                    best = Some(Candidate {
                        left_count, left_squares, right_squares, cut_col,
                        cut_val: if mapped { tree_cutoffs.unwrap().cutoff(cut_col, cut_val) } else { cut_val as u32 },
                    });
                }
            }
            continue;
        }
        scratch.bins.clear();
        scratch.bins.extend(
            (window.start..window.start + window.n_rows).map(|position| {
                let row = node.rows[position] as usize;
                Bin {
                    value: x[[row, cut_col]],
                    class: y[row],
                }
            }),
        );
        scratch.bins.sort_unstable_by_key(|bin| bin.value);
        scratch.left_classes.clear();
        scratch.left_classes.resize(n_classes, 0);
        let (mut left_count, mut left_squares, mut right_squares, mut position) =
            (0, 0_u64, window.total_squares, 0);
        while position < scratch.bins.len() {
            let value = scratch.bins[position].value;
            while position < scratch.bins.len() && scratch.bins[position].value == value {
                let count = &mut scratch.left_classes[scratch.bins[position].class as usize];
                left_squares += 2 * u64::from(*count) + 1;
                let right_count =
                    scratch.total_classes[scratch.bins[position].class as usize] - *count;
                right_squares -= 2 * u64::from(right_count) - 1;
                *count += 1;
                left_count += 1;
                position += 1;
            }
            if position == scratch.bins.len()
                || left_count < min_size
                || window.n_rows - left_count < min_size
            {
                continue;
            }
            let score = weighted_gini(left_count, left_squares, window.n_rows, right_squares);
            if score > criterion {
                criterion = score;
                best = Some(Candidate {
                    left_count,
                    left_squares,
                    right_squares,
                    cut_col,
                    cut_val: scratch.bins[position].value,
                });
            }
        }
    }
    finish(best, criterion, impurity, node, config, root_impurity, root_rows)
}

fn select_best(
    candidates: impl Iterator<Item = Candidate>,
    window: &EvaluationWindow,
    min_node_size: usize,
    node: NodeRows<'_>,
    config: &Config,
    root_impurity: f32,
    root_rows: usize,
) -> ClassSplit {
    let impurity = weighted_gini(0, 0, window.n_rows, window.total_squares);
    let min_size = sampled_min_size(window.n_rows, min_node_size);
    let (mut best, mut criterion) = (None, impurity);
    for candidate in candidates {
        if candidate.left_count < min_size || window.n_rows - candidate.left_count < min_size {
            continue;
        }
        let score = weighted_gini(
            candidate.left_count,
            candidate.left_squares,
            window.n_rows,
            candidate.right_squares,
        );
        if score > criterion {
            best = Some(candidate);
            criterion = score;
        }
    }
    finish(best, criterion, impurity, node, config, root_impurity, root_rows)
}

fn finish(
    best: Option<Candidate>,
    criterion: f32,
    impurity: f32,
    node: NodeRows<'_>,
    config: &Config,
    root_impurity: f32,
    root_rows: usize,
) -> ClassSplit {
    let gain = (criterion - impurity).max(0.0);
    let local_gain = gain / (-impurity).max(f32::MIN_POSITIVE);
    let global_gain = gain * node.n_rows as f32
        / (root_impurity.max(f32::MIN_POSITIVE) * root_rows as f32);
    let accepted = best.is_some()
        && local_gain >= config.min_local_gain
        && global_gain >= config.min_global_gain;
    let best = best.filter(|_| accepted);
    ClassSplit {
        cut_col: best.map(|candidate| candidate.cut_col),
        cut_val: best.map_or(0, |candidate| candidate.cut_val),
        gain: if accepted { gain } else { 0.0 },
    }
}

pub(crate) fn root_impurity(y: ArrayView1<'_, u32>, rows: &[u32], n_classes: usize) -> f32 {
    let mut counts = vec![0_u32; n_classes];
    for &row in rows { counts[y[row as usize] as usize] += 1 }
    let squares = counts.iter().map(|&count| u64::from(count) * u64::from(count)).sum();
    -weighted_gini(0, 0, rows.len(), squares)
}

fn weighted_gini(
    left_count: usize,
    left_squares: u64,
    total_count: usize,
    right_squares: u64,
) -> f32 {
    fn weighted_impurity(count: usize, squares: u64, total: usize) -> f64 {
        if count == 0 {
            0.0
        } else {
            (count as f64 - squares as f64 / count as f64) / total as f64
        }
    }
    let right_count = total_count - left_count;
    -(weighted_impurity(left_count, left_squares, total_count)
        + weighted_impurity(right_count, right_squares, total_count)) as f32
}

fn all_same(y: ArrayView1<'_, u32>, node: NodeRows<'_>, max_samples: usize) -> bool {
    let end = node.start + node.n_rows.min(max_samples);
    let first = y[node.rows[node.start] as usize];
    node.rows[node.start + 1..end]
        .iter()
        .all(|&row| y[row as usize] == first)
}
