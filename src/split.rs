use ndarray::{ArrayView1, ArrayView2};
use rand::RngExt;
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::Config;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct FeatureGroup {
    pub base: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Split {
    pub cut_col: Option<usize>,
    pub cut_val: u32,
    pub equality: bool,
    pub value: f32,
    pub gain: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct NodeRows<'a> {
    pub rows: &'a [u32],
    pub start: usize,
    pub n_rows: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Candidate {
    left_target: f32,
    left_sqr_target: f32,
    left_count: usize,
    cut_col: usize,
    cut_val: u32,
    equality: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseFeature {
    pub(crate) cut_col: usize,
    pub(crate) cardinality: usize,
    pub(crate) offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RankedRow {
    pub(crate) value: u32,
    pub(crate) row: usize,
}

#[derive(Default)]
pub(crate) struct SplitScratch {
    candidates: Vec<Candidate>,
    keys: HashSet<u64>,
    ranked_rows: Vec<RankedRow>,
    bin_counts: Vec<usize>,
    bin_targets: Vec<f32>,
    bin_squares: Vec<f32>,
    dense_features: Vec<DenseFeature>,
    dense_lookup: Vec<usize>,
}

struct EvaluationWindow {
    start: usize,
    n_rows: usize,
    sum_target: f32,
    sum_sqr_target: f32,
}

pub(crate) fn find_split(
    x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, node: NodeRows<'_>, config: &Config, cutoff_offsets: &[usize],
    feature_groups: Option<&[FeatureGroup]>, rng: &mut StdRng, scratch: &mut SplitScratch,
) -> Split {
    let max_samples = evaluation_rows(node.n_rows, config);
    if x.ncols() == 0 || node.n_rows < config.min_node_size || all_same(y, node, max_samples) {
        return leaf(y, node);
    }
    if config.random_splitter {
        random_split(x, y, node, config, feature_groups, rng, scratch)
    } else {
        histogram_split(x, y, node, config, cutoff_offsets, feature_groups, rng, scratch)
    }
}

fn finish_split(
    y: ArrayView1<'_, f32>, node: NodeRows<'_>, window: &EvaluationWindow, best: Option<Candidate>, criterion: f32, impurity: f32,
) -> Split {
    let gain = (criterion - impurity).max(0.0);
    Split {
        cut_col: best.map(|candidate| candidate.cut_col),
        cut_val: best.map_or(0, |candidate| candidate.cut_val),
        equality: best.is_some_and(|candidate| candidate.equality),
        value: if best.is_none() && window.n_rows < node.n_rows { node_mean(y, node) } else { window.sum_target / window.n_rows as f32 },
        gain: if best.is_some() { gain } else { 0.0 },
    }
}

fn leaf(y: ArrayView1<'_, f32>, node: NodeRows<'_>) -> Split {
    Split { cut_col: None, cut_val: 0, equality: false, value: node_mean(y, node), gain: 0.0 }
}

fn evaluation_window(y: ArrayView1<'_, f32>, node: NodeRows<'_>, max_samples: usize, rng: &mut StdRng) -> EvaluationWindow {
    let n_rows = node.n_rows.min(max_samples);
    let start = evaluation_start(node, n_rows, rng);
    let (mut sum_target, mut sum_sqr_target) = (0.0, 0.0);
    for &row in &node.rows[start..start + n_rows] {
        let target = y[row as usize];
        sum_target += target;
        sum_sqr_target += target * target;
    }
    EvaluationWindow { start, n_rows, sum_target, sum_sqr_target }
}

pub(crate) fn evaluation_rows(node_rows: usize, config: &Config) -> usize {
    node_rows.min(config.max_node_samples)
}

pub(crate) fn evaluation_start(node: NodeRows<'_>, n_rows: usize, rng: &mut StdRng) -> usize {
    if node.n_rows == n_rows { node.start } else { node.start + rng.random_range(0..=node.n_rows - n_rows) }
}

pub(crate) fn propose_cutoffs(
    x: ArrayView2<'_, u32>, node: NodeRows<'_>, used_n: usize, features: &[usize], divisor: f32, rng: &mut StdRng, keys: &mut HashSet<u64>,
    mut add: impl FnMut(usize, u32),
) {
    keys.clear();
    if features.is_empty() {
        return;
    }
    let attempts =
        ((used_n as f32 * (features.len() as f32).sqrt() / divisor) as usize).max(4).min(node.n_rows.saturating_mul(features.len()));
    for _ in 0..attempts {
        let cut_col = features[rng.random_range(0..features.len())];
        let position = rng.random_range(node.start..node.start + node.n_rows);
        let cut_val = x[[node.rows[position] as usize, cut_col]];
        let key = (cut_col as u64) << 32 | cut_val as u64;
        if keys.insert(key) {
            add(cut_col, cut_val)
        }
    }
}

pub(crate) fn dense_layout(
    features: &[usize], n_features: usize, cutoff_offsets: &[usize], dense: &mut Vec<DenseFeature>, lookup: &mut Vec<usize>,
    mut supported: impl FnMut(usize) -> bool,
) -> usize {
    dense.clear();
    lookup.clear();
    lookup.resize(n_features, usize::MAX);
    let mut total_bins = 0;
    for &cut_col in features {
        let cardinality = cutoff_offsets[cut_col + 1] - cutoff_offsets[cut_col];
        if supported(cardinality) {
            lookup[cut_col] = dense.len();
            dense.push(DenseFeature { cut_col, cardinality, offset: total_bins });
            total_bins += cardinality;
        }
    }
    total_bins
}

pub(crate) fn fill_dense_bins(
    x: ArrayView2<'_, u32>, node: NodeRows<'_>, start: usize, n_rows: usize, features: &[DenseFeature], mut add: impl FnMut(usize, usize),
) {
    if let Some(data) = x.as_slice() {
        for &row in &node.rows[start..start + n_rows] {
            let row = row as usize;
            let row_offset = row * x.ncols();
            for feature in features {
                add(feature.offset + data[row_offset + feature.cut_col] as usize, row);
            }
        }
    } else {
        for &row in &node.rows[start..start + n_rows] {
            let row = row as usize;
            for feature in features {
                add(feature.offset + x[[row, feature.cut_col]] as usize, row);
            }
        }
    }
}

pub(crate) fn ranked_rows(
    x: ArrayView2<'_, u32>, node: NodeRows<'_>, start: usize, n_rows: usize, cut_col: usize, rows: &mut Vec<RankedRow>,
) {
    rows.clear();
    rows.extend(node.rows[start..start + n_rows].iter().map(|&row| {
        let row = row as usize;
        RankedRow { value: x[[row, cut_col]], row }
    }));
    rows.sort_unstable_by_key(|row| row.value);
}

pub(crate) fn valid_children(left: usize, total: usize) -> bool {
    left > 0 && left < total
}

pub(crate) fn supported_equality(value: u32, cardinality: usize, equal: usize, total: usize) -> bool {
    value > 0 && value as usize + 1 < cardinality && valid_children(equal, total)
}

#[allow(clippy::too_many_arguments)]
fn random_split(
    x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, node: NodeRows<'_>, config: &Config, feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng, scratch: &mut SplitScratch,
) -> Split {
    let used_n = evaluation_rows(node.n_rows, config);
    let features = sample_features(x.ncols(), config, feature_groups, rng);
    propose_candidates(x, node, used_n, &features, config.cutoff_divisor, rng, scratch);
    let window = evaluation_window(y, node, used_n, rng);
    if let Some(data) = x.as_slice() {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let target = y[row];
            let offset = row * x.ncols();
            for candidate in &mut scratch.candidates {
                if data[offset + candidate.cut_col] < candidate.cut_val {
                    candidate.left_target += target;
                    candidate.left_sqr_target += target * target;
                    candidate.left_count += 1;
                }
            }
        }
    } else {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let target = y[row];
            for candidate in &mut scratch.candidates {
                if x[[row, candidate.cut_col]] < candidate.cut_val {
                    candidate.left_target += target;
                    candidate.left_sqr_target += target * target;
                    candidate.left_count += 1;
                }
            }
        }
    }
    let (best, criterion, impurity) = select_best(scratch.candidates.iter().copied(), &window, config);
    finish_split(y, node, &window, best, criterion, impurity)
}

fn propose_candidates(
    x: ArrayView2<'_, u32>, node: NodeRows<'_>, used_n: usize, features: &[usize], divisor: f32, rng: &mut StdRng,
    scratch: &mut SplitScratch,
) {
    scratch.candidates.clear();
    let (keys, candidates) = (&mut scratch.keys, &mut scratch.candidates);
    propose_cutoffs(x, node, used_n, features, divisor, rng, keys, |cut_col, cut_val| {
        candidates.push(Candidate { cut_col, cut_val, ..Candidate::default() })
    });
}

fn histogram_split(
    x: ArrayView2<'_, u32>, y: ArrayView1<'_, f32>, node: NodeRows<'_>, config: &Config, cutoff_offsets: &[usize],
    feature_groups: Option<&[FeatureGroup]>, rng: &mut StdRng, scratch: &mut SplitScratch,
) -> Split {
    let features = sample_features(x.ncols(), config, feature_groups, rng);
    let window = evaluation_window(y, node, evaluation_rows(node.n_rows, config), rng);
    let impurity = weighted_loss(0.0, 0.0, 0, window.sum_target, window.sum_sqr_target, window.n_rows, config.split_prior_rows);
    let mut criterion = impurity;
    let mut best = None;
    let total_bins =
        dense_layout(&features, x.ncols(), cutoff_offsets, &mut scratch.dense_features, &mut scratch.dense_lookup, |cardinality| {
            cardinality <= window.n_rows
        });
    scratch.bin_counts.clear();
    scratch.bin_targets.clear();
    scratch.bin_squares.clear();
    scratch.bin_counts.resize(total_bins, 0);
    scratch.bin_targets.resize(total_bins, 0.0);
    scratch.bin_squares.resize(total_bins, 0.0);
    let (counts, targets, squares) = (&mut scratch.bin_counts, &mut scratch.bin_targets, &mut scratch.bin_squares);
    fill_dense_bins(x, node, window.start, window.n_rows, &scratch.dense_features, |bin, row| {
        let target = y[row];
        counts[bin] += 1;
        targets[bin] += target;
        squares[bin] += target * target;
    });
    for cut_col in features {
        let dense = scratch.dense_lookup[cut_col];
        if dense != usize::MAX {
            let DenseFeature { cardinality, offset, .. } = scratch.dense_features[dense];
            let (mut left_target, mut left_sqr_target, mut left_count) = (0.0, 0.0, 0);
            for cut_val in 1..cardinality {
                if cut_val + 1 < cardinality {
                    let left_count = scratch.bin_counts[offset + cut_val];
                    if valid_children(left_count, window.n_rows) {
                        let left_target = scratch.bin_targets[offset + cut_val];
                        let left_sqr_target = scratch.bin_squares[offset + cut_val];
                        let candidate_loss = weighted_loss(
                            left_target,
                            left_sqr_target,
                            left_count,
                            window.sum_target,
                            window.sum_sqr_target,
                            window.n_rows,
                            config.split_prior_rows,
                        );
                        if candidate_loss > criterion {
                            criterion = candidate_loss;
                            best = Some(Candidate {
                                left_target,
                                left_sqr_target,
                                left_count,
                                cut_col,
                                cut_val: cut_val as u32,
                                equality: true,
                            });
                        }
                    }
                }
                left_count += scratch.bin_counts[offset + cut_val - 1];
                left_target += scratch.bin_targets[offset + cut_val - 1];
                left_sqr_target += scratch.bin_squares[offset + cut_val - 1];
                if scratch.bin_counts[offset + cut_val] == 0 {
                    continue;
                }
                if !valid_children(left_count, window.n_rows) {
                    continue;
                }
                let candidate_loss = weighted_loss(
                    left_target,
                    left_sqr_target,
                    left_count,
                    window.sum_target,
                    window.sum_sqr_target,
                    window.n_rows,
                    config.split_prior_rows,
                );
                if candidate_loss > criterion {
                    criterion = candidate_loss;
                    best = Some(Candidate { left_target, left_sqr_target, left_count, cut_col, cut_val: cut_val as u32, equality: false });
                }
            }
            continue;
        }
        ranked_rows(x, node, window.start, window.n_rows, cut_col, &mut scratch.ranked_rows);
        let (mut left_target, mut left_sqr_target, mut left_count) = (0.0, 0.0, 0);
        let mut position = 0;
        while position < scratch.ranked_rows.len() {
            let value = scratch.ranked_rows[position].value;
            let (mut equal_target, mut equal_sqr_target, mut equal_count) = (0.0, 0.0, 0);
            while position < scratch.ranked_rows.len() && scratch.ranked_rows[position].value == value {
                let target = y[scratch.ranked_rows[position].row];
                equal_target += target;
                equal_sqr_target += target * target;
                equal_count += 1;
                left_target += target;
                left_sqr_target += target * target;
                left_count += 1;
                position += 1;
            }
            if supported_equality(value, cutoff_offsets[cut_col + 1] - cutoff_offsets[cut_col], equal_count, window.n_rows) {
                let candidate_loss = weighted_loss(
                    equal_target,
                    equal_sqr_target,
                    equal_count,
                    window.sum_target,
                    window.sum_sqr_target,
                    window.n_rows,
                    config.split_prior_rows,
                );
                if candidate_loss > criterion {
                    criterion = candidate_loss;
                    best = Some(Candidate {
                        left_target: equal_target,
                        left_sqr_target: equal_sqr_target,
                        left_count: equal_count,
                        cut_col,
                        cut_val: value,
                        equality: true,
                    });
                }
            }
            if position == scratch.ranked_rows.len() || !valid_children(left_count, window.n_rows) {
                continue;
            }
            let candidate_loss = weighted_loss(
                left_target,
                left_sqr_target,
                left_count,
                window.sum_target,
                window.sum_sqr_target,
                window.n_rows,
                config.split_prior_rows,
            );
            if candidate_loss > criterion {
                criterion = candidate_loss;
                best = Some(Candidate {
                    left_target,
                    left_sqr_target,
                    left_count,
                    cut_col,
                    cut_val: scratch.ranked_rows[position].value,
                    equality: false,
                });
            }
        }
    }
    finish_split(y, node, &window, best, criterion, impurity)
}

pub(crate) fn sample_features(n_features: usize, config: &Config, feature_groups: Option<&[FeatureGroup]>, rng: &mut StdRng) -> Vec<usize> {
    if let Some(groups) = feature_groups {
        let selected = config.max_features.resolve(groups.len()).min(groups.len());
        rand::seq::index::sample(rng, groups.len(), selected).into_iter().flat_map(|group| groups[group].base.iter().copied()).collect()
    } else {
        let selected = config.max_features.resolve(n_features).min(n_features);
        rand::seq::index::sample(rng, n_features, selected).into_vec()
    }
}

fn select_best(candidates: impl Iterator<Item = Candidate>, window: &EvaluationWindow, config: &Config) -> (Option<Candidate>, f32, f32) {
    let impurity = weighted_loss(0.0, 0.0, 0, window.sum_target, window.sum_sqr_target, window.n_rows, config.split_prior_rows);
    let mut criterion = impurity;
    let mut best = None;
    for candidate in candidates {
        if !valid_children(candidate.left_count, window.n_rows) {
            continue;
        }
        let candidate_loss = weighted_loss(
            candidate.left_target,
            candidate.left_sqr_target,
            candidate.left_count,
            window.sum_target,
            window.sum_sqr_target,
            window.n_rows,
            config.split_prior_rows,
        );
        if candidate_loss > criterion {
            criterion = candidate_loss;
            best = Some(candidate);
        }
    }
    (best, criterion, impurity)
}

fn node_mean(y: ArrayView1<'_, f32>, node: NodeRows<'_>) -> f32 {
    node.rows[node.start..node.start + node.n_rows].iter().map(|&row| y[row as usize]).sum::<f32>() / node.n_rows as f32
}

fn all_same(y: ArrayView1<'_, f32>, node: NodeRows<'_>, max_samples: usize) -> bool {
    let end = node.start + node.n_rows.min(max_samples);
    let first = y[node.rows[node.start] as usize];
    node.rows[node.start + 1..end].iter().all(|&row| (y[row as usize] - first).abs() <= 1.0e-8 + 1.0e-5 * first.abs())
}

pub(crate) fn partition(
    x: ArrayView2<'_, u32>, rows: &mut [u32], start: usize, n_rows: usize, cut_col: usize, cut_val: u32, equality: bool,
) -> usize {
    let mut left = start;
    let mut right = start + n_rows;
    if let Some(data) = x.as_slice() {
        let cols = x.ncols();
        while left < right {
            let value = data[rows[left] as usize * cols + cut_col];
            if if equality { value == cut_val } else { value < cut_val } {
                left += 1;
            } else {
                right -= 1;
                rows.swap(left, right);
            }
        }
        return left - start;
    }
    while left < right {
        let value = x[[rows[left] as usize, cut_col]];
        if if equality { value == cut_val } else { value < cut_val } {
            left += 1;
        } else {
            right -= 1;
            rows.swap(left, right);
        }
    }
    left - start
}

fn weighted_loss(
    left_target: f32, left_sqr_target: f32, left_count: usize, sum_target: f32, sum_sqr_target: f32, total_count: usize, prior_rows: f32,
) -> f32 {
    let parent_mean = sum_target / total_count as f32;
    let mut result = 0.0;
    if left_count > 0 {
        result +=
            regularized_loss(left_target, left_sqr_target, left_count, parent_mean, prior_rows) * left_count as f32 / total_count as f32;
    }
    let right_count = total_count - left_count;
    if right_count > 0 {
        result += regularized_loss(sum_target - left_target, sum_sqr_target - left_sqr_target, right_count, parent_mean, prior_rows)
            * right_count as f32
            / total_count as f32;
    }
    result
}

fn regularized_loss(sum_target: f32, sum_sqr_target: f32, n: usize, parent_mean: f32, prior_rows: f32) -> f32 {
    if prior_rows == 0.0 {
        return loss(sum_target, sum_sqr_target, n);
    }
    let n = n as f32;
    let mean = sum_target / n;
    let variance = if n == 1.0 { 0.0 } else { ((sum_sqr_target - sum_target * mean) / (n - 1.0)).max(0.0) };
    let shrink = prior_rows / (n + prior_rows);
    let residual = mean - parent_mean;
    -(variance + shrink * shrink * residual * residual).sqrt()
}

fn loss(sum_target: f32, sum_sqr_target: f32, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let variance = ((sum_sqr_target - sum_target * sum_target / n as f32) / (n - 1) as f32).max(0.0);
    -variance.sqrt()
}
