use ndarray::{ArrayView1, ArrayView2};
use rand::RngExt;
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::Config;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct FeatureGroup {
    pub base: Vec<usize>,
    pub frequent: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Split {
    pub cut_col: Option<usize>,
    pub cut_val: u32,
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
}

#[derive(Clone, Copy, Debug)]
struct Bin {
    value: u32,
    target: f32,
}

#[derive(Clone, Copy, Debug)]
struct DenseFeature {
    cut_col: usize,
    cardinality: usize,
    offset: usize,
    mapped: bool,
}

pub(crate) struct FeatureCutoffs {
    values_start: usize,
    values_len: usize,
    bins_start: usize,
}

pub(crate) struct TreeCutoffs {
    features: Vec<Option<FeatureCutoffs>>,
    values: Vec<u32>,
    bins: Vec<u16>,
}

impl TreeCutoffs {
    pub(crate) fn fit(
        x: ArrayView2<'_, u32>,
        _rows: &[u32],
        cutoff_offsets: &[usize],
        samples: Option<usize>,
        rng: &mut StdRng,
    ) -> Option<Self> {
        let samples = samples.filter(|&samples| samples > 0)?;
        let mut features = Vec::with_capacity(x.ncols());
        let mut all_values = Vec::new();
        let mut all_bins = Vec::new();
        for cut_col in 0..x.ncols() {
            let cardinality = cutoff_offsets[cut_col + 1] - cutoff_offsets[cut_col];
            if cardinality <= samples + 1 {
                features.push(None);
                continue;
            }
            let mut values = Vec::with_capacity(samples);
            for _ in 0..samples { values.push(rng.random_range(1..cardinality as u32)) }
            values.sort_unstable();
            values.dedup();
            if values.is_empty() {
                features.push(None);
                continue;
            }
            let bins_start = all_bins.len();
            let mut bin = 0;
            for rank in 0..cardinality as u32 {
                while values.get(bin).is_some_and(|&cutoff| rank >= cutoff) { bin += 1 }
                all_bins.push(bin as u16);
            }
            let values_start = all_values.len();
            let values_len = values.len();
            all_values.extend(values);
            features.push(Some(FeatureCutoffs { values_start, values_len, bins_start }));
        }
        Some(Self { features, values: all_values, bins: all_bins })
    }

    pub(crate) fn feature(&self, cut_col: usize) -> Option<&FeatureCutoffs> {
        self.features[cut_col].as_ref()
    }

    pub(crate) fn bin(&self, cut_col: usize, rank: u32) -> usize {
        let feature = self.feature(cut_col).unwrap();
        self.bins[feature.bins_start + rank as usize] as usize
    }

    pub(crate) fn cutoff(&self, cut_col: usize, bin: usize) -> u32 {
        let feature = self.feature(cut_col).unwrap();
        self.values[feature.values_start + bin - 1]
    }

    pub(crate) fn cardinality(&self, cut_col: usize) -> usize {
        self.feature(cut_col).unwrap().values_len + 1
    }

}

#[derive(Default)]
pub(crate) struct SplitScratch {
    candidates: Vec<Candidate>,
    keys: HashSet<u64>,
    bins: Vec<Bin>,
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
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    config: &Config,
    cutoff_offsets: &[usize],
    tree_cutoffs: Option<&TreeCutoffs>,
    feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng,
    scratch: &mut SplitScratch,
    root_impurity: f32,
    root_rows: usize,
) -> Split {
    let max_samples = evaluation_rows(node.n_rows, config);
    if x.ncols() == 0
        || node.n_rows < config.min_node_size
        || all_same(y, node, max_samples)
    {
        return leaf(y, node);
    }
    if config.random_splitter {
        random_split(x, y, node, config, feature_groups, rng, scratch, root_impurity, root_rows)
    } else {
        histogram_split(x, y, node, config, cutoff_offsets, tree_cutoffs, feature_groups, rng, scratch, root_impurity, root_rows)
    }
}

fn finish_split(
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    window: &EvaluationWindow,
    best: Option<Candidate>,
    criterion: f32,
    impurity: f32,
    config: &Config,
    root_impurity: f32,
    root_rows: usize,
) -> Split {
    let gain = (criterion - impurity).max(0.0);
    let local_gain = gain / (-impurity).max(f32::MIN_POSITIVE);
    let global_gain = gain * node.n_rows as f32
        / (root_impurity.max(f32::MIN_POSITIVE) * root_rows as f32);
    let accepted = best.is_some()
        && local_gain >= config.min_local_gain
        && global_gain >= config.min_global_gain;
    let best = best.filter(|_| accepted);
    Split {
        cut_col: best.map(|candidate| candidate.cut_col),
        cut_val: best.map_or(0, |candidate| candidate.cut_val),
        value: if best.is_none() && window.n_rows < node.n_rows {
            node_mean(y, node)
        } else {
            window.sum_target / window.n_rows as f32
        },
        gain: if accepted { gain } else { 0.0 },
    }
}

fn leaf(y: ArrayView1<'_, f32>, node: NodeRows<'_>) -> Split {
    Split {
        cut_col: None,
        cut_val: 0,
        value: node_mean(y, node),
        gain: 0.0,
    }
}

fn evaluation_window(
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    max_samples: usize,
    rng: &mut StdRng,
) -> EvaluationWindow {
    let n_rows = node.n_rows.min(max_samples);
    let start = if node.n_rows == n_rows {
        node.start
    } else {
        node.start + rng.random_range(0..=node.n_rows - n_rows)
    };
    let (mut sum_target, mut sum_sqr_target) = (0.0, 0.0);
    for &row in &node.rows[start..start + n_rows] {
        let target = y[row as usize];
        sum_target += target;
        sum_sqr_target += target * target;
    }
    EvaluationWindow {
        start,
        n_rows,
        sum_target,
        sum_sqr_target,
    }
}

pub(crate) fn evaluation_rows(node_rows: usize, config: &Config) -> usize {
    node_rows.min(config.max_node_samples)
}

#[allow(clippy::too_many_arguments)]
fn random_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    config: &Config,
    feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng,
    scratch: &mut SplitScratch,
    root_impurity: f32,
    root_rows: usize,
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
    let (best, criterion, impurity) = select_best(
        scratch.candidates.iter().copied(),
        &window,
        config.min_node_size,
    );
    finish_split(y, node, &window, best, criterion, impurity, config, root_impurity, root_rows)
}

fn propose_candidates(
    x: ArrayView2<'_, u32>,
    node: NodeRows<'_>,
    used_n: usize,
    features: &[usize],
    divisor: f32,
    rng: &mut StdRng,
    scratch: &mut SplitScratch,
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

fn histogram_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    config: &Config,
    cutoff_offsets: &[usize],
    tree_cutoffs: Option<&TreeCutoffs>,
    feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng,
    scratch: &mut SplitScratch,
    root_impurity: f32,
    root_rows: usize,
) -> Split {
    let features = sample_features(x.ncols(), config, feature_groups, rng);
    let window = evaluation_window(y, node, evaluation_rows(node.n_rows, config), rng);
    let impurity = weighted_loss(
        0.0,
        0.0,
        0,
        window.sum_target,
        window.sum_sqr_target,
        window.n_rows,
    );
    let min_size = sampled_min_size(window.n_rows, config.min_node_size);
    let mut criterion = impurity;
    let mut best = None;
    scratch.dense_features.clear();
    scratch.dense_lookup.clear();
    scratch.dense_lookup.resize(x.ncols(), usize::MAX);
    let mut total_bins = 0;
    for &cut_col in &features {
        let mapped = tree_cutoffs.and_then(|cutoffs| cutoffs.feature(cut_col));
        let cardinality = mapped.map_or(cutoff_offsets[cut_col + 1] - cutoff_offsets[cut_col], |_| tree_cutoffs.unwrap().cardinality(cut_col));
        if mapped.is_some() || cardinality <= window.n_rows {
            scratch.dense_lookup[cut_col] = scratch.dense_features.len();
            scratch.dense_features.push(DenseFeature { cut_col, cardinality, offset: total_bins, mapped: mapped.is_some() });
            total_bins += cardinality;
        }
    }
    scratch.bin_counts.clear();
    scratch.bin_targets.clear();
    scratch.bin_squares.clear();
    scratch.bin_counts.resize(total_bins, 0);
    scratch.bin_targets.resize(total_bins, 0.0);
    scratch.bin_squares.resize(total_bins, 0.0);
    if let Some(data) = x.as_slice() {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let target = y[row];
            let row_offset = row * x.ncols();
            for feature in &scratch.dense_features {
                let rank = data[row_offset + feature.cut_col];
                let local = if feature.mapped { tree_cutoffs.unwrap().bin(feature.cut_col, rank) } else { rank as usize };
                let bin = feature.offset + local;
                scratch.bin_counts[bin] += 1;
                scratch.bin_targets[bin] += target;
                scratch.bin_squares[bin] += target * target;
            }
        }
    } else {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let target = y[row];
            for feature in &scratch.dense_features {
                let rank = x[[row, feature.cut_col]];
                let local = if feature.mapped { tree_cutoffs.unwrap().bin(feature.cut_col, rank) } else { rank as usize };
                let bin = feature.offset + local;
                scratch.bin_counts[bin] += 1;
                scratch.bin_targets[bin] += target;
                scratch.bin_squares[bin] += target * target;
            }
        }
    }
    for cut_col in features {
        let dense = scratch.dense_lookup[cut_col];
        if dense != usize::MAX {
            let DenseFeature { cardinality, offset, mapped, .. } = scratch.dense_features[dense];
            let (mut left_target, mut left_sqr_target, mut left_count) = (0.0, 0.0, 0);
            for cut_val in 1..cardinality {
                left_count += scratch.bin_counts[offset + cut_val - 1];
                left_target += scratch.bin_targets[offset + cut_val - 1];
                left_sqr_target += scratch.bin_squares[offset + cut_val - 1];
                if scratch.bin_counts[offset + cut_val] == 0 { continue }
                if left_count < min_size || window.n_rows - left_count < min_size { continue }
                let candidate_loss = weighted_loss(
                    left_target, left_sqr_target, left_count,
                    window.sum_target, window.sum_sqr_target, window.n_rows,
                );
                if candidate_loss > criterion {
                    criterion = candidate_loss;
                    best = Some(Candidate {
                        left_target, left_sqr_target, left_count, cut_col,
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
                    target: y[row],
                }
            }),
        );
        scratch.bins.sort_unstable_by_key(|bin| bin.value);
        let (mut left_target, mut left_sqr_target, mut left_count) = (0.0, 0.0, 0);
        let mut position = 0;
        while position < scratch.bins.len() {
            let value = scratch.bins[position].value;
            while position < scratch.bins.len() && scratch.bins[position].value == value {
                let target = scratch.bins[position].target;
                left_target += target;
                left_sqr_target += target * target;
                left_count += 1;
                position += 1;
            }
            if position == scratch.bins.len()
                || left_count < min_size
                || window.n_rows - left_count < min_size
            {
                continue;
            }
            let candidate_loss = weighted_loss(
                left_target,
                left_sqr_target,
                left_count,
                window.sum_target,
                window.sum_sqr_target,
                window.n_rows,
            );
            if candidate_loss > criterion {
                criterion = candidate_loss;
                best = Some(Candidate {
                    left_target,
                    left_sqr_target,
                    left_count,
                    cut_col,
                    cut_val: scratch.bins[position].value,
                });
            }
        }
    }
    finish_split(y, node, &window, best, criterion, impurity, config, root_impurity, root_rows)
}

pub(crate) fn sample_features(
    n_features: usize,
    config: &Config,
    feature_groups: Option<&[FeatureGroup]>,
    rng: &mut StdRng,
) -> Vec<usize> {
    if let Some(groups) = feature_groups {
        let selected = config.max_features.resolve(groups.len())
            .min(groups.len());
        rand::seq::index::sample(rng, groups.len(), selected)
            .into_iter()
            .flat_map(|group| {
                let group = &groups[group];
                let extra = (!group.frequent.is_empty())
                    .then(|| group.frequent[rng.random_range(0..group.frequent.len())]);
                group.base.iter().copied().chain(extra)
            })
            .collect()
    } else {
        let selected = config.max_features.resolve(n_features)
            .min(n_features);
        rand::seq::index::sample(rng, n_features, selected).into_vec()
    }
}

fn select_best(
    candidates: impl Iterator<Item = Candidate>,
    window: &EvaluationWindow,
    min_node_size: usize,
) -> (Option<Candidate>, f32, f32) {
    let impurity = weighted_loss(
        0.0,
        0.0,
        0,
        window.sum_target,
        window.sum_sqr_target,
        window.n_rows,
    );
    let mut criterion = impurity;
    let mut best = None;
    let min_size = sampled_min_size(window.n_rows, min_node_size);
    for candidate in candidates {
        let right_count = window.n_rows - candidate.left_count;
        if candidate.left_count < min_size || right_count < min_size {
            continue;
        }
        let candidate_loss = weighted_loss(
            candidate.left_target,
            candidate.left_sqr_target,
            candidate.left_count,
            window.sum_target,
            window.sum_sqr_target,
            window.n_rows,
        );
        if candidate_loss > criterion {
            criterion = candidate_loss;
            best = Some(candidate);
        }
    }
    (best, criterion, impurity)
}

pub(crate) fn sampled_min_size(sampled_rows: usize, min_node_size: usize) -> usize {
    ((sampled_rows as f32 * 0.05) as usize)
        .max(min_node_size / 3)
        .max(1)
}

fn node_mean(y: ArrayView1<'_, f32>, node: NodeRows<'_>) -> f32 {
    node.rows[node.start..node.start + node.n_rows]
        .iter()
        .map(|&row| y[row as usize])
        .sum::<f32>()
        / node.n_rows as f32
}

pub(crate) fn root_impurity(y: ArrayView1<'_, f32>, rows: &[u32]) -> f32 {
    let (sum, sum_squares) = rows.iter().fold((0.0, 0.0), |(sum, squares), &row| {
        let value = y[row as usize];
        (sum + value, squares + value * value)
    });
    -loss(sum, sum_squares, rows.len())
}

fn all_same(y: ArrayView1<'_, f32>, node: NodeRows<'_>, max_samples: usize) -> bool {
    let end = node.start + node.n_rows.min(max_samples);
    let first = y[node.rows[node.start] as usize];
    node.rows[node.start + 1..end]
        .iter()
        .all(|&row| (y[row as usize] - first).abs() <= 1.0e-8 + 1.0e-5 * first.abs())
}

pub(crate) fn partition(
    x: ArrayView2<'_, u32>,
    rows: &mut [u32],
    start: usize,
    n_rows: usize,
    cut_col: usize,
    cut_val: u32,
) -> usize {
    let mut left = start;
    let mut right = start + n_rows;
    if let Some(data) = x.as_slice() {
        let cols = x.ncols();
        while left < right {
            if data[rows[left] as usize * cols + cut_col] < cut_val {
                left += 1;
            } else {
                right -= 1;
                rows.swap(left, right);
            }
        }
        return left - start;
    }
    while left < right {
        if x[[rows[left] as usize, cut_col]] < cut_val {
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

#[cfg(test)]
pub(crate) fn test_weighted_loss(
    left_target: f32,
    left_sqr_target: f32,
    left_count: usize,
    sum_target: f32,
    sum_sqr_target: f32,
    total_count: usize,
) -> f32 {
    weighted_loss(
        left_target,
        left_sqr_target,
        left_count,
        sum_target,
        sum_sqr_target,
        total_count,
    )
}

#[cfg(test)]
pub(crate) fn test_loss(sum_target: f32, sum_sqr_target: f32, n: usize) -> f32 {
    loss(sum_target, sum_sqr_target, n)
}
