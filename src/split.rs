use ndarray::{ArrayView1, ArrayView2};
use rand::RngExt;
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::{Config, Splitter};

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

#[derive(Default)]
pub(crate) struct SplitScratch {
    candidates: Vec<Candidate>,
    keys: HashSet<u64>,
    bins: Vec<Bin>,
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
    feature_groups: Option<&[Vec<usize>]>,
    rng: &mut StdRng,
    scratch: &mut SplitScratch,
) -> Split {
    if x.ncols() == 0
        || node.n_rows < config.min_node_size
        || all_same(y, node, config.max_node_samples)
    {
        return leaf(y, node);
    }
    match config.workbench.splitter {
        Splitter::Random => random_split(x, y, node, config, rng, scratch),
        Splitter::Histogram => histogram_split(x, y, node, config, feature_groups, rng, scratch),
    }
}

fn finish_split(
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    window: &EvaluationWindow,
    best: Option<Candidate>,
    criterion: f32,
    impurity: f32,
) -> Split {
    Split {
        cut_col: best.map(|candidate| candidate.cut_col),
        cut_val: best.map_or(0, |candidate| candidate.cut_val),
        value: if best.is_none() && window.n_rows < node.n_rows {
            node_mean(y, node)
        } else {
            window.sum_target / window.n_rows as f32
        },
        gain: (criterion - impurity).max(0.0),
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

#[allow(clippy::too_many_arguments)]
fn random_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    config: &Config,
    rng: &mut StdRng,
    scratch: &mut SplitScratch,
) -> Split {
    let used_n = node.n_rows.min(config.max_node_samples);
    let candidate_rows = used_n.max(config.min_candidate_rows) as f64;
    let n_candidates =
        ((candidate_rows * (x.ncols() as f64).sqrt()) / config.cutoff_divisor as f64) as usize;
    let target_candidates = n_candidates
        .max(4)
        .min(node.n_rows.saturating_mul(x.ncols()));
    scratch.candidates.clear();
    scratch.keys.clear();
    for _ in 0..target_candidates.saturating_mul(config.candidate_attempt_factor) {
        let cut_col = rng.random_range(0..x.ncols());
        let split_pos = rng.random_range(node.start..node.start + node.n_rows);
        let cut_val = x[[node.rows[split_pos] as usize, cut_col]];
        let key = (cut_col as u64) << 32 | cut_val as u64;
        if scratch.keys.insert(key) {
            scratch.candidates.push(Candidate {
                cut_col,
                cut_val,
                ..Candidate::default()
            });
            if scratch.candidates.len() == target_candidates {
                break;
            }
        }
    }
    let window = evaluation_window(y, node, config.max_node_samples, rng);
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
    let (best, criterion, impurity) = select_best(
        scratch.candidates.iter().copied(),
        &window,
        config.min_node_size,
    );
    finish_split(y, node, &window, best, criterion, impurity)
}

fn histogram_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, f32>,
    node: NodeRows<'_>,
    config: &Config,
    feature_groups: Option<&[Vec<usize>]>,
    rng: &mut StdRng,
    scratch: &mut SplitScratch,
) -> Split {
    let features = sample_features(x.ncols(), config, feature_groups, rng);
    let window = evaluation_window(y, node, config.max_node_samples, rng);
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
    for cut_col in features {
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
    finish_split(y, node, &window, best, criterion, impurity)
}

fn sample_features(
    n_features: usize,
    config: &Config,
    feature_groups: Option<&[Vec<usize>]>,
    rng: &mut StdRng,
) -> Vec<usize> {
    if let Some(groups) = feature_groups {
        let selected = config.workbench.max_features.resolve(groups.len());
        rand::seq::index::sample(rng, groups.len(), selected)
            .into_iter()
            .flat_map(|group| groups[group].iter().copied())
            .collect()
    } else {
        let selected = config.workbench.max_features.resolve(n_features);
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

fn sampled_min_size(sampled_rows: usize, min_node_size: usize) -> usize {
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
