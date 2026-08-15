use ndarray::{ArrayView1, ArrayView2};
use rand::RngExt;
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::split::{NodeRows, sample_features, sampled_min_size};
use crate::{Config, Splitter};

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

#[derive(Default)]
pub(crate) struct ClassSplitScratch {
    candidates: Vec<Candidate>,
    candidate_classes: Vec<u32>,
    total_classes: Vec<u32>,
    left_classes: Vec<u32>,
    keys: HashSet<u64>,
    bins: Vec<Bin>,
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
    feature_groups: Option<&[Vec<usize>]>,
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
) -> ClassSplit {
    if x.ncols() == 0
        || node.n_rows < config.min_node_size
        || all_same(y, node, config.max_node_samples)
    {
        return leaf();
    }
    match config.workbench.splitter {
        Splitter::Random => random_split(x, y, node, n_classes, config, rng, scratch),
        Splitter::Histogram => {
            histogram_split(x, y, node, n_classes, config, feature_groups, rng, scratch)
        }
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
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
) -> ClassSplit {
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
    let window = evaluation_window(y, node, n_classes, config.max_node_samples, rng, scratch);
    scratch.candidate_classes.clear();
    scratch
        .candidate_classes
        .resize(scratch.candidates.len() * n_classes, 0);
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
    )
}

#[allow(clippy::too_many_arguments)]
fn histogram_split(
    x: ArrayView2<'_, u32>,
    y: ArrayView1<'_, u32>,
    node: NodeRows<'_>,
    n_classes: usize,
    config: &Config,
    feature_groups: Option<&[Vec<usize>]>,
    rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
) -> ClassSplit {
    let features = sample_features(x.ncols(), config, feature_groups, rng);
    let window = evaluation_window(y, node, n_classes, config.max_node_samples, rng, scratch);
    let impurity = weighted_gini(0, 0, window.n_rows, window.total_squares);
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
    finish(best, criterion, impurity)
}

fn select_best(
    candidates: impl Iterator<Item = Candidate>,
    window: &EvaluationWindow,
    min_node_size: usize,
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
    finish(best, criterion, impurity)
}

fn finish(best: Option<Candidate>, criterion: f32, impurity: f32) -> ClassSplit {
    ClassSplit {
        cut_col: best.map(|candidate| candidate.cut_col),
        cut_val: best.map_or(0, |candidate| candidate.cut_val),
        gain: (criterion - impurity).max(0.0),
    }
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
