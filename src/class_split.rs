use ndarray::{ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use std::collections::HashSet;

use crate::Config;
use crate::split::{
    DenseFeature, NodeRows, RankedRow, dense_layout, evaluation_rows, evaluation_start, fill_dense_bins, propose_cutoffs, ranked_rows,
    sample_features, supported_equality, valid_children,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ClassSplit {
    pub cut_col: Option<usize>,
    pub cut_val: u32,
    pub equality: bool,
    pub missing_right: bool,
    pub gain: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Candidate {
    left_count: usize,
    left_mass: f64,
    child_class_score: f64,
    cut_col: usize,
    cut_val: u32,
    equality: bool,
    missing_right: bool,
}

#[derive(Default)]
pub(crate) struct ClassSplitScratch {
    candidates: Vec<Candidate>,
    candidate_classes: Vec<u32>,
    candidate_missing_classes: Vec<u32>,
    total_classes: Vec<u32>,
    left_classes: Vec<u32>,
    equal_classes: Vec<u32>,
    missing_classes: Vec<u32>,
    keys: HashSet<u64>,
    ranked_rows: Vec<RankedRow>,
    bin_classes: Vec<u32>,
    dense_features: Vec<DenseFeature>,
    dense_lookup: Vec<usize>,
    class_score_table: Vec<f64>,
    class_weights: Vec<f64>,
    score_stride: usize,
}

struct EvaluationWindow {
    start: usize,
    n_rows: usize,
    total_mass: f64,
    total_class_score: f64,
}

impl ClassSplitScratch {
    pub(crate) fn new(tree_classes: &[u32], max_samples: usize, weight_power: f32) -> Self {
        let score_stride = max_samples + 1;
        let total = tree_classes.iter().map(|&count| count as usize).sum::<usize>() as f64;
        let raw_weights: Vec<_> =
            tree_classes.iter().map(|&count| if count == 0 { 0.0 } else { (count as f64 / total).powf(-(weight_power as f64)) }).collect();
        let normalizer = tree_classes.iter().zip(&raw_weights).map(|(&count, &weight)| count as f64 / total * weight).sum::<f64>();
        let class_weights: Vec<_> = raw_weights.into_iter().map(|weight| weight / normalizer).collect();
        let mut class_score_table = Vec::with_capacity(tree_classes.len() * score_stride);
        for &weight in &class_weights {
            class_score_table.extend((0..=max_samples).map(|count| {
                let mass = count as f64 * weight;
                if mass == 0.0 { 0.0 } else { mass * mass.ln() }
            }));
        }
        Self { class_score_table, class_weights, score_stride, ..Self::default() }
    }

    fn class_score(&self, class: usize, count: u32) -> f64 {
        self.class_score_table[class * self.score_stride + count as usize]
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn find_class_split(
    x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, node: NodeRows<'_>, n_classes: usize, config: &Config, cutoff_offsets: &[usize],
    missing_ranks: &[u32], rng: &mut StdRng, scratch: &mut ClassSplitScratch,
) -> ClassSplit {
    let max_samples = evaluation_rows(node.n_rows, config);
    if x.ncols() == 0 || node.n_rows < config.min_node_size || all_same(y, node, max_samples) {
        return leaf();
    }
    if config.random_splitter {
        random_split(x, y, node, n_classes, config, missing_ranks, rng, scratch)
    } else {
        histogram_split(x, y, node, n_classes, config, cutoff_offsets, missing_ranks, rng, scratch)
    }
}

fn leaf() -> ClassSplit {
    ClassSplit { cut_col: None, cut_val: 0, equality: false, missing_right: false, gain: 0.0 }
}

fn evaluation_window(
    y: ArrayView1<'_, u32>, node: NodeRows<'_>, n_classes: usize, max_samples: usize, rng: &mut StdRng, scratch: &mut ClassSplitScratch,
) -> EvaluationWindow {
    let n_rows = node.n_rows.min(max_samples);
    let start = evaluation_start(node, n_rows, rng);
    scratch.total_classes.clear();
    scratch.total_classes.resize(n_classes, 0);
    for &row in &node.rows[start..start + n_rows] {
        scratch.total_classes[y[row as usize] as usize] += 1;
    }
    let total_class_score = scratch.total_classes.iter().enumerate().map(|(class, &count)| scratch.class_score(class, count)).sum();
    let total_mass = scratch.total_classes.iter().zip(&scratch.class_weights).map(|(&count, &weight)| count as f64 * weight).sum();
    EvaluationWindow { start, n_rows, total_mass, total_class_score }
}

fn move_class_score(table: &[f64], stride: usize, score: &mut f64, class: usize, left: u32, total: u32, added: u32) {
    let at = |count: u32| table[class * stride + count as usize];
    *score += at(left + added) - at(left) + at(total - left - added) - at(total - left);
}

#[allow(clippy::too_many_arguments)]
fn random_split(
    x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, node: NodeRows<'_>, n_classes: usize, config: &Config, missing_ranks: &[u32],
    rng: &mut StdRng, scratch: &mut ClassSplitScratch,
) -> ClassSplit {
    let used_n = evaluation_rows(node.n_rows, config);
    let features = sample_features(x.ncols(), config, rng);
    propose_candidates(x, node, used_n, &features, config.cutoff_divisor, rng, scratch);
    let window = evaluation_window(y, node, n_classes, used_n, rng, scratch);
    let (score_table, score_stride, total_classes, class_weights) =
        (&scratch.class_score_table, scratch.score_stride, &scratch.total_classes, &scratch.class_weights);
    scratch.candidates.iter_mut().for_each(|candidate| candidate.child_class_score = window.total_class_score);
    scratch.candidate_classes.clear();
    scratch.candidate_classes.resize(scratch.candidates.len() * n_classes, 0);
    scratch.candidate_missing_classes.clear();
    scratch.candidate_missing_classes.resize(scratch.candidates.len() * n_classes, 0);
    if let Some(data) = x.as_slice() {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let class = y[row] as usize;
            let offset = row * x.ncols();
            for (candidate_idx, candidate) in scratch.candidates.iter_mut().enumerate() {
                let value = data[offset + candidate.cut_col];
                if value == missing_ranks[candidate.cut_col] {
                    scratch.candidate_missing_classes[candidate_idx * n_classes + class] += 1;
                } else if value < candidate.cut_val {
                    let count = &mut scratch.candidate_classes[candidate_idx * n_classes + class];
                    move_class_score(score_table, score_stride, &mut candidate.child_class_score, class, *count, total_classes[class], 1);
                    *count += 1;
                    candidate.left_count += 1;
                    candidate.left_mass += class_weights[class];
                }
            }
        }
    } else {
        for &row in &node.rows[window.start..window.start + window.n_rows] {
            let row = row as usize;
            let class = y[row] as usize;
            for (candidate_idx, candidate) in scratch.candidates.iter_mut().enumerate() {
                let value = x[[row, candidate.cut_col]];
                if value == missing_ranks[candidate.cut_col] {
                    scratch.candidate_missing_classes[candidate_idx * n_classes + class] += 1;
                } else if value < candidate.cut_val {
                    let count = &mut scratch.candidate_classes[candidate_idx * n_classes + class];
                    move_class_score(score_table, score_stride, &mut candidate.child_class_score, class, *count, total_classes[class], 1);
                    *count += 1;
                    candidate.left_count += 1;
                    candidate.left_mass += class_weights[class];
                }
            }
        }
    }
    select_best(scratch, &window, n_classes)
}

fn propose_candidates(
    x: ArrayView2<'_, u32>, node: NodeRows<'_>, used_n: usize, features: &[usize], divisor: f32, rng: &mut StdRng,
    scratch: &mut ClassSplitScratch,
) {
    scratch.candidates.clear();
    let (keys, candidates) = (&mut scratch.keys, &mut scratch.candidates);
    propose_cutoffs(x, node, used_n, features, divisor, rng, keys, |cut_col, cut_val| {
        candidates.push(Candidate { cut_col, cut_val, ..Candidate::default() })
    });
}

#[allow(clippy::too_many_arguments)]
fn histogram_split(
    x: ArrayView2<'_, u32>, y: ArrayView1<'_, u32>, node: NodeRows<'_>, n_classes: usize, config: &Config, cutoff_offsets: &[usize],
    missing_ranks: &[u32], rng: &mut StdRng, scratch: &mut ClassSplitScratch,
) -> ClassSplit {
    let features = sample_features(x.ncols(), config, rng);
    let window = evaluation_window(y, node, n_classes, evaluation_rows(node.n_rows, config), rng, scratch);
    let (score_table, score_stride) = (&scratch.class_score_table, scratch.score_stride);
    let impurity = log_score(0.0, window.total_class_score, &window);
    let mut criterion = impurity;
    let mut best = None;
    let sort_work = window.n_rows * (window.n_rows.ilog2() as usize + 1);
    let total_bins =
        dense_layout(&features, x.ncols(), cutoff_offsets, &mut scratch.dense_features, &mut scratch.dense_lookup, |cardinality| {
            cardinality <= window.n_rows && cardinality * n_classes <= sort_work
        });
    scratch.bin_classes.clear();
    scratch.bin_classes.resize(total_bins * n_classes, 0);
    let (dense_features, bin_classes) = (&scratch.dense_features, &mut scratch.bin_classes);
    fill_dense_bins(x, node, window.start, window.n_rows, dense_features, |bin, row| {
        bin_classes[bin * n_classes + y[row] as usize] += 1;
    });
    for cut_col in features {
        let dense = scratch.dense_lookup[cut_col];
        if dense != usize::MAX {
            let DenseFeature { cardinality, offset, .. } = scratch.dense_features[dense];
            let has_missing = missing_ranks[cut_col] != u32::MAX;
            let observed_cardinality = if has_missing { missing_ranks[cut_col] as usize } else { cardinality };
            let missing_classes: &[u32] = if has_missing {
                let start = (offset + observed_cardinality) * n_classes;
                let counts = &scratch.bin_classes[start..start + n_classes];
                if counts.iter().any(|&count| count > 0) { counts } else { &[] }
            } else {
                &[]
            };
            scratch.left_classes.clear();
            scratch.left_classes.resize(n_classes, 0);
            let (mut left_count, mut left_mass, mut child_class_score) = (0, 0.0, window.total_class_score);
            let end = observed_cardinality + usize::from(has_missing);
            for cut_val in 1..end {
                if cut_val + 1 < observed_cardinality {
                    let start = (offset + cut_val) * n_classes;
                    let counts = &scratch.bin_classes[start..start + n_classes];
                    let left_count = counts.iter().map(|&count| count as usize).sum::<usize>();
                    let left_mass = counts.iter().zip(&scratch.class_weights).map(|(&count, &weight)| count as f64 * weight).sum();
                    if valid_children(left_count, window.n_rows) {
                        let mut equality_score = window.total_class_score;
                        for (class, &count) in counts.iter().enumerate() {
                            move_class_score(score_table, score_stride, &mut equality_score, class, 0, scratch.total_classes[class], count);
                        }
                        let score = log_score(left_mass, equality_score, &window);
                        if score > criterion {
                            criterion = score;
                            best = Some(Candidate {
                                left_count,
                                left_mass,
                                child_class_score: equality_score,
                                cut_col,
                                cut_val: cut_val as u32,
                                equality: true,
                                missing_right: !missing_classes.is_empty() || window.n_rows - left_count >= left_count,
                            });
                        }
                    }
                }
                let start = (offset + cut_val - 1) * n_classes;
                let counts = &scratch.bin_classes[start..start + n_classes];
                for (class, &added) in counts.iter().enumerate() {
                    move_class_score(
                        score_table,
                        score_stride,
                        &mut child_class_score,
                        class,
                        scratch.left_classes[class],
                        scratch.total_classes[class],
                        added,
                    );
                    scratch.left_classes[class] += added;
                    left_count += added as usize;
                    left_mass += added as f64 * scratch.class_weights[class];
                }
                let next = (offset + cut_val) * n_classes;
                if scratch.bin_classes[next..next + n_classes].iter().all(|&count| count == 0) {
                    continue;
                }
                if !valid_children(left_count, window.n_rows) {
                    continue;
                }
                let score = log_score(left_mass, child_class_score, &window);
                if score > criterion {
                    criterion = score;
                    best = Some(Candidate {
                        left_count,
                        left_mass,
                        child_class_score,
                        cut_col,
                        cut_val: cut_val as u32,
                        equality: false,
                        missing_right: !missing_classes.is_empty() || window.n_rows - left_count >= left_count,
                    });
                }
            }
            if !missing_classes.is_empty() {
                scratch.left_classes.clear();
                scratch.left_classes.extend_from_slice(missing_classes);
                let mut left_count = missing_classes.iter().map(|&count| count as usize).sum::<usize>();
                let mut left_mass =
                    missing_classes.iter().zip(&scratch.class_weights).map(|(&count, &weight)| count as f64 * weight).sum::<f64>();
                let mut child_class_score = window.total_class_score;
                for (class, &count) in missing_classes.iter().enumerate() {
                    move_class_score(score_table, score_stride, &mut child_class_score, class, 0, scratch.total_classes[class], count);
                }
                for cut_val in 1..observed_cardinality {
                    let start = (offset + cut_val - 1) * n_classes;
                    let counts = &scratch.bin_classes[start..start + n_classes];
                    for (class, &added) in counts.iter().enumerate() {
                        move_class_score(
                            score_table,
                            score_stride,
                            &mut child_class_score,
                            class,
                            scratch.left_classes[class],
                            scratch.total_classes[class],
                            added,
                        );
                        scratch.left_classes[class] += added;
                        left_count += added as usize;
                        left_mass += added as f64 * scratch.class_weights[class];
                    }
                    let next = (offset + cut_val) * n_classes;
                    if scratch.bin_classes[next..next + n_classes].iter().all(|&count| count == 0)
                        || !valid_children(left_count, window.n_rows)
                    {
                        continue;
                    }
                    let score = log_score(left_mass, child_class_score, &window);
                    if score > criterion {
                        criterion = score;
                        best = Some(Candidate {
                            left_count,
                            left_mass,
                            child_class_score,
                            cut_col,
                            cut_val: cut_val as u32,
                            equality: false,
                            missing_right: false,
                        });
                    }
                }
            }
            continue;
        }
        ranked_rows(x, node, window.start, window.n_rows, cut_col, &mut scratch.ranked_rows);
        let missing_rank = missing_ranks[cut_col];
        let observed_end = if missing_rank == u32::MAX {
            scratch.ranked_rows.len()
        } else {
            scratch.ranked_rows.partition_point(|row| row.value != missing_rank)
        };
        scratch.missing_classes.clear();
        if missing_rank != u32::MAX {
            scratch.missing_classes.resize(n_classes, 0);
            for row in &scratch.ranked_rows[observed_end..] {
                scratch.missing_classes[y[row.row] as usize] += 1;
            }
            if scratch.missing_classes.iter().all(|&count| count == 0) {
                scratch.missing_classes.clear();
            }
        }
        scratch.left_classes.clear();
        scratch.left_classes.resize(n_classes, 0);
        scratch.equal_classes.clear();
        scratch.equal_classes.resize(n_classes, 0);
        let (mut left_count, mut left_mass, mut child_class_score, mut position) = (0, 0.0, window.total_class_score, 0);
        while position < observed_end {
            let value = scratch.ranked_rows[position].value;
            scratch.equal_classes.fill(0);
            let mut equal_count = 0;
            while position < observed_end && scratch.ranked_rows[position].value == value {
                let class = y[scratch.ranked_rows[position].row] as usize;
                scratch.equal_classes[class] += 1;
                equal_count += 1;
                let count = &mut scratch.left_classes[class];
                move_class_score(score_table, score_stride, &mut child_class_score, class, *count, scratch.total_classes[class], 1);
                *count += 1;
                left_count += 1;
                left_mass += scratch.class_weights[class];
                position += 1;
            }
            let observed_cardinality =
                if missing_rank == u32::MAX { cutoff_offsets[cut_col + 1] - cutoff_offsets[cut_col] } else { missing_rank as usize };
            if supported_equality(value, observed_cardinality, equal_count, window.n_rows) {
                let mut equality_score = window.total_class_score;
                let equality_mass =
                    scratch.equal_classes.iter().zip(&scratch.class_weights).map(|(&count, &weight)| count as f64 * weight).sum();
                for (class, &count) in scratch.equal_classes.iter().enumerate() {
                    move_class_score(score_table, score_stride, &mut equality_score, class, 0, scratch.total_classes[class], count);
                }
                let score = log_score(equality_mass, equality_score, &window);
                if score > criterion {
                    criterion = score;
                    best = Some(Candidate {
                        left_count: equal_count,
                        left_mass: equality_mass,
                        child_class_score: equality_score,
                        cut_col,
                        cut_val: value,
                        equality: true,
                        missing_right: !scratch.missing_classes.is_empty() || window.n_rows - equal_count >= equal_count,
                    });
                }
            }
            if position == observed_end || !valid_children(left_count, window.n_rows) {
                continue;
            }
            let score = log_score(left_mass, child_class_score, &window);
            if score > criterion {
                criterion = score;
                best = Some(Candidate {
                    left_count,
                    left_mass,
                    child_class_score,
                    cut_col,
                    cut_val: scratch.ranked_rows[position].value,
                    equality: false,
                    missing_right: !scratch.missing_classes.is_empty() || window.n_rows - left_count >= left_count,
                });
            }
        }
        if !scratch.missing_classes.is_empty() {
            scratch.left_classes.clone_from(&scratch.missing_classes);
            let mut left_count = scratch.missing_classes.iter().map(|&count| count as usize).sum::<usize>();
            let mut left_mass =
                scratch.missing_classes.iter().zip(&scratch.class_weights).map(|(&count, &weight)| count as f64 * weight).sum::<f64>();
            let mut child_class_score = window.total_class_score;
            for (class, &count) in scratch.missing_classes.iter().enumerate() {
                move_class_score(score_table, score_stride, &mut child_class_score, class, 0, scratch.total_classes[class], count);
            }
            let mut position = 0;
            while position < observed_end {
                let value = scratch.ranked_rows[position].value;
                while position < observed_end && scratch.ranked_rows[position].value == value {
                    let class = y[scratch.ranked_rows[position].row] as usize;
                    let count = &mut scratch.left_classes[class];
                    move_class_score(score_table, score_stride, &mut child_class_score, class, *count, scratch.total_classes[class], 1);
                    *count += 1;
                    left_count += 1;
                    left_mass += scratch.class_weights[class];
                    position += 1;
                }
                if position == observed_end || !valid_children(left_count, window.n_rows) {
                    continue;
                }
                let score = log_score(left_mass, child_class_score, &window);
                if score > criterion {
                    criterion = score;
                    best = Some(Candidate {
                        left_count,
                        left_mass,
                        child_class_score,
                        cut_col,
                        cut_val: scratch.ranked_rows[position].value,
                        equality: false,
                        missing_right: false,
                    });
                }
            }
        }
    }
    finish(best, criterion, impurity)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn consider_candidate(
    mut candidate: Candidate, left_classes: &[u32], missing_classes: &[u32], window: &EvaluationWindow, score_table: &[f64],
    score_stride: usize, total_classes: &[u32], class_weights: &[f64], criterion: &mut f32, best: &mut Option<Candidate>,
) {
    candidate.missing_right = window.n_rows - candidate.left_count >= candidate.left_count;
    if valid_children(candidate.left_count, window.n_rows) {
        let score = log_score(candidate.left_mass, candidate.child_class_score, window);
        if score > *criterion {
            *criterion = score;
            *best = Some(candidate);
        }
    }
    if missing_classes.is_empty() {
        return;
    }
    let missing_count = missing_classes.iter().map(|&count| count as usize).sum::<usize>();
    if missing_count == 0 {
        return;
    }
    candidate.missing_right = true;
    if candidate.equality {
        return;
    }
    let mut left_score = candidate.child_class_score;
    for (class, (&left, &added)) in left_classes.iter().zip(missing_classes).enumerate() {
        move_class_score(score_table, score_stride, &mut left_score, class, left, total_classes[class], added);
    }
    let left_count = candidate.left_count + missing_count;
    if !valid_children(left_count, window.n_rows) {
        return;
    }
    let missing_mass = missing_classes.iter().zip(class_weights).map(|(&count, &weight)| count as f64 * weight).sum::<f64>();
    let score = log_score(candidate.left_mass + missing_mass, left_score, window);
    if score > *criterion {
        *criterion = score;
        candidate.missing_right = false;
        *best = Some(candidate);
    }
}

fn select_best(scratch: &ClassSplitScratch, window: &EvaluationWindow, n_classes: usize) -> ClassSplit {
    let impurity = log_score(0.0, window.total_class_score, window);
    let (mut best, mut criterion) = (None, impurity);
    for (index, &candidate) in scratch.candidates.iter().enumerate() {
        let start = index * n_classes;
        consider_candidate(
            candidate,
            &scratch.candidate_classes[start..start + n_classes],
            &scratch.candidate_missing_classes[start..start + n_classes],
            window,
            &scratch.class_score_table,
            scratch.score_stride,
            &scratch.total_classes,
            &scratch.class_weights,
            &mut criterion,
            &mut best,
        );
    }
    finish(best, criterion, impurity)
}

fn finish(best: Option<Candidate>, criterion: f32, impurity: f32) -> ClassSplit {
    let gain = (criterion - impurity).max(0.0);
    let best = best.filter(|_| gain > 0.0);
    ClassSplit {
        cut_col: best.map(|candidate| candidate.cut_col),
        cut_val: best.map_or(0, |candidate| candidate.cut_val),
        equality: best.is_some_and(|candidate| candidate.equality),
        missing_right: best.is_some_and(|candidate| candidate.missing_right),
        gain: if best.is_some() { gain } else { 0.0 },
    }
}

fn log_score(left_mass: f64, child_class_score: f64, window: &EvaluationWindow) -> f32 {
    let right_mass = window.total_mass - left_mass;
    let size_score = |mass: f64| if mass == 0.0 { 0.0 } else { mass * mass.ln() };
    ((child_class_score - size_score(left_mass) - size_score(right_mass)) / window.total_mass) as f32
}

fn all_same(y: ArrayView1<'_, u32>, node: NodeRows<'_>, max_samples: usize) -> bool {
    let end = node.start + node.n_rows.min(max_samples);
    let first = y[node.rows[node.start] as usize];
    node.rows[node.start + 1..end].iter().all(|&row| y[row as usize] == first)
}
