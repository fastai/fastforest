use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::{Config, ForestError};

pub(crate) fn tree_seeds(config: &Config) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
    (0..config.n_trees).map(|_| rng.random()).collect()
}

pub(crate) fn variant_groups(alias: &[u32], n_columns: usize) -> Result<(usize, Vec<Vec<u32>>), ForestError> {
    if alias.len() != n_columns {
        return Err(ForestError::new(format!("expected {n_columns} feature aliases, got {}", alias.len())));
    }
    let canonical = alias.iter().enumerate().take_while(|&(index, &target)| target as usize == index).count();
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for (index, &target) in alias.iter().enumerate().skip(canonical) {
        if target as usize >= canonical {
            return Err(ForestError::new("feature aliases must map extra columns to canonical features"));
        }
        match groups.iter_mut().find(|group| group[0] == target) {
            Some(group) => group.push(index as u32),
            None => groups.push(vec![target, index as u32]),
        }
    }
    if groups.windows(2).any(|pair| pair[0].len() != pair[1].len()) {
        return Err(ForestError::new("feature alias groups must share one variant count"));
    }
    Ok((canonical, groups))
}

pub(crate) fn tree_universe(canonical: usize, groups: &[Vec<u32>], rng: &mut StdRng) -> Vec<usize> {
    let mut universe: Vec<usize> = (0..canonical).collect();
    if let Some(width) = groups.first().map(Vec::len) {
        let variant = rng.random_range(0..width);
        for group in groups {
            universe[group[0] as usize] = group[variant] as usize;
        }
    }
    universe
}

pub(crate) fn fold_variant_importance(importance: &mut Vec<f32>, canonical: usize, groups: &[Vec<u32>]) {
    for group in groups {
        for &column in &group[1..] {
            importance[group[0] as usize] += importance[column as usize];
        }
    }
    importance.truncate(canonical);
}

pub(crate) fn add_importance(total: &mut [f32], importance: &[f32]) {
    total.iter_mut().zip(importance).for_each(|(total, value)| *total += value);
}

pub(crate) fn normalize_importance(importance: &mut [f32]) {
    let sum = importance.iter().sum::<f32>();
    if sum > 0.0 {
        importance.iter_mut().for_each(|value| *value /= sum)
    }
}

pub(crate) fn combined_importance(left: &[f32], right: &[f32], left_trees: usize, right_trees: usize) -> Vec<f32> {
    let total = (left_trees + right_trees) as f32;
    left.iter().zip(right).map(|(left, right)| (left * left_trees as f32 + right * right_trees as f32) / total).collect()
}

pub(crate) fn normalize_oob(values: &mut [f32], counts: &[u32], outputs: usize) {
    for (row, &count) in values.chunks_exact_mut(outputs).zip(counts) {
        if count == 0 { row.fill(f32::NAN) } else { row.iter_mut().for_each(|value| *value /= count as f32) }
    }
}

pub(crate) type BuiltTree<T> = (T, Option<Vec<bool>>, Vec<f32>);
pub(crate) type AssembledForest<T> = (Vec<T>, Vec<f32>, Option<Vec<f32>>, Option<Vec<u32>>, Option<Vec<usize>>);

pub(crate) fn assemble_forest<T, N>(
    built: Vec<BuiltTree<T>>, n_features: usize, outputs: usize, oob_indices: Option<Vec<usize>>, track_oob: bool,
    mut add_oob: impl FnMut(&T, usize, &mut [f32]), mut into_native: impl FnMut(T) -> N,
) -> AssembledForest<N> {
    let mut trees = Vec::with_capacity(built.len());
    let mut values = oob_indices.as_ref().map(|indices| vec![0.0; indices.len() * outputs]);
    let mut counts = oob_indices.as_ref().map(|indices| vec![0; indices.len()]);
    let mut importance = vec![0.0; n_features];
    for (tree, in_bag, tree_importance) in built {
        add_importance(&mut importance, &tree_importance);
        if track_oob {
            let in_bag = in_bag.as_ref().unwrap();
            for (output_idx, &row_idx) in oob_indices.as_ref().unwrap().iter().enumerate() {
                if in_bag[row_idx] {
                    continue;
                }
                add_oob(&tree, row_idx, &mut values.as_mut().unwrap()[output_idx * outputs..(output_idx + 1) * outputs]);
                counts.as_mut().unwrap()[output_idx] += 1;
            }
        }
        trees.push(into_native(tree));
    }
    normalize_importance(&mut importance);
    if let (Some(values), Some(counts)) = (&mut values, &counts) {
        normalize_oob(values, counts, outputs)
    }
    (trees, importance, values, counts, oob_indices)
}

pub(crate) fn combined_oob(
    left: Option<(&[f32], &[u32])>, right: Option<(&[f32], &[u32])>, outputs: usize,
) -> Result<(Option<Vec<f32>>, Option<Vec<u32>>), ForestError> {
    match (left, right) {
        (Some((left, left_counts)), Some((right, right_counts))) => {
            let counts: Vec<_> = left_counts.iter().zip(right_counts).map(|(a, b)| a + b).collect();
            let values = left
                .chunks_exact(outputs)
                .zip(left_counts)
                .zip(right.chunks_exact(outputs).zip(right_counts))
                .zip(&counts)
                .flat_map(|(((left, &lc), (right, &rc)), &count)| {
                    (0..outputs).map(move |output| {
                        if count == 0 {
                            f32::NAN
                        } else {
                            (if lc == 0 { 0.0 } else { left[output] * lc as f32 } + if rc == 0 { 0.0 } else { right[output] * rc as f32 })
                                / count as f32
                        }
                    })
                })
                .collect();
            Ok((Some(values), Some(counts)))
        }
        (None, None) => Ok((None, None)),
        _ => Err(ForestError::new("forests have incompatible OOB results")),
    }
}
