use ndarray::ArrayView2;
use rayon::prelude::*;

const PREDICTION_CACHE_BYTES: usize = 1 << 19;

pub(crate) trait PredictionTree: Sync {
    fn prediction_bytes(&self) -> usize;
    fn add_prediction_by(&self, value: impl Fn(usize) -> f32, output: &mut [f32]);
}

pub(crate) fn trees_per_batch<T: PredictionTree>(trees: &[T]) -> usize {
    let bytes: usize = trees.iter().map(PredictionTree::prediction_bytes).sum();
    let mean_bytes = bytes.div_ceil(trees.len()).max(1);
    (PREDICTION_CACHE_BYTES / mean_bytes).clamp(1, trees.len())
}

pub(crate) fn row_block_size(n_rows: usize) -> usize {
    n_rows.div_ceil(rayon::current_num_threads().saturating_mul(4).max(1)).clamp(1, 4_096)
}

#[inline]
pub(crate) fn add_block_by<T: PredictionTree>(
    trees: &[T], n_rows: usize, outputs: usize, result: &mut [f32], trees_per_batch: usize, value: impl Fn(usize, usize) -> f32,
) {
    result.fill(0.0);
    for trees in trees.chunks(trees_per_batch) {
        for row in 0..n_rows {
            let output = &mut result[row * outputs..(row + 1) * outputs];
            trees.iter().for_each(|tree| tree.add_prediction_by(|col| value(row, col), output));
        }
    }
}

pub(crate) fn predict_outputs<T: PredictionTree>(trees: &[T], n_features: usize, outputs: usize, x: ArrayView2<'_, f32>) -> Vec<f32> {
    let mut result = vec![0.0; x.nrows() * outputs];
    if x.nrows() == 0 {
        return result;
    }
    let block_rows = row_block_size(x.nrows());
    let output_block = block_rows * outputs;
    let trees_per_batch = trees_per_batch(trees);
    let n_trees = trees.len() as f32;
    if let Some(data) = x.as_slice() {
        result.par_chunks_mut(output_block).enumerate().for_each(|(block, output)| {
            let row_start = block * block_rows;
            let n_rows = output.len() / outputs;
            add_block_by(trees, n_rows, outputs, output, trees_per_batch, |row, col| data[(row_start + row) * n_features + col]);
            output.iter_mut().for_each(|value| *value /= n_trees);
        });
    } else {
        result.par_chunks_mut(output_block).enumerate().for_each(|(block, output)| {
            let row_start = block * block_rows;
            let n_rows = output.len() / outputs;
            add_block_by(trees, n_rows, outputs, output, trees_per_batch, |row, col| x[[row_start + row, col]]);
            output.iter_mut().for_each(|value| *value /= n_trees);
        });
    }
    result
}
