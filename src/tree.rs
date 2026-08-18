use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use crate::split::{NodeRows, partition};

pub(crate) const LEAF_COL: u32 = u32::MAX;
pub(crate) const EQUALITY_BIT: u32 = 1 << 31;
pub(crate) const MISSING_RIGHT_BIT: u32 = 1 << 30;
pub(crate) const FEATURE_MASK: u32 = MISSING_RIGHT_BIT - 1;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Branch {
    pub(crate) cut_col: usize,
    pub(crate) cut_val: u32,
    pub(crate) equality: bool,
    pub(crate) missing_right: bool,
    pub(crate) gain: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct TreeNode<C, P> {
    pub(crate) cut_val: C,
    pub(crate) value: P,
    pub(crate) child: u32,
    pub(crate) cut_col: u32,
}

impl<C: Default, P: Default> TreeNode<C, P> {
    pub(crate) fn new() -> Self {
        Self { cut_val: C::default(), value: P::default(), child: 0, cut_col: LEAF_COL }
    }
}

impl<C, P> TreeNode<C, P> {
    pub(crate) fn is_leaf(&self) -> bool {
        self.cut_col == LEAF_COL
    }
    pub(crate) fn equality(&self) -> bool {
        !self.is_leaf() && self.cut_col & EQUALITY_BIT != 0
    }
    pub(crate) fn missing_right(&self) -> bool {
        !self.is_leaf() && self.cut_col & MISSING_RIGHT_BIT != 0
    }
    pub(crate) fn feature(&self) -> usize {
        (self.cut_col & FEATURE_MASK) as usize
    }
}

pub(crate) fn leaf_index<C: Copy + PartialEq, P>(
    nodes: &[TreeNode<C, P>], value: impl Fn(usize) -> C, missing: impl Fn(usize, C) -> bool, ordered_right: impl Fn(C, C) -> bool,
) -> usize {
    let mut node_idx = 0;
    loop {
        let node = &nodes[node_idx];
        if node.is_leaf() {
            return node_idx;
        }
        let observed = value(node.feature());
        let go_right = usize::from(if missing(node.feature(), observed) {
            node.missing_right()
        } else if node.equality() {
            observed != node.cut_val
        } else {
            ordered_right(observed, node.cut_val)
        });
        node_idx = node.child as usize + go_right;
    }
}

pub(crate) fn structure<C, P>(nodes: &[TreeNode<C, P>]) -> (usize, usize, usize) {
    let mut leaves = 0;
    let mut depth = 0;
    let mut stack = vec![(0, 0)];
    while let Some((index, node_depth)) = stack.pop() {
        let node = &nodes[index];
        depth = depth.max(node_depth);
        if node.is_leaf() {
            leaves += 1
        } else {
            stack.push((node.child as usize, node_depth + 1));
            stack.push((node.child as usize + 1, node_depth + 1));
        }
    }
    (nodes.len(), leaves, depth)
}

pub(crate) fn native_node<P>(node: TreeNode<u32, P>, cutoff_values: &[f32], cutoff_offsets: &[usize]) -> TreeNode<f32, P> {
    TreeNode {
        cut_val: if node.is_leaf() {
            0.0
        } else if node.equality() {
            cutoff_values[cutoff_offsets[node.feature()] + node.cut_val as usize + 1]
        } else {
            cutoff_values[cutoff_offsets[node.feature()] + node.cut_val as usize]
        },
        value: node.value,
        child: node.child,
        cut_col: node.cut_col,
    }
}

pub(crate) fn split_children<C: Copy, P: Default>(
    nodes: &mut Vec<TreeNode<C, P>>, node_idx: usize, cut_col: usize, cut_val: C, equality: bool, missing_right: bool,
) -> (usize, usize)
where
    C: Default,
{
    let left = nodes.len();
    nodes.push(TreeNode::new());
    nodes.push(TreeNode::new());
    nodes[node_idx].child = u32::try_from(left).expect("tree has too many nodes");
    nodes[node_idx].cut_col = u32::try_from(cut_col).expect("matrix has too many columns")
        | if equality { EQUALITY_BIT } else { 0 }
        | if missing_right { MISSING_RIGHT_BIT } else { 0 };
    nodes[node_idx].cut_val = cut_val;
    (left, left + 1)
}

pub(crate) fn grow_tree<P: Default>(
    x: ArrayView2<'_, u32>, rows: &mut [u32], nodes: &mut Vec<TreeNode<u32, P>>, importance: &mut [f32], missing_ranks: &[u32],
    mut visit: impl FnMut(NodeRows<'_>, &mut TreeNode<u32, P>) -> Option<Branch>,
) {
    let mut work = vec![(0, 0, rows.len())];
    while let Some((node_idx, start, n_rows)) = work.pop() {
        let branch = visit(NodeRows { rows, start, n_rows }, &mut nodes[node_idx]);
        let Some(branch) = branch else { continue };
        importance[branch.cut_col] += branch.gain * n_rows as f32;
        let left_n = partition(
            x,
            rows,
            start,
            n_rows,
            branch.cut_col,
            branch.cut_val,
            branch.equality,
            missing_ranks[branch.cut_col],
            branch.missing_right,
        );
        debug_assert!(left_n > 0 && left_n < n_rows);
        let (left_idx, right_idx) = split_children(nodes, node_idx, branch.cut_col, branch.cut_val, branch.equality, branch.missing_right);
        work.push((right_idx, start + left_n, n_rows - left_n));
        work.push((left_idx, start, left_n));
    }
}

const _: () = assert!(std::mem::size_of::<TreeNode<f32, f32>>() == 16);
const _: () = assert!(std::mem::size_of::<TreeNode<u32, f32>>() == 16);
const _: () = assert!(std::mem::size_of::<TreeNode<f32, u32>>() == 16);
const _: () = assert!(std::mem::size_of::<TreeNode<u32, u32>>() == 16);
