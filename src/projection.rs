use ndarray::ArrayView2;

use crate::ForestError;

#[derive(Clone, Debug, Default)]
pub struct Projections {
    pub phys: Vec<u32>,
    pub base: Vec<u32>,
    pub tables: Vec<u32>,
    pub offsets: Vec<usize>,
}

impl Projections {
    pub fn direct(n_features: usize) -> Self {
        Self { phys: (0..n_features as u32).collect(), base: vec![0; n_features], tables: Vec::new(), offsets: vec![0; n_features + 1] }
    }

    pub fn n_features(&self) -> usize {
        self.phys.len()
    }

    pub(crate) fn push_direct(&mut self, column: u32) {
        self.phys.push(column);
        self.base.push(0);
        self.offsets.push(self.tables.len());
    }

    pub(crate) fn push_projected(&mut self, base: u32, table: &[u32]) {
        self.phys.push(u32::MAX);
        self.base.push(base);
        self.tables.extend_from_slice(table);
        self.offsets.push(self.tables.len());
    }

    pub(crate) fn validate(&self, columns: usize) -> Result<(), ForestError> {
        let n = self.phys.len();
        if self.base.len() != n || self.offsets.len() != n + 1 || self.offsets.first() != Some(&0) {
            return Err(ForestError::new("projection dimensions are invalid"));
        }
        for feature in 0..n {
            let table = self.offsets[feature + 1] - self.offsets[feature];
            if self.phys[feature] == u32::MAX {
                if table == 0 || self.base[feature] as usize >= columns {
                    return Err(ForestError::new("projection base is invalid"));
                }
            } else if self.phys[feature] as usize >= columns || table != 0 {
                return Err(ForestError::new("projection column is invalid"));
            }
        }
        if self.offsets.last() != Some(&self.tables.len()) || self.offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(ForestError::new("projection tables are invalid"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TrainingData<'a> {
    pub matrix: ArrayView2<'a, u32>,
    pub projections: &'a Projections,
}

impl<'a> TrainingData<'a> {
    pub(crate) fn n_features(&self) -> usize {
        self.projections.n_features()
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.matrix.nrows()
    }

    #[inline]
    pub(crate) fn source(&self, feature: usize) -> (usize, &'a [u32]) {
        let phys = self.projections.phys[feature];
        if phys == u32::MAX {
            let table = &self.projections.tables[self.projections.offsets[feature]..self.projections.offsets[feature + 1]];
            (self.projections.base[feature] as usize, table)
        } else {
            (phys as usize, &[])
        }
    }

    #[inline]
    pub(crate) fn value(&self, row: usize, feature: usize) -> u32 {
        let (column, table) = self.source(feature);
        let raw = self.matrix[[row, column]];
        if table.is_empty() { raw } else { table[raw as usize] }
    }
}
