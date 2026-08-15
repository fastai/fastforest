use crate::ForestError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Splitter {
    Random,
    Histogram,
}

impl Default for Splitter {
    fn default() -> Self {
        Self::Histogram
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxFeatures {
    Sqrt,
    All,
    Fraction(f32),
    Count(usize),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum FeatureSampling {
    #[default]
    Encoded,
    Columns,
}

impl Default for MaxFeatures {
    fn default() -> Self {
        Self::Fraction(0.75)
    }
}

impl MaxFeatures {
    pub(crate) fn resolve(self, total: usize) -> usize {
        let selected = match self {
            Self::Sqrt => (total as f64).sqrt() as usize,
            Self::All => total,
            Self::Fraction(fraction) => (total as f32 * fraction) as usize,
            Self::Count(count) => count,
        };
        selected.clamp(1, total)
    }

    fn validate(self) -> Result<(), ForestError> {
        match self {
            Self::Fraction(fraction)
                if !(fraction.is_finite() && 0.0 < fraction && fraction <= 1.0) =>
            {
                Err(ForestError::new(
                    "max_features fraction must be finite and in (0, 1]",
                ))
            }
            Self::Count(0) => Err(ForestError::new(
                "max_features count must be greater than zero",
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Workbench {
    pub splitter: Splitter,
    pub max_features: MaxFeatures,
    pub leaf_regularization: f32,
    pub feature_sampling: FeatureSampling,
}

impl Default for Workbench {
    fn default() -> Self {
        Self {
            splitter: Splitter::default(),
            max_features: MaxFeatures::default(),
            leaf_regularization: 0.0,
            feature_sampling: FeatureSampling::default(),
        }
    }
}

impl Workbench {
    pub(crate) fn validate(&self) -> Result<(), ForestError> {
        self.max_features.validate()?;
        if !self.leaf_regularization.is_finite() || self.leaf_regularization < 0.0 {
            return Err(ForestError::new(
                "leaf_regularization must be finite and non-negative",
            ));
        }
        Ok(())
    }

    pub(crate) fn leaf_value(&self, mean: f32, rows: usize, parent_mean: f32) -> f32 {
        let regularization = self.leaf_regularization;
        (mean * rows as f32 + parent_mean * regularization) / (rows as f32 + regularization)
    }
}
