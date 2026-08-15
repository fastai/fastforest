//! Fast approximate-forest regression and classification.

mod class_split;
mod classification;
mod forest;
mod preprocessing;
#[cfg(feature = "python")]
mod python;
mod split;
mod workbench;

pub use classification::{ClassificationAdaptiveScore, ClassifierForest};
pub use forest::{AdaptiveScore, Config, Forest, ForestError};
pub use preprocessing::{Column, Encoder, Encoding, RawColumn};
pub use workbench::{FeatureSampling, MaxFeatures, Splitter, Workbench};
