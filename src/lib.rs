//! Fast approximate random-forest regression.

mod forest;
mod preprocessing;
#[cfg(feature = "python")]
mod python;
mod split;
mod workbench;

pub use forest::{AdaptiveScore, Config, Forest, ForestError};
pub use preprocessing::{Column, Encoder, Encoding, RawColumn};
pub use workbench::{FeatureSampling, MaxFeatures, Splitter, Workbench};
