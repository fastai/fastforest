//! Fast approximate random-forest regression.

mod forest;
#[cfg(feature = "python")]
mod python;

pub use forest::{Config, Forest, ForestError};
