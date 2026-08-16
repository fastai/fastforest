//! Fast approximate-forest regression and classification.

mod class_split;
pub mod cli;
mod compiled;
mod classification;
mod forest;
mod file;
mod model;
mod preprocessing;
#[cfg(feature = "python")]
mod python;
mod split;

pub use classification::ClassifierForest;
pub use compiled::compile_model;
pub use forest::{Config, FitPlan, Forest, ForestError, MaxFeatures, plan_fit};
pub use file::{
    FileFitOptions, Task, convert_csv_to_arrow, fit_arrow, fit_csv, fit_file, predict_arrow,
    predict_csv, predict_file,
};
pub use model::{ModelMetadata, SavedModel, SavedValue};
pub use preprocessing::{Column, Encoder, Encoding};
