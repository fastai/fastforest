//! Fast approximate-forest regression and classification.

mod class_split;
mod classification;
pub mod cli;
mod compiled;
mod csv_view;
mod ensemble;
mod file;
mod forest;
mod model;
mod prediction;
mod preprocessing;
mod projection;
mod split;
mod tree;

pub use classification::ClassifierForest;
pub use compiled::compile_model;
pub use csv_view::{CsvSample, CsvViewOptions, view_csv};
pub use file::{FileFitOptions, Task, convert_csv_to_arrow, fit_arrow, fit_csv, fit_file, predict_arrow, predict_csv, predict_file};
pub use forest::{Config, FitPlan, Forest, ForestError, MaxFeatures, plan_fit, resolve_replacement, uniform_sample_indices};
pub use model::{ModelMetadata, SavedModel, SavedValue};
pub use preprocessing::{Column, Encoder, EncoderOptions, Encoding, StatContext, StatTarget, TrendFit, detect_dates};
pub use projection::Projections;
