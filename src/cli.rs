use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::{
    CsvSample, CsvViewOptions, FileFitOptions, ForestError, MaxFeatures, SavedModel, SavedValue, Task, compile_model, convert_csv_to_arrow,
    fit_file, predict_file, view_csv,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TaskArg {
    Regression,
    Classification,
}

#[derive(Debug, Parser)]
#[command(version, about = "Fit a FastForest model from a tabular file")]
struct FitCommand {
    input: PathBuf,
    #[arg(long)]
    target: String,
    #[arg(long)]
    task: TaskArg,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    n_trees: Option<usize>,
    #[arg(long)]
    min_node_size: Option<usize>,
    #[arg(long)]
    bootstrap_fraction: Option<f32>,
    #[arg(long)]
    bootstrap_max: Option<String>,
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    replacement: Option<bool>,
    #[arg(long)]
    max_node_samples: Option<usize>,
    #[arg(long)]
    split_prior_rows: Option<f32>,
    #[arg(long)]
    class_weight_power: Option<f32>,
    #[arg(long)]
    cutoff_divisor: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    oob: bool,
    #[arg(long)]
    random_splitter: bool,
    #[arg(long)]
    max_features: Option<String>,
    #[arg(long)]
    allow_new_missing: bool,
    #[arg(long = "missing-value")]
    missing_values: Vec<String>,
    #[arg(long = "date-column")]
    date_columns: Vec<String>,
}

#[derive(Debug, Parser)]
#[command(version, about = "Stream predictions from a saved FastForest model")]
struct PredictCommand {
    model: PathBuf,
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 65_536)]
    batch_size: usize,
    #[arg(long)]
    proba: bool,
}

#[derive(Debug, Parser)]
#[command(version, about = "Convert a numeric CSV file to Arrow IPC")]
struct ConvertCommand {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 65_536)]
    batch_size: usize,
}

#[derive(Debug, Parser)]
#[command(version, about = "Display a CSV compactly with constants separated")]
struct ViewCommand {
    input: PathBuf,
    #[arg(long, value_delimiter = ',')]
    cols: Vec<String>,
    #[arg(long, conflicts_with = "sample")]
    rows: Option<usize>,
    #[arg(long, default_value_t = 0, conflicts_with = "sample")]
    start: usize,
    #[arg(long, conflicts_with = "sample")]
    end: Option<usize>,
    #[arg(long, conflicts_with = "rows")]
    sample: Option<CsvSample>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Debug, Parser)]
#[command(version, about = "Build a standalone predictor containing a saved FastForest model")]
struct CompileCommand {
    model: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Debug, Parser)]
#[command(version, about = "Stream predictions from an embedded FastForest model")]
struct EmbeddedPredictCommand {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 65_536)]
    batch_size: usize,
    #[arg(long)]
    proba: bool,
}

fn parse_bootstrap_max(value: &str) -> Result<Option<usize>, ForestError> {
    if value.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let value = value.parse::<usize>().map_err(|_| ForestError::new("bootstrap_max must be a positive integer or 'none'"))?;
    if value == 0 {
        return Err(ForestError::new("bootstrap_max must be greater than zero"));
    }
    Ok(Some(value))
}

fn parse_assignment(value: &str, option: &str) -> Result<(String, String), ForestError> {
    let (name, value) = value.split_once('=').ok_or_else(|| ForestError::new(format!("{option} must use COLUMN=VALUE")))?;
    if name.is_empty() {
        return Err(ForestError::new(format!("{option} column cannot be empty")));
    }
    Ok((name.to_owned(), value.to_owned()))
}

fn parse_max_features(value: &str) -> Result<MaxFeatures, ForestError> {
    match value {
        "sqrt" => Ok(MaxFeatures::Sqrt),
        value => value
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
            .map(MaxFeatures::Fraction)
            .ok_or_else(|| ForestError::new("max_features must be 'sqrt' or a fraction in (0, 1]")),
    }
}

fn parse_command<C, I, T>(args: I) -> Result<Option<C>, ForestError>
where
    C: Parser,
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match C::try_parse_from(args) {
        Ok(command) => Ok(Some(command)),
        Err(error) if matches!(error.kind(), clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion) => {
            print!("{error}");
            Ok(None)
        }
        Err(error) => Err(ForestError::new(error.to_string())),
    }
}

fn fit_options(command: FitCommand) -> Result<(PathBuf, PathBuf, FileFitOptions), ForestError> {
    let missing_values = command
        .missing_values
        .iter()
        .map(|value| parse_assignment(value, "missing-value").map(|(name, value)| (name, SavedValue { kind: 5, value })))
        .collect::<Result<_, _>>()?;
    let date_columns = command.date_columns.iter().map(|value| parse_assignment(value, "date-column")).collect::<Result<_, _>>()?;
    let task = match command.task {
        TaskArg::Regression => Task::Regression,
        TaskArg::Classification => Task::Classification,
    };
    let mut options = FileFitOptions::for_task(task);
    options.target = command.target;
    options.n_trees = command.n_trees;
    if let Some(value) = command.min_node_size {
        options.min_node_size = value
    }
    options.bootstrap_fraction = command.bootstrap_fraction;
    if let Some(value) = command.bootstrap_max.as_deref() {
        options.bootstrap_max = parse_bootstrap_max(value)?
    }
    options.replacement = command.replacement;
    if let Some(value) = command.max_node_samples {
        options.max_node_samples = value
    }
    if let Some(value) = command.split_prior_rows {
        options.split_prior_rows = value
    }
    if let Some(value) = command.class_weight_power {
        options.class_weight_power = value
    }
    if let Some(value) = command.cutoff_divisor {
        options.cutoff_divisor = value
    }
    options.seed = command.seed;
    options.oob = command.oob;
    options.random_splitter = command.random_splitter;
    if let Some(value) = command.max_features.as_deref() {
        options.max_features = parse_max_features(value)?
    }
    options.allow_new_missing = command.allow_new_missing;
    options.missing_values = missing_values;
    options.date_columns = date_columns;
    Ok((command.input, command.output, options))
}

pub fn run_fit<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<FitCommand, _, _>(args)? else {
        return Ok(());
    };
    let (input, output, options) = fit_options(command)?;
    fit_file(input, &options)?.save(output)
}

pub fn run_predict<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<PredictCommand, _, _>(args)? else {
        return Ok(());
    };
    let model = SavedModel::load(command.model)?;
    predict_file(&model, command.input, command.output, command.batch_size, command.proba)
}

pub fn run_convert<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<ConvertCommand, _, _>(args)? else {
        return Ok(());
    };
    convert_csv_to_arrow(command.input, command.output, command.batch_size)
}

pub fn run_view<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<ViewCommand, _, _>(args)? else {
        return Ok(());
    };
    print!(
        "{}",
        view_csv(
            command.input,
            &CsvViewOptions {
                columns: command.cols,
                rows: command.rows,
                start: command.start,
                end: command.end,
                sample: command.sample,
                seed: command.seed,
            }
        )?
    );
    Ok(())
}

pub fn run_compile<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<CompileCommand, _, _>(args)? else {
        return Ok(());
    };
    compile_model(&SavedModel::load(command.model)?, command.output)
}

pub fn run_embedded_predict<I, T>(model: &[u8], args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<EmbeddedPredictCommand, _, _>(args)? else {
        return Ok(());
    };
    predict_file(&SavedModel::from_bytes(model)?, command.input, command.output, command.batch_size, command.proba)
}
