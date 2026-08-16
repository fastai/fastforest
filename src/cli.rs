use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::{
    FileFitOptions, ForestError, MaxFeatures, SavedModel, SavedValue, Task, compile_model,
    convert_csv_to_arrow, fit_file, predict_file,
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
    #[arg(long, default_value_t = 8)]
    min_node_size: usize,
    #[arg(long)]
    bootstrap_fraction: Option<f32>,
    #[arg(long, default_value = "40000")]
    bootstrap_max: String,
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    replacement: Option<bool>,
    #[arg(long, default_value_t = 320)]
    max_node_samples: usize,
    #[arg(long)]
    tree_cutoff_samples: Option<usize>,
    #[arg(long, default_value_t = 0.0)]
    min_local_gain: f32,
    #[arg(long, default_value_t = 0.0)]
    min_global_gain: f32,
    #[arg(long, default_value_t = 10.0)]
    cutoff_divisor: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    oob: bool,
    #[arg(long)]
    random_splitter: bool,
    #[arg(long, default_value = "0.6")]
    max_features: String,
    #[arg(long, default_value_t = 4)]
    max_dummy_cardinality: usize,
    #[arg(long, default_value_t = 0.08)]
    frequent_value_fraction: f32,
    #[arg(long)]
    allow_new_missing: bool,
    #[arg(long = "missing-value")]
    missing_values: Vec<String>,
    #[arg(long = "one-hot-group")]
    one_hot_groups: Vec<String>,
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
    let value = value
        .parse::<usize>()
        .map_err(|_| ForestError::new("bootstrap_max must be a positive integer or 'none'"))?;
    if value == 0 {
        return Err(ForestError::new("bootstrap_max must be greater than zero"));
    }
    Ok(Some(value))
}

fn parse_assignment(value: &str, option: &str) -> Result<(String, String), ForestError> {
    let (name, value) = value
        .split_once('=')
        .ok_or_else(|| ForestError::new(format!("{option} must use COLUMN=VALUE")))?;
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
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) => {
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
        .map(|value| {
            parse_assignment(value, "missing-value")
                .map(|(name, value)| (name, SavedValue { kind: 5, value }))
        })
        .collect::<Result<_, _>>()?;
    let one_hot_groups = command
        .one_hot_groups
        .iter()
        .map(|value| {
            parse_assignment(value, "one-hot-group").map(|(name, columns)| {
                (
                    name,
                    columns.split(',').filter(|column| !column.is_empty()).map(str::to_owned).collect(),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let date_columns = command
        .date_columns
        .iter()
        .map(|value| parse_assignment(value, "date-column"))
        .collect::<Result<_, _>>()?;
    let options = FileFitOptions {
        task: match command.task {
            TaskArg::Regression => Task::Regression,
            TaskArg::Classification => Task::Classification,
        },
        target: command.target,
        n_trees: command.n_trees,
        min_node_size: command.min_node_size,
        bootstrap_fraction: command.bootstrap_fraction,
        bootstrap_max: parse_bootstrap_max(&command.bootstrap_max)?,
        replacement: command.replacement,
        max_node_samples: command.max_node_samples,
        tree_cutoff_samples: command.tree_cutoff_samples,
        min_local_gain: command.min_local_gain,
        min_global_gain: command.min_global_gain,
        cutoff_divisor: command.cutoff_divisor,
        seed: command.seed,
        oob: command.oob,
        random_splitter: command.random_splitter,
        max_features: parse_max_features(&command.max_features)?,
        max_dummy_cardinality: command.max_dummy_cardinality,
        frequent_value_fraction: command.frequent_value_fraction,
        allow_new_missing: command.allow_new_missing,
        missing_values,
        one_hot_groups,
        date_columns,
    };
    Ok((command.input, command.output, options))
}

pub fn run_fit<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<FitCommand, _, _>(args)? else { return Ok(()); };
    let (input, output, options) = fit_options(command)?;
    fit_file(input, &options)?.save(output)
}

pub fn run_predict<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<PredictCommand, _, _>(args)? else { return Ok(()); };
    let model = SavedModel::load(command.model)?;
    predict_file(&model, command.input, command.output, command.batch_size, command.proba)
}

pub fn run_convert<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<ConvertCommand, _, _>(args)? else { return Ok(()); };
    convert_csv_to_arrow(command.input, command.output, command.batch_size)
}

pub fn run_compile<I, T>(args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<CompileCommand, _, _>(args)? else { return Ok(()); };
    compile_model(&SavedModel::load(command.model)?, command.output)
}

pub fn run_embedded_predict<I, T>(model: &[u8], args: I) -> Result<(), ForestError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let Some(command) = parse_command::<EmbeddedPredictCommand, _, _>(args)? else { return Ok(()); };
    predict_file(
        &SavedModel::from_bytes(model)?,
        command.input,
        command.output,
        command.batch_size,
        command.proba,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_exposes_help() {
        run_fit(["fastforest-fit", "--help"]).unwrap();
        run_predict(["fastforest-predict", "--help"]).unwrap();
        run_convert(["fastforest-convert", "--help"]).unwrap();
        run_compile(["fastforest-compile", "--help"]).unwrap();
    }
}
