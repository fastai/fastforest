use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch, StringArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_cast::display::array_value_to_string;
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use arrow_select::{concat::concat, take::take};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::forest::uniform_sample_indices;
use crate::{
    ClassifierForest, Config, Encoder, Forest, ForestError, MaxFeatures, ModelMetadata, SavedModel, SavedValue, plan_fit,
    resolve_replacement,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Regression,
    Classification,
}

#[derive(Clone, Debug)]
pub struct FileFitOptions {
    pub task: Task,
    pub target: String,
    pub n_trees: Option<usize>,
    pub min_node_size: usize,
    pub bootstrap_fraction: Option<f32>,
    pub bootstrap_max: Option<usize>,
    pub replacement: Option<bool>,
    pub max_node_samples: usize,
    pub split_prior_rows: f32,
    pub class_weight_power: f32,
    pub cutoff_divisor: f32,
    pub seed: Option<u64>,
    pub oob: bool,
    pub random_splitter: bool,
    pub max_features: MaxFeatures,
    pub allow_new_missing: bool,
    pub missing_values: Vec<(String, SavedValue)>,
    pub date_columns: Vec<(String, String)>,
}

impl Default for FileFitOptions {
    fn default() -> Self {
        Self::for_task(Task::Regression)
    }
}

impl FileFitOptions {
    pub fn for_task(task: Task) -> Self {
        let config = if task == Task::Classification { Config::classification() } else { Config::default() };
        Self {
            task,
            target: String::new(),
            n_trees: None,
            min_node_size: config.min_node_size,
            bootstrap_fraction: config.bootstrap_fraction,
            bootstrap_max: config.bootstrap_max,
            replacement: None,
            max_node_samples: config.max_node_samples,
            split_prior_rows: config.split_prior_rows,
            class_weight_power: config.class_weight_power,
            cutoff_divisor: config.cutoff_divisor,
            seed: config.seed,
            oob: config.oob,
            random_splitter: config.random_splitter,
            max_features: config.max_features,
            allow_new_missing: false,
            missing_values: Vec::new(),
            date_columns: Vec::new(),
        }
    }

    fn resolved_replacement(&self, rows: usize) -> bool {
        resolve_replacement(rows, self.replacement, self.task == Task::Classification)
    }
}

fn file_error(context: &str, error: impl std::fmt::Display) -> ForestError {
    ForestError::new(format!("{context}: {error}"))
}

fn headers(path: &Path) -> Result<Vec<String>, ForestError> {
    let mut reader = csv::Reader::from_path(path).map_err(|error| file_error("could not open CSV", error))?;
    let names: Vec<_> =
        reader.headers().map_err(|error| file_error("could not read CSV header", error))?.iter().map(str::to_owned).collect();
    if names.is_empty() || names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return Err(ForestError::new("CSV column names must be present and unique"));
    }
    Ok(names)
}

fn predictor_layout(
    names: &[String], target: usize, options: &FileFitOptions,
) -> Result<(Vec<usize>, Vec<String>, ModelMetadata, Vec<(usize, String)>), ForestError> {
    let sources: Vec<_> = (0..names.len()).filter(|&index| index != target).collect();
    let predictor_names: Vec<_> = sources.iter().map(|&index| names[index].clone()).collect();
    let positions: HashMap<_, _> = predictor_names.iter().enumerate().map(|(index, name)| (name.as_str(), index)).collect();
    let mut markers = vec![SavedValue { kind: 5, value: String::new() }; predictor_names.len()];
    for (name, marker) in &options.missing_values {
        let index =
            positions.get(name.as_str()).copied().ok_or_else(|| ForestError::new(format!("unknown missing-value column {name:?}")))?;
        marker.validate()?;
        markers[index] = marker.clone();
    }
    let mut dates = Vec::new();
    for (name, format) in &options.date_columns {
        let index = positions.get(name.as_str()).copied().ok_or_else(|| ForestError::new(format!("unknown date column {name:?}")))?;
        dates.push((index, format.clone()));
    }
    let metadata = ModelMetadata { markers, date_columns: dates.clone(), parameters: Vec::new() };
    Ok((sources, predictor_names, metadata, dates))
}

fn target_sample_and_rows(path: &Path, target: usize, seed: Option<u64>) -> Result<(usize, Vec<String>), ForestError> {
    let mut reader = csv::Reader::from_path(path).map_err(|error| file_error("could not open CSV", error))?;
    let mut sample = Vec::with_capacity(1_000);
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(rand::random) ^ 0x517c_c1b7_2722_0a95);
    let mut rows = 0;
    for record in reader.records() {
        let record = record.map_err(|error| file_error("could not read CSV", error))?;
        let value = record.get(target).ok_or_else(|| ForestError::new("CSV row has the wrong number of columns"))?;
        if rows < 1_000 {
            sample.push(value.to_owned());
        } else {
            let replace = rng.random_range(0..=rows);
            if replace < 1_000 {
                sample[replace] = value.to_owned();
            }
        }
        rows += 1;
    }
    if rows == 0 {
        return Err(ForestError::new("training CSV must contain at least one row"));
    }
    Ok((rows, sample))
}

fn selected_csv_rows(
    path: &Path, selected: &[usize], sources: &[usize], target: usize, names: &[String],
) -> Result<(RecordBatch, Vec<String>), ForestError> {
    let mut selected = selected.to_vec();
    selected.sort_unstable();
    let mut next = 0;
    let mut columns: Vec<Vec<String>> = sources.iter().map(|_| Vec::with_capacity(selected.len())).collect();
    let mut targets = Vec::with_capacity(selected.len());
    let mut reader = csv::Reader::from_path(path).map_err(|error| file_error("could not open CSV", error))?;
    for (row, record) in reader.records().enumerate() {
        if next == selected.len() {
            break;
        }
        if row != selected[next] {
            continue;
        }
        let record = record.map_err(|error| file_error("could not read CSV", error))?;
        for (output, &source) in columns.iter_mut().zip(sources) {
            let value = record.get(source).ok_or_else(|| ForestError::new("CSV row has the wrong number of columns"))?;
            output.push(value.to_owned());
        }
        targets.push(record.get(target).ok_or_else(|| ForestError::new("CSV row has the wrong number of columns"))?.to_owned());
        next += 1;
    }
    if next != selected.len() {
        return Err(ForestError::new("CSV ended before all sampled rows were found"));
    }
    let fields: Vec<_> = names.iter().map(|name| Field::new(name, DataType::Utf8, false)).collect();
    let arrays: Vec<ArrayRef> = columns.into_iter().map(|values| Arc::new(StringArray::from(values)) as ArrayRef).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|error| file_error("could not build sampled Arrow table", error))?;
    Ok((batch, targets))
}

fn fit_config(options: &FileFitOptions, replacement: bool, n_trees: usize, sample_rows: usize) -> Config {
    Config {
        n_trees,
        min_node_size: options.min_node_size,
        bootstrap_fraction: options.bootstrap_fraction,
        bootstrap_max: options.bootstrap_max,
        sample_rows: Some(sample_rows),
        replacement,
        max_node_samples: options.max_node_samples,
        split_prior_rows: options.split_prior_rows,
        class_weight_power: options.class_weight_power,
        cutoff_divisor: options.cutoff_divisor,
        seed: options.seed,
        oob: options.oob,
        random_splitter: options.random_splitter,
        max_features: options.max_features,
    }
}

fn fit_sampled(
    predictors: &RecordBatch, targets: &[Option<SavedValue>], total_rows: usize, options: &FileFitOptions, metadata: ModelMetadata,
    dates: Vec<(usize, String)>,
) -> Result<SavedModel, ForestError> {
    let (encoder, x) = Encoder::fit_arrow(predictors, &metadata.markers, options.allow_new_missing, dates, options.seed)?;
    let replacement = options.resolved_replacement(total_rows);
    match options.task {
        Task::Regression => {
            let y: Result<Vec<_>, _> = targets
                .iter()
                .map(|value| {
                    let value = value.as_ref().ok_or_else(|| ForestError::new("regression targets cannot be missing"))?;
                    value
                        .value
                        .parse::<f32>()
                        .ok()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| ForestError::new(format!("invalid regression target {:?}", value.value)))
                })
                .collect();
            let plan =
                plan_fit(total_rows, options.n_trees, options.bootstrap_fraction, options.bootstrap_max, replacement, options.oob, 1)?;
            let config = fit_config(options, replacement, plan.n_trees, plan.rows_per_tree.min(x.nrows()));
            let forest = Forest::fit(
                x.view(),
                Array1::from_vec(y?).view(),
                encoder.cutoff_values(),
                encoder.cutoff_offsets(),
                Some(encoder.feature_group_ids()),
                &config,
            )?;
            Ok(SavedModel::regression(encoder, forest, metadata))
        }
        Task::Classification => {
            if targets.iter().any(Option::is_none) {
                return Err(ForestError::new("classification targets cannot be missing"));
            }
            let classes: Vec<_> = targets.iter().flatten().cloned().collect::<BTreeSet<_>>().into_iter().collect();
            if classes.len() < 2 {
                return Err(ForestError::new("classification requires at least two classes"));
            }
            let lookup: HashMap<_, _> = classes.iter().cloned().enumerate().map(|(index, value)| (value, index as u32)).collect();
            let y = Array1::from_iter(targets.iter().flatten().map(|value| lookup[value]));
            let dimensions = classes.len().saturating_sub(1).max(1);
            let plan = plan_fit(
                total_rows,
                options.n_trees,
                options.bootstrap_fraction,
                options.bootstrap_max,
                replacement,
                options.oob,
                dimensions,
            )?;
            let config = fit_config(options, replacement, plan.n_trees, plan.rows_per_tree.min(x.nrows()));
            let forest = ClassifierForest::fit(
                x.view(),
                y.view(),
                classes.len(),
                encoder.cutoff_values(),
                encoder.cutoff_offsets(),
                Some(encoder.feature_group_ids()),
                &config,
            )?;
            Ok(SavedModel::classification(encoder, forest, metadata, classes))
        }
    }
}

pub fn fit_csv(path: impl AsRef<Path>, options: &FileFitOptions) -> Result<SavedModel, ForestError> {
    let path = path.as_ref();
    let names = headers(path)?;
    let target = names
        .iter()
        .position(|name| name == &options.target)
        .ok_or_else(|| ForestError::new(format!("target column {:?} was not found", options.target)))?;
    let (n_rows, target_sample) = target_sample_and_rows(path, target, options.seed)?;
    let estimated_outputs =
        if options.task == Task::Classification { target_sample.iter().collect::<BTreeSet<_>>().len().saturating_sub(1).max(1) } else { 1 };
    let estimated = plan_fit(
        n_rows,
        options.n_trees,
        options.bootstrap_fraction,
        options.bootstrap_max,
        options.resolved_replacement(n_rows),
        options.oob,
        estimated_outputs,
    )?;
    let selected = uniform_sample_indices(n_rows, estimated.pool_rows, options.seed, 2);
    let (sources, predictor_names, metadata, dates) = predictor_layout(&names, target, options)?;
    let (batch, targets) = selected_csv_rows(path, &selected, &sources, target, &predictor_names)?;
    let targets: Vec<_> =
        targets.into_iter().map(|value| if value.is_empty() { None } else { Some(SavedValue { kind: 5, value }) }).collect();
    fit_sampled(&batch, &targets, n_rows, options, metadata, dates)
}

fn prediction_columns(
    path: &Path, names: &[String], batch_size: usize, mut predict: impl FnMut(RecordBatch) -> Result<(), ForestError>,
) -> Result<(), ForestError> {
    if batch_size == 0 {
        return Err(ForestError::new("batch_size must be greater than zero"));
    }
    let mut reader = csv::Reader::from_path(path).map_err(|error| file_error("could not open CSV", error))?;
    let source_names: Vec<_> =
        reader.headers().map_err(|error| file_error("could not read CSV header", error))?.iter().map(str::to_owned).collect();
    let positions: HashMap<_, _> = source_names.iter().enumerate().map(|(index, name)| (name.as_str(), index)).collect();
    let sources: Result<Vec<_>, _> = names
        .iter()
        .map(|name| positions.get(name.as_str()).copied().ok_or_else(|| ForestError::new(format!("prediction column {name:?} is missing"))))
        .collect();
    let sources = sources?;
    let batch = |columns: Vec<Vec<String>>| {
        let fields: Vec<_> = names.iter().map(|name| Field::new(name, DataType::Utf8, false)).collect();
        let arrays: Vec<ArrayRef> = columns.into_iter().map(|values| Arc::new(StringArray::from(values)) as ArrayRef).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
            .map_err(|error| file_error("could not build prediction Arrow table", error))
    };
    let mut columns: Vec<Vec<String>> = names.iter().map(|_| Vec::with_capacity(batch_size)).collect();
    for record in reader.records() {
        let record = record.map_err(|error| file_error("could not read CSV", error))?;
        for (output, &source) in columns.iter_mut().zip(&sources) {
            let value = record.get(source).ok_or_else(|| ForestError::new("CSV row has the wrong number of columns"))?;
            output.push(value.to_owned());
        }
        if columns[0].len() == batch_size {
            let full = std::mem::replace(&mut columns, names.iter().map(|_| Vec::with_capacity(batch_size)).collect());
            predict(batch(full)?)?;
        }
    }
    if !columns[0].is_empty() {
        predict(batch(columns)?)?;
    }
    Ok(())
}

fn class_text(value: &SavedValue) -> String {
    match value.kind {
        0 => String::new(),
        1 => "nan".to_owned(),
        2 => if value.value == "1" { "true" } else { "false" }.to_owned(),
        _ => value.value.clone(),
    }
}

pub fn predict_csv(
    model: &SavedModel, input: impl AsRef<Path>, output: impl AsRef<Path>, batch_size: usize, proba: bool,
) -> Result<(), ForestError> {
    let input = input.as_ref();
    let output = output.as_ref();
    let encoder = model.encoder();
    let markers = &model.metadata().markers;
    let mut writer = csv::Writer::from_path(output).map_err(|error| file_error("could not create output CSV", error))?;
    match model {
        SavedModel::Regression { forest, .. } => {
            writer.write_record(["prediction"]).map_err(|error| file_error("could not write CSV", error))?;
            prediction_columns(input, encoder.input_names(), batch_size, |batch| {
                let x = encoder.transform_arrow(&batch, markers)?;
                for value in forest.predict(x.view())? {
                    writer.serialize([value]).map_err(|error| file_error("could not write CSV", error))?;
                }
                Ok(())
            })?;
        }
        SavedModel::Classification { forest, classes, .. } if proba => {
            writer
                .write_record(classes.iter().map(|class| format!("proba_{}", class_text(class))))
                .map_err(|error| file_error("could not write CSV", error))?;
            prediction_columns(input, encoder.input_names(), batch_size, |batch| {
                let x = encoder.transform_arrow(&batch, markers)?;
                for row in forest.predict_proba(x.view())?.chunks_exact(classes.len()) {
                    writer.serialize(row).map_err(|error| file_error("could not write CSV", error))?;
                }
                Ok(())
            })?;
        }
        SavedModel::Classification { forest, classes, .. } => {
            writer.write_record(["prediction"]).map_err(|error| file_error("could not write CSV", error))?;
            prediction_columns(input, encoder.input_names(), batch_size, |batch| {
                let x = encoder.transform_arrow(&batch, markers)?;
                for class in forest.predict(x.view())? {
                    writer
                        .write_record([class_text(&classes[class as usize])])
                        .map_err(|error| file_error("could not write CSV", error))?;
                }
                Ok(())
            })?;
        }
    }
    writer.flush().map_err(|error| file_error("could not finish output CSV", error))
}

fn is_arrow(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("arrow") || extension.eq_ignore_ascii_case("feather"))
}

fn arrow_reader(path: &Path) -> Result<FileReader<File>, ForestError> {
    let file = File::open(path).map_err(|error| file_error("could not open Arrow file", error))?;
    FileReader::try_new(file, None).map_err(|error| file_error("could not read Arrow file", error))
}

macro_rules! arrow_number {
    ($array:expr, $row:expr, $ty:ty) => {
        $array.as_any().downcast_ref::<$ty>().unwrap().value($row)
    };
}

fn arrow_saved_value(array: &dyn Array, row: usize) -> Result<Option<SavedValue>, ForestError> {
    if array.is_null(row) {
        return Ok(None);
    }
    let (kind, value) = match array.data_type() {
        DataType::Int8 => (3, arrow_number!(array, row, Int8Array).to_string()),
        DataType::Int16 => (3, arrow_number!(array, row, Int16Array).to_string()),
        DataType::Int32 => (3, arrow_number!(array, row, Int32Array).to_string()),
        DataType::Int64 => (3, arrow_number!(array, row, Int64Array).to_string()),
        DataType::UInt8 => (3, arrow_number!(array, row, UInt8Array).to_string()),
        DataType::UInt16 => (3, arrow_number!(array, row, UInt16Array).to_string()),
        DataType::UInt32 => (3, arrow_number!(array, row, UInt32Array).to_string()),
        DataType::UInt64 => {
            let value = arrow_number!(array, row, UInt64Array);
            if value > i64::MAX as u64 { (4, (value as f64).to_string()) } else { (3, value.to_string()) }
        }
        DataType::Float32 => (4, arrow_number!(array, row, Float32Array).to_string()),
        DataType::Float64 => (4, arrow_number!(array, row, Float64Array).to_string()),
        DataType::Boolean => (2, if arrow_number!(array, row, BooleanArray) { "1" } else { "0" }.to_owned()),
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Dictionary(_, _)
        | DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Time32(_)
        | DataType::Time64(_) => (5, array_value_to_string(array, row).map_err(|error| file_error("could not read Arrow target", error))?),
        data_type => {
            return Err(ForestError::new(format!("unsupported Arrow target type {data_type}")));
        }
    };
    Ok(Some(SavedValue { kind, value }))
}

fn concat_batches(batches: &[RecordBatch], schema: Arc<Schema>) -> Result<RecordBatch, ForestError> {
    let columns: Result<Vec<_>, _> = (0..schema.fields().len())
        .map(|column| {
            let arrays: Vec<&dyn Array> = batches.iter().map(|batch| batch.column(column).as_ref()).collect();
            concat(&arrays).map_err(|error| file_error("could not join sampled Arrow batches", error))
        })
        .collect();
    RecordBatch::try_new(schema, columns?).map_err(|error| file_error("could not build sampled Arrow table", error))
}

pub fn fit_arrow(path: impl AsRef<Path>, options: &FileFitOptions) -> Result<SavedModel, ForestError> {
    let path = path.as_ref();
    let mut reader = arrow_reader(path)?;
    let schema = reader.schema();
    let names: Vec<_> = schema.fields().iter().map(|field| field.name().clone()).collect();
    if names.is_empty() || names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return Err(ForestError::new("Arrow column names must be present and unique"));
    }
    let target = names
        .iter()
        .position(|name| name == &options.target)
        .ok_or_else(|| ForestError::new(format!("target column {:?} was not found", options.target)))?;
    let mut rows = 0;
    let mut target_sample = Vec::with_capacity(1_000);
    let mut rng = StdRng::seed_from_u64(options.seed.unwrap_or_else(rand::random) ^ 0x517c_c1b7_2722_0a95);
    for batch in &mut reader {
        let batch = batch.map_err(|error| file_error("could not read Arrow batch", error))?;
        let array = batch.column(target);
        for row in 0..batch.num_rows() {
            let value = arrow_saved_value(array.as_ref(), row)?;
            if rows < 1_000 {
                target_sample.push(value);
            } else {
                let replace = rng.random_range(0..=rows);
                if replace < 1_000 {
                    target_sample[replace] = value;
                }
            }
            rows += 1;
        }
    }
    if rows == 0 {
        return Err(ForestError::new("training Arrow file must contain at least one row"));
    }
    let estimated_outputs = if options.task == Task::Classification {
        target_sample.iter().flatten().collect::<BTreeSet<_>>().len().saturating_sub(1).max(1)
    } else {
        1
    };
    let estimated = plan_fit(
        rows,
        options.n_trees,
        options.bootstrap_fraction,
        options.bootstrap_max,
        options.resolved_replacement(rows),
        options.oob,
        estimated_outputs,
    )?;
    let mut selected = uniform_sample_indices(rows, estimated.pool_rows, options.seed, 2);
    selected.sort_unstable();
    let (sources, _, metadata, dates) = predictor_layout(&names, target, options)?;
    let mut batches = Vec::new();
    let mut reader = arrow_reader(path)?;
    let mut global_row = 0;
    let mut next = 0;
    for batch in &mut reader {
        let batch = batch.map_err(|error| file_error("could not read Arrow batch", error))?;
        let end = global_row + batch.num_rows();
        let start = next;
        while next < selected.len() && selected[next] < end {
            next += 1
        }
        if next > start {
            let indices = UInt32Array::from(selected[start..next].iter().map(|row| (row - global_row) as u32).collect::<Vec<_>>());
            let columns: Result<Vec<_>, _> = batch
                .columns()
                .iter()
                .map(|array| take(array.as_ref(), &indices, None).map_err(|error| file_error("could not select sampled Arrow rows", error)))
                .collect();
            batches.push(
                RecordBatch::try_new(batch.schema(), columns?).map_err(|error| file_error("could not build sampled Arrow batch", error))?,
            );
        }
        global_row = end;
    }
    if next != selected.len() {
        return Err(ForestError::new("Arrow input ended before all sampled rows were found"));
    }
    let selected_batch = concat_batches(&batches, schema)?;
    let targets: Result<Vec<_>, _> =
        (0..selected_batch.num_rows()).map(|row| arrow_saved_value(selected_batch.column(target).as_ref(), row)).collect();
    let targets = targets?;
    let predictors = selected_batch.project(&sources).map_err(|error| file_error("could not select predictor columns", error))?;
    fit_sampled(&predictors, &targets, rows, options, metadata, dates)
}

fn arrow_output_writer(path: &Path, schema: Arc<Schema>) -> Result<FileWriter<File>, ForestError> {
    let file = File::create(path).map_err(|error| file_error("could not create Arrow output", error))?;
    FileWriter::try_new(file, &schema).map_err(|error| file_error("could not create Arrow writer", error))
}

pub fn predict_arrow(
    model: &SavedModel, input: impl AsRef<Path>, output: impl AsRef<Path>, batch_size: usize, proba: bool,
) -> Result<(), ForestError> {
    if batch_size == 0 {
        return Err(ForestError::new("batch_size must be greater than zero"));
    }
    let input = input.as_ref();
    let output = output.as_ref();
    let encoder = model.encoder();
    let mut reader = arrow_reader(input)?;
    let schema = reader.schema();
    let positions: HashMap<_, _> = schema.fields().iter().enumerate().map(|(index, field)| (field.name().as_str(), index)).collect();
    let sources: Result<Vec<_>, _> = encoder
        .input_names()
        .iter()
        .map(|name| positions.get(name.as_str()).copied().ok_or_else(|| ForestError::new(format!("prediction column {name:?} is missing"))))
        .collect();
    let sources = sources?;
    let output_schema = match model {
        SavedModel::Regression { .. } => Arc::new(Schema::new(vec![Field::new("prediction", DataType::Float32, false)])),
        SavedModel::Classification { classes, .. } if proba => Arc::new(Schema::new(
            classes.iter().map(|class| Field::new(format!("proba_{}", class_text(class)), DataType::Float32, false)).collect::<Vec<_>>(),
        )),
        SavedModel::Classification { .. } => Arc::new(Schema::new(vec![Field::new("prediction", DataType::Utf8, false)])),
    };
    let mut writer = arrow_output_writer(output, output_schema.clone())?;
    for batch in &mut reader {
        let batch = batch.map_err(|error| file_error("could not read Arrow batch", error))?;
        for offset in (0..batch.num_rows()).step_by(batch_size) {
            let batch = batch.slice(offset, batch_size.min(batch.num_rows() - offset));
            let predictors = batch.project(&sources).map_err(|error| file_error("could not select prediction columns", error))?;
            let x = encoder.transform_arrow(&predictors, &model.metadata().markers)?;
            let arrays: Vec<ArrayRef> = match model {
                SavedModel::Regression { forest, .. } => {
                    vec![Arc::new(Float32Array::from(forest.predict(x.view())?))]
                }
                SavedModel::Classification { forest, classes, .. } if proba => {
                    let probabilities = forest.predict_proba(x.view())?;
                    (0..classes.len())
                        .map(|class| {
                            Arc::new(Float32Array::from(
                                probabilities.chunks_exact(classes.len()).map(|row| row[class]).collect::<Vec<_>>(),
                            )) as ArrayRef
                        })
                        .collect()
                }
                SavedModel::Classification { forest, classes, .. } => vec![Arc::new(StringArray::from(
                    forest.predict(x.view())?.into_iter().map(|class| class_text(&classes[class as usize])).collect::<Vec<_>>(),
                ))],
            };
            let output_batch = RecordBatch::try_new(output_schema.clone(), arrays)
                .map_err(|error| file_error("could not construct Arrow output", error))?;
            writer.write(&output_batch).map_err(|error| file_error("could not write Arrow output", error))?;
        }
    }
    writer.finish().map_err(|error| file_error("could not finish Arrow output", error))
}

pub fn fit_file(path: impl AsRef<Path>, options: &FileFitOptions) -> Result<SavedModel, ForestError> {
    let path = path.as_ref();
    if is_arrow(path) { fit_arrow(path, options) } else { fit_csv(path, options) }
}

pub fn predict_file(
    model: &SavedModel, input: impl AsRef<Path>, output: impl AsRef<Path>, batch_size: usize, proba: bool,
) -> Result<(), ForestError> {
    let input = input.as_ref();
    let output = output.as_ref();
    if is_arrow(input) || is_arrow(output) {
        if !is_arrow(input) || !is_arrow(output) {
            return Err(ForestError::new("Arrow prediction currently requires both input and output to be Arrow"));
        }
        predict_arrow(model, input, output, batch_size, proba)
    } else {
        predict_csv(model, input, output, batch_size, proba)
    }
}

#[derive(Clone, Copy)]
enum ConvertedKind {
    Integer,
    Float,
}

pub fn convert_csv_to_arrow(input: impl AsRef<Path>, output: impl AsRef<Path>, batch_size: usize) -> Result<(), ForestError> {
    if batch_size == 0 {
        return Err(ForestError::new("batch_size must be greater than zero"));
    }
    let input = input.as_ref();
    let output = output.as_ref();
    let names = headers(input)?;
    let mut kinds = vec![ConvertedKind::Integer; names.len()];
    let mut reader = csv::Reader::from_path(input).map_err(|error| file_error("could not open CSV", error))?;
    for (row, record) in reader.records().enumerate() {
        let record = record.map_err(|error| file_error("could not read CSV", error))?;
        for (column, value) in record.iter().enumerate() {
            if value.is_empty() || value.parse::<i64>().is_ok() {
                continue;
            }
            if value.parse::<f64>().is_ok_and(f64::is_finite) {
                kinds[column] = ConvertedKind::Float;
            } else {
                return Err(ForestError::new(format!("column {:?} has nonnumeric value {value:?} at row {row}", names[column])));
            }
        }
    }
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .zip(&kinds)
            .map(|(name, kind)| {
                Field::new(name, if matches!(kind, ConvertedKind::Integer) { DataType::Int64 } else { DataType::Float64 }, true)
            })
            .collect::<Vec<_>>(),
    ));
    let mut writer = arrow_output_writer(output, schema.clone())?;
    let mut reader = csv::Reader::from_path(input).map_err(|error| file_error("could not open CSV", error))?;
    let mut rows: Vec<Vec<Option<String>>> = names.iter().map(|_| Vec::with_capacity(batch_size)).collect();
    let write_rows = |rows: &mut Vec<Vec<Option<String>>>, writer: &mut FileWriter<File>| -> Result<(), ForestError> {
        if rows[0].is_empty() {
            return Ok(());
        }
        let arrays: Result<Vec<ArrayRef>, ForestError> = rows
            .iter()
            .zip(&kinds)
            .map(|(values, kind)| match kind {
                ConvertedKind::Integer => Ok(Arc::new(Int64Array::from(
                    values.iter().map(|value| value.as_ref().map(|value| value.parse::<i64>().unwrap())).collect::<Vec<_>>(),
                )) as ArrayRef),
                ConvertedKind::Float => Ok(Arc::new(Float64Array::from(
                    values.iter().map(|value| value.as_ref().map(|value| value.parse::<f64>().unwrap())).collect::<Vec<_>>(),
                )) as ArrayRef),
            })
            .collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays?).map_err(|error| file_error("could not construct Arrow batch", error))?;
        writer.write(&batch).map_err(|error| file_error("could not write Arrow output", error))?;
        rows.iter_mut().for_each(Vec::clear);
        Ok(())
    };
    for record in reader.records() {
        let record = record.map_err(|error| file_error("could not read CSV", error))?;
        for (column, value) in record.iter().enumerate() {
            rows[column].push((!value.is_empty()).then(|| value.to_owned()));
        }
        if rows[0].len() == batch_size {
            write_rows(&mut rows, &mut writer)?;
        }
    }
    write_rows(&mut rows, &mut writer)?;
    writer.finish().map_err(|error| file_error("could not finish Arrow output", error))
}
