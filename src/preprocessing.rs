use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_cast::display::array_value_to_string;
use arrow_schema::DataType;
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::projection::Projections;
use crate::{ForestError, SavedValue};

const DATE_PARTS: [&str; 16] = [
    "Year",
    "Month",
    "Week",
    "Day",
    "Dayofweek",
    "Dayofyear",
    "Is_month_end",
    "Is_month_start",
    "Is_quarter_end",
    "Is_quarter_start",
    "Is_year_end",
    "Is_year_start",
    "Hour",
    "Minute",
    "Second",
    "Elapsed",
];
const MIN_STAT_CARDINALITY: usize = 6;

#[cfg(feature = "python")]
fn date_formats() -> Vec<String> {
    const BASES: [&str; 15] = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%m/%d/%y",
        "%d/%m/%y",
        "%m-%d-%y",
        "%d-%m-%y",
        "%d-%b-%Y",
        "%d %b %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %B %Y",
    ];
    const TIMES: [&str; 5] = [" %H:%M:%S", " %H:%M:%S%.f", " %H:%M", " %I:%M:%S %p", " %I:%M %p"];
    let mut formats = Vec::new();
    for base in BASES {
        formats.extend(TIMES.iter().map(|time| format!("{base}{time}")));
        formats.push(base.to_owned());
    }
    formats.extend(TIMES[..3].iter().map(|time| format!("%Y-%m-%dT{}", time.trim())));
    formats.extend(
        [
            "%Y-%m-%dT%H:%M:%S%.f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M",
            "%Y%m%d",
            "%H:%M:%S",
            "%H:%M:%S%.f",
            "%H:%M",
            "%I:%M:%S %p",
            "%I:%M %p",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    formats
}

#[derive(Clone, Debug)]
pub(crate) enum RawColumn {
    Numeric(Vec<Option<f32>>),
    Text(Vec<Option<String>>),
    Categorical { codes: Vec<i32>, categories: Arc<[Option<String>]>, null_value: Option<String> },
    Bundle { codes: Vec<Option<u32>>, categories: Arc<[String]> },
}

impl RawColumn {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Text(values) => values.len(),
            Self::Categorical { codes, .. } => codes.len(),
            Self::Bundle { codes, .. } => codes.len(),
        }
    }

    fn expand_categories(codes: Vec<i32>, categories: Arc<[Option<String>]>, null_value: Option<String>) -> Vec<Option<String>> {
        codes
            .into_iter()
            .map(|code| if code < 0 { null_value.clone() } else { categories.get(code as usize).cloned().unwrap_or(None) })
            .collect()
    }

    fn into_simple(self) -> Self {
        match self {
            Self::Categorical { codes, categories, null_value } => Self::Text(Self::expand_categories(codes, categories, null_value)),
            Self::Bundle { .. } => self,
            simple => simple,
        }
    }

    fn missing(&self) -> Vec<bool> {
        match self {
            Self::Numeric(values) => values.iter().map(Option::is_none).collect(),
            Self::Text(values) => values.iter().map(Option::is_none).collect(),
            Self::Categorical { codes, categories, null_value } => codes
                .iter()
                .map(|code| if *code < 0 { null_value.is_none() } else { categories.get(*code as usize).is_none_or(Option::is_none) })
                .collect(),
            Self::Bundle { codes, .. } => codes.iter().map(Option::is_none).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Encoding {
    Ordered,
    Statistic,
    Counter,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Values {
    Numeric(Vec<f32>),
    Text(Vec<String>),
    Categorical(Vec<String>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Column {
    name: String,
    values: Values,
    all_int: bool,
    median_numeric: Option<f32>,
    median_text: Option<String>,
    had_missing: bool,
    encodings: Vec<Encoding>,
    stats: Option<LevelStats>,
}

impl Column {
    fn cardinality(&self) -> usize {
        match &self.values {
            Values::Numeric(values) => values.len(),
            Values::Text(values) | Values::Categorical(values) => values.len(),
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self.values, Values::Numeric(_))
    }

    pub fn all_int(&self) -> bool {
        self.all_int
    }

    pub fn had_missing(&self) -> bool {
        self.had_missing
    }

    pub fn numeric_values(&self) -> &[f32] {
        match &self.values {
            Values::Numeric(values) => values,
            Values::Text(_) | Values::Categorical(_) => &[],
        }
    }

    pub fn text_values(&self) -> &[String] {
        match &self.values {
            Values::Numeric(_) => &[],
            Values::Text(values) | Values::Categorical(values) => values,
        }
    }

    pub fn median_numeric(&self) -> Option<f32> {
        self.median_numeric
    }

    pub fn median_text(&self) -> Option<&str> {
        self.median_text.as_deref()
    }

    pub fn encodings(&self) -> &[Encoding] {
        &self.encodings
    }
}

#[derive(Debug)]
struct FittedColumn {
    column: Column,
    encodings: Vec<FittedEncoding>,
}

#[derive(Debug)]
struct FittedEncoding {
    kind: Encoding,
    bounds: Vec<f32>,
    codes: FitCodes,
}

#[derive(Debug)]
enum FitCodes {
    Materialized(Arc<[u32]>),
    Projected { base: Arc<[u32]>, table: Vec<u32> },
}

impl FitCodes {
    fn len(&self) -> usize {
        match self {
            Self::Materialized(codes) => codes.len(),
            Self::Projected { base, .. } => base.len(),
        }
    }

    fn at(&self, row: usize) -> u32 {
        match self {
            Self::Materialized(codes) => codes[row],
            Self::Projected { base, table } => table[base[row] as usize],
        }
    }

    fn projected(&self, table: Vec<u32>) -> Self {
        match self {
            Self::Materialized(base) => Self::Projected { base: base.clone(), table },
            Self::Projected { base, table: first } => {
                Self::Projected { base: base.clone(), table: first.iter().map(|&code| table[code as usize]).collect() }
            }
        }
    }

    fn base_and_table(&self) -> (&Arc<[u32]>, Option<&[u32]>) {
        match self {
            Self::Materialized(base) => (base, None),
            Self::Projected { base, table } => (base, Some(table)),
        }
    }
}

impl FittedColumn {
    fn training_codes(&self) -> Option<&FitCodes> {
        self.encodings.first().map(|encoding| &encoding.codes)
    }
}

#[derive(Clone)]
struct DateBase {
    codes: Arc<[u32]>,
    distinct: Arc<[NaiveDateTime]>,
    had_missing: bool,
}

enum ArrangedColumn<T> {
    Raw(RawColumn),
    DatePart { date: T, part: u8 },
}

struct Parts<T> {
    unique: Vec<T>,
    codes: Vec<u32>,
    counts: Vec<usize>,
    median: T,
}

struct PreparedColumn {
    values: Values,
    codes: Vec<u32>,
    counts: Vec<usize>,
    median_numeric: Option<f32>,
    median_text: Option<String>,
    missing: Vec<bool>,
    all_int: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Encoder {
    columns: Vec<Column>,
    input_names: Vec<String>,
    input_columns: Vec<InputColumn>,
    allow_new_missing: bool,
    cutoff_values: Vec<f32>,
    cutoff_offsets: Vec<usize>,
    encoded_to_raw: Vec<usize>,
    natural_sort: bool,
    #[serde(skip)]
    projections: Projections,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum InputColumn {
    Direct(usize),
    Bundle { indices: Vec<usize>, categories: Vec<String> },
    DatePart { index: usize, format: String, part: u8 },
}

fn invalid(message: impl Into<String>) -> ForestError {
    ForestError::new(message)
}

pub(crate) fn marker_matches(value: &str, marker: &SavedValue) -> bool {
    match marker.kind {
        0 => value.is_empty(),
        1 => value.parse::<f64>().is_ok_and(f64::is_nan),
        2 => value == if marker.value == "1" { "true" } else { "false" },
        3 => value.parse::<i64>().ok().is_some_and(|parsed| parsed.to_string() == marker.value),
        4 => value.parse::<f64>().ok() == marker.value.parse::<f64>().ok(),
        5 => value == marker.value,
        _ => false,
    }
}

macro_rules! numeric_arrow_column {
    ($array:expr, $marker:expr, $ty:ty) => {{
        let array = $array.as_any().downcast_ref::<$ty>().unwrap();
        RawColumn::Numeric(
            (0..array.len())
                .map(|row| {
                    if array.is_null(row) {
                        return None;
                    }
                    let value = array.value(row) as f64;
                    (!marker_matches(&value.to_string(), $marker)).then_some(value as f32)
                })
                .collect(),
        )
    }};
}

macro_rules! dictionary_arrow_column {
    ($array:expr, $marker:expr, $ty:ty) => {{
        let array = $array.as_any().downcast_ref::<DictionaryArray<$ty>>().unwrap();
        let categories: Result<Vec<_>, ForestError> = (0..array.values().len())
            .map(|row| {
                if array.values().is_null(row) {
                    return Ok(None);
                }
                let value = array_value_to_string(array.values().as_ref(), row)
                    .map_err(|error| invalid(format!("could not read Arrow category: {error}")))?;
                Ok((!marker_matches(&value, $marker)).then_some(value))
            })
            .collect();
        RawColumn::Categorical {
            codes: (0..array.len()).map(|row| if array.is_null(row) { -1 } else { array.keys().value(row) as i32 }).collect(),
            categories: categories?.into(),
            null_value: None,
        }
    }};
}

fn arrow_column(array: &dyn Array, marker: &SavedValue) -> Result<RawColumn, ForestError> {
    let column = match array.data_type() {
        DataType::Float32 => numeric_arrow_column!(array, marker, Float32Array),
        DataType::Float64 => numeric_arrow_column!(array, marker, Float64Array),
        DataType::Int8 => numeric_arrow_column!(array, marker, Int8Array),
        DataType::Int16 => numeric_arrow_column!(array, marker, Int16Array),
        DataType::Int32 => numeric_arrow_column!(array, marker, Int32Array),
        DataType::Int64 => numeric_arrow_column!(array, marker, Int64Array),
        DataType::UInt8 => numeric_arrow_column!(array, marker, UInt8Array),
        DataType::UInt16 => numeric_arrow_column!(array, marker, UInt16Array),
        DataType::UInt32 => numeric_arrow_column!(array, marker, UInt32Array),
        DataType::UInt64 => numeric_arrow_column!(array, marker, UInt64Array),
        DataType::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            RawColumn::Numeric(
                (0..array.len())
                    .map(|row| {
                        if array.is_null(row) {
                            return None;
                        }
                        let value = array.value(row);
                        (!marker_matches(if value { "true" } else { "false" }, marker)).then_some(f32::from(value))
                    })
                    .collect(),
            )
        }
        DataType::Dictionary(key, _) => match key.as_ref() {
            DataType::Int8 => dictionary_arrow_column!(array, marker, Int8Type),
            DataType::Int16 => dictionary_arrow_column!(array, marker, Int16Type),
            DataType::Int32 => dictionary_arrow_column!(array, marker, Int32Type),
            DataType::Int64 => dictionary_arrow_column!(array, marker, Int64Type),
            DataType::UInt8 => dictionary_arrow_column!(array, marker, UInt8Type),
            DataType::UInt16 => dictionary_arrow_column!(array, marker, UInt16Type),
            DataType::UInt32 => dictionary_arrow_column!(array, marker, UInt32Type),
            DataType::UInt64 => dictionary_arrow_column!(array, marker, UInt64Type),
            data_type => {
                return Err(invalid(format!("unsupported Arrow dictionary key type {data_type}")));
            }
        },
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Time32(_)
        | DataType::Time64(_) => RawColumn::Text(
            (0..array.len())
                .map(|row| {
                    if array.is_null(row) {
                        return Ok(None);
                    }
                    let value =
                        array_value_to_string(array, row).map_err(|error| invalid(format!("could not read Arrow value: {error}")))?;
                    Ok((!marker_matches(&value, marker)).then_some(value))
                })
                .collect::<Result<_, ForestError>>()?,
        ),
        data_type => {
            return Err(invalid(format!("unsupported Arrow column type {data_type}")));
        }
    };
    Ok(column)
}

fn numeric_arrow_value(array: &dyn Array, row: usize) -> Option<f32> {
    if array.is_null(row) {
        return None;
    }
    Some(match array.data_type() {
        DataType::Float32 => array.as_any().downcast_ref::<Float32Array>().unwrap().value(row),
        DataType::Float64 => array.as_any().downcast_ref::<Float64Array>().unwrap().value(row) as f32,
        DataType::Int8 => array.as_any().downcast_ref::<Int8Array>().unwrap().value(row) as f32,
        DataType::Int16 => array.as_any().downcast_ref::<Int16Array>().unwrap().value(row) as f32,
        DataType::Int32 => array.as_any().downcast_ref::<Int32Array>().unwrap().value(row) as f32,
        DataType::Int64 => array.as_any().downcast_ref::<Int64Array>().unwrap().value(row) as f32,
        DataType::UInt8 => array.as_any().downcast_ref::<UInt8Array>().unwrap().value(row) as f32,
        DataType::UInt16 => array.as_any().downcast_ref::<UInt16Array>().unwrap().value(row) as f32,
        DataType::UInt32 => array.as_any().downcast_ref::<UInt32Array>().unwrap().value(row) as f32,
        DataType::UInt64 => array.as_any().downcast_ref::<UInt64Array>().unwrap().value(row) as f32,
        DataType::Boolean => f32::from(array.as_any().downcast_ref::<BooleanArray>().unwrap().value(row)),
        _ => return None,
    })
}

fn numeric_arrow_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
    )
}

fn arrow_text_column(array: &dyn Array, marker: &SavedValue) -> Result<RawColumn, ForestError> {
    if matches!(array.data_type(), DataType::Dictionary(_, _)) {
        return arrow_column(array, marker);
    }
    Ok(RawColumn::Text(
        (0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    return Ok(None);
                }
                let value = array_value_to_string(array, row).map_err(|error| invalid(format!("could not read Arrow value: {error}")))?;
                Ok((!marker_matches(&value, marker)).then_some(value))
            })
            .collect::<Result<_, ForestError>>()?,
    ))
}

fn arrow_columns(batch: &RecordBatch, markers: &[SavedValue], date_indices: &[usize]) -> Result<Vec<RawColumn>, ForestError> {
    if batch.num_columns() != markers.len() {
        return Err(invalid("missing_values must have one value per column"));
    }
    batch
        .columns()
        .par_iter()
        .zip(markers)
        .enumerate()
        .map(
            |(index, (array, marker))| {
                if date_indices.contains(&index) { arrow_text_column(array.as_ref(), marker) } else { arrow_column(array.as_ref(), marker) }
            },
        )
        .collect()
}

fn validate_rows(columns: &[RawColumn], names: &[String]) -> Result<usize, ForestError> {
    if columns.len() != names.len() {
        return Err(invalid("X must have one column name per column"));
    }
    let rows = columns.first().map_or(0, RawColumn::len);
    if rows == 0 {
        return Err(invalid("X must contain at least one row"));
    }
    if columns.iter().any(|column| column.len() != rows) {
        return Err(invalid("X columns must have the same number of rows"));
    }
    Ok(rows)
}

fn indicator_bit(value: f32) -> Option<bool> {
    if value == 0. {
        Some(false)
    } else if value == 1. {
        Some(true)
    } else {
        None
    }
}

fn binary_sample(column: &RawColumn, sample: &[usize]) -> Option<Vec<bool>> {
    let mut seen = [false; 2];
    let mut record = |active: bool| {
        seen[usize::from(active)] = true;
        active
    };
    let sampled = match column {
        RawColumn::Numeric(values) => {
            for value in values {
                record(indicator_bit((*value)?)?);
            }
            sample.iter().map(|&row| indicator_bit(values[row].unwrap()).unwrap()).collect()
        }
        RawColumn::Text(values) => {
            for value in values {
                record(indicator_bit(value.as_ref()?.parse().ok()?)?);
            }
            sample.iter().map(|&row| indicator_bit(values[row].as_ref().unwrap().parse().unwrap()).unwrap()).collect()
        }
        RawColumn::Categorical { codes, categories, null_value } => {
            let parsed: Option<Vec<_>> =
                categories.iter().map(|value| value.as_ref().and_then(|value| value.parse().ok()).and_then(indicator_bit)).collect();
            let parsed = parsed?;
            let parsed_null = null_value.as_ref().and_then(|value| value.parse().ok()).and_then(indicator_bit);
            let value = |row: usize| {
                let code = codes[row];
                if code < 0 { parsed_null } else { parsed.get(code as usize).copied() }
            };
            for row in 0..codes.len() {
                record(value(row)?);
            }
            sample.iter().map(|&row| value(row).unwrap()).collect()
        }
        RawColumn::Bundle { .. } => return None,
    };
    (seen[0] && seen[1]).then_some(sampled)
}

fn bundle_prefix(names: &[String]) -> String {
    let Some(first) = names.first() else { return String::new() };
    let mut prefix = first.clone();
    for name in &names[1..] {
        let bytes = prefix.chars().zip(name.chars()).take_while(|(left, right)| left == right).map(|(value, _)| value.len_utf8()).sum();
        prefix.truncate(bytes);
    }
    prefix.trim_end_matches(|value: char| value.is_ascii_digit() || matches!(value, '_' | '-' | '.' | ' ')).to_owned()
}

fn automatic_bundles(
    columns: &[RawColumn], names: &[String], excluded: &[bool], seed: Option<u64>,
) -> Vec<(String, Vec<usize>, Vec<String>)> {
    let rows = columns[0].len();
    let sample = crate::forest::uniform_sample_indices(rows, rows.min(10_000), seed, 0xb3e7_68d1);
    let candidates: Vec<_> = columns
        .iter()
        .enumerate()
        .filter(|(index, _)| !excluded[*index])
        .filter_map(|(index, column)| binary_sample(column, &sample).map(|values| (index, values)))
        .collect();
    let mut conflicts: Vec<HashSet<usize>> = (0..candidates.len()).map(|_| HashSet::new()).collect();
    for row in 0..sample.len() {
        let active: Vec<_> = candidates.iter().enumerate().filter_map(|(index, (_, values))| values[row].then_some(index)).collect();
        for (position, &left) in active.iter().enumerate() {
            for &right in &active[position + 1..] {
                conflicts[left].insert(right);
                conflicts[right].insert(left);
            }
        }
    }
    let mut order: Vec<_> = (0..candidates.len()).collect();
    order.sort_unstable_by_key(|&index| (std::cmp::Reverse(conflicts[index].len()), candidates[index].0));
    let mut colors: Vec<Vec<usize>> = Vec::new();
    for candidate in order {
        if let Some(color) = colors.iter_mut().find(|color| color.iter().all(|member| !conflicts[candidate].contains(member))) {
            color.push(candidate);
        } else {
            colors.push(vec![candidate]);
        }
    }
    let mut used: HashSet<_> = names.iter().cloned().collect();
    let mut fallback = 1;
    let mut bundles = Vec::new();
    for mut color in colors {
        if color.len() < 2 {
            continue;
        }
        let covered = (0..sample.len()).filter(|&row| color.iter().any(|&candidate| candidates[candidate].1[row])).count();
        if covered * 2 <= sample.len() {
            continue;
        }
        color.sort_unstable_by_key(|&candidate| {
            let active = candidates[candidate].1.iter().filter(|value| **value).count();
            (active, candidates[candidate].0)
        });
        let indices: Vec<_> = color.iter().map(|&candidate| candidates[candidate].0).collect();
        let categories: Vec<_> = indices.iter().map(|&index| names[index].clone()).collect();
        let mut name = bundle_prefix(&categories);
        if name.is_empty() || used.contains(&name) {
            loop {
                name = format!("bundle_{fallback}");
                fallback += 1;
                if !used.contains(&name) {
                    break;
                }
            }
        }
        used.insert(name.clone());
        bundles.push((name, indices, categories));
    }
    bundles.sort_unstable_by_key(|(_, indices, _)| *indices.iter().min().unwrap());
    bundles
}

fn input_layout(
    names: &[String], bundles: &[(String, Vec<usize>, Vec<String>)], date_parts: &[(usize, String, u8, String)],
) -> Result<(Vec<InputColumn>, Vec<String>), ForestError> {
    let mut grouped = vec![false; names.len()];
    for (index, _, _, _) in date_parts {
        if *index >= names.len() {
            return Err(invalid("date column is out of range"));
        }
        grouped[*index] = true;
    }
    for (bundle, indices, _) in bundles {
        for &index in indices {
            if index >= names.len() || std::mem::replace(&mut grouped[index], true) {
                return Err(invalid(format!("automatic bundle {bundle:?} contains an unavailable column")));
            }
        }
    }
    let mut input_columns = Vec::new();
    let mut logical_names = Vec::new();
    for (index, name) in names.iter().enumerate() {
        if !grouped[index] {
            input_columns.push(InputColumn::Direct(index));
            logical_names.push(name.clone());
        }
    }
    for (bundle, indices, categories) in bundles {
        input_columns.push(InputColumn::Bundle { indices: indices.clone(), categories: categories.clone() });
        logical_names.push(bundle.clone());
    }
    for (index, format, part, name) in date_parts {
        input_columns.push(InputColumn::DatePart { index: *index, format: format.clone(), part: *part });
        logical_names.push(name.clone());
    }
    let mut unique = logical_names.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != logical_names.len() {
        return Err(invalid("automatic bundle and feature names must be unique"));
    }
    Ok((input_columns, logical_names))
}

fn date_layout(names: &[String], date_columns: &[(usize, String)]) -> Result<Vec<(usize, String, u8, String)>, ForestError> {
    let mut result = Vec::with_capacity(date_columns.len() * DATE_PARTS.len());
    for (index, format) in date_columns {
        let name = names.get(*index).ok_or_else(|| invalid("date column is out of range"))?;
        if format.is_empty() {
            return Err(invalid(format!("date column {name:?} must have a non-empty format")));
        }
        let prefix = name.strip_suffix("Date").or_else(|| name.strip_suffix("date")).unwrap_or(name);
        result
            .extend(DATE_PARTS.iter().enumerate().map(|(part, suffix)| (*index, format.clone(), part as u8, format!("{prefix}{suffix}"))));
    }
    Ok(result)
}

#[cfg(feature = "python")]
pub fn detect_dates(batch: &RecordBatch, markers: &[SavedValue], seed: Option<u64>) -> Result<Vec<(usize, String)>, ForestError> {
    if batch.num_columns() != markers.len() {
        return Err(invalid("missing_values must have one value per column"));
    }
    let sample = crate::forest::uniform_sample_indices(batch.num_rows(), batch.num_rows().min(200), seed, 0x2d4a_7f18);
    let formats = date_formats();
    let detected: Result<Vec<_>, ForestError> = batch
        .columns()
        .par_iter()
        .zip(markers)
        .enumerate()
        .map(|(column, (array, marker))| {
            let mut candidates = formats.clone();
            let mut observed = false;
            for &row in &sample {
                if array.is_null(row) {
                    continue;
                }
                let value = array_value_to_string(array.as_ref(), row)
                    .map_err(|error| invalid(format!("could not inspect Arrow date value: {error}")))?;
                if marker_matches(&value, marker) {
                    continue;
                }
                observed = true;
                candidates.retain(|format| parse_date(&value, format).is_some());
                if candidates.is_empty() {
                    break;
                }
            }
            Ok((observed && !candidates.is_empty()).then(|| (column, candidates.remove(0))))
        })
        .collect();
    Ok(detected?.into_iter().flatten().collect())
}

fn parse_numeric(values: &[Option<String>], name: &str) -> Result<Option<Vec<Option<f32>>>, ForestError> {
    let mut parsed = Vec::with_capacity(values.len());
    for value in values {
        let Some(value) = value else {
            parsed.push(None);
            continue;
        };
        let Ok(value) = value.parse::<f32>() else {
            return Ok(None);
        };
        if !value.is_finite() {
            return Err(invalid(format!("column {name:?} contains a non-finite numeric value")));
        }
        parsed.push(Some(value));
    }
    Ok(Some(parsed))
}

fn indicator_values(raw: RawColumn, name: &str) -> Result<Vec<Option<f32>>, ForestError> {
    match raw.into_simple() {
        RawColumn::Numeric(values) => Ok(values),
        RawColumn::Text(values) => {
            parse_numeric(&values, name)?.ok_or_else(|| invalid(format!("bundle {name:?} must contain only numeric indicator columns")))
        }
        RawColumn::Categorical { .. } | RawColumn::Bundle { .. } => unreachable!(),
    }
}

fn collapse_bundle(columns: Vec<RawColumn>, categories: Vec<String>, name: &str) -> Result<RawColumn, ForestError> {
    let indicators: Result<Vec<_>, _> = columns.into_iter().map(|column| indicator_values(column, name)).collect();
    let indicators = indicators?;
    let rows = indicators.first().map_or(0, Vec::len);
    let codes: Result<Vec<_>, _> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let mut active = None;
            let mut missing = false;
            for (member, values) in indicators.iter().enumerate() {
                match values[row] {
                    None => missing = true,
                    Some(0.) => {}
                    Some(1.) if active.is_none() => active = Some(member as u32 + 1),
                    Some(1.) => {}
                    Some(_) => return Err(invalid(format!("bundle {name:?} has a value other than 0 or 1 at row {row}"))),
                }
            }
            Ok(active.or((!missing).then_some(0)))
        })
        .collect();
    let mut values = Vec::with_capacity(categories.len() + 1);
    values.push("(none)".to_owned());
    values.extend(categories);
    Ok(RawColumn::Bundle { codes: codes?, categories: values.into() })
}

fn parse_date(value: &str, format: &str) -> Option<NaiveDateTime> {
    let parsed = DateTime::parse_from_str(value, format)
        .map(|value| value.naive_utc())
        .ok()
        .or_else(|| NaiveDateTime::parse_from_str(value, format).ok())
        .or_else(|| NaiveDate::parse_from_str(value, format).ok().and_then(|date| date.and_hms_opt(0, 0, 0)))
        .or_else(|| {
            NaiveTime::parse_from_str(value, format).ok().map(|time| NaiveDate::from_ymd_opt(1900, 1, 1).unwrap().and_time(time))
        })?;
    (!(format.contains("%Y") || format.contains("%y")) || (1900..=2100).contains(&parsed.year())).then_some(parsed)
}

fn parse_dates(raw: &RawColumn, format: &str, name: &str) -> Result<Vec<Option<NaiveDateTime>>, ForestError> {
    match raw {
        RawColumn::Text(values) => {
            Ok(values.par_iter().map(|value| value.as_deref().and_then(|value| parse_date(value, format))).collect())
        }
        RawColumn::Categorical { codes, categories, null_value } => {
            let parsed: Vec<_> = categories.iter().map(|value| value.as_deref().and_then(|value| parse_date(value, format))).collect();
            let parsed_null = null_value.as_deref().and_then(|value| parse_date(value, format));
            codes
                .par_iter()
                .map(|code| {
                    if *code < 0 {
                        return Ok(parsed_null);
                    }
                    parsed.get(*code as usize).copied().ok_or_else(|| invalid(format!("date column {name:?} has an invalid category code")))
                })
                .collect()
        }
        RawColumn::Numeric(_) | RawColumn::Bundle { .. } => Err(invalid(format!("date column {name:?} must contain strings"))),
    }
}

fn date_value(value: NaiveDateTime, part: u8) -> f32 {
    let date = value.date();
    let month_start = date.day() == 1;
    let month_end = date.succ_opt().is_none_or(|next| next.month() != date.month());
    match part {
        0 => date.year() as f32,
        1 => date.month() as f32,
        2 => date.iso_week().week() as f32,
        3 => date.day() as f32,
        4 => date.weekday().num_days_from_monday() as f32,
        5 => date.ordinal() as f32,
        6 => f32::from(month_end),
        7 => f32::from(month_start),
        8 => f32::from(month_end && date.month().is_multiple_of(3)),
        9 => f32::from(month_start && date.month() % 3 == 1),
        10 => f32::from(month_end && date.month() == 12),
        11 => f32::from(month_start && date.month() == 1),
        12 => value.hour() as f32,
        13 => value.minute() as f32,
        14 => value.second() as f32,
        15 => value.and_utc().timestamp() as f32,
        _ => unreachable!(),
    }
}

fn arrange_columns<T: Clone>(
    columns: Vec<RawColumn>, input_columns: &[InputColumn], logical_names: &[String],
    mut prepare_date: impl FnMut(&RawColumn, &str, &str) -> Result<T, ForestError>,
) -> Result<Vec<ArrangedColumn<T>>, ForestError> {
    let mut dates = HashMap::new();
    for (position, source) in input_columns.iter().enumerate() {
        if let InputColumn::DatePart { index, format, .. } = source {
            if !dates.contains_key(index) {
                dates.insert(*index, prepare_date(&columns[*index], format, &logical_names[position])?);
            }
        }
    }
    let mut columns: Vec<_> = columns.into_iter().map(Some).collect();
    let arranged: Result<Vec<_>, _> = input_columns
        .iter()
        .zip(logical_names)
        .map(|(source, name)| match source {
            InputColumn::Direct(index) => Ok(ArrangedColumn::Raw(columns[*index].take().unwrap())),
            InputColumn::Bundle { indices, categories } => {
                collapse_bundle(indices.iter().map(|index| columns[*index].take().unwrap()).collect(), categories.clone(), name)
                    .map(ArrangedColumn::Raw)
            }
            InputColumn::DatePart { index, part, .. } => Ok(ArrangedColumn::DatePart { date: dates[index].clone(), part: *part }),
        })
        .collect();
    arranged
}

fn arrange_training_columns(
    columns: Vec<RawColumn>, input_columns: &[InputColumn], logical_names: &[String],
) -> Result<Vec<ArrangedColumn<DateBase>>, ForestError> {
    arrange_columns(columns, input_columns, logical_names, |raw, format, name| {
        let parsed = parse_dates(raw, format, name)?;
        let had_missing = parsed.iter().any(Option::is_none);
        let (codes, distinct) = date_base_codes(&parsed);
        Ok(DateBase { codes: codes.into(), distinct: distinct.into(), had_missing })
    })
}

fn arrange_inference_columns(
    columns: Vec<RawColumn>, input_columns: &[InputColumn], logical_names: &[String],
) -> Result<Vec<RawColumn>, ForestError> {
    arrange_columns(columns, input_columns, logical_names, parse_dates).map(|columns| {
        columns
            .into_iter()
            .map(|column| match column {
                ArrangedColumn::Raw(raw) => raw,
                ArrangedColumn::DatePart { date, part } => {
                    RawColumn::Numeric(date.into_iter().map(|value| value.map(|value| date_value(value, part))).collect())
                }
            })
            .collect()
    })
}

fn date_base_codes(values: &[Option<NaiveDateTime>]) -> (Vec<u32>, Vec<NaiveDateTime>) {
    let mut distinct: Vec<NaiveDateTime> = values.iter().flatten().copied().collect();
    distinct.sort_unstable();
    distinct.dedup();
    let missing = distinct.len() as u32;
    let codes = values.iter().map(|value| value.map_or(missing, |value| distinct.binary_search(&value).unwrap() as u32)).collect();
    (codes, distinct)
}

fn numeric_parts(values: Vec<Option<f32>>, name: &str) -> Result<Parts<f32>, ForestError> {
    let observed: Vec<_> = values.iter().flatten().copied().collect();
    if observed.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("column {name:?} contains a non-finite numeric value")));
    }
    let mut unique = observed.clone();
    unique.sort_unstable_by(|a, b| a.total_cmp(b));
    unique.dedup_by(|left, right| left.total_cmp(right).is_eq());
    let mut counts = vec![0; unique.len()];
    let mut codes = Vec::with_capacity(values.len());
    for value in &values {
        if let Some(value) = value {
            let code = unique.binary_search_by(|candidate| candidate.total_cmp(value)).unwrap();
            counts[code] += 1;
            codes.push(code as u32);
        } else {
            codes.push(u32::MAX);
        }
    }
    let mut ordered = observed;
    let middle = ordered.len() / 2;
    ordered.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
    let median = ordered[middle];
    Ok(Parts { unique, codes, counts, median })
}

fn digit_run(bytes: &[u8]) -> usize {
    bytes.iter().take_while(|byte| byte.is_ascii_digit()).count()
}

pub(crate) fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut shared = 0;
    while shared + 8 <= limit && left[shared..shared + 8] == right[shared..shared + 8] {
        shared += 8;
    }
    while shared < limit && left[shared] == right[shared] {
        shared += 1;
    }
    shared
}

pub(crate) fn natural_cmp(left: &str, right: &str) -> std::cmp::Ordering {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    let shared = common_prefix_len(left, right);
    if shared == left.len() && shared == right.len() {
        return std::cmp::Ordering::Equal;
    }
    let start = left[..shared].iter().rposition(|byte| !byte.is_ascii_digit()).map_or(0, |position| position + 1);
    natural_cmp_suffix(&left[start..], &right[start..])
}

fn natural_cmp_suffix(mut l: &[u8], mut r: &[u8]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    loop {
        match (l.first(), r.first()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(&a), Some(&b)) => {
                if a.is_ascii_digit() && b.is_ascii_digit() {
                    let (run_l, run_r) = (digit_run(l), digit_run(r));
                    let (digits_l, rest_l) = l.split_at(run_l);
                    let (digits_r, rest_r) = r.split_at(run_r);
                    let trimmed_l = &digits_l[digits_l.iter().take_while(|byte| **byte == b'0').count()..];
                    let trimmed_r = &digits_r[digits_r.iter().take_while(|byte| **byte == b'0').count()..];
                    let ordering = trimmed_l.len().cmp(&trimmed_r.len()).then_with(|| trimmed_l.cmp(trimmed_r)).then(run_l.cmp(&run_r));
                    if ordering != Ordering::Equal {
                        return ordering;
                    }
                    (l, r) = (rest_l, rest_r);
                } else {
                    let ordering = a.cmp(&b);
                    if ordering != Ordering::Equal {
                        return ordering;
                    }
                    (l, r) = (&l[1..], &r[1..]);
                }
            }
        }
    }
}

pub(crate) fn level_cmp(natural: bool, left: &str, right: &str) -> std::cmp::Ordering {
    if natural { natural_cmp(left, right) } else { left.cmp(right) }
}

fn sorted_levels(mut values: Vec<String>, natural: bool) -> Vec<String> {
    values.sort_unstable_by(|left, right| level_cmp(natural, left, right));
    values.dedup();
    values
}

fn level_code(levels: &[String], value: &str, natural: bool) -> Result<usize, usize> {
    levels.binary_search_by(|candidate| level_cmp(natural, candidate, value))
}

fn text_parts(values: Vec<Option<String>>, natural: bool) -> Parts<String> {
    let mut observed: Vec<_> = values.iter().flatten().cloned().collect();
    let unique = sorted_levels(observed.clone(), natural);
    let mut counts = vec![0; unique.len()];
    let mut codes = Vec::with_capacity(values.len());
    for value in &values {
        if let Some(value) = value {
            let code = level_code(&unique, value, natural).unwrap();
            counts[code] += 1;
            codes.push(code as u32);
        } else {
            codes.push(u32::MAX);
        }
    }
    let middle = observed.len() / 2;
    observed.select_nth_unstable_by(middle, |left, right| level_cmp(natural, left, right));
    let median = observed[middle].clone();
    Parts { unique, codes, counts, median }
}

fn empty_fitted_column(name: String) -> FittedColumn {
    FittedColumn {
        column: Column {
            name,
            values: Values::Text(Vec::new()),
            all_int: false,
            median_numeric: None,
            median_text: None,
            had_missing: true,
            encodings: Vec::new(),
            stats: None,
        },
        encodings: Vec::new(),
    }
}

fn finish_column(name: String, prepared: PreparedColumn) -> FittedColumn {
    let PreparedColumn { values, mut codes, counts, median_numeric, median_text, missing, all_int } = prepared;
    let had_missing = missing.iter().any(|value| *value);
    if had_missing {
        let missing_rank = counts.len() as u32;
        codes.iter_mut().filter(|code| **code == u32::MAX).for_each(|code| *code = missing_rank);
    }
    let cardinality = counts.len();
    let mut fitted = Vec::new();
    if cardinality > 1 || had_missing {
        let mut cutoff: Vec<_> = match &values {
            Values::Numeric(unique) => rank_bounds(unique),
            Values::Text(unique) | Values::Categorical(unique) => (0..unique.len()).map(|index| index.saturating_sub(1) as f32).collect(),
        };
        if had_missing {
            cutoff.push(f32::MAX);
        }
        fitted.push(FittedEncoding { kind: Encoding::Ordered, bounds: cutoff, codes: FitCodes::Materialized(codes.into()) });
    }
    FittedColumn {
        column: Column { name, values, all_int, median_numeric, median_text, had_missing, encodings: Vec::new(), stats: None },
        encodings: fitted,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EncoderOptions {
    pub target_statistics: bool,
    pub min_rows_per_level: usize,
    pub min_stat_agreement: f32,
    pub frequency: bool,
    pub natural_sort: bool,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self { target_statistics: true, min_rows_per_level: 20, min_stat_agreement: 0.75, frequency: false, natural_sort: false }
    }
}

impl EncoderOptions {
    fn validate(&self) -> Result<(), ForestError> {
        if self.min_rows_per_level == 0 {
            return Err(invalid("min_rows_per_level must be positive"));
        }
        if !self.min_stat_agreement.is_finite() || !(0.0..=1.0).contains(&self.min_stat_agreement) {
            return Err(invalid("min_stat_agreement must be between zero and one"));
        }
        Ok(())
    }
}

pub enum StatTarget<'a> {
    Regression(&'a [f32]),
    Binary(&'a [u32]),
}

#[derive(Default)]
pub struct StatContext<'a> {
    pub target: Option<StatTarget<'a>>,
    pub order_column: Option<usize>,
    pub agreement_sample_rows: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LevelStats {
    counts: Vec<u32>,
    sums: Vec<f64>,
    prior: f32,
    min_rows: u32,
}

impl LevelStats {
    fn statistic(&self, code: Option<usize>) -> f32 {
        let value = match code {
            Some(code) if self.counts[code] >= self.min_rows => (self.sums[code] / self.counts[code] as f64) as f32,
            _ => self.prior,
        };
        value
    }

    fn count(&self, code: Option<usize>) -> f32 {
        code.map_or(0.0, |code| self.counts[code] as f32)
    }
}

fn rank_bounds(distinct: &[f32]) -> Vec<f32> {
    distinct.iter().enumerate().map(|(index, value)| if index == 0 { *value } else { distinct[index - 1] }).collect()
}

fn fit_counter(counts: &[u32]) -> (Vec<u32>, Vec<f32>) {
    let mut distinct = counts.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let floats: Vec<f32> = distinct.iter().map(|&count| count as f32).collect();
    let bounds = rank_bounds(&floats);
    let table: Vec<u32> = counts.iter().map(|count| distinct.partition_point(|level| level < count) as u32).collect();
    (table, bounds)
}

fn fit_float_ranks(values: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut distinct = values.to_vec();
    distinct.sort_unstable_by(f32::total_cmp);
    distinct.dedup_by(|left, right| left.total_cmp(right).is_eq());
    let bounds = rank_bounds(&distinct);
    let table = values.iter().map(|value| distinct.binary_search_by(|candidate| candidate.total_cmp(value)).unwrap() as u32).collect();
    (table, bounds)
}

fn grouped_order(codes: &[u32]) -> Vec<usize> {
    let mut counts = vec![0usize; codes.iter().copied().max().unwrap_or(0) as usize + 1];
    for &code in codes {
        counts[code as usize] += 1;
    }
    let mut starts = vec![0usize; counts.len() + 1];
    for index in 0..counts.len() {
        starts[index + 1] = starts[index] + counts[index];
    }
    let mut next = starts[..counts.len()].to_vec();
    let mut rows = vec![0usize; codes.len()];
    for (row, &code) in codes.iter().enumerate() {
        rows[next[code as usize]] = row;
        next[code as usize] += 1;
    }
    rows
}

fn stat_agrees(codes: &FitCodes, y: &[f32], levels: usize, floor: u32, early: &[usize], later: &[usize], minimum: f32) -> bool {
    if minimum == 0.0 {
        return true;
    }
    if early.is_empty() || later.is_empty() {
        return false;
    }
    let level_totals = |rows: &[usize]| {
        let (mut counts, mut sums) = (vec![0u32; levels], vec![0f64; levels]);
        for &row in rows {
            let code = codes.at(row) as usize;
            counts[code] += 1;
            sums[code] += y[row] as f64;
        }
        (counts, sums)
    };
    let prior = early.iter().map(|&row| y[row] as f64).sum::<f64>() / early.len() as f64;
    let (early_counts, early_sums) = level_totals(early);
    let (late_counts, late_sums) = level_totals(later);
    let mut predicted = Vec::new();
    let mut observed = Vec::new();
    let mut weights = Vec::new();
    for code in 0..levels {
        let n_late = late_counts[code];
        if n_late == 0 {
            continue;
        }
        let n_early = early_counts[code];
        let value = if n_early >= floor { early_sums[code] / n_early as f64 } else { prior };
        predicted.push(value as f32);
        observed.push((late_sums[code] / n_late as f64) as f32);
        weights.push(n_late as f64);
    }
    let (predicted, predicted_bounds) = fit_float_ranks(&predicted);
    let (observed, observed_bounds) = fit_float_ranks(&observed);
    if predicted_bounds.len() < 2 || observed_bounds.len() < 2 {
        return false;
    }
    let weight = weights.iter().sum::<f64>();
    let mean_predicted = predicted.iter().zip(&weights).map(|(&rank, &weight)| rank as f64 * weight).sum::<f64>() / weight;
    let mean_observed = observed.iter().zip(&weights).map(|(&rank, &weight)| rank as f64 * weight).sum::<f64>() / weight;
    let mut covariance = 0.0;
    let mut predicted_variance = 0.0;
    let mut observed_variance = 0.0;
    for ((&predicted, &observed), &weight) in predicted.iter().zip(&observed).zip(&weights) {
        let predicted = predicted as f64 - mean_predicted;
        let observed = observed as f64 - mean_observed;
        covariance += weight * predicted * observed;
        predicted_variance += weight * predicted * predicted;
        observed_variance += weight * observed * observed;
    }
    let scale = (predicted_variance * observed_variance).sqrt();
    scale > 0.0 && covariance >= minimum as f64 * scale
}

fn derived_features(
    fitted: &mut FittedColumn, y: Option<&[f32]>, agreement_rows: Option<(&[usize], &[usize])>, prior: f32, options: &EncoderOptions,
) {
    let levels = fitted.column.cardinality() + usize::from(fitted.column.had_missing);
    let Some(codes) = fitted.training_codes() else { return };
    if levels < MIN_STAT_CARDINALITY {
        return;
    }
    let mut counts = vec![0u32; levels];
    for row in 0..codes.len() {
        counts[codes.at(row) as usize] += 1;
    }
    let floor = options.min_rows_per_level as u32;
    if counts.iter().filter(|&&count| count >= floor).count() < MIN_STAT_CARDINALITY {
        return;
    }
    let mut sums = if y.is_some() { vec![0f64; levels] } else { Vec::new() };
    let mut statistic = None;
    let agreeing = y.is_none_or(|y| {
        agreement_rows.is_none_or(|(early, later)| {
            let sampled = early.len() + later.len();
            let agreement_floor = (floor as usize * sampled).div_ceil(codes.len()).max(1) as u32;
            stat_agrees(codes, y, levels, agreement_floor, early, later, options.min_stat_agreement)
        })
    });
    if let Some(y) = y
        && agreeing
    {
        for row in 0..codes.len() {
            sums[codes.at(row) as usize] += y[row] as f64;
        }
        let values: Vec<f32> =
            counts.iter().zip(&sums).map(|(&count, &sum)| if count >= floor { (sum / count as f64) as f32 } else { prior }).collect();
        let (table, bounds) = fit_float_ranks(&values);
        if bounds.len() > 1 {
            statistic = Some(FittedEncoding { kind: Encoding::Statistic, bounds, codes: codes.projected(table) });
        }
    }
    let counter = options.frequency.then(|| {
        let (table, bounds) = fit_counter(&counts);
        FittedEncoding { kind: Encoding::Counter, bounds, codes: codes.projected(table) }
    });
    if let Some(statistic) = statistic {
        fitted.encodings.push(statistic);
    }
    if let Some(counter) = counter {
        fitted.encodings.push(counter);
    }
    fitted.column.stats = Some(LevelStats { counts, sums, prior, min_rows: floor });
}

fn fit_date_part(base: DateBase, part: u8, name: String) -> FittedColumn {
    if base.distinct.is_empty() {
        return empty_fitted_column(name);
    }
    let values: Vec<f32> = base.distinct.iter().map(|&value| date_value(value, part)).collect();
    let mut unique = values.clone();
    unique.sort_unstable_by(f32::total_cmp);
    unique.dedup_by(|left, right| left.total_cmp(right).is_eq());
    let mut table: Vec<u32> =
        values.iter().map(|value| unique.binary_search_by(|candidate| candidate.total_cmp(value)).unwrap() as u32).collect();
    let mut counts = vec![0usize; unique.len()];
    for &code in base.codes.iter() {
        if let Some(&rank) = table.get(code as usize) {
            counts[rank as usize] += 1;
        }
    }
    let observed = counts.iter().sum::<usize>();
    let mut cumulative = 0;
    let median = counts
        .iter()
        .position(|count| {
            cumulative += count;
            cumulative > observed / 2
        })
        .map(|index| unique[index]);
    let mut encodings = Vec::new();
    if unique.len() > 1 || base.had_missing {
        let mut cutoff = rank_bounds(&unique);
        if base.had_missing {
            cutoff.push(f32::MAX);
            table.push(unique.len() as u32);
        }
        encodings.push(FittedEncoding { kind: Encoding::Ordered, bounds: cutoff, codes: FitCodes::Projected { base: base.codes, table } });
    }
    FittedColumn {
        column: Column {
            name,
            values: Values::Numeric(unique),
            all_int: true,
            median_numeric: median,
            median_text: None,
            had_missing: base.had_missing,
            encodings: Vec::new(),
            stats: None,
        },
        encodings,
    }
}

fn fit_arranged(column: ArrangedColumn<DateBase>, name: String, natural: bool) -> Result<FittedColumn, ForestError> {
    match column {
        ArrangedColumn::Raw(raw) => fit_column(raw, name, natural),
        ArrangedColumn::DatePart { date, part } => Ok(fit_date_part(date, part, name)),
    }
}

fn fit_column(raw: RawColumn, name: String, natural: bool) -> Result<FittedColumn, ForestError> {
    let raw = match raw {
        RawColumn::Bundle { codes, categories } => {
            let mut counts = vec![0; categories.len()];
            for code in codes.iter().flatten() {
                counts[*code as usize] += 1;
            }
            let observed = codes.iter().flatten().count();
            let mut cumulative = 0;
            let median_code = counts
                .iter()
                .position(|count| {
                    cumulative += count;
                    cumulative > observed / 2
                })
                .unwrap_or(0) as u32;
            let missing = codes.iter().map(Option::is_none).collect();
            return Ok(finish_column(
                name,
                PreparedColumn {
                    values: Values::Categorical(categories.to_vec()),
                    codes: codes.into_iter().map(|code| code.unwrap_or(u32::MAX)).collect(),
                    counts,
                    median_numeric: None,
                    median_text: Some(categories[median_code as usize].clone()),
                    missing,
                    all_int: false,
                },
            ));
        }
        RawColumn::Categorical { codes, categories, null_value } => {
            let labels: Vec<_> = codes
                .iter()
                .map(|code| if *code < 0 { null_value.as_ref() } else { categories.get(*code as usize).and_then(Option::as_ref) })
                .collect();
            let unique = sorted_levels(labels.iter().filter_map(|value| (*value).cloned()).collect(), natural);
            if unique.is_empty() {
                return Ok(empty_fitted_column(name));
            }
            let mut counts = vec![0; unique.len()];
            let ranked: Vec<_> = labels
                .iter()
                .map(|value| {
                    value.map_or(u32::MAX, |value| {
                        let code = level_code(&unique, value, natural).unwrap();
                        counts[code] += 1;
                        code as u32
                    })
                })
                .collect();
            let missing: Vec<_> = labels.iter().map(|value| value.is_none()).collect();
            let observed = ranked.len() - missing.iter().filter(|value| **value).count();
            let mut cumulative = 0;
            let median_code = counts
                .iter()
                .position(|count| {
                    cumulative += count;
                    cumulative > observed / 2
                })
                .unwrap() as u32;
            return Ok(finish_column(
                name,
                PreparedColumn {
                    median_text: Some(unique[median_code as usize].clone()),
                    values: Values::Text(unique),
                    codes: ranked,
                    counts,
                    median_numeric: None,
                    missing,
                    all_int: false,
                },
            ));
        }
        raw => raw,
    };
    let observed = match &raw {
        RawColumn::Numeric(values) => values.iter().flatten().count(),
        RawColumn::Text(values) => values.iter().flatten().count(),
        RawColumn::Categorical { .. } | RawColumn::Bundle { .. } => unreachable!(),
    };
    if observed == 0 {
        return Ok(empty_fitted_column(name));
    }
    let numeric = match raw {
        RawColumn::Numeric(values) => Some(values),
        RawColumn::Text(values) => match parse_numeric(&values, &name)? {
            Some(parsed) => Some(parsed),
            None => {
                let missing: Vec<_> = values.iter().map(Option::is_none).collect();
                let parts = text_parts(values, natural);
                return Ok(finish_column(
                    name,
                    PreparedColumn {
                        values: Values::Text(parts.unique),
                        codes: parts.codes,
                        counts: parts.counts,
                        median_numeric: None,
                        median_text: Some(parts.median),
                        missing,
                        all_int: false,
                    },
                ));
            }
        },
        RawColumn::Categorical { .. } | RawColumn::Bundle { .. } => unreachable!(),
    };
    let values = numeric.unwrap();
    let missing: Vec<_> = values.iter().map(Option::is_none).collect();
    let all_int = values.iter().flatten().all(|value| value.fract() == 0.0);
    let parts = numeric_parts(values, &name)?;
    Ok(finish_column(
        name,
        PreparedColumn {
            values: Values::Numeric(parts.unique),
            codes: parts.codes,
            counts: parts.counts,
            median_numeric: Some(parts.median),
            median_text: None,
            missing,
            all_int,
        },
    ))
}

fn assemble_range<T: Copy + Default + Send + Sync>(features: &[Vec<T>], start: usize, rows: usize) -> Array2<T> {
    let cols = features.len();
    let mut data = vec![T::default(); rows * cols];
    data.par_chunks_mut(cols.max(1)).enumerate().for_each(|(row, output)| {
        for (col, feature) in features.iter().enumerate() {
            output[col] = feature[start + row];
        }
    });
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn assemble<T: Copy + Default + Send + Sync>(features: &[Vec<T>], rows: usize) -> Array2<T> {
    assemble_range(features, 0, rows)
}

fn agreement_partition(rows: usize, order: Option<&[usize]>, sample_rows: usize, seed: Option<u64>) -> (Vec<usize>, Vec<usize>) {
    let total = sample_rows.min(rows);
    let n_early = total / 2;
    let n_late = total - n_early;
    if let Some(order) = order {
        let middle = rows / 2;
        let early = crate::forest::uniform_sample_indices(middle, n_early.min(middle), seed, 0x51a7_0001)
            .into_iter()
            .map(|row| order[row])
            .collect();
        let later_rows = rows - middle;
        let later = crate::forest::uniform_sample_indices(later_rows, n_late.min(later_rows), seed, 0x51a7_0002)
            .into_iter()
            .map(|row| order[middle + row])
            .collect();
        return (early, later);
    }
    let selected = crate::forest::uniform_sample_indices(rows, total, seed, 0x51a7_0003);
    let early_positions = crate::forest::uniform_sample_indices(total, n_early, seed, 0x51a7_0004);
    let mut is_early = vec![false; total];
    for position in early_positions {
        is_early[position] = true;
    }
    let mut early = Vec::with_capacity(n_early);
    let mut later = Vec::with_capacity(n_late);
    for (position, row) in selected.into_iter().enumerate() {
        if is_early[position] {
            early.push(row);
        } else {
            later.push(row);
        }
    }
    (early, later)
}

fn append_codes(codes: &FitCodes, physical: &mut Vec<Vec<u32>>, bases: &mut HashMap<usize, u32>, projections: &mut Projections) {
    let (base, table) = codes.base_and_table();
    // Bases are shared `Arc`s, so pointer equality identifies one physical column; a defensive copy upstream would silently duplicate it.
    let key = base.as_ptr() as usize;
    let column = *bases.entry(key).or_insert_with(|| {
        physical.push(base.to_vec());
        physical.len() as u32 - 1
    });
    match table {
        Some(table) => projections.push_projected(column, table),
        None => projections.push_direct(column),
    }
}

impl Encoder {
    pub fn fit_arrow(
        batch: &RecordBatch, markers: &[SavedValue], allow_new_missing: bool, date_columns: Vec<(usize, String)>, seed: Option<u64>,
        stats: &EncoderOptions, context: StatContext<'_>,
    ) -> Result<(Self, Array2<u32>), ForestError> {
        let names = batch.schema().fields().iter().map(|field| field.name().clone()).collect();
        let date_indices: Vec<_> = date_columns.iter().map(|(index, _)| *index).collect();
        let columns = arrow_columns(batch, markers, &date_indices)?;
        Self::fit(columns, names, allow_new_missing, date_columns, seed, stats, context)
    }

    pub fn transform_arrow(&self, batch: &RecordBatch, markers: &[SavedValue]) -> Result<Array2<f32>, ForestError> {
        let schema = batch.schema();
        let names: Vec<_> = schema.fields().iter().map(|field| field.name().as_str()).collect();
        if names != self.input_names.iter().map(String::as_str).collect::<Vec<_>>() {
            return Err(invalid("prediction columns must match training columns"));
        }
        let direct_numeric = markers.iter().all(|marker| marker.kind == 5 && marker.value.is_empty())
            && self.input_columns.iter().all(|source| matches!(source, InputColumn::Direct(_)))
            && self.columns.iter().all(|column| column.is_numeric() && !column.had_missing())
            && batch.columns().iter().all(|array| numeric_arrow_type(array.data_type()));
        if direct_numeric {
            let encoded_cols = self.encoded_to_raw.len();
            let mut data = vec![0.0; batch.num_rows() * encoded_cols];
            data.par_chunks_mut(encoded_cols.max(1)).enumerate().try_for_each(|(row, output)| {
                let mut encoded = 0;
                for (array, column) in batch.columns().iter().zip(&self.columns) {
                    let value = match numeric_arrow_value(array.as_ref(), row) {
                        Some(value) if !value.is_finite() => {
                            return Err(invalid(format!("column {:?} contains a non-finite numeric value", column.name)));
                        }
                        Some(value) => value,
                        None if self.allow_new_missing => f32::NAN,
                        None => {
                            return Err(invalid(format!(
                                "column {:?} has a missing value at row {row}, but had none during training",
                                column.name
                            )));
                        }
                    };
                    let Values::Numeric(unique) = &column.values else { unreachable!() };
                    let code =
                        column.stats.is_some().then(|| unique.binary_search_by(|candidate| candidate.total_cmp(&value)).ok()).flatten();
                    for encoding in &column.encodings {
                        output[encoded] = match encoding {
                            Encoding::Ordered => value,
                            Encoding::Statistic => column.stats.as_ref().unwrap().statistic(code),
                            Encoding::Counter => column.stats.as_ref().unwrap().count(code),
                        };
                        encoded += 1;
                    }
                }
                Ok::<_, ForestError>(())
            })?;
            return Ok(Array2::from_shape_vec((batch.num_rows(), encoded_cols), data).unwrap());
        }
        let date_indices: Vec<_> = self.date_columns().into_iter().map(|(index, _)| index).collect();
        self.transform(arrow_columns(batch, markers, &date_indices)?)
    }

    pub(crate) fn validate_loaded(&self) -> Result<(), ForestError> {
        if self.input_names.is_empty() || self.input_columns.len() != self.columns.len() {
            return Err(invalid("saved preprocessing schema dimensions are invalid"));
        }
        let encoded = self.encoded_to_raw.len();
        if self.cutoff_offsets.len() != encoded + 1
            || self.cutoff_offsets.first() != Some(&0)
            || self.cutoff_offsets.last() != Some(&self.cutoff_values.len())
            || self.cutoff_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self.encoded_to_raw.iter().any(|&column| column >= self.columns.len())
        {
            return Err(invalid("saved preprocessing feature mappings are invalid"));
        }
        for source in &self.input_columns {
            if let InputColumn::Bundle { indices, categories } = source
                && (indices.len() < 2 || indices.len() != categories.len() || indices.iter().any(|&index| index >= self.input_names.len()))
            {
                return Err(invalid("saved automatic bundle is invalid"));
            }
        }
        for column in &self.columns {
            let has_statistic = column.encodings.iter().any(|encoding| matches!(encoding, Encoding::Statistic));
            let needs_stats = has_statistic || column.encodings.iter().any(|encoding| matches!(encoding, Encoding::Counter));
            match &column.stats {
                None if needs_stats => return Err(invalid("saved statistic encoding has no level table")),
                Some(stats)
                    if stats.counts.len() != column.cardinality() + usize::from(column.had_missing)
                        || (has_statistic && stats.sums.len() != stats.counts.len()) =>
                {
                    return Err(invalid("saved level table is invalid"));
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(crate) fn fit(
        columns: Vec<RawColumn>, names: Vec<String>, allow_new_missing: bool, date_columns: Vec<(usize, String)>, seed: Option<u64>,
        stats: &EncoderOptions, context: StatContext<'_>,
    ) -> Result<(Self, Array2<u32>), ForestError> {
        let StatContext { target, order_column, agreement_sample_rows } = context;
        stats.validate()?;
        let rows = validate_rows(&columns, &names)?;
        let date_parts = date_layout(&names, &date_columns)?;
        let mut excluded = vec![false; names.len()];
        for (index, _, _, _) in &date_parts {
            excluded[*index] = true;
        }
        if let Some(index) = order_column {
            let excluded = excluded.get_mut(index).ok_or_else(|| invalid("order column is out of range"))?;
            *excluded = true;
        }
        let bundles = automatic_bundles(&columns, &names, &excluded, seed);
        let (input_columns, logical_names) = input_layout(&names, &bundles, &date_parts)?;
        let columns = arrange_training_columns(columns, &input_columns, &logical_names)?;
        let date_order = order_column.and_then(|order| {
            input_columns.iter().zip(&columns).find_map(|(source, arranged)| match (source, arranged) {
                (InputColumn::DatePart { index, .. }, ArrangedColumn::DatePart { date, .. }) if *index == order => Some(date.codes.clone()),
                _ => None,
            })
        });
        let natural = stats.natural_sort;
        let fitted: Result<Vec<_>, _> =
            columns.into_par_iter().zip(logical_names).map(|(column, name)| fit_arranged(column, name, natural)).collect();
        let mut fitted = fitted?;
        let explicit_order = if stats.target_statistics {
            match (order_column, date_order) {
                (_, Some(codes)) => Some(grouped_order(&codes)),
                (Some(order), None) => {
                    let position = input_columns
                        .iter()
                        .position(|source| matches!(source, InputColumn::Direct(index) if *index == order))
                        .ok_or_else(|| invalid("order column is unavailable after feature arrangement"))?;
                    let codes = fitted[position]
                        .training_codes()
                        .ok_or_else(|| invalid("order column must contain at least two distinct values"))?;
                    Some(grouped_order(&(0..codes.len()).map(|row| codes.at(row)).collect::<Vec<_>>()))
                }
                (None, None) => None,
            }
        } else {
            None
        };
        let owned;
        let y: Option<&[f32]> = match target {
            Some(StatTarget::Regression(y)) => Some(y),
            Some(StatTarget::Binary(y)) => {
                owned = y.iter().map(|&value| value as f32).collect::<Vec<_>>();
                Some(&owned)
            }
            None => None,
        };
        if let Some(y) = y
            && y.len() != rows
        {
            return Err(invalid("target length must match the training rows"));
        }
        let stat_y = if stats.target_statistics { y } else { None };
        let agreement_rows = stat_y.map(|_| agreement_partition(rows, explicit_order.as_deref(), agreement_sample_rows, seed));
        if stat_y.is_some() || stats.frequency {
            let prior = stat_y.map_or(0.0f32, |y| y.iter().sum::<f32>() / y.len().max(1) as f32);
            let agreement = agreement_rows.as_ref().map(|(early, later)| (early.as_slice(), later.as_slice()));
            fitted.par_iter_mut().for_each(|fitted| derived_features(fitted, stat_y, agreement, prior, stats));
        }
        let mut physical: Vec<Vec<u32>> = Vec::new();
        let mut projections = Projections { phys: Vec::new(), base: Vec::new(), tables: Vec::new(), offsets: vec![0] };
        let mut cutoff_values = Vec::new();
        let mut cutoff_offsets = vec![0];
        let mut encoded_to_raw = Vec::new();
        let mut bases = HashMap::new();
        for (raw, column) in fitted.iter().enumerate() {
            for encoding in &column.encodings {
                cutoff_values.extend(&encoding.bounds);
                cutoff_offsets.push(cutoff_values.len());
                encoded_to_raw.push(raw);
                append_codes(&encoding.codes, &mut physical, &mut bases, &mut projections);
            }
        }
        let matrix = assemble(&physical, rows);
        let encoder = Self {
            columns: fitted
                .into_iter()
                .map(|mut fitted| {
                    fitted.column.encodings = fitted.encodings.into_iter().map(|encoding| encoding.kind).collect();
                    fitted.column
                })
                .collect(),
            input_names: names,
            input_columns,
            allow_new_missing,
            natural_sort: natural,
            cutoff_values,
            cutoff_offsets,
            encoded_to_raw,
            projections,
        };
        Ok((encoder, matrix))
    }

    fn transform_features(&self, columns: Vec<RawColumn>) -> Result<(Vec<Vec<f32>>, usize), ForestError> {
        let rows = validate_rows(&columns, &self.input_names)?;
        let names: Vec<_> = self.columns.iter().map(|column| column.name.clone()).collect();
        let columns = arrange_inference_columns(columns, &self.input_columns, &names)?;
        let encoded: Result<Vec<_>, _> = columns
            .into_par_iter()
            .zip(&self.columns)
            .zip(&self.input_columns)
            .map(|((raw, fitted), source)| {
                transform_column(raw, fitted, self.allow_new_missing || matches!(source, InputColumn::DatePart { .. }), self.natural_sort)
            })
            .collect();
        let features: Vec<_> = encoded?.into_iter().flatten().collect();
        Ok((features, rows))
    }

    fn transform(&self, columns: Vec<RawColumn>) -> Result<Array2<f32>, ForestError> {
        let (features, rows) = self.transform_features(columns)?;
        Ok(assemble(&features, rows))
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn logical_names(&self) -> Vec<String> {
        self.columns.iter().map(|column| column.name.clone()).collect()
    }

    pub fn date_columns(&self) -> Vec<(usize, String)> {
        let mut result = Vec::new();
        for source in &self.input_columns {
            if let InputColumn::DatePart { index, format, .. } = source
                && !result.iter().any(|(seen, _)| seen == index)
            {
                result.push((*index, format.clone()));
            }
        }
        result
    }

    pub fn date_layout(&self) -> Vec<(usize, String, Vec<String>)> {
        let mut result: Vec<(usize, String, Vec<String>)> = Vec::new();
        for (source, column) in self.input_columns.iter().zip(&self.columns) {
            if let InputColumn::DatePart { index, format, .. } = source {
                if let Some((_, _, names)) = result.iter_mut().find(|(seen, _, _)| seen == index) {
                    names.push(column.name.clone());
                } else {
                    result.push((*index, format.clone(), vec![column.name.clone()]));
                }
            }
        }
        result
    }

    pub fn bundle_layout(&self) -> Vec<(String, Vec<usize>, Vec<String>)> {
        self.input_columns
            .iter()
            .zip(&self.columns)
            .filter_map(|(source, column)| match source {
                InputColumn::Bundle { indices, categories } => Some((column.name.clone(), indices.clone(), categories.clone())),
                _ => None,
            })
            .collect()
    }

    pub fn date_values_arrow(&self, batch: &RecordBatch, markers: &[SavedValue]) -> Result<Array2<f32>, ForestError> {
        let dates = self.date_columns();
        if dates.is_empty() {
            return Ok(Array2::zeros((batch.num_rows(), 0)));
        }
        let date_indices: Vec<_> = dates.iter().map(|(index, _)| *index).collect();
        let raw = arrow_columns(batch, markers, &date_indices)?;
        let parsed: Result<Vec<_>, _> =
            dates.iter().map(|(index, format)| parse_dates(&raw[*index], format, &self.input_names[*index])).collect();
        let parsed = parsed?;
        let mut data = Vec::with_capacity(batch.num_rows() * dates.len() * DATE_PARTS.len());
        for row in 0..batch.num_rows() {
            for values in &parsed {
                for part in 0..DATE_PARTS.len() {
                    data.push(values[row].map_or(f32::NAN, |value| date_value(value, part as u8)));
                }
            }
        }
        Ok(Array2::from_shape_vec((batch.num_rows(), dates.len() * DATE_PARTS.len()), data).unwrap())
    }

    pub fn training_layout(&self) -> (Vec<f32>, Vec<usize>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<usize>) {
        let projections = &self.projections;
        (
            self.cutoff_values.clone(),
            self.cutoff_offsets.clone(),
            self.missing_ranks(),
            projections.phys.clone(),
            projections.base.clone(),
            projections.tables.clone(),
            projections.offsets.clone(),
        )
    }

    pub fn projections(&self) -> &Projections {
        &self.projections
    }

    pub fn cutoff_values(&self) -> &[f32] {
        &self.cutoff_values
    }

    pub fn cutoff_offsets(&self) -> &[usize] {
        &self.cutoff_offsets
    }

    pub fn encoded_to_raw(&self) -> &[usize] {
        &self.encoded_to_raw
    }

    pub fn missing_ranks(&self) -> Vec<u32> {
        self.columns
            .iter()
            .flat_map(|column| {
                let rank = if column.had_missing { column.cardinality() as u32 } else { u32::MAX };
                column.encodings.iter().map(move |encoding| match encoding {
                    Encoding::Ordered => rank,
                    Encoding::Statistic | Encoding::Counter => u32::MAX,
                })
            })
            .collect()
    }
}

fn numeric_input(raw: RawColumn, fitted: &Column) -> Result<Vec<Option<f32>>, ForestError> {
    match raw.into_simple() {
        RawColumn::Numeric(values) => {
            if values.iter().flatten().any(|value| !value.is_finite()) {
                return Err(invalid(format!("column {:?} contains a non-finite numeric value", fitted.name)));
            }
            Ok(values)
        }
        RawColumn::Text(values) => {
            parse_numeric(&values, &fitted.name)?.ok_or_else(|| invalid(format!("column {:?} was numeric during training", fitted.name)))
        }
        RawColumn::Categorical { .. } | RawColumn::Bundle { .. } => unreachable!(),
    }
}

fn text_input(raw: RawColumn) -> Vec<Option<String>> {
    match raw.into_simple() {
        RawColumn::Text(values) => values,
        RawColumn::Numeric(values) => values
            .into_iter()
            .map(|value| value.map(|value| if value.fract() == 0.0 { format!("{value:.1}") } else { value.to_string() }))
            .collect(),
        RawColumn::Categorical { .. } | RawColumn::Bundle { .. } => unreachable!(),
    }
}

fn transform_column(raw: RawColumn, fitted: &Column, allow_new_missing: bool, natural: bool) -> Result<Vec<Vec<f32>>, ForestError> {
    if fitted.encodings.is_empty() {
        return Ok(Vec::new());
    }
    let missing = raw.missing();
    if !fitted.had_missing
        && !allow_new_missing
        && let Some(row) = missing.iter().position(|missing| *missing)
    {
        return Err(invalid(format!("column {:?} has a missing value at row {row}, but had none during training", fitted.name)));
    }
    let (ordered, codes): (Vec<f32>, Vec<Option<usize>>) = match &fitted.values {
        Values::Numeric(unique) => {
            let values = numeric_input(raw, fitted)?;
            let codes = values
                .iter()
                .map(|value| value.and_then(|value| unique.binary_search_by(|candidate| candidate.total_cmp(&value)).ok()))
                .collect();
            (values.iter().map(|value| value.unwrap_or(f32::NAN)).collect(), codes)
        }
        Values::Text(unique) => {
            let values = text_input(raw);
            let mut ordered = Vec::with_capacity(values.len());
            let codes = values
                .iter()
                .map(|value| match value {
                    None => {
                        ordered.push(f32::NAN);
                        None
                    }
                    Some(value) => match level_code(unique, value, natural) {
                        Ok(index) => {
                            ordered.push(index as f32);
                            Some(index)
                        }
                        Err(index) => {
                            ordered.push(index as f32 - 0.5);
                            None
                        }
                    },
                })
                .collect();
            (ordered, codes)
        }
        Values::Categorical(_) => {
            let RawColumn::Bundle { codes, .. } = raw else {
                return Err(invalid(format!("column {:?} expected bundled input", fitted.name)));
            };
            (
                codes.iter().map(|code| code.map_or(f32::NAN, |code| code as f32)).collect(),
                codes.into_iter().map(|code| code.map(|code| code as usize)).collect(),
            )
        }
    };
    let codes: Vec<Option<usize>> = if fitted.had_missing {
        let missing_code = fitted.cardinality();
        codes.iter().zip(&missing).map(|(&code, &is_missing)| if is_missing { Some(missing_code) } else { code }).collect()
    } else {
        codes
    };
    let result = fitted
        .encodings
        .iter()
        .map(|encoding| match encoding {
            Encoding::Ordered => ordered.clone(),
            Encoding::Statistic => {
                let stats = fitted.stats.as_ref().unwrap();
                codes.iter().map(|&code| stats.statistic(code)).collect()
            }
            Encoding::Counter => {
                let stats = fitted.stats.as_ref().unwrap();
                codes.iter().map(|&code| stats.count(code)).collect()
            }
        })
        .collect();
    Ok(result)
}
