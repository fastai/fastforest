use std::collections::HashMap;

use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::ForestError;

#[derive(Clone, Debug)]
pub enum RawColumn {
    Numeric(Vec<Option<f32>>),
    Text(Vec<Option<String>>),
    Categorical {
        codes: Vec<i32>,
        categories: Vec<Option<String>>,
        null_value: Option<String>,
    },
}

impl RawColumn {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Text(values) => values.len(),
            Self::Categorical { codes, .. } => codes.len(),
        }
    }

    fn expand_categories(
        codes: Vec<i32>,
        categories: Vec<Option<String>>,
        null_value: Option<String>,
    ) -> Vec<Option<String>> {
        codes
            .into_iter()
            .map(|code| {
                if code < 0 {
                    null_value.clone()
                } else {
                    categories.get(code as usize).cloned().unwrap_or(None)
                }
            })
            .collect()
    }

    fn into_simple(self) -> Self {
        match self {
            Self::Categorical {
                codes,
                categories,
                null_value,
            } => Self::Text(Self::expand_categories(codes, categories, null_value)),
            simple => simple,
        }
    }

    fn missing(&self) -> Vec<bool> {
        match self {
            Self::Numeric(values) => values.iter().map(Option::is_none).collect(),
            Self::Text(values) => values.iter().map(Option::is_none).collect(),
            Self::Categorical {
                codes,
                categories,
                null_value,
            } => codes
                .iter()
                .map(|code| {
                    if *code < 0 {
                        null_value.is_none()
                    } else {
                        categories.get(*code as usize).is_none_or(Option::is_none)
                    }
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Encoding {
    Ordered,
    Dummy(u32),
    Missing,
}

#[derive(Clone, Debug)]
enum Values {
    Numeric(Vec<f32>),
    Text(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct Column {
    name: String,
    values: Values,
    all_int: bool,
    median_numeric: Option<f32>,
    median_text: Option<String>,
    had_missing: bool,
    encodings: Vec<Encoding>,
}

impl Column {
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
            Values::Text(_) => &[],
        }
    }

    pub fn text_values(&self) -> &[String] {
        match &self.values {
            Values::Numeric(_) => &[],
            Values::Text(values) => values,
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
    ranked: Vec<Vec<u32>>,
    bounds: Vec<Vec<f32>>,
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
    median_code: u32,
    median_numeric: Option<f32>,
    median_text: Option<String>,
    missing: Vec<bool>,
    all_int: bool,
}

#[derive(Clone, Debug)]
pub struct Encoder {
    columns: Vec<Column>,
    input_names: Vec<String>,
    input_columns: Vec<InputColumn>,
    cutoff_values: Vec<f32>,
    cutoff_offsets: Vec<usize>,
    encoded_to_raw: Vec<usize>,
    feature_group_ids: Vec<usize>,
}

#[derive(Clone, Debug)]
enum InputColumn {
    Direct(usize),
    OneHot {
        indices: Vec<usize>,
        categories: Vec<String>,
    },
    DatePart {
        index: usize,
        format: String,
        part: u8,
    },
}

fn invalid(message: impl Into<String>) -> ForestError {
    ForestError::new(message)
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

fn input_layout(
    names: &[String],
    groups: &[(String, Vec<usize>)],
    date_parts: &[(usize, String, u8, String)],
) -> Result<(Vec<InputColumn>, Vec<String>), ForestError> {
    let mut grouped = vec![false; names.len()];
    for (group, indices) in groups {
        if indices.len() < 2 {
            return Err(invalid(format!(
                "one-hot group {group:?} must contain at least two columns"
            )));
        }
        for &index in indices {
            if index >= names.len() {
                return Err(invalid(format!(
                    "one-hot group {group:?} contains an out-of-range column"
                )));
            }
            if std::mem::replace(&mut grouped[index], true) {
                return Err(invalid(format!(
                    "column {:?} belongs to more than one one-hot group",
                    names[index]
                )));
            }
        }
    }
    let mut date_columns = vec![false; names.len()];
    for (index, _, _, _) in date_parts {
        if *index >= names.len() {
            return Err(invalid("date column is out of range"));
        }
        if grouped[*index] && !date_columns[*index] {
            return Err(invalid(format!(
                "column {:?} cannot be both grouped and expanded as a date",
                names[*index]
            )));
        }
        grouped[*index] = true;
        date_columns[*index] = true;
    }
    let mut input_columns = Vec::new();
    let mut logical_names = Vec::new();
    for (index, name) in names.iter().enumerate() {
        if !grouped[index] {
            input_columns.push(InputColumn::Direct(index));
            logical_names.push(name.clone());
        }
    }
    for (group, indices) in groups {
        input_columns.push(InputColumn::OneHot {
            indices: indices.clone(),
            categories: indices.iter().map(|index| names[*index].clone()).collect(),
        });
        logical_names.push(group.clone());
    }
    for (index, format, part, name) in date_parts {
        input_columns.push(InputColumn::DatePart {
            index: *index,
            format: format.clone(),
            part: *part,
        });
        logical_names.push(name.clone());
    }
    let mut unique = logical_names.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != logical_names.len() {
        return Err(invalid("one-hot group and feature names must be unique"));
    }
    Ok((input_columns, logical_names))
}

fn parse_numeric(
    values: &[Option<String>],
    name: &str,
) -> Result<Option<Vec<Option<f32>>>, ForestError> {
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
            return Err(invalid(format!(
                "column {name:?} contains a non-finite numeric value"
            )));
        }
        parsed.push(Some(value));
    }
    Ok(Some(parsed))
}

fn indicator_values(raw: RawColumn, group: &str) -> Result<Vec<Option<f32>>, ForestError> {
    match raw.into_simple() {
        RawColumn::Numeric(values) => Ok(values),
        RawColumn::Text(values) => parse_numeric(&values, group)?.ok_or_else(|| {
            invalid(format!(
                "one-hot group {group:?} must contain only numeric indicator columns"
            ))
        }),
        RawColumn::Categorical { .. } => unreachable!(),
    }
}

fn collapse_one_hot(
    columns: Vec<RawColumn>,
    categories: Vec<String>,
    group: &str,
) -> Result<RawColumn, ForestError> {
    let indicators: Result<Vec<_>, _> = columns
        .into_iter()
        .map(|column| indicator_values(column, group))
        .collect();
    let indicators = indicators?;
    let rows = indicators.first().map_or(0, Vec::len);
    let codes: Result<Vec<_>, _> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let mut active = None;
            for (category, values) in indicators.iter().enumerate() {
                let Some(value) = values[row] else {
                    return Err(invalid(format!(
                        "one-hot group {group:?} has a missing value at row {row}"
                    )));
                };
                if value != 0.0 && value != 1.0 {
                    return Err(invalid(format!(
                        "one-hot group {group:?} has a value other than 0 or 1 at row {row}"
                    )));
                }
                if value == 1.0 && active.replace(category).is_some() {
                    return Err(invalid(format!(
                        "one-hot group {group:?} has multiple active categories at row {row}"
                    )));
                }
            }
            active.map(|category| category as i32).ok_or_else(|| {
                invalid(format!(
                    "one-hot group {group:?} has no active category at row {row}"
                ))
            })
        })
        .collect();
    Ok(RawColumn::Categorical {
        codes: codes?,
        categories: categories.into_iter().map(Some).collect(),
        null_value: None,
    })
}

fn parse_dates(
    raw: &RawColumn,
    format: &str,
    name: &str,
) -> Result<Vec<Option<NaiveDateTime>>, ForestError> {
    let RawColumn::Text(values) = raw else {
        return Err(invalid(format!(
            "date column {name:?} must contain strings"
        )));
    };
    values
        .par_iter()
        .enumerate()
        .map(|(row, value)| {
            let Some(value) = value else { return Ok(None) };
            let parsed = NaiveDateTime::parse_from_str(value, format).or_else(|_| {
                NaiveDate::parse_from_str(value, format)
                    .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
            });
            parsed.map(Some).map_err(|_| {
                invalid(format!(
                    "date column {name:?} has an invalid value at row {row}"
                ))
            })
        })
        .collect()
}

fn date_value(value: NaiveDateTime, part: u8) -> f32 {
    let date = value.date();
    let month_start = date.day() == 1;
    let month_end = date
        .succ_opt()
        .is_none_or(|next| next.month() != date.month());
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

fn arrange_columns(
    columns: Vec<RawColumn>,
    input_columns: &[InputColumn],
    logical_names: &[String],
) -> Result<Vec<RawColumn>, ForestError> {
    let mut dates = HashMap::new();
    for (position, source) in input_columns.iter().enumerate() {
        if let InputColumn::DatePart { index, format, .. } = source {
            if !dates.contains_key(index) {
                dates.insert(
                    *index,
                    parse_dates(&columns[*index], format, &logical_names[position])?,
                );
            }
        }
    }
    let mut columns: Vec<_> = columns.into_iter().map(Some).collect();
    input_columns
        .iter()
        .zip(logical_names)
        .map(|(source, name)| match source {
            InputColumn::Direct(index) => Ok(columns[*index].take().unwrap()),
            InputColumn::OneHot {
                indices,
                categories,
            } => collapse_one_hot(
                indices
                    .iter()
                    .map(|index| columns[*index].take().unwrap())
                    .collect(),
                categories.clone(),
                name,
            ),
            InputColumn::DatePart { index, part, .. } => Ok(RawColumn::Numeric(
                dates[index]
                    .iter()
                    .map(|value| value.map(|value| date_value(value, *part)))
                    .collect(),
            )),
        })
        .collect()
}

fn numeric_parts(values: Vec<Option<f32>>, name: &str) -> Result<Parts<f32>, ForestError> {
    let observed: Vec<_> = values.iter().flatten().copied().collect();
    if observed.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "column {name:?} contains a non-finite numeric value"
        )));
    }
    let mut unique = observed.clone();
    unique.sort_unstable_by(|a, b| a.total_cmp(b));
    unique.dedup();
    let mut counts = vec![0; unique.len()];
    let mut codes = Vec::with_capacity(values.len());
    for value in &values {
        if let Some(value) = value {
            let code = unique
                .binary_search_by(|candidate| candidate.total_cmp(value))
                .unwrap();
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
    Ok(Parts {
        unique,
        codes,
        counts,
        median,
    })
}

fn text_parts(values: Vec<Option<String>>) -> Parts<String> {
    let mut observed: Vec<_> = values.iter().flatten().cloned().collect();
    let mut unique = observed.clone();
    unique.sort_unstable();
    unique.dedup();
    let mut counts = vec![0; unique.len()];
    let mut codes = Vec::with_capacity(values.len());
    for value in &values {
        if let Some(value) = value {
            let code = unique.binary_search(value).unwrap();
            counts[code] += 1;
            codes.push(code as u32);
        } else {
            codes.push(u32::MAX);
        }
    }
    let middle = observed.len() / 2;
    observed.select_nth_unstable(middle);
    let median = observed[middle].clone();
    Parts {
        unique,
        codes,
        counts,
        median,
    }
}

fn finish_column(
    name: String,
    prepared: PreparedColumn,
    max_dummy_cardinality: usize,
) -> FittedColumn {
    let PreparedColumn {
        values,
        mut codes,
        counts,
        median_code,
        median_numeric,
        median_text,
        missing,
        all_int,
    } = prepared;
    let had_missing = missing.iter().any(|value| *value);
    if had_missing {
        codes
            .iter_mut()
            .filter(|code| **code == u32::MAX)
            .for_each(|code| *code = median_code);
    }
    let cardinality = counts.len();
    let baseline = (cardinality <= max_dummy_cardinality).then(|| {
        let minimum = counts.iter().copied().min().unwrap();
        counts.iter().position(|count| *count == minimum).unwrap() as u32
    });
    let mut encodings = Vec::new();
    let mut ranked = Vec::new();
    let mut bounds = Vec::new();
    if let Some(baseline) = baseline {
        for category in 0..cardinality as u32 {
            if category == baseline {
                continue;
            }
            encodings.push(Encoding::Dummy(category));
            ranked.push(
                codes
                    .iter()
                    .map(|code| u32::from(*code == category))
                    .collect(),
            );
            bounds.push(vec![0.0, 0.0]);
        }
    } else {
        encodings.push(Encoding::Ordered);
        ranked.push(codes);
        let cutoff = match &values {
            Values::Numeric(unique) => unique
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    if index == 0 {
                        *value
                    } else {
                        unique[index - 1]
                    }
                })
                .collect(),
            Values::Text(unique) => (0..unique.len())
                .map(|index| index.saturating_sub(1) as f32)
                .collect(),
        };
        bounds.push(cutoff);
    }
    if had_missing {
        encodings.push(Encoding::Missing);
        ranked.push(missing.iter().map(|value| u32::from(*value)).collect());
        bounds.push(vec![0.0, 0.0]);
    }
    FittedColumn {
        column: Column {
            name,
            values,
            all_int,
            median_numeric,
            median_text,
            had_missing,
            encodings,
        },
        ranked,
        bounds,
    }
}

fn fit_column(
    raw: RawColumn,
    name: String,
    max_dummy_cardinality: usize,
) -> Result<FittedColumn, ForestError> {
    let raw = match raw {
        RawColumn::Categorical {
            codes,
            categories,
            null_value,
        } => {
            let labels: Vec<_> = codes
                .iter()
                .map(|code| {
                    if *code < 0 {
                        null_value.as_ref()
                    } else {
                        categories.get(*code as usize).and_then(Option::as_ref)
                    }
                })
                .collect();
            let mut unique: Vec<_> = labels
                .iter()
                .filter_map(|value| (*value).cloned())
                .collect();
            unique.sort_unstable();
            unique.dedup();
            if unique.is_empty() {
                return Ok(FittedColumn {
                    column: Column {
                        name,
                        values: Values::Text(Vec::new()),
                        all_int: false,
                        median_numeric: None,
                        median_text: None,
                        had_missing: true,
                        encodings: Vec::new(),
                    },
                    ranked: Vec::new(),
                    bounds: Vec::new(),
                });
            }
            let mut counts = vec![0; unique.len()];
            let ranked: Vec<_> = labels
                .iter()
                .map(|value| {
                    value.map_or(u32::MAX, |value| {
                        let code = unique.binary_search(value).unwrap();
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
                    median_text: missing
                        .iter()
                        .any(|value| *value)
                        .then(|| unique[median_code as usize].clone()),
                    values: Values::Text(unique),
                    codes: ranked,
                    counts,
                    median_code,
                    median_numeric: None,
                    missing,
                    all_int: false,
                },
                max_dummy_cardinality,
            ));
        }
        raw => raw,
    };
    let had_missing = match &raw {
        RawColumn::Numeric(values) => values.iter().any(Option::is_none),
        RawColumn::Text(values) => values.iter().any(Option::is_none),
        RawColumn::Categorical { .. } => unreachable!(),
    };
    let observed = match &raw {
        RawColumn::Numeric(values) => values.iter().flatten().count(),
        RawColumn::Text(values) => values.iter().flatten().count(),
        RawColumn::Categorical { .. } => unreachable!(),
    };
    if observed == 0 {
        return Ok(FittedColumn {
            column: Column {
                name,
                values: Values::Text(Vec::new()),
                all_int: false,
                median_numeric: None,
                median_text: None,
                had_missing: true,
                encodings: Vec::new(),
            },
            ranked: Vec::new(),
            bounds: Vec::new(),
        });
    }
    let numeric = match raw {
        RawColumn::Numeric(values) => Some(values),
        RawColumn::Text(values) => match parse_numeric(&values, &name)? {
            Some(parsed) => Some(parsed),
            None => {
                let missing: Vec<_> = values.iter().map(Option::is_none).collect();
                let parts = text_parts(values);
                let median_code = parts.unique.binary_search(&parts.median).unwrap() as u32;
                return Ok(finish_column(
                    name,
                    PreparedColumn {
                        values: Values::Text(parts.unique),
                        codes: parts.codes,
                        counts: parts.counts,
                        median_code,
                        median_numeric: None,
                        median_text: had_missing.then_some(parts.median),
                        missing,
                        all_int: false,
                    },
                    max_dummy_cardinality,
                ));
            }
        },
        RawColumn::Categorical { .. } => unreachable!(),
    };
    let values = numeric.unwrap();
    let missing: Vec<_> = values.iter().map(Option::is_none).collect();
    let all_int = values.iter().flatten().all(|value| value.fract() == 0.0);
    let parts = numeric_parts(values, &name)?;
    let median_code = parts
        .unique
        .binary_search_by(|value| value.total_cmp(&parts.median))
        .unwrap() as u32;
    Ok(finish_column(
        name,
        PreparedColumn {
            values: Values::Numeric(parts.unique),
            codes: parts.codes,
            counts: parts.counts,
            median_code,
            median_numeric: had_missing.then_some(parts.median),
            median_text: None,
            missing,
            all_int,
        },
        max_dummy_cardinality,
    ))
}

fn assemble<T: Copy + Default + Send + Sync>(features: &[Vec<T>], rows: usize) -> Array2<T> {
    let cols = features.len();
    let mut data = vec![T::default(); rows * cols];
    data.par_chunks_mut(cols.max(1))
        .enumerate()
        .for_each(|(row, output)| {
            for (col, feature) in features.iter().enumerate() {
                output[col] = feature[row];
            }
        });
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

impl Encoder {
    pub fn fit(
        columns: Vec<RawColumn>,
        names: Vec<String>,
        max_dummy_cardinality: usize,
        one_hot_groups: Vec<(String, Vec<usize>)>,
        date_parts: Vec<(usize, String, u8, String)>,
    ) -> Result<(Self, Array2<u32>), ForestError> {
        if max_dummy_cardinality == 0 {
            return Err(invalid("max_dummy_cardinality must be a positive integer"));
        }
        let rows = validate_rows(&columns, &names)?;
        let (input_columns, logical_names) = input_layout(&names, &one_hot_groups, &date_parts)?;
        let columns = arrange_columns(columns, &input_columns, &logical_names)?;
        let fitted: Result<Vec<_>, _> = columns
            .into_par_iter()
            .zip(logical_names)
            .map(|(column, name)| fit_column(column, name, max_dummy_cardinality))
            .collect();
        let fitted = fitted?;
        let mut features = Vec::new();
        let mut cutoff_values = Vec::new();
        let mut cutoff_offsets = vec![0];
        let mut encoded_to_raw = Vec::new();
        let mut feature_group_ids = Vec::new();
        let mut next_group = 0;
        for (raw, column) in fitted.iter().enumerate() {
            let atomic = column.column.is_numeric() && column.column.had_missing();
            let atomic_group = next_group;
            for (ranked, bounds) in column.ranked.iter().zip(&column.bounds) {
                features.push(ranked.clone());
                cutoff_values.extend(bounds);
                cutoff_offsets.push(cutoff_values.len());
                encoded_to_raw.push(raw);
                feature_group_ids.push(if atomic { atomic_group } else { next_group });
                if !atomic {
                    next_group += 1;
                }
            }
            if atomic && !column.ranked.is_empty() {
                next_group += 1;
            }
        }
        let matrix = assemble(&features, rows);
        let encoder = Self {
            columns: fitted.into_iter().map(|column| column.column).collect(),
            input_names: names,
            input_columns,
            cutoff_values,
            cutoff_offsets,
            encoded_to_raw,
            feature_group_ids,
        };
        Ok((encoder, matrix))
    }

    pub fn transform(&self, columns: Vec<RawColumn>) -> Result<Array2<f32>, ForestError> {
        let rows = validate_rows(&columns, &self.input_names)?;
        let names: Vec<_> = self
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let columns = arrange_columns(columns, &self.input_columns, &names)?;
        let encoded: Result<Vec<_>, _> = columns
            .into_par_iter()
            .zip(&self.columns)
            .map(|(raw, fitted)| transform_column(raw, fitted))
            .collect();
        let features: Vec<_> = encoded?.into_iter().flatten().collect();
        Ok(assemble(&features, rows))
    }

    pub fn transform_numeric<T>(
        &self,
        values: ArrayView2<'_, T>,
    ) -> Result<Array2<f32>, ForestError>
    where
        T: Copy + Into<f64> + Send + Sync,
    {
        if values.nrows() == 0 {
            return Err(invalid("X must contain at least one row"));
        }
        if values.ncols() != self.input_names.len() {
            return Err(invalid(format!(
                "expected {} features, got {}",
                self.input_names.len(),
                values.ncols()
            )));
        }
        let encoded_cols = self.encoded_to_raw.len();
        if encoded_cols == 0 {
            return Ok(Array2::zeros((values.nrows(), 0)));
        }
        let mut data = vec![0.0; values.nrows() * encoded_cols];
        data.par_chunks_mut(encoded_cols)
            .enumerate()
            .try_for_each(|(row, output)| {
                let mut encoded = 0;
                for (source, column) in self.input_columns.iter().zip(&self.columns) {
                    let (value, numeric) = match source {
                        InputColumn::Direct(raw) => {
                            let value = values[[row, *raw]].into() as f32;
                            if !value.is_finite() {
                                return Err(invalid(format!(
                                    "column {:?} contains a non-finite numeric value",
                                    column.name
                                )));
                            }
                            if !column.is_numeric() {
                                return Err(invalid(
                                    "numeric matrix transformation requires numeric fitted columns",
                                ));
                            }
                            (value, true)
                        }
                        InputColumn::OneHot {
                            indices,
                            categories,
                        } => {
                            let mut active = None;
                            for (category, raw) in indices.iter().enumerate() {
                                let value = values[[row, *raw]].into() as f32;
                                if !value.is_finite() || value != 0.0 && value != 1.0 {
                                    return Err(invalid(format!(
                                        "one-hot group {:?} has a value other than 0 or 1 at row {row}",
                                        column.name
                                    )));
                                }
                                if value == 1.0 && active.replace(category).is_some() {
                                    return Err(invalid(format!(
                                        "one-hot group {:?} has multiple active categories at row {row}",
                                        column.name
                                    )));
                                }
                            }
                            let category = active.ok_or_else(|| {
                                invalid(format!(
                                    "one-hot group {:?} has no active category at row {row}",
                                    column.name
                                ))
                            })?;
                            let Values::Text(unique) = &column.values else {
                                unreachable!()
                            };
                            (unique.binary_search(&categories[category]).unwrap() as f32, false)
                        }
                        InputColumn::DatePart { .. } => {
                            return Err(invalid("numeric matrix transformation cannot contain date columns"));
                        }
                    };
                    for encoding in &column.encodings {
                        output[encoded] = match encoding {
                            Encoding::Ordered => value,
                            Encoding::Dummy(category) if numeric => {
                                let Values::Numeric(unique) = &column.values else { unreachable!() };
                                f32::from(value == unique[*category as usize])
                            }
                            Encoding::Dummy(category) => f32::from(value == *category as f32),
                            Encoding::Missing => 0.0,
                        };
                        encoded += 1;
                    }
                }
                Ok::<_, ForestError>(())
            })?;
        Ok(Array2::from_shape_vec((values.nrows(), encoded_cols), data).unwrap())
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
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

    pub fn feature_group_ids(&self) -> &[usize] {
        &self.feature_group_ids
    }
}

fn numeric_input(raw: RawColumn, fitted: &Column) -> Result<Vec<Option<f32>>, ForestError> {
    match raw.into_simple() {
        RawColumn::Numeric(values) => {
            if values.iter().flatten().any(|value| !value.is_finite()) {
                return Err(invalid(format!(
                    "column {:?} contains a non-finite numeric value",
                    fitted.name
                )));
            }
            Ok(values)
        }
        RawColumn::Text(values) => parse_numeric(&values, &fitted.name)?.ok_or_else(|| {
            invalid(format!(
                "column {:?} was numeric during training",
                fitted.name
            ))
        }),
        RawColumn::Categorical { .. } => unreachable!(),
    }
}

fn text_input(raw: RawColumn) -> Vec<Option<String>> {
    match raw.into_simple() {
        RawColumn::Text(values) => values,
        RawColumn::Numeric(values) => values
            .into_iter()
            .map(|value| {
                value.map(|value| {
                    if value.fract() == 0.0 {
                        format!("{value:.1}")
                    } else {
                        value.to_string()
                    }
                })
            })
            .collect(),
        RawColumn::Categorical { .. } => unreachable!(),
    }
}

fn transform_column(raw: RawColumn, fitted: &Column) -> Result<Vec<Vec<f32>>, ForestError> {
    if fitted.encodings.is_empty() {
        return Ok(Vec::new());
    }
    let missing = raw.missing();
    if !fitted.had_missing
        && let Some(row) = missing.iter().position(|missing| *missing)
    {
        return Err(invalid(format!(
            "column {:?} has a missing value at row {row}, but had none during training",
            fitted.name
        )));
    }
    let result = match &fitted.values {
        Values::Numeric(unique) => {
            let mut values = numeric_input(raw, fitted)?;
            if let Some(median) = fitted.median_numeric {
                values
                    .iter_mut()
                    .filter(|value| value.is_none())
                    .for_each(|value| *value = Some(median));
            }
            fitted
                .encodings
                .iter()
                .map(|encoding| match encoding {
                    Encoding::Ordered => values.iter().map(|value| value.unwrap()).collect(),
                    Encoding::Dummy(category) => values
                        .iter()
                        .map(|value| {
                            f32::from(
                                value.is_some_and(|value| value == unique[*category as usize]),
                            )
                        })
                        .collect(),
                    Encoding::Missing => {
                        missing.iter().map(|missing| f32::from(*missing)).collect()
                    }
                })
                .collect()
        }
        Values::Text(unique) => {
            let mut values = text_input(raw);
            if let Some(median) = &fitted.median_text {
                values
                    .iter_mut()
                    .filter(|value| value.is_none())
                    .for_each(|value| *value = Some(median.clone()));
            }
            let codes: Vec<_> = values
                .iter()
                .map(
                    |value| match unique.binary_search(value.as_ref().unwrap()) {
                        Ok(index) | Err(index) => index as u32,
                    },
                )
                .collect();
            fitted
                .encodings
                .iter()
                .map(|encoding| match encoding {
                    Encoding::Ordered => codes.iter().map(|code| *code as f32).collect(),
                    Encoding::Dummy(category) => values
                        .iter()
                        .map(|value| {
                            f32::from(
                                value
                                    .as_ref()
                                    .is_some_and(|value| value == &unique[*category as usize]),
                            )
                        })
                        .collect(),
                    Encoding::Missing => {
                        missing.iter().map(|missing| f32::from(*missing)).collect()
                    }
                })
                .collect()
        }
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_encoding_story() {
        let columns = vec![
            RawColumn::Text(vec![
                Some("10".into()),
                None,
                Some("2".into()),
                Some("7".into()),
            ]),
            RawColumn::Text(vec![
                Some("b".into()),
                Some("a".into()),
                Some("c".into()),
                Some("b".into()),
            ]),
        ];
        let (encoder, ranked) = Encoder::fit(
            columns,
            vec!["number".into(), "label".into()],
            3,
            vec![],
            vec![],
        )
        .unwrap();
        assert_eq!(ranked.dim(), (4, 5));
        assert!(encoder.columns[0].is_numeric());
        assert!(encoder.columns[0].had_missing());
        assert_eq!(encoder.feature_group_ids, vec![0, 0, 0, 1, 2]);
        let predicted = encoder
            .transform(vec![
                RawColumn::Text(vec![Some("5".into()), None]),
                RawColumn::Text(vec![Some("new".into()), Some("a".into())]),
            ])
            .unwrap();
        assert_eq!(predicted.dim(), (2, 5));
        assert_eq!(predicted[[1, 2]], 1.0);
        assert_eq!(&predicted.row(0).as_slice().unwrap()[3..], &[0.0, 0.0]);

        let raw = vec![
            RawColumn::Numeric(vec![Some(0.0), Some(1.0), Some(2.0), Some(1.0)]),
            RawColumn::Numeric(vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)]),
        ];
        let (numeric, _) = Encoder::fit(
            raw,
            vec!["small".into(), "ordered".into()],
            3,
            vec![],
            vec![],
        )
        .unwrap();
        let matrix = ndarray::array![[2.0_f64, 15.0], [0.0, 50.0]];
        let direct = numeric.transform_numeric(matrix.view()).unwrap();
        let columns = vec![
            RawColumn::Numeric(vec![Some(2.0), Some(0.0)]),
            RawColumn::Numeric(vec![Some(15.0), Some(50.0)]),
        ];
        assert_eq!(direct, numeric.transform(columns).unwrap());

        let one_hot = vec![
            RawColumn::Numeric(vec![Some(1.0), Some(0.0), Some(0.0), Some(1.0)]),
            RawColumn::Numeric(vec![Some(0.0), Some(1.0), Some(0.0), Some(0.0)]),
            RawColumn::Numeric(vec![Some(0.0), Some(0.0), Some(1.0), Some(0.0)]),
        ];
        let groups = vec![("color".into(), vec![0, 1, 2])];
        let (grouped, fitted) = Encoder::fit(
            one_hot,
            vec!["red".into(), "green".into(), "blue".into()],
            2,
            groups,
            vec![],
        )
        .unwrap();
        assert_eq!(grouped.columns[0].text_values(), &["blue", "green", "red"]);
        assert_eq!(fitted.column(0).to_vec(), vec![2, 1, 0, 2]);
        let direct = grouped
            .transform_numeric(ndarray::array![[0.0_f32, 1.0, 0.0], [0.0, 0.0, 1.0]].view())
            .unwrap();
        assert_eq!(direct.column(0).to_vec(), vec![1.0, 0.0]);

        let dates = vec![RawColumn::Text(vec![
            Some("2023-12-31 23:30:15".into()),
            Some("2024-01-01 00:00:00".into()),
        ])];
        let format = "%Y-%m-%d %H:%M:%S".to_string();
        let parts = vec![
            (0, format.clone(), 0, "eventYear".into()),
            (0, format.clone(), 2, "eventWeek".into()),
            (0, format.clone(), 4, "eventDayofweek".into()),
            (0, format, 12, "eventHour".into()),
        ];
        let (dated, _) = Encoder::fit(dates, vec!["eventDate".into()], 4, vec![], parts).unwrap();
        assert_eq!(dated.columns[0].numeric_values(), &[2023.0, 2024.0]);
        assert_eq!(dated.columns[1].numeric_values(), &[1.0, 52.0]);
        assert_eq!(dated.columns[2].numeric_values(), &[0.0, 6.0]);
        assert_eq!(dated.columns[3].numeric_values(), &[0.0, 23.0]);
    }
}
