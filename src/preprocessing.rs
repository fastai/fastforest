use ndarray::Array2;
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
    cutoff_values: Vec<f32>,
    cutoff_offsets: Vec<usize>,
    encoded_to_raw: Vec<usize>,
    feature_group_ids: Vec<usize>,
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
    let raw = raw.into_simple();
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
    ) -> Result<(Self, Array2<u32>), ForestError> {
        if max_dummy_cardinality == 0 {
            return Err(invalid("max_dummy_cardinality must be a positive integer"));
        }
        let rows = validate_rows(&columns, &names)?;
        let fitted: Result<Vec<_>, _> = columns
            .into_par_iter()
            .zip(names)
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
            cutoff_values,
            cutoff_offsets,
            encoded_to_raw,
            feature_group_ids,
        };
        Ok((encoder, matrix))
    }

    pub fn transform(&self, columns: Vec<RawColumn>) -> Result<Array2<f32>, ForestError> {
        let names: Vec<_> = self
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let rows = validate_rows(&columns, &names)?;
        let encoded: Result<Vec<_>, _> = columns
            .into_par_iter()
            .zip(&self.columns)
            .map(|(raw, fitted)| transform_column(raw, fitted))
            .collect();
        let features: Vec<_> = encoded?.into_iter().flatten().collect();
        Ok(assemble(&features, rows))
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
        let (encoder, ranked) =
            Encoder::fit(columns, vec!["number".into(), "label".into()], 3).unwrap();
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
    }
}
