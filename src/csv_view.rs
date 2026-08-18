use std::fmt::Write;
use std::path::Path;
use std::str::FromStr;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::ForestError;

#[derive(Clone, Copy, Debug)]
pub enum CsvSample {
    Rows(usize),
    Fraction(f64),
}

impl FromStr for CsvSample {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if let Some(percent) = value.strip_suffix('%') {
            let percent = percent.parse::<f64>().map_err(|_| "sample must be a row count or percentage")?;
            if !percent.is_finite() || percent <= 0.0 || percent > 100.0 {
                return Err("sample percentage must be in (0, 100]".to_owned());
            }
            Ok(Self::Fraction(percent / 100.0))
        } else {
            let rows = value.parse::<usize>().map_err(|_| "sample must be a row count or percentage")?;
            if rows == 0 {
                return Err("sample row count must be positive".to_owned());
            }
            Ok(Self::Rows(rows))
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CsvViewOptions {
    pub columns: Vec<String>,
    pub rows: Option<usize>,
    pub sample: Option<CsvSample>,
    pub seed: u64,
}

#[derive(Clone, Copy)]
enum Kind {
    Integer,
    Float,
    Text,
}

fn kinds(rows: &[Vec<String>], columns: usize) -> Vec<Kind> {
    let mut result = vec![Kind::Integer; columns];
    for row in rows {
        for (kind, value) in result.iter_mut().zip(row) {
            if value.is_empty() || matches!(kind, Kind::Text) || value.parse::<i64>().is_ok() {
                continue;
            }
            *kind = if value.parse::<f64>().is_ok_and(f64::is_finite) { Kind::Float } else { Kind::Text };
        }
    }
    result
}

fn significant(value: f64, digits: usize) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    let exponent = value.abs().log10().floor() as isize;
    if exponent >= digits as isize || exponent < -3 {
        return format!("{value:.precision$e}", precision = digits.saturating_sub(1));
    }
    let decimals = (digits as isize - 1 - exponent).max(0) as usize;
    let mut result = format!("{value:.decimals$}");
    if result.contains('.') {
        while result.ends_with('0') {
            result.pop();
        }
        if result.ends_with('.') {
            result.pop();
        }
    }
    result
}

fn formatted(name: &str, value: &str, kind: Kind) -> String {
    match kind {
        Kind::Integer => value.parse::<i64>().map_or_else(|_| value.to_owned(), |value| value.to_string()),
        Kind::Float => value.parse::<f64>().map_or_else(
            |_| value.to_owned(),
            |value| {
                significant(
                    value,
                    if ["second", "time", "fit", "predict", "proba"].iter().any(|part| name.to_lowercase().contains(part)) { 3 } else { 4 },
                )
            },
        ),
        Kind::Text => value.to_owned(),
    }
}

fn file_error(context: &str, error: impl std::fmt::Display) -> ForestError {
    ForestError::new(format!("{context}: {error}"))
}

pub fn view_csv(input: impl AsRef<Path>, options: &CsvViewOptions) -> Result<String, ForestError> {
    if options.rows == Some(0) {
        return Err(ForestError::new("rows must be positive"));
    }
    if options.rows.is_some() && options.sample.is_some() {
        return Err(ForestError::new("rows and sample cannot be used together"));
    }
    let mut reader = csv::Reader::from_path(input).map_err(|error| file_error("could not open CSV", error))?;
    let headers = reader.headers().map_err(|error| file_error("could not read CSV headers", error))?.clone();
    let selected = if options.columns.is_empty() {
        (0..headers.len()).collect::<Vec<_>>()
    } else {
        options
            .columns
            .iter()
            .map(|name| {
                headers.iter().position(|header| header == name).ok_or_else(|| ForestError::new(format!("unknown column {name:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let names = selected.iter().map(|&column| headers[column].to_owned()).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut rows: Vec<(usize, Vec<String>)> = Vec::new();
    let mut total = 0;
    for (index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| file_error("could not read CSV", error))?;
        let values = || selected.iter().map(|&column| record[column].to_owned()).collect::<Vec<_>>();
        match options.sample {
            Some(CsvSample::Rows(limit)) if rows.len() < limit => rows.push((index, values())),
            Some(CsvSample::Rows(limit)) => {
                let replace = rng.random_range(0..=index);
                if replace < limit {
                    rows[replace] = (index, values())
                }
            }
            Some(CsvSample::Fraction(fraction)) => {
                if rng.random::<f64>() < fraction {
                    rows.push((index, values()))
                }
            }
            None if options.rows.is_none_or(|limit| index < limit) => rows.push((index, values())),
            None => {}
        }
        total += 1;
    }
    if total == 0 {
        return Err(ForestError::new("CSV contains no data rows"));
    }
    rows.sort_unstable_by_key(|(index, _)| *index);
    let rows = rows.into_iter().map(|(_, row)| row).collect::<Vec<_>>();
    let kinds = kinds(&rows, names.len());
    let constants = (0..names.len())
        .filter(|&column| rows.first().is_some_and(|first| rows.iter().all(|row| row[column] == first[column])))
        .collect::<Vec<_>>();
    let varying = (0..names.len()).filter(|column| !constants.contains(column)).collect::<Vec<_>>();
    let mut output = String::new();
    match options.sample {
        Some(CsvSample::Rows(_)) => writeln!(output, "{} randomly sampled rows from {total}", rows.len()).unwrap(),
        Some(CsvSample::Fraction(fraction)) => {
            writeln!(output, "{} rows sampled at {}% from {total}", rows.len(), significant(fraction * 100.0, 4)).unwrap()
        }
        None if options.rows.is_some() => writeln!(output, "first {} rows from {total}", rows.len()).unwrap(),
        None => {}
    }
    if varying.is_empty() {
        output.push_str("No varying columns\n");
    } else {
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.write_record(varying.iter().map(|&column| &names[column])).unwrap();
        for row in &rows {
            writer.write_record(varying.iter().map(|&column| formatted(&names[column], &row[column], kinds[column]))).unwrap();
        }
        output.push_str(&String::from_utf8(writer.into_inner().unwrap()).unwrap());
    }
    if !constants.is_empty() {
        output.push_str("Constants:\n");
        for &column in &constants {
            writeln!(output, "{}={}", names[column], formatted(&names[column], &rows[0][column], kinds[column])).unwrap();
        }
    }
    Ok(output)
}
