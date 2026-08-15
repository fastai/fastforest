use ndarray::Array2;
use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyString, PyTuple};

use crate::{
    ClassifierForest, Config, Encoder, Encoding, FeatureSampling, Forest, ForestError, MaxFeatures,
    RawColumn, Splitter, Workbench,
};

type PyExplanation<'py> = (Bound<'py, PyArray1<f32>>, f32, Bound<'py, PyArray2<f32>>);
type PyColumnMetadata = (
    bool,
    bool,
    bool,
    Option<f32>,
    Option<String>,
    Vec<f32>,
    Vec<String>,
    Vec<(u8, i64)>,
);

fn value_error(error: ForestError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn workbench(
    splitter: u8,
    max_features_kind: u8,
    max_features_value: f32,
    leaf_regularization: f32,
    feature_sampling: u8,
) -> PyResult<Workbench> {
    let splitter = match splitter {
        0 => Splitter::Random,
        1 => Splitter::Histogram,
        _ => return Err(PyValueError::new_err("unknown experimental splitter")),
    };
    let max_features = match max_features_kind {
        0 => MaxFeatures::Sqrt,
        1 => MaxFeatures::All,
        2 => MaxFeatures::Fraction(max_features_value),
        3 => MaxFeatures::Count(max_features_value as usize),
        _ => return Err(PyValueError::new_err("unknown max_features kind")),
    };
    let feature_sampling = match feature_sampling {
        0 => FeatureSampling::Encoded,
        1 => FeatureSampling::Columns,
        _ => return Err(PyValueError::new_err("unknown feature sampling unit")),
    };
    Ok(Workbench {
        splitter,
        max_features,
        leaf_regularization,
        feature_sampling,
    })
}

enum Marker {
    None,
    Nan,
    Text(String),
    Number(f64),
    Object(Py<PyAny>),
}

impl Marker {
    fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.is_none() {
            return Ok(Self::None);
        }
        if let Ok(value) = value.cast::<PyString>() {
            return Ok(Self::Text(value.to_str()?.to_owned()));
        }
        if let Ok(value) = value.extract::<f64>() {
            return Ok(if value.is_nan() {
                Self::Nan
            } else {
                Self::Number(value)
            });
        }
        Ok(Self::Object(value.clone().unbind()))
    }

    fn numeric_missing(&self, value: f64) -> bool {
        match self {
            Self::Nan => value.is_nan(),
            Self::Number(marker) => value == *marker,
            Self::None | Self::Text(_) | Self::Object(_) => false,
        }
    }

    fn nan_missing(&self) -> bool {
        matches!(self, Self::Nan)
    }

    fn object_value(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
        let missing = match self {
            Self::None => value.is_none(),
            Self::Nan => value.extract::<f64>().is_ok_and(f64::is_nan),
            Self::Text(marker) => value
                .cast::<PyString>()
                .is_ok_and(|value| value.to_str().is_ok_and(|value| value == marker)),
            Self::Number(marker) => value.extract::<f64>().is_ok_and(|value| value == *marker),
            Self::Object(marker) => value.eq(marker.bind(py))?,
        };
        if missing {
            Ok(None)
        } else {
            Ok(Some(value.str()?.to_str()?.to_owned()))
        }
    }
}

fn categorical_codes(column: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    macro_rules! codes {
        ($ty:ty) => {
            if let Ok(array) = column.cast::<PyArray1<$ty>>() {
                return Ok(array
                    .readonly()
                    .as_array()
                    .iter()
                    .map(|value| *value as i32)
                    .collect());
            }
        };
    }
    codes!(i8);
    codes!(i16);
    codes!(i32);
    codes!(i64);
    Err(PyValueError::new_err(
        "categorical codes must be a one-dimensional integer array",
    ))
}

fn categorical_column(
    py: Python<'_>,
    value: &Bound<'_, PyTuple>,
    marker: &Marker,
) -> PyResult<RawColumn> {
    let codes = categorical_codes(&value.get_item(0)?)?;
    let categories = value.get_item(1)?.cast_into::<PyArray1<Py<PyAny>>>()?;
    let categories = categories
        .readonly()
        .as_array()
        .iter()
        .map(|value| marker.object_value(py, value.bind(py)))
        .collect::<PyResult<_>>()?;
    Ok(RawColumn::Categorical {
        codes,
        categories,
        null_value: (!marker.nan_missing()).then(|| "nan".to_owned()),
    })
}

fn raw_column(py: Python<'_>, column: &Bound<'_, PyAny>, marker: &Marker) -> PyResult<RawColumn> {
    if let Ok(column) = column.cast::<PyTuple>() {
        return categorical_column(py, column, marker);
    }
    macro_rules! numeric {
        ($ty:ty) => {
            if let Ok(array) = column.cast::<PyArray1<$ty>>() {
                let readonly = array.readonly();
                return Ok(RawColumn::Numeric(
                    readonly
                        .as_array()
                        .iter()
                        .map(|value| {
                            let value = *value as f64;
                            (!marker.numeric_missing(value)).then_some(value as f32)
                        })
                        .collect(),
                ));
            }
        };
    }
    numeric!(f32);
    numeric!(f64);
    numeric!(i8);
    numeric!(i16);
    numeric!(i32);
    numeric!(i64);
    numeric!(u8);
    numeric!(u16);
    numeric!(u32);
    numeric!(u64);
    if let Ok(array) = column.cast::<PyArray1<bool>>() {
        let readonly = array.readonly();
        return Ok(RawColumn::Numeric(
            readonly
                .as_array()
                .iter()
                .map(|value| Some(u8::from(*value) as f32))
                .collect(),
        ));
    }
    let array = column.cast::<PyArray1<Py<PyAny>>>()?;
    let readonly = array.readonly();
    let mut cached: HashMap<usize, Option<String>> = HashMap::new();
    let mut values = Vec::with_capacity(readonly.as_array().len());
    for value in readonly.as_array() {
        let key = value.as_ptr() as usize;
        let canonical = if let Some(canonical) = cached.get(&key) {
            canonical.clone()
        } else {
            let canonical = marker.object_value(py, value.bind(py))?;
            cached.insert(key, canonical.clone());
            canonical
        };
        values.push(canonical);
    }
    Ok(RawColumn::Text(values))
}

fn raw_columns(
    py: Python<'_>,
    columns: &Bound<'_, PyList>,
    markers: &Bound<'_, PyList>,
) -> PyResult<Vec<RawColumn>> {
    if columns.len() != markers.len() {
        return Err(PyValueError::new_err(
            "missing_values must have one value per column",
        ));
    }
    columns
        .iter()
        .zip(markers.iter())
        .map(|(column, marker)| raw_column(py, &column, &Marker::from_python(&marker)?))
        .collect()
}

#[pyclass(name = "Encoder", frozen)]
struct PyEncoder {
    inner: Encoder,
}

#[pymethods]
impl PyEncoder {
    #[staticmethod]
    fn fit<'py>(
        py: Python<'py>,
        columns: &Bound<'py, PyList>,
        names: Vec<String>,
        markers: &Bound<'py, PyList>,
        max_dummy_cardinality: usize,
        one_hot_groups: Vec<(String, Vec<usize>)>,
        date_parts: Vec<(usize, String, u8, String)>,
    ) -> PyResult<(Self, Bound<'py, PyArray2<u32>>)> {
        let columns = raw_columns(py, columns, markers)?;
        let (inner, ranked) = py
            .detach(|| {
                Encoder::fit(
                    columns,
                    names,
                    max_dummy_cardinality,
                    one_hot_groups,
                    date_parts,
                )
            })
            .map_err(value_error)?;
        Ok((Self { inner }, ranked.into_pyarray(py)))
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        columns: &Bound<'py, PyList>,
        markers: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let columns = raw_columns(py, columns, markers)?;
        let transformed = py
            .detach(|| self.inner.transform(columns))
            .map_err(value_error)?;
        Ok(transformed.into_pyarray(py))
    }

    fn transform_numeric<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        macro_rules! transform {
            ($ty:ty) => {
                if let Ok(values) = values.cast::<PyArray2<$ty>>() {
                    let readonly = values.readonly();
                    let values = readonly.as_array();
                    let transformed = py
                        .detach(|| self.inner.transform_numeric(values))
                        .map_err(value_error)?;
                    return Ok(transformed.into_pyarray(py));
                }
            };
        }
        transform!(f32);
        transform!(f64);
        Err(PyValueError::new_err(
            "numeric matrix must have dtype float32 or float64",
        ))
    }

    fn metadata(&self, column: usize) -> PyResult<PyColumnMetadata> {
        let column = self
            .inner
            .columns()
            .get(column)
            .ok_or_else(|| PyValueError::new_err("column index is out of range"))?;
        let encodings = column
            .encodings()
            .iter()
            .map(|encoding| match encoding {
                Encoding::Ordered => (0, -1),
                Encoding::Dummy(category) => (1, i64::from(*category)),
                Encoding::Missing => (2, -1),
            })
            .collect();
        Ok((
            column.is_numeric(),
            column.all_int(),
            column.had_missing(),
            column.median_numeric(),
            column.median_text().map(str::to_owned),
            column.numeric_values().to_vec(),
            column.text_values().to_vec(),
            encodings,
        ))
    }

    #[getter]
    fn cutoff_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.cutoff_values().to_vec().into_pyarray(py)
    }

    #[getter]
    fn cutoff_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.inner.cutoff_offsets().to_vec().into_pyarray(py)
    }

    #[getter]
    fn encoded_to_raw<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.inner.encoded_to_raw().to_vec().into_pyarray(py)
    }

    #[getter]
    fn feature_group_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.inner.feature_group_ids().to_vec().into_pyarray(py)
    }
}

#[pyclass(name = "Forest", frozen)]
struct PyForest {
    inner: Forest,
}

#[pymethods]
impl PyForest {
    #[staticmethod]
    #[pyo3(signature = (
        x, y, cutoff_values, cutoff_offsets, encoded_to_raw, n_trees=50, min_node_size=4, bootstrap_fraction=None, bootstrap_max=Some(40_000), replacement=false,
        max_node_samples=320, min_candidate_rows=20, candidate_attempt_factor=2, cutoff_divisor=3.0, seed=None, oob=false, adaptive=true,
        splitter=1, max_features_kind=2, max_features_value=0.75, leaf_regularization=0.0, feature_sampling=0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, u32>,
        y: PyReadonlyArray1<'_, f32>,
        cutoff_values: PyReadonlyArray1<'_, f32>,
        cutoff_offsets: PyReadonlyArray1<'_, usize>,
        encoded_to_raw: PyReadonlyArray1<'_, usize>,
        n_trees: usize,
        min_node_size: usize,
        bootstrap_fraction: Option<f32>,
        bootstrap_max: Option<usize>,
        replacement: bool,
        max_node_samples: usize,
        min_candidate_rows: usize,
        candidate_attempt_factor: usize,
        cutoff_divisor: f32,
        seed: Option<u64>,
        oob: bool,
        adaptive: bool,
        splitter: u8,
        max_features_kind: u8,
        max_features_value: f32,
        leaf_regularization: f32,
        feature_sampling: u8,
    ) -> PyResult<Self> {
        let config = Config {
            n_trees,
            min_node_size,
            bootstrap_fraction,
            bootstrap_max,
            replacement,
            max_node_samples,
            min_candidate_rows,
            candidate_attempt_factor,
            cutoff_divisor,
            seed,
            oob,
            adaptive,
            workbench: workbench(
                splitter,
                max_features_kind,
                max_features_value,
                leaf_regularization,
                feature_sampling,
            )?,
        };
        let x = x.as_array();
        let y = y.as_array();
        let cutoff_values = cutoff_values.as_slice()?;
        let cutoff_offsets = cutoff_offsets.as_slice()?;
        let encoded_to_raw = encoded_to_raw.as_slice()?;
        let inner = py
            .detach(|| {
                Forest::fit(
                    x,
                    y,
                    cutoff_values,
                    cutoff_offsets,
                    Some(encoded_to_raw),
                    &config,
                )
            })
            .map_err(value_error)?;
        Ok(Self { inner })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x = x.as_array();
        let predictions = py.detach(|| self.inner.predict(x)).map_err(value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    fn predict_trees<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = x.as_array();
        let predictions = py
            .detach(|| self.inner.predict_trees(x))
            .map_err(value_error)?;
        let predictions = Array2::from_shape_vec((x.nrows(), self.inner.n_trees()), predictions)
            .expect("prediction matrix has the wrong size");
        Ok(predictions.into_pyarray(py))
    }

    fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<PyExplanation<'py>> {
        let x = x.as_array();
        let (predictions, bias, contributions) =
            py.detach(|| self.inner.explain(x)).map_err(value_error)?;
        let contributions =
            Array2::from_shape_vec((x.nrows(), self.inner.n_features()), contributions)
                .expect("contribution matrix has the wrong size");
        Ok((
            predictions.into_pyarray(py),
            bias,
            contributions.into_pyarray(py),
        ))
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.feature_importances().to_vec().into_pyarray(py)
    }

    #[getter]
    fn oob_prediction<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner
            .oob_prediction()
            .map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn oob_counts<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        self.inner
            .oob_counts()
            .map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn adaptive_scores(&self) -> Vec<(f64, usize, f64)> {
        self.inner
            .adaptive_scores()
            .iter()
            .map(|score| {
                (
                    f64::from((score.max_features * 10.0).round() as i32) / 10.0,
                    score.max_node_samples,
                    score.oob_mse,
                )
            })
            .collect()
    }

    #[getter]
    fn adaptive_choice(&self) -> Option<(f64, usize)> {
        self.inner.adaptive_choice().map(|score| {
            (
                f64::from((score.max_features * 10.0).round() as i32) / 10.0,
                score.max_node_samples,
            )
        })
    }
}

#[pyclass(name = "ClassifierForest", frozen)]
struct PyClassifierForest {
    inner: ClassifierForest,
}

#[pymethods]
impl PyClassifierForest {
    #[staticmethod]
    #[pyo3(signature = (
        x, y, n_classes, cutoff_values, cutoff_offsets, encoded_to_raw, n_trees=50, min_node_size=4, bootstrap_fraction=None,
        bootstrap_max=Some(40_000), replacement=false, max_node_samples=320, min_candidate_rows=20, candidate_attempt_factor=2,
        cutoff_divisor=3.0, seed=None, oob=false, adaptive=true, splitter=1, max_features_kind=2, max_features_value=0.75,
        leaf_regularization=0.0, feature_sampling=0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, u32>,
        y: PyReadonlyArray1<'_, u32>,
        n_classes: usize,
        cutoff_values: PyReadonlyArray1<'_, f32>,
        cutoff_offsets: PyReadonlyArray1<'_, usize>,
        encoded_to_raw: PyReadonlyArray1<'_, usize>,
        n_trees: usize,
        min_node_size: usize,
        bootstrap_fraction: Option<f32>,
        bootstrap_max: Option<usize>,
        replacement: bool,
        max_node_samples: usize,
        min_candidate_rows: usize,
        candidate_attempt_factor: usize,
        cutoff_divisor: f32,
        seed: Option<u64>,
        oob: bool,
        adaptive: bool,
        splitter: u8,
        max_features_kind: u8,
        max_features_value: f32,
        leaf_regularization: f32,
        feature_sampling: u8,
    ) -> PyResult<Self> {
        let config = Config {
            n_trees,
            min_node_size,
            bootstrap_fraction,
            bootstrap_max,
            replacement,
            max_node_samples,
            min_candidate_rows,
            candidate_attempt_factor,
            cutoff_divisor,
            seed,
            oob,
            adaptive,
            workbench: workbench(
                splitter,
                max_features_kind,
                max_features_value,
                leaf_regularization,
                feature_sampling,
            )?,
        };
        let x = x.as_array();
        let y = y.as_array();
        let cutoff_values = cutoff_values.as_slice()?;
        let cutoff_offsets = cutoff_offsets.as_slice()?;
        let encoded_to_raw = encoded_to_raw.as_slice()?;
        let inner = py
            .detach(|| {
                ClassifierForest::fit(
                    x,
                    y,
                    n_classes,
                    cutoff_values,
                    cutoff_offsets,
                    Some(encoded_to_raw),
                    &config,
                )
            })
            .map_err(value_error)?;
        Ok(Self { inner })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let x = x.as_array();
        let predictions = py.detach(|| self.inner.predict(x)).map_err(value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = x.as_array();
        let probabilities = py
            .detach(|| self.inner.predict_proba(x))
            .map_err(value_error)?;
        let probabilities =
            Array2::from_shape_vec((x.nrows(), self.inner.n_classes()), probabilities)
                .expect("probability matrix has the wrong size");
        Ok(probabilities.into_pyarray(py))
    }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.feature_importances().to_vec().into_pyarray(py)
    }

    #[getter]
    fn prediction_trees_per_batch(&self) -> usize {
        self.inner.prediction_trees_per_batch()
    }

    #[getter]
    fn oob_decision_function<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        self.inner.oob_decision().map(|values| {
            Array2::from_shape_vec(
                (
                    values.len() / self.inner.n_classes(),
                    self.inner.n_classes(),
                ),
                values.to_vec(),
            )
            .expect("OOB probability matrix has the wrong size")
            .into_pyarray(py)
        })
    }

    #[getter]
    fn oob_counts<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        self.inner
            .oob_counts()
            .map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn adaptive_scores(&self) -> Vec<(f64, usize, f64)> {
        self.inner
            .adaptive_scores()
            .iter()
            .map(|score| {
                (
                    f64::from((score.max_features * 10.0).round() as i32) / 10.0,
                    score.max_node_samples,
                    score.oob_brier,
                )
            })
            .collect()
    }

    #[getter]
    fn adaptive_choice(&self) -> Option<(f64, usize)> {
        self.inner.adaptive_choice().map(|score| {
            (
                f64::from((score.max_features * 10.0).round() as i32) / 10.0,
                score.max_node_samples,
            )
        })
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForest>()?;
    m.add_class::<PyClassifierForest>()?;
    m.add_class::<PyEncoder>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
