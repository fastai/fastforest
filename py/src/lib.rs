use arrow_array::RecordBatch;
use arrow_pyarrow::PyArrowType;
use ndarray::{Array2, ArrayView2};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fastforest::detect_dates;
use fastforest::{
    ClassifierForest, Config, Encoder, EncoderOptions, Encoding, Forest, ForestError, MaxFeatures, ModelMetadata, Projections, SavedModel,
    SavedValue, StatContext, StatTarget, plan_fit, resolve_replacement,
};
const PREDICTION_BLOCK_BYTES: usize = 64 << 20;

fn prediction_block_rows(rows: usize, features: usize, outputs: usize) -> usize {
    let bytes_per_row = 4 * (features + outputs).max(1);
    (PREDICTION_BLOCK_BYTES / bytes_per_row).max(1_024).min(rows.max(1))
}

fn predict_encoded<T>(
    encoder: &Encoder, batch: &RecordBatch, markers: &[SavedValue], outputs: usize,
    predict: impl Fn(ArrayView2<'_, f32>) -> Result<Vec<T>, ForestError>,
) -> Result<Vec<T>, ForestError> {
    let rows = batch.num_rows();
    if rows == 0 { return Err(ForestError::new("X must contain at least one row")); }
    let block_rows = prediction_block_rows(rows, encoder.encoded_to_raw().len(), outputs);
    let mut result = Vec::with_capacity(rows * outputs);
    for start in (0..rows).step_by(block_rows) {
        let n_rows = block_rows.min(rows - start);
        let block = encoder.transform_arrow(&batch.slice(start, n_rows), markers)?;
        result.extend(predict(block.view())?);
    }
    Ok(result)
}

#[pyfunction(name = "_fit_plan")]
#[pyo3(signature = (n_rows, n_trees, bootstrap_fraction, bootstrap_max, replacement, oob, output_dimensions))]
#[allow(clippy::too_many_arguments)]
fn py_fit_plan(
    n_rows: usize, n_trees: Option<usize>, bootstrap_fraction: Option<f32>, bootstrap_max: Option<usize>, replacement: bool, oob: bool,
    output_dimensions: usize,
) -> PyResult<(usize, usize, usize)> {
    let plan = plan_fit(n_rows, n_trees, bootstrap_fraction, bootstrap_max, replacement, oob, output_dimensions).map_err(value_error)?;
    Ok((plan.n_trees, plan.rows_per_tree, plan.pool_rows))
}

#[pyfunction(name = "_sample_indices")]
#[pyo3(signature = (n_rows, sample_rows, seed=None, stream=0))]
fn py_sample_indices<'py>(
    py: Python<'py>, n_rows: usize, sample_rows: usize, seed: Option<u64>, stream: u64,
) -> Bound<'py, PyArray1<usize>> { fastforest::uniform_sample_indices(n_rows, sample_rows.min(n_rows), seed, stream).into_pyarray(py) }

#[pyfunction(name = "_defaults")]
fn py_defaults(py: Python<'_>, classification: bool) -> PyResult<Py<PyDict>> {
    let config = if classification { Config::classification() } else { Config::default() };
    let result = PyDict::new(py);
    result.set_item("n_trees", py.None())?;
    result.set_item("min_node_size", config.min_node_size)?;
    result.set_item("bootstrap_fraction", config.bootstrap_fraction)?;
    result.set_item("bootstrap_max", config.bootstrap_max)?;
    result.set_item("replacement", py.None())?;
    result.set_item("max_node_samples", config.max_node_samples)?;
    result.set_item("split_prior_rows", config.split_prior_rows)?;
    result.set_item("class_weight_power", config.class_weight_power)?;
    result.set_item("cutoff_divisor", config.cutoff_divisor)?;
    result.set_item("random_splitter", config.random_splitter)?;
    match config.max_features {
        MaxFeatures::Sqrt => result.set_item("max_features", "sqrt")?,
        MaxFeatures::Fraction(value) => result.set_item("max_features", (value as f64 * 1e6).round() / 1e6)?,
    }
    result.set_item("seed", py.None())?;
    result.set_item("oob", config.oob)?;
    result.set_item("allow_new_missing", false)?;
    let stats = EncoderOptions::default();
    result.set_item("target_statistics", stats.target_statistics)?;
    result.set_item("min_rows_per_level", stats.min_rows_per_level)?;
    result.set_item("min_stat_agreement", stats.min_stat_agreement)?;
    result.set_item("stat_components", stats.stat_components)?;
    result.set_item("frequency", stats.frequency)?;
    result.set_item("natural_sort", stats.natural_sort)?;
    Ok(result.unbind())
}

#[pyfunction(name = "_resolve_replacement")]
fn py_resolve_replacement(n_rows: usize, replacement: Option<bool>, classification: bool) -> bool {
    resolve_replacement(n_rows, replacement, classification)
}

type PyExplanation<'py> = (Bound<'py, PyArray1<f32>>, f32, Bound<'py, PyArray2<f32>>);
type PyColumnMetadata = (bool, bool, bool, Option<f32>, Option<String>, Vec<f32>, Vec<String>, Vec<(u8, i64)>);
type PySavedMetadata = (Vec<(u8, String)>, Vec<(usize, String)>, Vec<(String, (u8, String))>);
type PyLoadedModel = (u8, PyEncoder, Option<PyForest>, Option<PyClassifierForest>, PySavedMetadata, Vec<(u8, String)>);

fn saved_values(values: Vec<(u8, String)>) -> Vec<SavedValue> {
    values.into_iter().map(|(kind, value)| SavedValue { kind, value }).collect()
}

fn saved_metadata(metadata: PySavedMetadata) -> ModelMetadata {
    ModelMetadata {
        markers: saved_values(metadata.0),
        date_columns: metadata.1,
        parameters: metadata.2.into_iter().map(|(name, (kind, value))| (name, SavedValue { kind, value })).collect(),
    }
}

fn python_metadata(metadata: ModelMetadata) -> PySavedMetadata {
    (
        metadata.markers.into_iter().map(|value| (value.kind, value.value)).collect(),
        metadata.date_columns,
        metadata.parameters.into_iter().map(|(name, value)| (name, (value.kind, value.value))).collect(),
    )
}

#[pyfunction(name = "_save_regression")]
fn py_save_regression(path: String, encoder: PyRef<'_, PyEncoder>, forest: PyRef<'_, PyForest>, metadata: PySavedMetadata) -> PyResult<()> {
    SavedModel::regression(encoder.inner.clone(), forest.inner.clone(), saved_metadata(metadata)).save(path).map_err(value_error)
}

#[pyfunction(name = "_save_classification")]
fn py_save_classification(
    path: String, encoder: PyRef<'_, PyEncoder>, forest: PyRef<'_, PyClassifierForest>, metadata: PySavedMetadata,
    classes: Vec<(u8, String)>,
) -> PyResult<()> {
    SavedModel::classification(encoder.inner.clone(), forest.inner.clone(), saved_metadata(metadata), saved_values(classes))
        .save(path)
        .map_err(value_error)
}

#[pyfunction(name = "_predict_regression_file")]
#[pyo3(signature = (encoder, forest, metadata, input, output, batch_size=65_536))]
fn py_predict_regression_file(
    encoder: PyRef<'_, PyEncoder>, forest: PyRef<'_, PyForest>, metadata: PySavedMetadata, input: String, output: String, batch_size: usize,
) -> PyResult<()> {
    let model = SavedModel::regression(encoder.inner.clone(), forest.inner.clone(), saved_metadata(metadata));
    fastforest::predict_file(&model, input, output, batch_size, false).map_err(value_error)
}

#[pyfunction(name = "_predict_classification_file")]
#[pyo3(signature = (encoder, forest, metadata, classes, input, output, batch_size=65_536, proba=false))]
#[allow(clippy::too_many_arguments)]
fn py_predict_classification_file(
    encoder: PyRef<'_, PyEncoder>, forest: PyRef<'_, PyClassifierForest>, metadata: PySavedMetadata, classes: Vec<(u8, String)>,
    input: String, output: String, batch_size: usize, proba: bool,
) -> PyResult<()> {
    let model = SavedModel::classification(encoder.inner.clone(), forest.inner.clone(), saved_metadata(metadata), saved_values(classes));
    fastforest::predict_file(&model, input, output, batch_size, proba).map_err(value_error)
}

#[pyfunction(name = "_compile_regression")]
fn py_compile_regression(
    encoder: PyRef<'_, PyEncoder>, forest: PyRef<'_, PyForest>, metadata: PySavedMetadata, output: String,
) -> PyResult<()> {
    fastforest::compile_model(&SavedModel::regression(encoder.inner.clone(), forest.inner.clone(), saved_metadata(metadata)), output)
        .map_err(value_error)
}

#[pyfunction(name = "_compile_classification")]
fn py_compile_classification(
    encoder: PyRef<'_, PyEncoder>, forest: PyRef<'_, PyClassifierForest>, metadata: PySavedMetadata, classes: Vec<(u8, String)>,
    output: String,
) -> PyResult<()> {
    fastforest::compile_model(
        &SavedModel::classification(encoder.inner.clone(), forest.inner.clone(), saved_metadata(metadata), saved_values(classes)),
        output,
    )
    .map_err(value_error)
}

#[pyfunction(name = "_load_model")]
fn py_load_model(path: String) -> PyResult<PyLoadedModel> {
    match SavedModel::load(path).map_err(value_error)? {
        SavedModel::Regression { encoder, forest, metadata } => {
            Ok((0, PyEncoder { inner: encoder }, Some(PyForest { inner: forest }), None, python_metadata(metadata), Vec::new()))
        }
        SavedModel::Classification { encoder, forest, metadata, classes } => Ok((
            1,
            PyEncoder { inner: encoder },
            None,
            Some(PyClassifierForest { inner: forest }),
            python_metadata(metadata),
            classes.into_iter().map(|value| (value.kind, value.value)).collect(),
        )),
    }
}

fn value_error(error: ForestError) -> PyErr { PyValueError::new_err(error.to_string()) }

fn max_features(kind: u8, value: f32) -> PyResult<MaxFeatures> {
    match kind {
        1 => Ok(MaxFeatures::Sqrt),
        2 => Ok(MaxFeatures::Fraction(value)),
        _ => Err(PyValueError::new_err("unknown max_features kind")),
    }
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct PyBatchConfig {
    n_trees: usize,
    min_node_size: usize,
    bootstrap_fraction: Option<f32>,
    bootstrap_max: Option<usize>,
    sample_rows: Option<usize>,
    replacement: bool,
    max_node_samples: usize,
    split_prior_rows: f32,
    class_weight_power: f32,
    cutoff_divisor: f32,
    seed: Option<u64>,
    oob: bool,
    random_splitter: bool,
    max_features_kind: u8,
    max_features_value: f32,
}

impl PyBatchConfig {
    fn into_config(self) -> PyResult<Config> {
        Ok(Config {
            n_trees: self.n_trees,
            min_node_size: self.min_node_size,
            bootstrap_fraction: self.bootstrap_fraction,
            bootstrap_max: self.bootstrap_max,
            sample_rows: self.sample_rows,
            replacement: self.replacement,
            max_node_samples: self.max_node_samples,
            split_prior_rows: self.split_prior_rows,
            class_weight_power: self.class_weight_power,
            cutoff_divisor: self.cutoff_divisor,
            seed: self.seed,
            oob: self.oob,
            random_splitter: self.random_splitter,
            max_features: max_features(self.max_features_kind, self.max_features_value)?,
        })
    }
}

#[pyclass(name = "Encoder", frozen)]
struct PyEncoder { inner: Encoder }

#[pymethods]
impl PyEncoder {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        py: Python<'py>, batch: PyArrowType<RecordBatch>, markers: Vec<(u8, String)>, allow_new_missing: bool,
        date_columns: Vec<(usize, String)>, seed: Option<u64>, y: Option<PyReadonlyArray1<'_, f32>>,
        y_class: Option<(PyReadonlyArray1<'_, u32>, usize)>, target_statistics: bool, min_rows_per_level: usize, min_stat_agreement: f32,
        stat_components: usize, frequency: bool, natural_sort: bool, order_column: Option<usize>, agreement_sample_rows: usize,
        detrend: bool,
    ) -> PyResult<(Self, Bound<'py, PyArray2<u32>>, Option<(usize, f64, f64, Bound<'py, PyArray1<f32>>)>)> {
        let markers = saved_values(markers);
        let stats = EncoderOptions { target_statistics, min_rows_per_level, min_stat_agreement, stat_components, frequency, natural_sort };
        let y = y.map(|y| y.as_slice().map(<[f32]>::to_vec)).transpose()?;
        let y_class = y_class.map(|(y, k)| y.as_slice().map(|y| (y.to_vec(), k))).transpose()?;
        let target = match (&y, &y_class) {
            (Some(y), _) => Some(StatTarget::Regression(y)),
            (None, Some((y, k))) => Some(StatTarget::Classes { codes: y, k: *k }),
            (None, None) => None,
        };
        let (inner, ranked, trend) = py
            .detach(|| {
                Encoder::fit_arrow(
                    &batch.0,
                    &markers,
                    allow_new_missing,
                    date_columns,
                    seed,
                    &stats,
                    StatContext { target, order_column, agreement_sample_rows, detrend },
                )
            })
            .map_err(value_error)?;
        let trend = trend.map(|trend| (trend.feature, trend.intercept, trend.slope, trend.values.into_pyarray(py)));
        Ok((Self { inner }, ranked.into_pyarray(py), trend))
    }

    #[allow(clippy::type_complexity)]
    fn training_layout<'py>(
        &self, py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
    )> {
        let (values, offsets, ranks, phys, base, tables, table_offsets) = self.inner.training_layout();
        Ok((
            values.into_pyarray(py),
            offsets.into_pyarray(py),
            ranks.into_pyarray(py),
            phys.into_pyarray(py),
            base.into_pyarray(py),
            tables.into_pyarray(py),
            table_offsets.into_pyarray(py),
        ))
    }

    #[staticmethod]
    #[pyo3(signature = (batch, markers, seed=None))]
    fn detect_dates(
        py: Python<'_>, batch: PyArrowType<RecordBatch>, markers: Vec<(u8, String)>, seed: Option<u64>,
    ) -> PyResult<Vec<(usize, String)>> {
        let markers = saved_values(markers);
        py.detach(|| detect_dates(&batch.0, &markers, seed)).map_err(value_error)
    }

    fn transform<'py>(
        &self, py: Python<'py>, batch: PyArrowType<RecordBatch>, markers: Vec<(u8, String)>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let markers = saved_values(markers);
        let transformed = py.detach(|| self.inner.transform_arrow(&batch.0, &markers)).map_err(value_error)?;
        Ok(transformed.into_pyarray(py))
    }

    fn metadata(&self, column: usize) -> PyResult<PyColumnMetadata> {
        let column = self.inner.columns().get(column).ok_or_else(|| PyValueError::new_err("column index is out of range"))?;
        let encodings = column
            .encodings()
            .iter()
            .map(|encoding| match encoding {
                Encoding::Ordered => (0, -1),
                Encoding::Statistic(component) => (1, *component as i64),
                Encoding::Counter => (2, -1),
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

    fn date_values<'py>(
        &self, py: Python<'py>, batch: PyArrowType<RecordBatch>, markers: Vec<(u8, String)>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let markers = saved_values(markers);
        let values = py.detach(|| self.inner.date_values_arrow(&batch.0, &markers)).map_err(value_error)?;
        Ok(values.into_pyarray(py))
    }

    #[getter]
    fn cutoff_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> { self.inner.cutoff_values().to_vec().into_pyarray(py) }

    #[getter]
    fn input_names(&self) -> Vec<String> { self.inner.input_names().to_vec() }

    #[getter]
    fn logical_names(&self) -> Vec<String> { self.inner.logical_names() }

    #[getter]
    fn date_layout(&self) -> Vec<(usize, String, Vec<String>)> { self.inner.date_layout() }

    #[getter]
    fn bundle_layout(&self) -> Vec<(String, Vec<usize>, Vec<String>)> { self.inner.bundle_layout() }

    #[getter]
    fn cutoff_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> { self.inner.cutoff_offsets().to_vec().into_pyarray(py) }

    #[getter]
    fn encoded_to_raw<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> { self.inner.encoded_to_raw().to_vec().into_pyarray(py) }

    #[getter]
    fn missing_ranks<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> { self.inner.missing_ranks().into_pyarray(py) }
}

#[pyclass(name = "Forest", frozen)]
struct PyForest { inner: Forest }

type PyLayout<'py> = (
    PyReadonlyArray1<'py, f32>,
    PyReadonlyArray1<'py, usize>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, usize>,
);

struct NativeLayout<'a> {
    cutoff_values: &'a [f32],
    cutoff_offsets: &'a [usize],
    missing_ranks: &'a [u32],
    projections: Projections,
}

fn native_layout<'a>(layout: &'a PyLayout<'_>) -> PyResult<NativeLayout<'a>> {
    let (cutoff_values, cutoff_offsets, missing_ranks, phys, base, tables, table_offsets) = layout;
    Ok(NativeLayout {
        cutoff_values: cutoff_values.as_slice()?,
        cutoff_offsets: cutoff_offsets.as_slice()?,
        missing_ranks: missing_ranks.as_slice()?,
        projections: Projections {
            phys: phys.as_slice()?.to_vec(),
            base: base.as_slice()?.to_vec(),
            tables: tables.as_slice()?.to_vec(),
            offsets: table_offsets.as_slice()?.to_vec(),
        },
    })
}

#[pymethods]
impl PyForest {
    #[staticmethod]
    fn fit(
        py: Python<'_>, x: PyReadonlyArray2<'_, u32>, y: PyReadonlyArray1<'_, f32>, layout: PyLayout<'_>, config: PyBatchConfig,
        tracking_indices: Option<PyReadonlyArray1<'_, usize>>, trend: Option<(usize, f64, f64)>,
    ) -> PyResult<Self> {
        let config = config.into_config()?;
        let x = x.as_array();
        let y = y.as_array();
        let layout = native_layout(&layout)?;
        let tracking_indices = tracking_indices.as_ref().map(PyReadonlyArray1::as_slice).transpose()?;
        let mut inner = py
            .detach(|| {
                let (values, offsets, ranks) = (layout.cutoff_values, layout.cutoff_offsets, layout.missing_ranks);
                match tracking_indices {
                    Some(indices) => Forest::fit_on_tracking(x, &layout.projections, y, values, offsets, ranks, &config, indices),
                    None => Forest::fit(x, &layout.projections, y, values, offsets, ranks, &config),
                }
            })
            .map_err(value_error)?;
        inner.trend = trend;
        Ok(Self { inner })
    }

    fn combined(&self, other: PyRef<'_, PyForest>) -> PyResult<Self> {
        Ok(Self { inner: self.inner.combined(&other.inner).map_err(value_error)? })
    }

    #[staticmethod]
    fn fit_batch(
        py: Python<'_>, x: PyReadonlyArray2<'_, u32>, y: PyReadonlyArray1<'_, f32>, layout: PyLayout<'_>, configs: Vec<PyBatchConfig>,
        oob_rows: Option<usize>,
    ) -> PyResult<Vec<Py<PyForest>>> {
        let configs = configs.into_iter().map(PyBatchConfig::into_config).collect::<PyResult<Vec<_>>>()?;
        let x = x.as_array();
        let y = y.as_array();
        let layout = native_layout(&layout)?;
        let forests = py
            .detach(|| {
                Forest::fit_batch(
                    x,
                    &layout.projections,
                    y,
                    layout.cutoff_values,
                    layout.cutoff_offsets,
                    layout.missing_ranks,
                    &configs,
                    oob_rows,
                )
            })
            .map_err(value_error)?;
        forests.into_iter().map(|inner| Py::new(py, PyForest { inner })).collect()
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x = x.as_array();
        let predictions = py.detach(|| self.inner.predict(x)).map_err(value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    fn predict_encoded<'py>(
        &self, py: Python<'py>, encoder: PyRef<'_, PyEncoder>, batch: PyArrowType<RecordBatch>, markers: Vec<(u8, String)>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let markers = saved_values(markers);
        let encoder_inner: &Encoder = &encoder.inner;
        let forest_inner: &Forest = &self.inner;
        let predictions = py
            .detach(|| predict_encoded(encoder_inner, &batch.0, &markers, 1, |block| forest_inner.predict(block)))
            .map_err(value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    fn predict_trees<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = x.as_array();
        let predictions = py.detach(|| self.inner.predict_trees(x)).map_err(value_error)?;
        let predictions =
            Array2::from_shape_vec((x.nrows(), self.inner.n_trees()), predictions).expect("prediction matrix has the wrong size");
        Ok(predictions.into_pyarray(py))
    }

    fn explain<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f32>) -> PyResult<PyExplanation<'py>> {
        let x = x.as_array();
        let (predictions, bias, contributions) = py.detach(|| self.inner.explain(x)).map_err(value_error)?;
        let contributions =
            Array2::from_shape_vec((x.nrows(), self.inner.n_features()), contributions).expect("contribution matrix has the wrong size");
        Ok((predictions.into_pyarray(py), bias, contributions.into_pyarray(py)))
    }

    #[getter]
    fn n_features(&self) -> usize { self.inner.n_features() }

    #[getter]
    fn n_trees(&self) -> usize { self.inner.n_trees() }

    #[getter]
    fn tree_structures(&self) -> Vec<(usize, usize, usize)> { self.inner.tree_structures() }

    #[getter]
    fn split_counts_by_depth(&self) -> Vec<Vec<(usize, usize)>> { self.inner.split_counts_by_depth() }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.feature_importances().to_vec().into_pyarray(py)
    }

    #[getter]
    fn oob_prediction<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.oob_prediction().map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn oob_counts<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        self.inner.oob_counts().map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn oob_indices<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<usize>>> {
        self.inner.oob_indices().map(|values| values.to_vec().into_pyarray(py))
    }
}

#[pyclass(name = "ClassifierForest", frozen)]
struct PyClassifierForest { inner: ClassifierForest }

#[pymethods]
impl PyClassifierForest {
    #[staticmethod]
    fn fit(
        py: Python<'_>, x: PyReadonlyArray2<'_, u32>, y: PyReadonlyArray1<'_, u32>, n_classes: usize, layout: PyLayout<'_>,
        config: PyBatchConfig, tracking_indices: Option<PyReadonlyArray1<'_, usize>>,
    ) -> PyResult<Self> {
        let config = config.into_config()?;
        let x = x.as_array();
        let y = y.as_array();
        let layout = native_layout(&layout)?;
        let tracking_indices = tracking_indices.as_ref().map(PyReadonlyArray1::as_slice).transpose()?;
        let inner = py
            .detach(|| {
                let (values, offsets, ranks) = (layout.cutoff_values, layout.cutoff_offsets, layout.missing_ranks);
                match tracking_indices {
                    Some(indices) => {
                        ClassifierForest::fit_on_tracking(x, &layout.projections, y, n_classes, values, offsets, ranks, &config, indices)
                    }
                    None => ClassifierForest::fit(x, &layout.projections, y, n_classes, values, offsets, ranks, &config),
                }
            })
            .map_err(value_error)?;
        Ok(Self { inner })
    }

    fn combined(&self, other: PyRef<'_, PyClassifierForest>) -> PyResult<Self> {
        Ok(Self { inner: self.inner.combined(&other.inner).map_err(value_error)? })
    }

    #[staticmethod]
    fn fit_batch(
        py: Python<'_>, x: PyReadonlyArray2<'_, u32>, y: PyReadonlyArray1<'_, u32>, n_classes: usize, layout: PyLayout<'_>,
        configs: Vec<PyBatchConfig>, oob_rows: Option<usize>,
    ) -> PyResult<Vec<Py<PyClassifierForest>>> {
        let configs = configs.into_iter().map(PyBatchConfig::into_config).collect::<PyResult<Vec<_>>>()?;
        let x = x.as_array();
        let y = y.as_array();
        let layout = native_layout(&layout)?;
        let forests = py
            .detach(|| {
                ClassifierForest::fit_batch(
                    x,
                    &layout.projections,
                    y,
                    n_classes,
                    layout.cutoff_values,
                    layout.cutoff_offsets,
                    layout.missing_ranks,
                    &configs,
                    oob_rows,
                )
            })
            .map_err(value_error)?;
        forests.into_iter().map(|inner| Py::new(py, PyClassifierForest { inner })).collect()
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f32>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let x = x.as_array();
        let predictions = py.detach(|| self.inner.predict(x)).map_err(value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    fn predict_encoded<'py>(
        &self, py: Python<'py>, encoder: PyRef<'_, PyEncoder>, batch: PyArrowType<RecordBatch>, markers: Vec<(u8, String)>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let markers = saved_values(markers);
        let encoder_inner: &Encoder = &encoder.inner;
        let forest_inner: &ClassifierForest = &self.inner;
        let predictions = py
            .detach(|| predict_encoded(encoder_inner, &batch.0, &markers, 1, |block| forest_inner.predict(block)))
            .map_err(value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = x.as_array();
        let probabilities = py.detach(|| self.inner.predict_proba(x)).map_err(value_error)?;
        let probabilities =
            Array2::from_shape_vec((x.nrows(), self.inner.n_classes()), probabilities).expect("probability matrix has the wrong size");
        Ok(probabilities.into_pyarray(py))
    }

    fn predict_proba_encoded<'py>(
        &self, py: Python<'py>, encoder: PyRef<'_, PyEncoder>, batch: PyArrowType<RecordBatch>, markers: Vec<(u8, String)>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let markers = saved_values(markers);
        let encoder_inner: &Encoder = &encoder.inner;
        let forest_inner: &ClassifierForest = &self.inner;
        let classes = forest_inner.n_classes();
        let probabilities = py
            .detach(|| predict_encoded(encoder_inner, &batch.0, &markers, classes, |block| forest_inner.predict_proba(block)))
            .map_err(value_error)?;
        let rows = probabilities.len() / self.inner.n_classes();
        let probabilities =
            Array2::from_shape_vec((rows, self.inner.n_classes()), probabilities).expect("probability matrix has the wrong size");
        Ok(probabilities.into_pyarray(py))
    }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.feature_importances().to_vec().into_pyarray(py)
    }

    #[getter]
    fn n_trees(&self) -> usize { self.inner.n_trees() }

    #[getter]
    fn tree_structures(&self) -> Vec<(usize, usize, usize)> { self.inner.tree_structures() }

    #[getter]
    fn prediction_trees_per_batch(&self) -> usize { self.inner.prediction_trees_per_batch() }

    #[getter]
    fn oob_decision_function<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        self.inner.oob_decision().map(|values| {
            Array2::from_shape_vec((values.len() / self.inner.n_classes(), self.inner.n_classes()), values.to_vec())
                .expect("OOB probability matrix has the wrong size")
                .into_pyarray(py)
        })
    }

    #[getter]
    fn oob_counts<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        self.inner.oob_counts().map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn oob_indices<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<usize>>> {
        self.inner.oob_indices().map(|values| values.to_vec().into_pyarray(py))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForest>()?;
    m.add_class::<PyClassifierForest>()?;
    m.add_class::<PyEncoder>()?;
    m.add_function(wrap_pyfunction!(py_fit_plan, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_indices, m)?)?;
    m.add_function(wrap_pyfunction!(py_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(py_resolve_replacement, m)?)?;
    m.add_function(wrap_pyfunction!(py_save_regression, m)?)?;
    m.add_function(wrap_pyfunction!(py_save_classification, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_predict_regression_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_predict_classification_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_regression, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_classification, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
