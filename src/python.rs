use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Config, Forest, ForestError};

type PyExplanation<'py> = (Bound<'py, PyArray1<f32>>, f32, Bound<'py, PyArray2<f32>>);

fn value_error(error: ForestError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "Forest", frozen)]
struct PyForest {
    inner: Forest,
}

#[pymethods]
impl PyForest {
    #[staticmethod]
    #[pyo3(signature = (
        x, y, n_trees=100, min_node_size=4, bootstrap_fraction=0.8, bootstrap_max=Some(40_000), replacement=false,
        max_node_samples=160, cutoff_divisor=3.0, seed=None, oob=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        n_trees: usize,
        min_node_size: usize,
        bootstrap_fraction: f32,
        bootstrap_max: Option<usize>,
        replacement: bool,
        max_node_samples: usize,
        cutoff_divisor: f32,
        seed: Option<u64>,
        oob: bool,
    ) -> PyResult<Self> {
        let config = Config {
            n_trees,
            min_node_size,
            bootstrap_fraction,
            bootstrap_max,
            replacement,
            max_node_samples,
            cutoff_divisor,
            seed,
            oob,
        };
        let x = x.as_array();
        let y = y.as_array();
        let inner = py
            .detach(|| Forest::fit(x, y, &config))
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
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForest>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
