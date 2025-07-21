mod core;
mod raw_estimation;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::raw_estimation::raw_estimate_single;

#[pymodule]
fn emsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_estimate_single, m)?)?;
    Ok(())
}
