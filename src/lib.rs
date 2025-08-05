mod core;
mod raw_estimation;
mod raw_estimation_udv;
mod exact;
mod estimation;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::raw_estimation::{raw_estimate_single, raw_estimate_batch, raw_estimate_reuse};
use crate::raw_estimation_udv::{raw_estimate_udv_single, raw_estimate_udv_batch};
use crate::exact::exact_calculation;
use crate::estimation::estimate;

#[pymodule]
fn _lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_estimate_single, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(exact_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(estimate, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_udv_single, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_udv_batch, m)?)?;
    Ok(())
}
