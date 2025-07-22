mod core;
mod raw_estimation;
mod exact;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::raw_estimation::{raw_estimate_single, raw_estimate_batch, raw_estimate_reuse};
use crate::exact::exact_calculation;

#[pymodule]
fn emsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_estimate_single, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(exact_calculation, m)?)?;
    Ok(())
}
