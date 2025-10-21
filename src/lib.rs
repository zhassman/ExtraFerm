#![allow(clippy::too_many_arguments)]

mod core;
mod estimation;
mod exact;
mod raw_estimation;
mod raw_estimation_lucj;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::estimation::{estimate_batch, estimate_single};
use crate::exact::exact_calculation;
use crate::raw_estimation::{raw_estimate_batch, raw_estimate_reuse, raw_estimate_single};
use crate::raw_estimation_lucj::{raw_estimate_lucj_batch, raw_estimate_lucj_single};

#[pymodule]
fn _lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_estimate_single, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(exact_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_single, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_lucj_single, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate_lucj_batch, m)?)?;
    Ok(())
}
