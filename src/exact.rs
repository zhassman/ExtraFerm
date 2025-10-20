use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::core::{build_v_matrix, calculate_expectation};

#[pyfunction]
pub fn exact_calculation(
    py: Python,
    num_qubits: usize,
    angles: &Bound<'_, PyArray1<f64>>,
    initial_state: u128,
    outcome_states: &Bound<'_, PyAny>,
    gate_types: &Bound<'_, PyArray1<u8>>,
    params: &Bound<'_, PyArray2<f64>>,
    qubits: &Bound<'_, PyArray2<usize>>,
    orb_indices: &Bound<'_, PyArray1<i64>>,
    orb_mats: &Bound<'_, PyArray3<Complex64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    let outs: Vec<u128> = outcome_states.extract()?;
    let n_out = outs.len();
    let mut probabilities = vec![0.0f64; n_out];

    let in_hw = initial_state.count_ones();
    let valid: Vec<(usize, u128)> = outs
        .iter()
        .enumerate()
        .filter(|&(_i, &o)| o.count_ones() == in_hw)
        .map(|(i, &o)| (i, o))
        .collect();

    if !valid.is_empty() {
        let num_mask = 1u128 << raw.len();

        let accum: Vec<Complex64> = (0..num_mask)
            .into_par_iter()
            .map(|mask| {
                let mut weight = 1.0;
                let mut coeff = Complex64::new(1.0, 0.0);
                for (j, &theta) in raw.iter().enumerate() {
                    let t = theta / 4.0;
                    if (mask >> j) & 1 == 1 {
                        weight *= t.sin();
                        coeff *= Complex64::new(0.0, 1.0);
                    } else {
                        weight *= t.cos();
                    }
                }

                let v = build_v_matrix(
                    num_qubits,
                    raw,
                    mask,
                    gts,
                    &pmat,
                    &qmat,
                    orb_idx,
                    &orb_mats_arr,
                );

                valid
                    .iter()
                    .map(|&(_i, out)| {
                        coeff * calculate_expectation(&v, initial_state, out) * weight
                    })
                    .collect::<Vec<_>>()
            })
            .reduce(
                || vec![Complex64::new(0.0, 0.0); valid.len()],
                |mut acc, vec| {
                    for (a, b) in acc.iter_mut().zip(&vec) {
                        *a += *b;
                    }
                    acc
                },
            );

        for ((idx, _), alpha) in valid.into_iter().zip(accum) {
            probabilities[idx] = alpha.norm_sqr();
        }
    }

    Ok(PyArray1::from_vec(py, probabilities).to_owned().into())
}
