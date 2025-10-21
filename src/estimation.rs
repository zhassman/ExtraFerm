use ndarray::{Array2, Array3};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::{E, PI};

use crate::raw_estimation::raw_estimate_internal;
use crate::raw_estimation_lucj::raw_estimate_lucj_internal;

fn calculate_epsilon(p: f64, delta: f64, s: usize, extent: f64) -> f64 {
    let sqrt_p = p.sqrt();
    let sqrt_extent = extent.sqrt();
    let log_term = (2.0 * E.powi(2) / delta).ln();
    let last_term = (2.0 * log_term / (s as f64)).sqrt();
    -p + (sqrt_p + (sqrt_extent + sqrt_p) * last_term).powi(2)
}

fn calculate_trajectory_count(epsilon: f64, delta: f64, extent: f64, p: f64) -> usize {
    let root_e = extent.sqrt();
    let root_p = p.sqrt();
    let numerator = (root_e + root_p).powi(2);
    let log_term = (2.0 * E.powi(2) / delta).ln();
    let denominator = ((p + epsilon).sqrt() - root_p).powi(2);
    (2.0 * numerator * log_term / denominator).ceil() as usize
}

pub fn estimate_internal(
    num_qubits: usize,
    angles: &[f64],
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_state: u128,
    epsilon_total: f64,
    delta_total: f64,
    use_lucj: bool,
    gate_types: &[u8],
    pmat: &Array2<f64>,
    qmat: &Array2<usize>,
    orb_idx: &[i64],
    orb_mats_arr: &Array3<Complex64>,
    seed: u64,
) -> f64 {
    let mut p_star = 1.0;
    let mut p_hat = 1.0;
    let mut exit_condition = false;
    let mut e_star = 1.0;
    let mut k = 1_usize;
    let mut delta_k = 6.0 * delta_total / (PI * (k as f64)).powi(2);
    let mut s = calculate_trajectory_count(e_star, delta_k, extent, p_star);

    while !exit_condition {
        e_star = calculate_epsilon(p_star, delta_k, s, extent);
        if e_star < epsilon_total {
            exit_condition = true;
        }

        let estimate_val = if initial_state.count_ones() != outcome_state.count_ones() {
            0.0
        } else if use_lucj {
            raw_estimate_lucj_internal(
                num_qubits,
                angles,
                negative_mask,
                extent,
                initial_state,
                outcome_state,
                s,
                gate_types,
                pmat,
                qmat,
                orb_idx,
                orb_mats_arr,
                seed,
            )
        } else {
            raw_estimate_internal(
                num_qubits,
                angles,
                negative_mask,
                extent,
                initial_state,
                outcome_state,
                s,
                gate_types,
                pmat,
                qmat,
                orb_idx,
                orb_mats_arr,
                seed,
            )
        };
        p_hat = estimate_val;
        // println!("k: {}, pHat: {}, eStar: {}, epsTot: {}", k, p_hat, e_star, epsilon_total);

        p_star = (p_star.min(p_hat + e_star)).clamp(0.0, 1.0);

        k += 1;
        delta_k = 6.0 * delta_total / (PI * (k as f64)).powi(2);
        s *= 2;
    }

    p_hat
}

#[pyfunction]
pub fn estimate_single(
    num_qubits: usize,
    angles: &Bound<'_, PyArray1<f64>>,
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_state: u128,
    epsilon_total: f64,
    delta_total: f64,
    use_lucj: bool,
    gate_types: &Bound<'_, PyArray1<u8>>,
    params: &Bound<'_, PyArray2<f64>>,
    qubits: &Bound<'_, PyArray2<usize>>,
    orb_indices: &Bound<'_, PyArray1<i64>>,
    orb_mats: &Bound<'_, PyArray3<Complex64>>,
    seed: u64,
) -> PyResult<f64> {
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    Ok(estimate_internal(
        num_qubits,
        raw,
        negative_mask,
        extent,
        initial_state,
        outcome_state,
        epsilon_total,
        delta_total,
        use_lucj,
        gts,
        &pmat,
        &qmat,
        orb_idx,
        &orb_mats_arr,
        seed,
    ))
}

#[pyfunction]
pub fn estimate_batch(
    py: Python,
    num_qubits: usize,
    angles: &Bound<'_, PyArray1<f64>>,
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_states: &Bound<'_, PyAny>,
    epsilon_total: f64,
    delta_total: f64,
    use_lucj: bool,
    gate_types: &Bound<'_, PyArray1<u8>>,
    params: &Bound<'_, PyArray2<f64>>,
    qubits: &Bound<'_, PyArray2<usize>>,
    orb_indices: &Bound<'_, PyArray1<i64>>,
    orb_mats: &Bound<'_, PyArray3<Complex64>>,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    let outs: Vec<u128> = outcome_states.extract()?;

    let results: Vec<f64> = outs
        .into_par_iter()
        .enumerate()
        .map(|(idx, outcome_state)| {
            estimate_internal(
                num_qubits,
                raw,
                negative_mask,
                extent,
                initial_state,
                outcome_state,
                epsilon_total,
                delta_total,
                use_lucj,
                gts,
                &pmat,
                &qmat,
                orb_idx,
                &orb_mats_arr,
                seed.wrapping_add(idx as u64),
            )
        })
        .collect();

    Ok(PyArray1::from_vec(py, results).to_owned().into())
}
