use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray3};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use ndarray::{Array2, Array3};

use crate::core::build_v_matrix;
use crate::core::calculate_expectation;


/// Internal function performing the full Monte Carlo logic.
/// Accepts Rust-native types and returns the estimated probability.
pub fn raw_estimate_internal(
    num_qubits: usize,
    raw: &[f64],
    negative_mask: u128,
    extent: f64,
    in_state: u128,
    out_state: u128,
    trajectory_count: usize,
    gts: &[u8],
    pmat: &Array2<f64>,
    qmat: &Array2<usize>,
    orb_idx: &[i64],
    orb_mats_arr: &Array3<Complex64>,
) -> f64 {
    // Precompute sin/cos splitting probabilities
    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| {
            let t = theta.abs() / 4.0;
            (t.sin(), t.cos())
        })
        .unzip();

    // Parallel Monte Carlo across trajectories
    let sum_alpha: Complex64 = (0..trajectory_count)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut x_mask: u128 = 0;
            for i in 0..pre_sin.len() {
                if rng.gen::<f64>() < pre_sin[i] / (pre_sin[i] + pre_cos[i]) {
                    x_mask |= 1 << i;
                }
            }

            // Build the V-matrix for this trajectory
            let v_mat = build_v_matrix(
                num_qubits,
                raw,
                x_mask,
                gts,
                pmat,
                qmat,
                orb_idx,
                orb_mats_arr,
            );

            // Compute the amplitude for this bitstring
            let amp = calculate_expectation(&v_mat, in_state, out_state);

            // Phase and sign corrections
            let j_phase = match x_mask.count_ones() % 4 {
                0 => Complex64::new(1.0, 0.0),
                1 => Complex64::new(0.0, 1.0),
                2 => Complex64::new(-1.0, 0.0),
                3 => Complex64::new(0.0, -1.0),
                _ => unreachable!(),
            };
            let sign = if (negative_mask & x_mask).count_ones() % 2 == 1 {
                -1.0
            } else {
                1.0
            };

            j_phase * amp * sign
        })
        .reduce(|| Complex64::new(0.0, 0.0), |a, b| a + b);

    // Normalize by trajectory count squared
    let t2 = (trajectory_count * trajectory_count) as f64;
    (sum_alpha.norm_sqr() / t2) * extent
}

/// Python-facing binding: extracts all PyArray arguments,
/// calls the internal estimator, and returns a PyResult.
#[pyfunction]
pub fn raw_estimate_single(
    num_qubits: usize,
    angles: &PyArray1<f64>,
    negative_mask: u128,
    extent: f64,
    in_state: u128,
    out_state: u128,
    trajectory_count: usize,
    gate_types: &PyArray1<u8>,
    params: &PyArray2<f64>,
    qubits: &PyArray2<usize>,
    orb_indices: &PyArray1<i64>,
    orb_mats: &PyArray3<Complex64>,
) -> PyResult<f64> {
    // Early exit if bit-counts differ
    if in_state.count_ones() != out_state.count_ones() {
        return Ok(0.0);
    }

    // Extract Rust-native slices and owned arrays
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    // Delegate to the internal function
    let result = raw_estimate_internal(
        num_qubits,
        raw,
        negative_mask,
        extent,
        in_state,
        out_state,
        trajectory_count,
        gts,
        &pmat,
        &qmat,
        orb_idx,
        &orb_mats_arr,
    );

    Ok(result)
}


#[pyfunction(allow_threads)]
pub fn raw_estimate_batch(
    py: Python,
    num_qubits: usize,
    angles: &PyArray1<f64>,
    negative_mask: u128,
    extent: f64,
    in_state: u128,
    out_states: &PyAny,  // any Python iterable of ints
    trajectory_count: usize,
    gate_types: &PyArray1<u8>,
    params: &PyArray2<f64>,
    qubits: &PyArray2<usize>,
    orb_indices: &PyArray1<i64>,
    orb_mats: &PyArray3<Complex64>,
) -> PyResult<Py<PyArray1<f64>>> {
    // extract core arrays
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    // extract out_states into Vec<u128>
    let outs: Vec<u128> = out_states.extract()?;

    // parallel compute
    let results: Vec<f64> = outs
        .into_par_iter()
        .map(|out_state| {
            if in_state.count_ones() != out_state.count_ones() {
                0.0
            } else {
                raw_estimate_internal(
                    num_qubits,
                    raw,
                    negative_mask,
                    extent,
                    in_state,
                    out_state,
                    trajectory_count,
                    gts,
                    &pmat,
                    &qmat,
                    orb_idx,
                    &orb_mats_arr,
                )
            }
        })
        .collect();

    Ok(PyArray1::from_vec(py, results).to_owned())
}


#[pyfunction(allow_threads)]
pub fn raw_estimate_reuse(
    py: Python,
    num_qubits: usize,
    angles: &PyArray1<f64>,
    negative_mask: u128,
    extent: f64,
    in_state: u128,
    out_states: &PyAny,
    trajectory_count: usize,
    gate_types: &PyArray1<u8>,
    params: &PyArray2<f64>,
    qubits: &PyArray2<usize>,
    orb_indices: &PyArray1<i64>,
    orb_mats: &PyArray3<Complex64>,
) -> PyResult<Py<PyArray1<f64>>> {
    // 1) extract static data from Python
    let raw: &[f64]      = unsafe { angles.as_slice()? };
    let gts: &[u8]       = unsafe { gate_types.as_slice()? };
    let pmat             = unsafe { params.as_array().to_owned() };
    let qmat             = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64]  = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr     = unsafe { orb_mats.as_array().to_owned() };

    // 2) collect all out_states as u128
    let outs: Vec<u128> = out_states.extract()?;
    let n_out = outs.len();

    // 3) precompute sin/cos splitting
    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| { let t = theta.abs() / 4.0; (t.sin(), t.cos()) })
        .unzip();

    // 4) full nested parallelism: outer over trajectories
    let accum: Vec<Complex64> = (0..trajectory_count)
        .into_par_iter()
        .map(|_| {
            // sample x_mask
            let mut rng = rand::thread_rng();
            let mut x_mask: u128 = 0;
            for i in 0..pre_sin.len() {
                if rng.gen::<f64>() < pre_sin[i] / (pre_sin[i] + pre_cos[i]) {
                    x_mask |= 1 << i;
                }
            }
            // build V-matrix for this trajectory
            let v_mat = build_v_matrix(
                num_qubits, raw, x_mask, gts,
                &pmat, &qmat, orb_idx, &orb_mats_arr
            );
            // inner parallel: compute contributions for each out_state
            outs.par_iter()
                .map(|&out_state| {
                    if in_state.count_ones() != out_state.count_ones() {
                        Complex64::new(0.0, 0.0)
                    } else {
                        let amp = calculate_expectation(&v_mat, in_state, out_state);
                        let j_phase = match x_mask.count_ones() % 4 {
                            0 => Complex64::new(1.0, 0.0),
                            1 => Complex64::new(0.0, 1.0),
                            2 => Complex64::new(-1.0, 0.0),
                            3 => Complex64::new(0.0, -1.0),
                            _ => unreachable!(),
                        };
                        let sign = if (negative_mask & x_mask).count_ones() % 2 == 1 {
                            -1.0
                        } else { 1.0 };
                        j_phase * amp * sign
                    }
                })
                .collect::<Vec<_>>()
        })
        // combine per-trajectory Vecs into one accumulator
        .reduce(
            || vec![Complex64::new(0.0, 0.0); n_out],
            |mut acc, vec| {
                for (a, &c) in acc.iter_mut().zip(&vec) {
                    *a += c;
                }
                acc
            }
        );

    // 5) normalize into probabilities
    let norm = (trajectory_count * trajectory_count) as f64;
    let probs: Vec<f64> = accum
        .into_iter()
        .map(|alpha| (alpha.norm_sqr() / norm) * extent)
        .collect();

    // 6) return as NumPy array
    Ok(PyArray1::from_vec(py, probs).to_owned())
}
