use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use ndarray::{Array2, Array3};
use std::vec;

// use std::time;

use crate::core::build_v_matrix;
use crate::core::calculate_expectation;


pub fn raw_estimate_internal(
    num_qubits: usize,
    raw: &[f64],
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_state: u128,
    trajectory_count: usize,
    gts: &[u8],
    pmat: &Array2<f64>,
    qmat: &Array2<usize>,
    orb_idx: &[i64],
    orb_mats_arr: &Array3<Complex64>,
) -> f64 {
    // let start = std::time::Instant::now();
    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| {
            let t = theta.abs() / 4.0;
            (t.sin(), t.cos())
        })
        .unzip();


    let sum_alpha: Complex64 = (0..trajectory_count)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::rng();
            let mut x_mask: u128 = 0;
            for i in 0..pre_sin.len() {
                if rng.random::<f64>() < pre_sin[i] / (pre_sin[i] + pre_cos[i]) {
                    x_mask |= 1 << i;
                }
            }


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


            let amp = calculate_expectation(&v_mat, initial_state, outcome_state);


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

    // let end = std::time::Instant::now();
    // println!("total computation time (internal) = {}", end.duration_since(start).as_secs_f64());
    let t2 = (trajectory_count * trajectory_count) as f64;
    (sum_alpha.norm_sqr() / t2) * extent
}


#[pyfunction]
pub fn raw_estimate_single(
    num_qubits: usize,
    angles: &Bound<'_, PyArray1<f64>>,
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_state: u128,
    trajectory_count: usize,
    gate_types: &Bound<'_, PyArray1<u8>>,
    params: &Bound<'_, PyArray2<f64>>,
    qubits: &Bound<'_, PyArray2<usize>>,
    orb_indices: &Bound<'_, PyArray1<i64>>,
    orb_mats: &Bound<'_, PyArray3<Complex64>>,
) -> PyResult<f64> {
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    if initial_state.count_ones() != outcome_state.count_ones() {
        Ok(0.0)
    } else {
        Ok(raw_estimate_internal(
            num_qubits,
            raw,
            negative_mask,
            extent,
            initial_state,
            outcome_state,
            trajectory_count,
            gts,
            &pmat,
            &qmat,
            orb_idx,
            &orb_mats_arr,
        ))
    }
}


#[pyfunction]
pub fn raw_estimate_batch(
    py: Python,
    num_qubits: usize,
    angles: &Bound<'_, PyArray1<f64>>,
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_states: &Bound<'_, PyAny>,
    trajectory_count: usize,
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

    let results: Vec<f64> = outs
        .into_par_iter()
        .map(|outcome_state| {
            if initial_state.count_ones() != outcome_state.count_ones() {
                0.0
            } else {
                raw_estimate_internal(
                    num_qubits,
                    raw,
                    negative_mask,
                    extent,
                    initial_state,
                    outcome_state,
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

    Ok(PyArray1::from_vec(py, results).to_owned().into())
}


#[pyfunction]
pub fn raw_estimate_reuse(
    py: Python,
    num_qubits: usize,
    angles: &Bound<'_, PyArray1<f64>>,
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_states: &Bound<'_, PyAny>,
    trajectory_count: usize,
    gate_types: &Bound<'_, PyArray1<u8>>,
    params: &Bound<'_, PyArray2<f64>>,
    qubits: &Bound<'_, PyArray2<usize>>,
    orb_indices: &Bound<'_, PyArray1<i64>>,
    orb_mats: &Bound<'_, PyArray3<Complex64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let raw: &[f64]      = unsafe { angles.as_slice()? };
    let gts: &[u8]       = unsafe { gate_types.as_slice()? };
    let pmat             = unsafe { params.as_array().to_owned() };
    let qmat             = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64]  = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr     = unsafe { orb_mats.as_array().to_owned() };

    let outs: Vec<u128> = outcome_states.extract()?;
    let n_out = outs.len();

    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| { let t = theta.abs() / 4.0; (t.sin(), t.cos()) })
        .unzip();

    let accum: Vec<Complex64> = (0..trajectory_count)
        .into_par_iter()
        .map(|_| {

            let mut rng = rand::rng();
            let mut x_mask: u128 = 0;
            for i in 0..pre_sin.len() {
                if rng.random::<f64>() < pre_sin[i] / (pre_sin[i] + pre_cos[i]) {
                    x_mask |= 1 << i;
                }
            }

            let v_mat = build_v_matrix(
                num_qubits, raw, x_mask, gts,
                &pmat, &qmat, orb_idx, &orb_mats_arr
            );

            outs.par_iter()
                .map(|&outcome_state| {
                    if initial_state.count_ones() != outcome_state.count_ones() {
                        Complex64::new(0.0, 0.0)
                    } else {
                        let amp = calculate_expectation(&v_mat, initial_state, outcome_state);
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

        .reduce(
            || vec![Complex64::new(0.0, 0.0); n_out],
            |mut acc, vec| {
                for (a, &c) in acc.iter_mut().zip(&vec) {
                    *a += c;
                }
                acc
            }
        );

    let norm = (trajectory_count * trajectory_count) as f64;
    let probs: Vec<f64> = accum
        .into_iter()
        .map(|alpha| (alpha.norm_sqr() / norm) * extent)
        .collect();

    Ok(PyArray1::from_vec(py, probs).to_owned().into())
}
