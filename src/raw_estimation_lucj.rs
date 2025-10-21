use ndarray::{s, Array2, Array3, Axis};
use ndarray_linalg::Determinant;
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

pub fn raw_estimate_lucj_internal(
    num_qubits: usize,
    raw: &[f64],
    negative_mask: u128,
    extent: f64,
    initial_state: u128,
    outcome_state: u128,
    trajectory_count: usize,
    gts: &[u8],
    _pmat: &Array2<f64>,
    qmat: &Array2<usize>,
    orb_idx: &[i64],
    orb_mats_arr: &Array3<Complex64>,
    seed: u64,
) -> f64 {
    let u: Array2<Complex64> = orb_mats_arr
        .index_axis(Axis(0), orb_idx[0].try_into().unwrap())
        .to_owned();
    let v: Array2<Complex64> = orb_mats_arr
        .index_axis(Axis(0), orb_idx[orb_idx.len() - 1].try_into().unwrap())
        .to_owned();

    let mut u_desired_cols: Vec<usize> = Vec::new();
    let mut v_desired_rows: Vec<usize> = Vec::new();
    for q in 0..num_qubits {
        if (initial_state >> q) & 1 == 1 {
            u_desired_cols.push(q);
        }
        if (outcome_state >> q) & 1 == 1 {
            v_desired_rows.push(q);
        }
    }

    let num_fermions = u_desired_cols.len();
    if num_fermions != v_desired_rows.len() {
        return 0.;
    }

    let mut u_subview = u.select(Axis(1), &u_desired_cols);
    let v_subview = v.select(Axis(0), &v_desired_rows);

    let mut cp_idx = 0;
    for (k, &gt) in gts.iter().enumerate() {
        if gt == 1 {
            let theta = raw[cp_idx];
            let q1 = qmat[(k, 0)];
            let q2 = qmat[(k, 1)];
            let phase = Complex64::from_polar(1.0, theta * 0.5);
            let mut r1 = u_subview.index_axis_mut(Axis(0), q1);
            r1 *= phase;
            let mut r2 = u_subview.index_axis_mut(Axis(0), q2);
            r2 *= phase;
            cp_idx += 1;
        }
    }

    let base_mat = v_subview.dot(&u_subview);

    let mut corrections: Array3<Complex64> =
        Array3::zeros((num_qubits, num_fermions, num_fermions));
    for i in 0..num_qubits {
        let u_row_vec = u_subview.index_axis(Axis(0), i);
        let v_col_vec = v_subview.index_axis(Axis(1), i);
        let u_reshaped = u_row_vec.to_shape((1, num_fermions)).unwrap();
        let v_reshaped = v_col_vec.to_shape((num_fermions, 1)).unwrap();
        let outer: Array2<Complex64> = v_reshaped.dot(&u_reshaped) * (-2.0);
        corrections.slice_mut(s![i, .., ..]).assign(&outer);
    }

    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| {
            let t = theta.abs() / 4.0;
            (t.sin(), t.cos())
        })
        .unzip();
    let probs: Vec<f64> = pre_sin
        .iter()
        .zip(pre_cos.iter())
        .map(|(s, c)| s / (s + c))
        .collect();

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut rand_buf: Vec<u64> = vec![0u64; probs.len()];
    const INV_U64: f64 = 1.0 / ((u64::MAX as f64) + 1.0);

    let mut x_masks_counts: HashMap<u128, i64> = HashMap::new();
    for _ in 0..trajectory_count {
        rng.fill(rand_buf.as_mut_slice());
        let mut x_mask: u128 = 0;
        for (i, &r) in rand_buf.iter().enumerate() {
            let u = (r as f64) * INV_U64;
            if u < probs[i] {
                x_mask |= 1 << i;
            }
        }
        match x_masks_counts.get_mut(&x_mask) {
            Some(count) => *count += 1,
            None => {
                x_masks_counts.insert(x_mask, 1);
            }
        }
    }

    let mut diags: HashSet<u128> = HashSet::new();
    let mut x_masks_diags: HashMap<u128, u128> = HashMap::new();
    for (&x_mask, _) in x_masks_counts.iter() {
        let mut diag_mask: u128 = 0;
        let mut cp_idx: i64 = 0;
        for (k, &gt) in gts.iter().enumerate() {
            if gt == 1 {
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];
                if (x_mask >> cp_idx) & 1 == 1 {
                    diag_mask ^= 1 << q1;
                    diag_mask ^= 1 << q2;
                }
                cp_idx += 1;
            }
        }
        x_masks_diags.insert(x_mask, diag_mask);
        diags.insert(diag_mask);
    }

    let diags_dets: HashMap<u128, Complex64> = diags
        .into_par_iter()
        .map(|d| {
            let mut mat = base_mat.clone();
            for i in 0..num_qubits {
                if (d >> i) & 1 == 1 {
                    mat += &corrections.slice(s![i, .., ..]);
                }
            }
            (d, mat.det().unwrap())
        })
        .collect::<HashMap<u128, Complex64>>();

    let sum_alpha: Complex64 = x_masks_diags
        .into_iter()
        .map(|(x_mask, diag)| {
            let amp = diags_dets.get(&diag).unwrap();
            let count: f64 = *x_masks_counts.get(&x_mask).unwrap() as f64;
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
            j_phase * amp * sign * count
        })
        .sum();

    let t2 = (trajectory_count * trajectory_count) as f64;
    (sum_alpha.norm_sqr() / t2) * extent
}

#[pyfunction]
pub fn raw_estimate_lucj_single(
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
    seed: u64,
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
        Ok(raw_estimate_lucj_internal(
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
            seed,
        ))
    }
}

#[pyfunction]
pub fn raw_estimate_lucj_batch(
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
            if initial_state.count_ones() != outcome_state.count_ones() {
                0.0
            } else {
                raw_estimate_lucj_internal(
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
                    seed.wrapping_add(idx as u64),
                )
            }
        })
        .collect();

    Ok(PyArray1::from_vec(py, results).to_owned().into())
}
