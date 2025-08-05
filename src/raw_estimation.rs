use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distr::StandardUniform;
use rayon::prelude::*;
use ndarray::{Array2, Array3, Axis, s};
use ndarray_linalg::Determinant;
use std::vec;
use std::collections::HashMap;
use std::collections::HashSet;

use std::time;

use crate::core::build_v_matrix;
use crate::core::calculate_expectation;


#[pyfunction]
pub fn raw_estimate_udv_single(
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
    let start = std::time::Instant::now();
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let _pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    let u : Array2<Complex64> = orb_mats_arr.index_axis(Axis(0), orb_idx[0].try_into().unwrap()).to_owned();
    let v : Array2<Complex64> = orb_mats_arr.index_axis(Axis(0), orb_idx[orb_idx.len()-1].try_into().unwrap()).to_owned();

    
    let mut u_desired_cols : Vec<usize> = Vec::new();
    let mut v_desired_rows : Vec<usize> = Vec::new();

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
	return Ok(0.);
    }
    
    let mut u_subview = u.select(Axis(1), &u_desired_cols);
    let v_subview = v.select(Axis(0), &v_desired_rows);
    
    //update u  with the free operations d_0(theta) for each cphase gate
    let mut cp_idx = 0;
    for (k, &gt) in gts.iter().enumerate() {
        match gt {
            1 => {		
                let theta = raw[cp_idx];
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];
		
                let phase = Complex64::from_polar(1.0, theta * 0.5);

		let mut r1 = u_subview.index_axis_mut(Axis(0), q1);		
		r1 *= phase;
		let mut r2 = u_subview.index_axis_mut(Axis(0), q2);
		r2 *= phase;
                cp_idx += 1;
            },
	    _ => {}
	}
    }
    
    let base_mat = v_subview.dot(&u_subview);
    
    let mut corrections : Array3<Complex64> = Array3::zeros((num_qubits, num_fermions, num_fermions));
    
    for i in 0..num_qubits {
	let u_row_vec = u_subview.index_axis(Axis(0), i);
	let v_col_vec = v_subview.index_axis(Axis(1), i);
	let u_reshaped = u_row_vec.to_shape((1, num_fermions)).unwrap();
	let v_reshaped = v_col_vec.to_shape((num_fermions, 1)).unwrap();
	let outer : Array2<Complex64> = v_reshaped.dot(&u_reshaped)*(-2.0);

	corrections.slice_mut(s![i,..,..]).assign(&outer);
    }


    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| {
            let t = theta.abs() / 4.0;
            (t.sin(), t.cos())
        })
        .unzip();

    let end_precomputation = std::time::Instant::now();
    
    //let mut baserng = StdRng::from_os_rng();
    //let seeds: Vec<u64> = (&mut baserng).sample_iter(StandardUniform).take(trajectory_count).collect();
    let mut rng = StdRng::from_os_rng();

    
    let mut diags :HashSet<u128> = HashSet::new();
    let mut x_masks_diags :HashMap<u128, u128> = HashMap::new();
    let mut x_masks_counts :HashMap<u128, i64> = HashMap::new();
    for _ in 0..trajectory_count {
	
	let mut x_mask: u128 = 0;
	for i in 0..pre_sin.len() {
            if rng.random::<f64>() < pre_sin[i] / (pre_sin[i] + pre_cos[i]) {
                    x_mask |= 1 << i;
            }
        }
	let mut diag_mask: u128 = 0;
	let mut cp_idx :i64 = 0;
	for (k, &gt) in gts.iter().enumerate() {
	    match gt {
		1 => {
		    let q1 = qmat[(k, 0)];
		    let q2 = qmat[(k, 1)];
		    if (x_mask >> cp_idx) & 1 == 1 {
			diag_mask ^= 1 << q1;
			diag_mask ^= 1 << q2;
		    }
		    cp_idx += 1;
		},
		_ => {}
	    }
	}
	x_masks_diags.insert(x_mask, diag_mask);
	diags.insert(diag_mask);
	match x_masks_counts.get_mut(&x_mask){
	    Some(count) => {*count += 1;},
	    None => {x_masks_counts.insert(x_mask, 1);}
	}
    }

    let end_hashmap_computation = std::time::Instant::now();
    let diags_dets : HashMap<u128, Complex64> = diags.into_par_iter()
	.map(|d| {
	    
	let mut mat = base_mat.clone();

	for i in 0..num_qubits {
	    if (d >> i) & 1 == 1 {
		mat += &corrections.slice(s![i,..,..]);
	    }
	}

	(d, mat.det().unwrap())
    }).collect::<HashMap<u128, Complex64>>();

    let end_determinant_computation = std::time::Instant::now();
    
    let sum_alpha: Complex64 = x_masks_diags
	.into_iter()
        .map(|(x_mask, diag)| {	    

            let amp = diags_dets.get(&diag).unwrap();
	    let count :f64 = *x_masks_counts.get(&x_mask).unwrap() as f64;
	    
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

            j_phase * amp * sign *count
        })
        .sum();
    
    let end_full_computation = std::time::Instant::now();

    println!("total computation time (udv) = {}", end_full_computation.duration_since(start).as_secs_f64());
    println!("precomputation time = {}", end_precomputation.duration_since(start).as_secs_f64());
    println!("hashmap time = {}", end_hashmap_computation.duration_since(end_precomputation).as_secs_f64());
    println!("determinant time = {}", end_determinant_computation.duration_since(end_hashmap_computation).as_secs_f64());    
    println!("reduction time = {}", end_full_computation.duration_since(end_determinant_computation).as_secs_f64());
    
    
    let t2 = (trajectory_count * trajectory_count) as f64;
    let val = (sum_alpha.norm_sqr() / t2) * extent;
    Ok(val)
}


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
    let start = std::time::Instant::now();
    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| {
            let t = theta.abs() / 4.0;
            (t.sin(), t.cos())
        })
        .unzip();
    
    let mut baserng = StdRng::from_os_rng();
    let seeds: Vec<u64> = (&mut baserng).sample_iter(StandardUniform).take(trajectory_count).collect();

    /*
    let diags :HashSet<u128> = HashSet::from_iter(seeds.iter().map(|seed| {
	let mut rng = StdRng::seed_from_u64(*seed);
	let mut x_mask: u128 = 0;
	for i in 0..pre_sin.len() {
            if rng.random::<f64>() < pre_sin[i] / (pre_sin[i] + pre_cos[i]) {
                    x_mask |= 1 << i;
            }
        }
	let mut diag: Vec<i64> = vec![1; num_qubits];
	let mut diag_mask: u128 = 0;
	let mut cp_idx :i64 = 0;
	for (k, &gt) in gts.iter().enumerate() {
	    match gt {
		1 => {
		    let q1 = qmat[(k, 0)];
		    let q2 = qmat[(k, 1)];
		    if (x_mask >> cp_idx) & 1 == 1 {
			diag_mask ^= 1 << q1;
			diag_mask ^= 1 << q2;
			
			diag[q1] *= -1;
			diag[q2] *= -1;
		    }
		    cp_idx += 1;
		},
		_ => {}
	    }
	}
	diag_mask
    }));*/
    
    let sum_alpha: Complex64 = seeds
        .into_par_iter()
        .map(|seed| {	    
            let mut rng = StdRng::seed_from_u64(seed);
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
        .sum();
    let end = std::time::Instant::now();
    println!("total computation time (internal) = {}", end.duration_since(start).as_secs_f64());
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
