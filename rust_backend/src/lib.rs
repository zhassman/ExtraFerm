use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray3};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use ndarray::{Array2, Array3, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::Determinant;
use std::f64::consts::PI;
use pyo3::wrap_pyfunction;


/// Direct determinant calculation without copying submatrices
///
/// Given an n×m matrix `v`, and two basis‐state bitmasks `in_state` and
/// `out_state`, build the submatrix (rows where out_state has a 1, and
/// columns where in_state has a 1 if n=m, else *all* columns), then return
/// its determinant (or 0 if something goes wrong).
fn calculate_expectation_direct<S>(
    v: &ArrayBase<S, Ix2>,
    in_state: usize,
    out_state: usize,
) -> Complex64
where
    S: Data<Elem = Complex64>,
{
    let (nrows, ncols) = (v.shape()[0], v.shape()[1]);

    // how many rows/columns we’ll extract
    let row_count = out_state.count_ones() as usize;
    let col_count = if nrows == ncols {
        in_state.count_ones() as usize
    } else {
        ncols
    };

    // short‐circuit if there’s nothing to do
    if row_count == 0 || col_count == 0 {
        return Complex64::new(0.0, 0.0);
    }

    // pick out the row indices
    let mut rows = Vec::with_capacity(row_count);
    for i in 0..nrows {
        if (out_state >> i) & 1 == 1 {
            rows.push(i);
        }
    }

    // pick out the column indices
    let mut cols = Vec::with_capacity(col_count);
    if nrows == ncols {
        for j in 0..ncols {
            if (in_state >> j) & 1 == 1 {
                cols.push(j);
            }
        }
    } else {
        cols.extend(0..ncols);
    }

    // copy the submatrix into a contiguous buffer
    let mut buffer = Vec::with_capacity(row_count * col_count);
    for &r in &rows {
        for &c in &cols {
            buffer.push(v[(r, c)]);
        }
    }

    // build an Array2 from it and take the determinant
    let sub = Array2::from_shape_vec((row_count, col_count), buffer)
        .expect("buffer length must match submatrix dimensions");
    sub.det().unwrap_or(Complex64::new(0.0, 0.0))
}

/// Build the n-qubit state matrix V by applying each gate in sequence.
///
/// This is marked inline(always) so LLVM will fuse it into the hot loop.
#[inline(always)]
fn build_v_matrix(
    nqubits: usize,
    raw: &[f64],
    x_mask: usize,
    gts: &[u8],
    pmat: &Array2<f64>,
    qmat: &Array2<usize>,
    orb_idx: &[i64],
    orb_mats: &Array3<Complex64>,
) -> Array2<Complex64> {
    let mut v = Array2::<Complex64>::eye(nqubits);
    let mut cp_idx = 0;

    for (k, &gt) in gts.iter().enumerate() {
        match gt {
            1 => {
                let theta = raw[k];
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];
                let phase = if ((x_mask >> cp_idx) & 1) == 1 {
                    Complex64::from_polar(1.0, theta * 0.5 + PI)
                } else {
                    Complex64::from_polar(1.0, theta * 0.5)
                };
                for j in 0..nqubits {
                    v[(q1, j)] *= phase;
                    v[(q2, j)] *= phase;
                }
                cp_idx += 1;
            }
            2 => {
                let theta = pmat[(k, 0)];
                let beta  = pmat[(k, 1)];
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];

                let ph = Complex64::from_polar(1.0, beta + PI / 2.0);
                for j in 0..nqubits {
                    v[(q1, j)] *= ph;
                }
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                for j in 0..nqubits {
                    let a = v[(q1, j)];
                    let b = v[(q2, j)];
                    v[(q1, j)] = Complex64::new(c, 0.0) * a + Complex64::new(s, 0.0) * b;
                    v[(q2, j)] = Complex64::new(-s, 0.0) * a + Complex64::new(c, 0.0) * b;
                }
                let iph = Complex64::from_polar(1.0, -beta - PI / 2.0);
                for j in 0..nqubits {
                    v[(q1, j)] *= iph;
                }
            }
            3 => {
                let theta = pmat[(k, 0)];
                let q1 = qmat[(k, 0)];
                let phase = Complex64::from_polar(1.0, theta);
                for j in 0..nqubits {
                    v[(q1, j)] *= phase;
                }
            }
            4 => {
                let idx = orb_idx[k] as usize;
                let mat = orb_mats.index_axis(Axis(0), idx);
                v = mat.dot(&v);
            }
            _ => {}
        }
    }

    v
}


#[pyfunction]
fn raw_estimate(
    py: Python,
    angles: &PyArray1<f64>,
    negative_mask: usize,
    extent: f64,
    in_state: usize,
    out_state: usize,
    trajectory_count: usize,
    gate_types: &PyArray1<u8>,
    params: &PyArray2<f64>,
    qubits: &PyArray2<usize>,
    orb_indices: &PyArray1<i64>,
    orb_mats: &PyArray3<Complex64>,
) -> PyResult<f64> {
    if in_state.count_ones() != out_state.count_ones() {
        return Ok(0.0);
    }

    let raw: &[f64] = unsafe { angles.as_slice()? };
    let gts: &[u8] = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let orb_idx: &[i64] = unsafe { orb_indices.as_slice()? };
    let orb_mats_arr = unsafe { orb_mats.as_array().to_owned() };

    let (pre_sin, pre_cos): (Vec<f64>, Vec<f64>) = raw
        .iter()
        .map(|&theta| {
            let t = theta.abs() / 4.0;
            (t.sin(), t.cos())
        })
        .unzip();

    let max_q = qmat.iter().cloned().max().unwrap_or(0);
    let nqubits = max_q + 1;

    let sum_alpha: Complex64 = (0..trajectory_count)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut x_mask = 0;
            for i in 0..pre_sin.len() {
                if rng.gen::<f64>() < pre_sin[i] / (pre_sin[i] + pre_cos[i]) {
                    x_mask |= 1 << i;
                }
            }

            let v_mat = build_v_matrix(
                nqubits,
                raw,
                x_mask,
                gts,
                &pmat,
                &qmat,
                orb_idx,
                &orb_mats_arr,
            );

            let amp = calculate_expectation_direct(&v_mat, in_state, out_state);

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

    let t2 = (trajectory_count * trajectory_count) as f64;
    Ok((sum_alpha.norm_sqr() / t2) * extent)
}


#[pymodule]
fn emsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raw_estimate, m)?)?;
    Ok(())
}