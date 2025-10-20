use ndarray::{Array2, Array3, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::Determinant;
use num_complex::Complex64;
use std::f64::consts::PI;

pub fn calculate_expectation<S>(
    v: &ArrayBase<S, Ix2>,
    initial_state: u128,
    outcome_state: u128,
) -> Complex64
where
    S: Data<Elem = Complex64>,
{
    let (nrows, ncols) = (v.shape()[0], v.shape()[1]);

    let row_count = outcome_state.count_ones() as usize;
    let col_count = if nrows == ncols {
        initial_state.count_ones() as usize
    } else {
        ncols
    };

    if row_count == 0 || col_count == 0 {
        return Complex64::new(0.0, 0.0);
    }

    let mut rows = Vec::with_capacity(row_count);
    for i in 0..nrows {
        if (outcome_state >> i) & 1 == 1 {
            rows.push(i);
        }
    }

    let mut cols = Vec::with_capacity(col_count);
    if nrows == ncols {
        for j in 0..ncols {
            if (initial_state >> j) & 1 == 1 {
                cols.push(j);
            }
        }
    } else {
        cols.extend(0..ncols);
    }

    let mut buffer = Vec::with_capacity(row_count * col_count);
    for &r in &rows {
        for &c in &cols {
            buffer.push(v[(r, c)]);
        }
    }

    let sub = Array2::from_shape_vec((row_count, col_count), buffer)
        .expect("buffer length must match submatrix dimensions");
    sub.det().unwrap_or(Complex64::new(0.0, 0.0))
}

#[inline(always)]
pub fn build_v_matrix(
    num_qubits: usize,
    raw: &[f64],
    x_mask: u128,
    gts: &[u8],
    pmat: &Array2<f64>,
    qmat: &Array2<usize>,
    orb_idx: &[i64],
    orb_mats: &Array3<Complex64>,
) -> Array2<Complex64> {
    let mut v = Array2::<Complex64>::eye(num_qubits);
    let mut cp_idx = 0;

    for (k, &gt) in gts.iter().enumerate() {
        match gt {
            1 => {
                let theta = raw[cp_idx];
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];
                let phase = if ((x_mask >> cp_idx) & 1) == 1 {
                    Complex64::from_polar(1.0, theta * 0.5 + PI)
                } else {
                    Complex64::from_polar(1.0, theta * 0.5)
                };
                for j in 0..num_qubits {
                    v[(q1, j)] *= phase;
                    v[(q2, j)] *= phase;
                }
                cp_idx += 1;
            }
            2 => {
                let theta = pmat[(k, 0)];
                let beta = pmat[(k, 1)];
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];

                let ph = Complex64::from_polar(1.0, beta + PI / 2.0);
                for j in 0..num_qubits {
                    v[(q1, j)] *= ph;
                }
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                for j in 0..num_qubits {
                    let a = v[(q1, j)];
                    let b = v[(q2, j)];
                    v[(q1, j)] = Complex64::new(c, 0.0) * a + Complex64::new(s, 0.0) * b;
                    v[(q2, j)] = Complex64::new(-s, 0.0) * a + Complex64::new(c, 0.0) * b;
                }
                let iph = Complex64::from_polar(1.0, -beta - PI / 2.0);
                for j in 0..num_qubits {
                    v[(q1, j)] *= iph;
                }
            }
            3 => {
                let theta = pmat[(k, 0)];
                let q1 = qmat[(k, 0)];
                let phase = Complex64::from_polar(1.0, theta);
                for j in 0..num_qubits {
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
