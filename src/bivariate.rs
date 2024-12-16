//! Bivariate polynomial compatible with arkwork's trait and backend

use ark_ff::{Field, Zero};
use ark_poly::Polynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    fmt,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
    rand::Rng,
};
use p3_maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

// NOTE: I have avoid implmenting `trait DenseMVPolynomial`, because the knowledge of "dense+bivariate" obviates
// the need to define `BVTerm` and `num_vars()` etc. I will consider adding them if necessary down the road.
// TODO: avoid panic and define custom error type

/// A dense bivariate polynomial
///
/// # Representation
///
/// Dense polynomial assumes coefficients for all possible monimals.
/// General Matrix Form (Row-wise Storage):
///
/// [  X^0 * Y^0   X^0 * Y^1   X^0 * Y^2   ...   X^0 * Y^d_y  ]
/// [  X^1 * Y^0   X^1 * Y^1   X^1 * Y^2   ...   X^1 * Y^d_y  ]
/// [  X^2 * Y^0   X^2 * Y^1   X^2 * Y^2   ...   X^2 * Y^d_y  ]
/// [     ...          ...          ...    ...       ...      ]
/// [ X^d_x * Y^0  X^d_x * Y^1  X^d_x * Y^2  ...  X^d_x * Y^d_y ]
///
/// Matrix Dimensions: (d_x + 1) x (d_y + 1)
///
/// Row-wise iteration:
/// First fix the power of X (row), then iterate over powers of Y (columns).
///
/// Example: For d_x = 2, d_y = 2:
/// Matrix:
/// [ 1     1 * Y      1 * Y^2   ]
/// [ X     X * Y      X * Y^2   ]
/// [ X^2   X^2 * Y   X^2 * Y^2  ]
#[derive(Clone, PartialEq, Eq, Hash, CanonicalSerialize, CanonicalDeserialize)]
pub struct DensePolynomial<F: Field> {
    /// coefficients matrix for monomial terms:
    pub coeffs: Vec<Vec<F>>,
    /// degree in X
    pub deg_x: usize,
    /// degree in Y
    pub deg_y: usize,
}

impl<F: Field> Polynomial<F> for DensePolynomial<F> {
    type Point = [F; 2];
    fn degree(&self) -> usize {
        self.deg_x * self.deg_y
    }
    // full evaluation
    fn evaluate(&self, [x, y]: &Self::Point) -> F {
        self.coeffs
            .par_iter()
            .enumerate()
            .map(|(row_idx, row)| {
                row.par_iter()
                    .enumerate()
                    .map(|(col_idx, coeff)| {
                        *coeff * x.pow([row_idx as u64]) * y.pow([col_idx as u64])
                    })
                    .reduce(|| F::ZERO, |acc, term| acc + term)
            })
            .reduce(|| F::ZERO, |acc, row_sum| acc + row_sum)
    }
}

impl<F: Field> DensePolynomial<F> {
    /// constructor with check
    pub fn new(coeffs: Vec<Vec<F>>, deg_x: usize, deg_y: usize) -> Self {
        assert!(
            !coeffs.is_empty(),
            "empty coeffs, use ::zero() or ::default() for zero poly"
        );
        assert!(coeffs.len() > 0 && coeffs.len() == deg_x + 1);
        assert!(coeffs[0].len() > 0 && coeffs.par_iter().all(|row| row.len() == deg_y + 1));
        Self {
            coeffs,
            deg_x,
            deg_y,
        }
    }
    /// internal use when we know inputs are safe, thus forgo further checks
    #[allow(dead_code)]
    pub(crate) fn new_unchecked(coeffs: Vec<Vec<F>>, deg_x: usize, deg_y: usize) -> Self {
        Self {
            coeffs,
            deg_x,
            deg_y,
        }
    }

    /// generate a random bivariate poly with `d_x` degree in X, `d_y` degree in Y
    pub fn rand<R: Rng>(deg_x: usize, deg_y: usize, rng: &mut R) -> Self {
        let coeffs = (0..deg_x + 1)
            .map(|_| (0..deg_y + 1).map(|_| F::rand(rng)).collect())
            .collect();
        Self {
            coeffs,
            deg_x,
            deg_y,
        }
    }

    // adjust/decrease degree in case there are leading zeros in X or Y
    fn update_degree(&mut self) {
        // adjust deg_x
        while let Some(row) = self.coeffs.last() {
            if row.par_iter().all(|c| c.is_zero()) {
                self.coeffs.pop();
                self.deg_x -= 1;
            } else {
                break; // Stop when a non-zero row is encountered
            }
        }
        // adjust deg_y
        let mut cols = self.deg_y + 1;
        while cols > 0 {
            cols -= 1;
            // Check if the column is entirely zero
            if self.coeffs.par_iter().all(|row| row[cols].is_zero()) {
                // remove the trailing zero in all rows
                self.coeffs.par_iter_mut().for_each(|row| {
                    row.pop();
                });
                self.deg_y -= 1;
            } else {
                // Stop if we find a non-zero (trailing) column
                break;
            }
        }
    }
}

// === Basic Operations for Bivarate Polynomial ====
// =================================================
// TODO: impl Mul, Div
impl<F: Field> Add for DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<'a, 'b, F: Field> Add<&'a DensePolynomial<F>> for &'b DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    #[inline]
    fn add(self, rhs: &'a DensePolynomial<F>) -> Self::Output {
        if rhs.is_zero() {
            self.clone()
        } else if self.is_zero() {
            rhs.clone()
        } else if self.deg_y == rhs.deg_y && self.deg_x == rhs.deg_x {
            // slightly more memory efficient than the next general case
            let mut result = DensePolynomial {
                coeffs: self
                    .coeffs
                    .par_iter()
                    .zip(rhs.coeffs.par_iter())
                    .map(|(row_a, row_b)| {
                        row_a
                            .par_iter()
                            .zip(row_b.par_iter())
                            .map(|(a, b)| *a + *b)
                            .collect()
                    })
                    .collect(),
                deg_x: self.deg_x,
                deg_y: self.deg_y,
            };
            result.update_degree();
            result
        } else {
            let deg_x = self.deg_x.max(rhs.deg_x);
            let deg_y = self.deg_y.max(rhs.deg_y);

            let mut sum_coeffs = vec![vec![F::default(); deg_y + 1]; deg_x + 1];
            sum_coeffs
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_idx, row)| {
                    if row_idx <= self.deg_x {
                        row.par_iter_mut()
                            .zip(self.coeffs[row_idx].par_iter())
                            .for_each(|(a, b)| *a += b);
                    }
                    if row_idx <= rhs.deg_x {
                        row.par_iter_mut()
                            .zip(rhs.coeffs[row_idx].par_iter())
                            .for_each(|(a, b)| *a += b);
                    }
                });
            let mut result = DensePolynomial {
                coeffs: sum_coeffs,
                deg_x,
                deg_y,
            };
            result.update_degree();
            result
        }
    }
}

impl<'a, F: Field> AddAssign<&'a DensePolynomial<F>> for DensePolynomial<F> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a DensePolynomial<F>) {
        if self.is_zero() {
            *self = rhs.clone();
        } else if rhs.is_zero() {
        } else if self.deg_x == rhs.deg_x && self.deg_y == rhs.deg_y {
            self.coeffs
                .par_iter_mut()
                .zip(rhs.coeffs.par_iter())
                .for_each(|(row_a, row_b)| {
                    row_a
                        .par_iter_mut()
                        .zip(row_b.par_iter())
                        .for_each(|(a, b)| *a += b);
                });
            self.update_degree();
        } else {
            let deg_x = self.deg_x.max(rhs.deg_x);
            let deg_y = self.deg_y.max(rhs.deg_y);

            // first resize self.coeffs to max of both dimension, pad with zeros
            if self.deg_x < deg_x {
                self.coeffs
                    .resize(deg_x + 1, vec![F::default(); self.deg_y + 1]);
            }
            if self.deg_y < deg_y {
                self.coeffs
                    .par_iter_mut()
                    .for_each(|row| row.resize(deg_y + 1, F::default()));
            }
            // go through rhs's rows to add each items
            self.coeffs
                .par_iter_mut()
                .take(rhs.deg_x + 1)
                .enumerate()
                .for_each(|(row_idx, row)| {
                    row.par_iter_mut()
                        .zip(rhs.coeffs[row_idx].par_iter())
                        .for_each(|(a, b)| *a += b)
                });
            self.deg_x = deg_x;
            self.deg_y = deg_y;
            self.update_degree();
        }
    }
}

impl<'a, F: Field> AddAssign<(F, &'a DensePolynomial<F>)> for DensePolynomial<F> {
    #[inline]
    fn add_assign(&mut self, (f, rhs): (F, &'a DensePolynomial<F>)) {
        if rhs.is_zero() || f.is_zero() {
        } else if self.is_zero() {
            *self = rhs.clone();
            self.coeffs
                .par_iter_mut()
                .for_each(|row| row.par_iter_mut().for_each(|c| *c *= &f));
            self.update_degree();
        } else {
            let deg_x = self.deg_x.max(rhs.deg_x);
            let deg_y = self.deg_y.max(rhs.deg_y);
            // first resize self.coeffs to max of both dimension, pad with zeros
            if self.deg_x < deg_x {
                self.coeffs
                    .resize(deg_x + 1, vec![F::default(); self.deg_y + 1]);
            }
            if self.deg_y < deg_y {
                self.coeffs
                    .par_iter_mut()
                    .for_each(|row| row.resize(deg_y + 1, F::default()));
            }
            // go through rhs's rows to scale-and-add each items
            self.coeffs
                .par_iter_mut()
                .take(rhs.deg_x + 1)
                .enumerate()
                .for_each(|(row_idx, row)| {
                    row.par_iter_mut()
                        .zip(rhs.coeffs[row_idx].par_iter())
                        .for_each(|(a, b)| *a += *b * &f)
                });
            self.deg_x = deg_x;
            self.deg_y = deg_y;
            self.update_degree();
        }
    }
}

impl<F: Field> Neg for DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    #[inline]
    fn neg(mut self) -> Self::Output {
        self.coeffs.par_iter_mut().for_each(|row| {
            row.par_iter_mut().for_each(|c| *c = -*c);
        });
        self
    }
}

impl<'a, 'b, F: Field> Sub<&'a DensePolynomial<F>> for &'b DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    #[inline]
    fn sub(self, rhs: &'a DensePolynomial<F>) -> Self::Output {
        if rhs.is_zero() {
            self.clone()
        } else if self.is_zero() {
            -rhs.clone()
        } else if self.deg_y == rhs.deg_y && self.deg_x == rhs.deg_x {
            let mut result = self.clone();
            result
                .coeffs
                .par_iter_mut()
                .zip(rhs.coeffs.par_iter())
                .for_each(|(row_a, row_b)| {
                    row_a
                        .par_iter_mut()
                        .zip(row_b.par_iter())
                        .for_each(|(a, b)| *a -= b)
                });
            result.update_degree();
            result
        } else {
            let deg_x = self.deg_x.max(rhs.deg_x);
            let deg_y = self.deg_y.max(rhs.deg_y);
            let mut coeffs = vec![vec![F::default(); deg_y + 1]; deg_x + 1];
            // 0 + self - rhs (on row that exists)
            coeffs
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_idx, row)| {
                    if row_idx <= self.deg_x {
                        row.par_iter_mut()
                            .zip(self.coeffs[row_idx].par_iter())
                            .for_each(|(a, b)| *a += b);
                    }
                    if row_idx <= rhs.deg_x {
                        row.par_iter_mut()
                            .zip(rhs.coeffs[row_idx].par_iter())
                            .for_each(|(a, b)| *a -= b);
                    }
                });
            let mut result = DensePolynomial {
                coeffs,
                deg_x,
                deg_y,
            };
            result.update_degree();
            result
        }
    }
}

impl<'a, F: Field> SubAssign<&'a DensePolynomial<F>> for DensePolynomial<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a DensePolynomial<F>) {
        if rhs.is_zero() {
        } else if self.is_zero() {
            *self = -rhs.clone();
        } else if self.deg_y == rhs.deg_y && self.deg_x == rhs.deg_x {
            self.coeffs
                .par_iter_mut()
                .zip(rhs.coeffs.par_iter())
                .for_each(|(row_a, row_b)| {
                    row_a
                        .par_iter_mut()
                        .zip(row_b.par_iter())
                        .for_each(|(a, b)| *a -= b)
                });
            self.update_degree();
        } else {
            let deg_x = self.deg_x.max(rhs.deg_x);
            let deg_y = self.deg_y.max(rhs.deg_y);

            // first resize self.coeffs to max of both dimension, pad with zeros
            if self.deg_x < deg_x {
                self.coeffs
                    .resize(deg_x + 1, vec![F::default(); self.deg_y + 1]);
            }
            if self.deg_y < deg_y {
                self.coeffs
                    .par_iter_mut()
                    .for_each(|row| row.resize(deg_y + 1, F::default()));
            }

            // go through rhs's rows to subtract each items
            self.coeffs
                .par_iter_mut()
                .take(rhs.deg_x + 1)
                .enumerate()
                .for_each(|(row_idx, row)| {
                    row.par_iter_mut()
                        .zip(rhs.coeffs[row_idx].par_iter())
                        .for_each(|(a, b)| *a -= b);
                });
            self.deg_x = deg_x;
            self.deg_y = deg_y;
            self.update_degree();
        }
    }
}

impl<F: Field> Zero for DensePolynomial<F> {
    /// Returns the zero polynomial.
    fn zero() -> Self {
        Self {
            coeffs: vec![vec![F::ZERO]],
            deg_x: 0,
            deg_y: 0,
        }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().flatten().all(|coeff| coeff.is_zero())
    }
}

impl<F: Field> Default for DensePolynomial<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: Field> fmt::Debug for DensePolynomial<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first_monomial_written = false;
        for (row_idx, row) in self.coeffs.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                if !cell.is_zero() {
                    if first_monomial_written {
                        write!(f, " + {}*X^{}*Y^{}", cell, row_idx, col_idx)?;
                    } else {
                        write!(
                            f,
                            "f^[deg_x={}, deg_y={}](X,Y) = {}*X^{}*Y^{}",
                            self.deg_x, self.deg_y, cell, row_idx, col_idx
                        )?;
                        first_monomial_written = true;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_rng;
    use ark_bn254::Fr;
    use ark_std::UniformRand;

    #[test]
    fn add_polys() {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let dx1 = rng.gen_range(5..20) as usize;
            let dy1 = rng.gen_range(5..20) as usize;
            let dx2 = rng.gen_range(5..20) as usize;
            let dy2 = rng.gen_range(5..20) as usize;
            let p1 = DensePolynomial::<Fr>::rand(dx1, dy1, rng);
            let p2 = DensePolynomial::<Fr>::rand(dx2, dy2, rng);
            assert_eq!(&p1 + &p2, &p2 + &p1);
        }
    }

    #[test]
    fn add_assign_polys() {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let dx1 = rng.gen_range(5..20) as usize;
            let dy1 = rng.gen_range(5..20) as usize;
            let dx2 = rng.gen_range(5..20) as usize;
            let dy2 = rng.gen_range(5..20) as usize;

            let mut p1 = DensePolynomial::<Fr>::rand(dx1, dy1, rng);
            let p1_copy = p1.clone();
            let mut p2 = DensePolynomial::<Fr>::rand(dx2, dy2, rng);
            p1 += &p2;
            p2 += &p1_copy;
            assert_eq!(p1, p2);
        }
    }

    #[test]
    fn add_assign_scaled_polys() {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let dx1 = rng.gen_range(5..20) as usize;
            let dy1 = rng.gen_range(5..20) as usize;
            let dx2 = rng.gen_range(5..20) as usize;
            let dy2 = rng.gen_range(5..20) as usize;
            let s = rng.gen_range(0..10);

            let mut p1 = DensePolynomial::<Fr>::rand(dx1, dy1, rng);
            let mut p1_copy = p1.clone();
            let p2 = DensePolynomial::<Fr>::rand(dx2, dy2, rng);

            for _ in 0..s {
                p1_copy += &p2;
            }
            p1 += (Fr::from(s), &p2);

            assert_eq!(p1, p1_copy);
        }
    }

    #[test]
    fn sub_polys() {
        let rng = &mut test_rng();
        for _ in 0..50 {
            let dx1 = rng.gen_range(5..20) as usize;
            let dy1 = rng.gen_range(5..20) as usize;
            let dx2 = rng.gen_range(5..20) as usize;
            let dy2 = rng.gen_range(5..20) as usize;

            let p1 = DensePolynomial::<Fr>::rand(dx1, dy1, rng);
            let p2 = DensePolynomial::<Fr>::rand(dx2, dy2, rng);
            let p1_minus_p2 = &p1 - &p2;
            let p2_minus_p1 = &p2 - &p1;
            assert_eq!(p1_minus_p2, -p2_minus_p1.clone());
            assert_eq!(&p1_minus_p2 + &p2, p1);
            assert_eq!(&p2_minus_p1 + &p1, p2);
        }
    }

    #[test]
    fn test_additive_identity() {
        // Test adding polynomials with its negative equals 0
        let rng = &mut test_rng();
        for _ in 0..10 {
            let dx1 = rng.gen_range(5..20) as usize;
            let dy1 = rng.gen_range(5..20) as usize;

            let p1 = DensePolynomial::<Fr>::rand(dx1, dy1, rng);
            assert_eq!(p1, -(-p1.clone()));
        }
    }

    #[test]
    fn evaluate_polys() {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let dx = rng.gen_range(0..20) as usize;
            let dy = rng.gen_range(0..20) as usize;
            let point = [Fr::rand(rng), Fr::rand(rng)];

            let p1 = DensePolynomial::<Fr>::rand(dx, dy, rng);
            let mut expected = Fr::zero();
            for (row_idx, row) in p1.coeffs.iter().enumerate() {
                for (col_idx, coeff) in row.iter().enumerate() {
                    expected +=
                        *coeff * point[0].pow(&[row_idx as u64]) * point[1].pow(&[col_idx as u64]);
                }
            }
            assert_eq!(p1.evaluate(&point), expected);
        }
    }
}
