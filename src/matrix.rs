//! Matrix operations using vector-storage

use anyhow::{ensure, Result};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
use ark_std::sync::Arc;
use p3_maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSlice,
};

/// Row-based Matrix of data type `T`, abstracted `[[T; width]; height]`
///
/// # struct contracts
/// - `data.len() = width * height` at all time
pub struct Matrix<F: Field> {
    data: Arc<Vec<F>>,
    width: usize,
    height: usize,
}

impl<F: Field> Matrix<F> {
    /// Construct a row-based matrix, padded to full `width * height` matrix.
    pub fn new(mut data: Vec<F>, width: usize, height: usize) -> Result<Self> {
        ensure!(data.len() <= width * height, "Matrix dim too small");
        ensure!(width > 0 && height > 0, "Matrix dim should be positive");
        data.resize(width * height, F::default());

        Ok(Self {
            data: Arc::new(data),
            width,
            height,
        })
    }

    /// Returns the width
    pub fn width(&self) -> usize {
        self.width
    }
    /// Returns the height
    pub fn height(&self) -> usize {
        self.height
    }
    /// Returns the (row, col) cell value reference
    pub fn get_cell(&self, row: usize, col: usize) -> &F {
        &self.data[row * self.width + col]
    }

    /// Convert to `height` number of univariate polynomials of degree `width-1`
    pub fn to_row_uv_polys(&self) -> Vec<DensePolynomial<F>> {
        self.data
            .par_chunks(self.width)
            .map(|row| DensePolynomial::from_coefficients_slice(row))
            .collect()
    }

    /// Returns a parallel column enumerator
    pub fn par_col_enumerate(&self) -> impl IndexedParallelIterator<Item = (usize, Vec<F>)> + '_ {
        let width = self.width;
        let height = self.height;

        (0..width).into_par_iter().map(move |col| {
            let mut column = Vec::with_capacity(height);
            for row in 0..height {
                column.push(self.data.clone()[row * width + col].clone());
            }
            (col, column)
        })
    }
    /// Returns a parallel column iterator
    pub fn par_col(&self) -> impl ParallelIterator<Item = Vec<F>> + '_ {
        let width = self.width;
        let height = self.height;

        (0..width).into_par_iter().map(move |col| {
            let mut column = Vec::with_capacity(height);
            for row in 0..height {
                column.push(self.data.clone()[row * width + col].clone());
            }
            column
        })
    }

    /// Returns a parallel row enumerator
    pub fn par_row_enumerate(&self) -> impl IndexedParallelIterator<Item = (usize, &[F])> + '_ {
        self.data.par_chunks(self.width).enumerate()
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, UniformRand};
    use p3_maybe_rayon::prelude::*;

    use super::*;

    #[test]
    fn test_par_iter() {
        let rng = &mut ark_std::test_rng();

        for _ in 0..10 {
            let width = rng.gen_range(5..10);
            let height = rng.gen_range(5..10);

            let mut data = Vec::new();
            for _ in 0..width * height {
                data.push(Fr::rand(rng));
            }
            assert_eq!(data.len(), width * height);

            let matrix = Matrix::new(data, width, height).unwrap();
            matrix.par_col_enumerate().for_each(|(col_idx, col)| {
                for row_idx in 0..height {
                    assert_eq!(&col[row_idx], matrix.get_cell(row_idx, col_idx));
                }
            });
            matrix.par_row_enumerate().for_each(|(row_idx, row)| {
                for col_idx in 0..width {
                    assert_eq!(&row[col_idx], matrix.get_cell(row_idx, col_idx));
                }
            })
        }
    }
}
