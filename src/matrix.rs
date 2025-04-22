//! Matrix operations using vector-storage

use anyhow::{anyhow, ensure, Result};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
use ark_serialize::CanonicalSerialize;
use ark_std::collections::BTreeSet;
use ark_std::rand::{CryptoRng, RngCore};
use ark_std::slice::Chunks;
use ark_std::sync::Arc;
use p3_maybe_rayon::prelude::*;

/// Row-based Matrix of data type `T`, abstracted `[[T; width]; height]`
///
/// # struct contracts
/// - `data.len() = width * height` at all time
#[derive(Clone, Debug, PartialEq, CanonicalSerialize)]
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

    pub fn rand<R: RngCore + CryptoRng>(rng: &mut R, width: usize, height: usize) -> Self {
        let data = (0..width * height).map(|_| F::rand(rng)).collect();
        Matrix::new(data, width, height).unwrap()
    }

    /// Reshape the matrix, since we are vec-storage, no memory copy
    pub fn reshape(&mut self, width: usize, height: usize) -> Result<()> {
        ensure!(
            width * height != self.width * self.height,
            "Matrix cannot reshape to diff size"
        );
        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Returns the width
    pub const fn width(&self) -> usize {
        self.width
    }
    /// Returns the height
    pub const fn height(&self) -> usize {
        self.height
    }
    /// Returns the (row, col) cell value reference
    #[inline]
    pub fn cell(&self, row: usize, col: usize) -> &F {
        &self.data[row * self.width + col]
    }

    /// Convert to `height` number of univariate polynomials of degree `width-1`
    pub fn to_row_uv_polys(&self) -> Vec<DensePolynomial<F>> {
        self.data
            .par_chunks(self.width)
            .map(|row| DensePolynomial::from_coefficients_slice(row))
            .collect()
    }

    /// Convert to `width` number of univariate polynomials of degree `height-1`
    pub fn to_col_uv_polys(&self) -> Vec<DensePolynomial<F>> {
        self.par_col()
            .map(|col| DensePolynomial::from_coefficients_slice(&col))
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

    /// Returns a sequential column iterator
    pub fn col_iter(&self) -> impl Iterator<Item = Vec<F>> + '_ {
        let width = self.width;
        let height = self.height;

        (0..width).into_iter().map(move |col| {
            let mut column = Vec::with_capacity(height);
            for row in 0..height {
                column.push(self.data[row * width + col].clone());
            }
            column
        })
    }

    /// Returns a sequential row iterator
    pub fn row_iter(&self) -> Chunks<'_, F> {
        self.data.chunks(self.width)
    }

    pub fn cell_iter(&self) -> impl Iterator<Item = &F> + Clone {
        self.data.iter()
    }

    /// Owned copy of all cells
    pub fn all_cells(&self) -> Vec<F> {
        (*self.data).clone()
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

    /// Returns a parallel row iterator
    pub fn par_row(&self) -> impl ParallelIterator<Item = &[F]> {
        self.data.par_chunks(self.width)
    }

    /// Returns an owned parallel row iterator
    pub fn into_par_row(&self) -> impl ParallelIterator<Item = Vec<F>> {
        let data = self.data.clone();
        let width = self.width;

        (0..self.height).into_par_iter().map(move |row_idx| {
            let start = row_idx * width;
            let end = start + width;
            data[start..end].to_vec()
        })
    }

    /// Returns a parallel row enumerator
    pub fn par_row_enumerate(&self) -> impl IndexedParallelIterator<Item = (usize, &[F])> + '_ {
        self.data.par_chunks(self.width).enumerate()
    }

    /// Scale each row
    pub fn scale_rows(&mut self, scaling_factors: &[F]) -> Result<()> {
        ensure!(
            scaling_factors.len() == self.height,
            "scaling factors len != height"
        );
        let data = Arc::make_mut(&mut self.data);

        for row in 0..self.height {
            let scaling_factor = &scaling_factors[row];
            for col in 0..self.width {
                let index = row * self.width + col;
                data[index] *= *scaling_factor;
            }
        }
        Ok(())
    }

    /// Scale each col
    pub fn scale_cols(&mut self, scaling_factors: &[F]) -> Result<()> {
        ensure!(
            scaling_factors.len() == self.width,
            "scaling factors len != width"
        );

        let data = Arc::make_mut(&mut self.data);
        data.par_chunks_mut(self.width).for_each(|row| {
            row.iter_mut()
                .zip(scaling_factors.iter())
                .for_each(|(v, s)| *v = *v * s);
        });
        Ok(())
    }

    /// Performs an in-place transpose of the matrix
    pub fn transpose_in_place(&mut self) {
        // For non-square matrices, we need to create a new storage
        if self.width != self.height {
            let mut new_data = Vec::with_capacity(self.width * self.height);

            for col in 0..self.width {
                for row in 0..self.height {
                    let idx = row * self.width + col;
                    new_data.push(self.data[idx].clone());
                }
            }

            self.data = Arc::new(new_data);
            std::mem::swap(&mut self.width, &mut self.height);
        } else {
            // For square matrices, we can do a true in-place transpose
            let data = Arc::make_mut(&mut self.data);
            let n = self.width; // same as height for square matrices

            for i in 0..n {
                for j in i + 1..n {
                    let idx1 = i * n + j;
                    let idx2 = j * n + i;
                    data.swap(idx1, idx2);
                }
            }
            // dimension stays the same
        }
    }

    /// Get a certain column
    pub fn col(&self, col_idx: usize) -> Result<Vec<F>> {
        if col_idx >= self.width {
            return Err(anyhow!(
                "width: {}, but requested col: {}",
                self.width,
                col_idx
            ));
        }
        let mut col = vec![];
        for row in 0..self.height {
            col.push(*self.cell(row, col_idx));
        }
        Ok(col)
    }

    /// Get a certain row
    pub fn row(&self, row_idx: usize) -> Result<Vec<F>> {
        if row_idx >= self.height {
            return Err(anyhow!(
                "height: {}, but requested row: {}",
                self.height,
                row_idx
            ));
        }
        let start_idx = row_idx * self.width;
        Ok(self.data[start_idx..start_idx + self.width].to_vec())
    }

    /// consume self, only col_idx in `col_indices`, return a new, trimmed matrix
    pub fn retain_cols(self, col_indices: &BTreeSet<usize>) -> Result<Self> {
        if col_indices.iter().any(|idx| *idx >= self.width) {
            return Err(anyhow!("try to retain col_idx out-of-bound"));
        }
        let trimmed = self
            .par_row()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(col_idx, _)| col_indices.contains(col_idx))
                    .map(|(_, f)| *f)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();
        Ok(Self::new(trimmed, col_indices.len(), self.height)?)
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
                    assert_eq!(&col[row_idx], matrix.cell(row_idx, col_idx));
                }
            });
            matrix.par_row_enumerate().for_each(|(row_idx, row)| {
                for col_idx in 0..width {
                    assert_eq!(&row[col_idx], matrix.cell(row_idx, col_idx));
                }
            })
        }
    }

    #[test]
    fn test_scaling() {
        let rng = &mut ark_std::test_rng();
        let width = 5;
        let height = 5;
        let data = (0..width * height).map(|_| Fr::rand(rng)).collect();

        let data = Matrix::new(data, width, height).unwrap();
        let mut scaled = data.clone();
        let mut col_scaled = data.clone();
        let scaling_factors = [
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
        ];

        scaled.scale_rows(&scaling_factors).unwrap();
        col_scaled.scale_cols(&scaling_factors).unwrap();

        // Verify each row is scaled correctly
        for row in 0..height {
            for col in 0..width {
                assert_eq!(
                    *scaled.cell(row, col),
                    *data.cell(row, col) * scaling_factors[row],
                );
                assert_eq!(
                    *col_scaled.cell(row, col),
                    *data.cell(row, col) * scaling_factors[col],
                );
            }
        }
    }

    #[test]
    fn test_transpose() {
        let rng = &mut ark_std::test_rng();
        for _ in 0..10 {
            let width = rng.gen_range(5..10);
            let height = rng.gen_range(5..10);
            let mut data = Vec::new();
            for _ in 0..width * height {
                data.push(Fr::rand(rng));
            }

            let matrix = Matrix::new(data, width, height).unwrap();
            let mut transposed = matrix.clone();
            transposed.transpose_in_place();

            assert_eq!(matrix.width, transposed.height);
            assert_eq!(matrix.height, transposed.width);
            for i in 0..height {
                for j in 0..width {
                    assert_eq!(matrix.cell(i, j), transposed.cell(j, i));
                }
            }
        }
    }

    #[test]
    fn test_retain_cols() {
        let data = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
        ];
        let mut retain_cols = BTreeSet::new();
        retain_cols.insert(1);
        assert_eq!(
            Matrix::new(vec![Fr::from(2), Fr::from(4), Fr::from(6)], 1, 3).unwrap(),
            Matrix::new(data.clone(), 2, 3)
                .unwrap()
                .retain_cols(&retain_cols)
                .unwrap()
        );

        assert_eq!(
            Matrix::new(vec![Fr::from(2), Fr::from(5)], 1, 2).unwrap(),
            Matrix::new(data.clone(), 3, 2)
                .unwrap()
                .retain_cols(&retain_cols)
                .unwrap()
        );
    }
}
