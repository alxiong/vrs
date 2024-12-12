//! Matrix operations using vector-storage

use anyhow::{ensure, Result};
use ark_ff::Field;

/// Row-based Matrix of data type `T`, abstracted `[[T; width]; height]`
pub struct Matrix<F: Field> {
    data: Vec<F>,
    width: usize,
    height: usize,
}

impl<F: Field> Matrix<F> {
    /// Construct a row-based matrix, padded to full `width * height` matrix.
    pub fn new(data: Vec<F>, width: usize, height: usize) -> Result<Self> {
        ensure!(data.len() < width * height, "Matrix dim too small");
        ensure!(width > 0 && height > 0, "Matrix dim should be positive");

        let mut mat = Self {
            data,
            width,
            height,
        };
        mat.data.resize(width * height, F::default());
        Ok(mat)
    }
}
