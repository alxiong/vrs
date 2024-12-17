//! testing utilities

use ark_ff::Field;
use ark_std::rand::{rngs::StdRng, SeedableRng};

/// a CryptoRng
pub fn test_rng() -> StdRng {
    // arbitrary seed
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

/// A simple convolution between two vectors of field elements
pub fn conv<F: Field>(signal: &[F], kernel: &[F]) -> Vec<F> {
    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let output_len = signal_len + kernel_len - 1;

    let mut output = vec![F::default(); output_len];

    for i in 0..output_len {
        for j in 0..kernel_len {
            if i >= j && i - j < signal_len {
                output[i] = output[i] + signal[i - j] * kernel[j];
            }
        }
    }

    output
}
