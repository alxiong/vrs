//! Implementations of fast multi-evaluations and multi-partial-evaluations proofs generations

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use p3_maybe_rayon::prelude::*;

pub mod bivariate;
pub mod univariate;

/// An O(n)-time algorithm to compute the FFT(v.reverse()) from results of FFT(v)
/// where n is the domain size (i.e. size of FFT(v) results).
/// This is faster than recomputing FFT(v.reverse()) from scratch which is O(n*logn)
///
/// - `num_coeffs`: length of `v` (number of coefficients of the polynomial under FFT)
pub fn fft_rev<F: FftField>(
    domain: &Radix2EvaluationDomain<F>,
    num_coeffs: usize,
    fft_results: &[F],
) -> Vec<F> {
    // omega^{-(k-1)}
    let phase_gen = domain.group_gen_inv.pow(&[num_coeffs as u64 - 1]);
    // phase factors: (phase_gen^0, phase_gen^1, ..., phase_gen^{n-1})
    let mut phase_factors = vec![F::ONE; domain.size as usize];
    Radix2EvaluationDomain::distribute_powers(&mut phase_factors, phase_gen);

    let mut fft_reverse: Vec<F> = fft_results.to_owned();
    // FFT(v.rev())[j] = FFT(v)[n-j] * phase_factors[j] for j=1..n-1
    fft_reverse
        .par_iter_mut()
        .zip(phase_factors.par_iter())
        .for_each(|(v, phase_factor)| *v *= phase_factor);
    // Reverse elements except the first
    fft_reverse[1..].reverse();
    fft_reverse
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{conv, test_rng};
    use ark_bn254::Fr;
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use ark_std::UniformRand;

    // This demonstrates that a * b = IFFT(FFT(a) \odot FFT(b))
    // where * is convolution, \odot is hadamard product
    #[test]
    fn convolution_via_fft() {
        let rng = &mut test_rng();
        let k = 8; // vector size
        let n = 16; // eval domain size

        let a: Vec<Fr> = (0..k).map(|_| Fr::rand(rng)).collect();
        let b: Vec<Fr> = (0..k).map(|_| Fr::rand(rng)).collect();
        let c = conv(&a, &b);

        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let fft_a = domain.fft(&a);
        let fft_b = domain.fft(&b);
        let res = domain.ifft(
            &fft_a
                .iter()
                .zip(fft_b.iter())
                .map(|(a, b)| *a * b)
                .collect::<Vec<_>>(),
        );

        // conv returns 2n-1 value, the
        assert_eq!(res.len(), c.len() + 1);
        assert_eq!(res[..c.len()], c);
    }

    #[test]
    // test derivation of FFT(v.reverse()) from FFT(v)
    fn test_fft_rev() {
        let num_coeffs = 8;
        let rng = &mut test_rng();

        for domain_size in [8, 16, 32] {
            let domain = Radix2EvaluationDomain::<Fr>::new(domain_size).unwrap();
            let mut v = vec![];
            for _ in 0..num_coeffs {
                v.push(Fr::rand(rng));
            }

            let fft_result = domain.fft(&v);
            v.reverse();
            let expected_fft_reverse = domain.fft(&v);

            assert_eq!(
                super::fft_rev(&domain, num_coeffs, &fft_result),
                expected_fft_reverse
            );
        }
    }
}
