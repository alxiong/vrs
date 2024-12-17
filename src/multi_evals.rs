//! Implementations of fast multi-evaluations and multi-partial-evaluations proofs generations

pub mod bivariate;
pub mod univariate;

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
}
