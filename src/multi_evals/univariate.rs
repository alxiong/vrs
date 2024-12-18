//! Multi-evaluations for Univariate Polynomials
//!
//! # References:
//! - Algorithm 2 in https://www.usenix.org/conference/usenixsecurity22/presentation/zhang-jiaheng
//! - https://eprint.iacr.org/2023/033
//! - https://eprint.iacr.org/2020/1516

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain};
use jf_pcs::prelude::{UnivariateKzgProof, UnivariateProverParam};
use p3_maybe_rayon::prelude::*;

/// Implement multi-evaluation for `jf_pcs::UnivariateKzgPCS`
///
/// Let poly degree = k-1 with coefficient c_0,...,c_k-1
/// Let C = (c_k-1, c_k-2, ... , c_0), T = ([\tau^0], [\tau^1], .., [\tau^{k-1}])
/// where [\tau^i] is a group element in SRS
///
/// Our multi-eval proof computation boils down to:
///     [\vec{H}] = IFFT(FFT(C) \odot FFT(T))
/// where \odot is element-wise/hadamard product, FFT(C) is over field elements,
/// FFT(T) is over group elements, final IFFT is also over group elements
pub fn multi_eval<E: Pairing>(
    pk: &UnivariateProverParam<E>,
    poly: &DensePolynomial<E::ScalarField>,
    domain: &Radix2EvaluationDomain<E::ScalarField>,
) -> Vec<UnivariateKzgProof<E>> {
    assert!(
        domain.size >= poly.degree() as u64 * 2,
        "domain_size < degree * 2, consider double the domain_size then truncate the result"
    );
    let mut c = poly.coeffs.clone().into_iter().rev().collect::<Vec<_>>();
    // NOTE: arkwork's EC-NNT only works over Projective
    let mut t = pk.powers_of_g[..=poly.degree()]
        .par_iter()
        .map(|affine| affine.into_group())
        .collect::<Vec<_>>();

    domain.fft_in_place(&mut c);
    let fft_c = c; // no memory copy/clone, just renaming (ownership transfer) to a clearer name
    domain.fft_in_place(&mut t);
    let mut fft_t = t;

    fft_t
        .par_iter_mut()
        .zip(fft_c.par_iter())
        .for_each(|(t, c)| *t *= c);
    let mut prod = fft_t; // no memcpy, only rename

    domain.ifft_in_place(&mut prod);
    let mut h = prod; // no memcpy, only rename
    h.truncate(poly.degree());
    h.reverse();

    domain.fft_in_place(&mut h);
    let proofs_proj = h; // no memcpy, only rename
    proofs_proj
        .into_par_iter()
        .map(|proof| UnivariateKzgProof {
            proof: proof.into_affine(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use ark_poly::{
        univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
    };
    use ark_std::rand::Rng;
    use jf_pcs::prelude::*;
    use p3_maybe_rayon::prelude::*;

    use crate::test_utils::test_rng;

    #[test]
    fn test_uv_multi_eval() {
        let rng = &mut test_rng();
        let max_degree = 32;
        let pp = UnivariateUniversalParams::<Bn254>::gen_srs_for_testing(rng, max_degree).unwrap();

        for _ in 0..10 {
            let degree = rng.gen_range(4..max_degree);

            let (pk, vk) = UnivariateUniversalParams::trim(&pp, degree).unwrap();
            let domain =
                Radix2EvaluationDomain::new(rng.gen_range(2 * degree..8 * degree)).unwrap();
            let poly = DensePolynomial::<Fr>::rand(degree, rng);
            let cm = UnivariateKzgPCS::commit(&pk, &poly).unwrap();

            let proofs = super::multi_eval(&pk, &poly, &domain);
            let elements = domain.elements().collect::<Vec<_>>();
            proofs
                .par_iter()
                .zip(elements.par_iter())
                .for_each(|(proof, point)| {
                    // test match the single-eval/open proof
                    let (expected_proof, eval) = UnivariateKzgPCS::open(&pk, &poly, point).unwrap();
                    assert_eq!(&expected_proof, proof);
                    // should pass verification
                    assert!(UnivariateKzgPCS::verify(&vk, &cm, point, &eval, proof).unwrap());
                });
        }
    }
}
