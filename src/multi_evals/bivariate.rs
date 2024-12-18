//! Multi-partial-evaluations for Bivariate Polynomials

use ark_ec::pairing::Pairing;
use ark_poly::Radix2EvaluationDomain;

use crate::{bivariate::DensePolynomial, bkzg::BivariateProverParam};

/// Implement multi-partial-evaluation, partially opened at x.
/// g(Y) = f(x, Y) for Y \in {\omega, \omega^2, ..., \omega^n} where n is domain size
/// Outputs all proofs for all correct g(Y)
pub fn multi_partial_eval<E: Pairing>(
    pk: &BivariateProverParam<E>,
    poly: &DensePolynomial<E::ScalarField>,
    x: &E::ScalarField,
    domain: &Radix2EvaluationDomain<E::ScalarField>,
) -> Vec<E::G1Affine> {
    assert!(
        domain.size >= poly.deg_y as u64 * 2,
        "domain_size < degree * 2, consider double the domain_size then truncate the result"
    );

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bkzg::*, test_utils::test_rng};
    use ark_bn254::{Bn254, Fr};
    use ark_poly::EvaluationDomain;
    use ark_std::{rand::Rng, UniformRand};
    use jf_pcs::prelude::*;
    use p3_maybe_rayon::prelude::*;

    #[ignore]
    #[test]
    fn test_bv_multi_partial_eval() {
        let rng = &mut test_rng();
        let max_deg_x = 128;
        let max_deg_y = 16;
        let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
        let pp =
            BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
                .unwrap();
        for _ in 0..5 {
            let deg_x = rng.gen_range(8..max_deg_x);
            let deg_y = rng.gen_range(1..max_deg_y);
            let supported_degree = combine_u32(deg_x, deg_y);
            let (pk, vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
            let domain =
                Radix2EvaluationDomain::<Fr>::new(rng.gen_range(2 * deg_y..8 * deg_y) as usize)
                    .unwrap();

            let poly = DensePolynomial::rand(deg_x as usize, deg_y as usize, rng);
            let cm = BivariateKzgPCS::commit(&pk, &poly).unwrap();
            let x = Fr::rand(rng);

            let proofs = super::multi_partial_eval(&pk, &poly, &x, &domain);
            let elements = domain.elements().collect::<Vec<_>>();
            proofs
                .par_iter()
                .zip(elements.par_iter())
                .for_each(|(proof, &y)| {
                    // test match the single-eval/open proof
                    let (expected_proof, _eval) =
                        BivariateKzgPCS::open(&pk, &poly, &(x, y)).unwrap();
                    assert_eq!(&expected_proof.partial_eval_x_proof, proof);
                });
        }
    }
}
