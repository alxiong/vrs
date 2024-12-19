//! Multi-partial-evaluations for Bivariate Polynomials

use ark_ec::pairing::Pairing;
use ark_poly::Radix2EvaluationDomain;

use crate::{bivariate::DensePolynomial, bkzg::BivariateProverParam};

/// Implement multi-partial-evaluation, partially opened at y.
/// g(X) = f(X, y) for y \in `domain`
/// Outputs all proofs for all correct g(X)
pub fn multi_partial_eval<E: Pairing>(
    pk: &BivariateProverParam<E>,
    poly: &DensePolynomial<E::ScalarField>,
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
    use ark_std::rand::Rng;
    use jf_pcs::prelude::*;
    use p3_maybe_rayon::prelude::*;

    #[test]
    fn test_bv_multi_partial_eval() {
        let rng = &mut test_rng();
        let max_deg_x = 16;
        let max_deg_y = 2u32.pow(12);
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

            let proofs = super::multi_partial_eval(&pk, &poly, &domain);
            let elements = domain.elements().collect::<Vec<_>>();
            let at_x = false;
            proofs
                .par_iter()
                .zip(elements.par_iter())
                .for_each(|(proof, y)| {
                    let (expected_proof, partial_eval) =
                        BivariateKzgPCS::partial_eval(&pk, &poly, y, at_x).unwrap();
                    assert_eq!(proof, &expected_proof);
                    assert!(BivariateKzgPCS::verify_partial(
                        &vk,
                        &cm,
                        y,
                        at_x,
                        &partial_eval,
                        proof
                    )
                    .unwrap());
                });
        }
    }
}
