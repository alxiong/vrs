//! Multi-partial-evaluations for Bivariate Polynomials

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_poly::{univariate, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain};
use p3_maybe_rayon::prelude::*;

use crate::{
    bivariate::DensePolynomial,
    bkzg::{BivariateProverParam, PartialEvalProof},
};

/// Implement multi-partial-evaluation, partially opened at y.
/// g(X) = f(X, y) for y \in `domain`
/// Outputs all (partial-eval proofs, partial-eval g(X))
pub fn multi_partial_eval<E: Pairing>(
    pk: &BivariateProverParam<E>,
    poly: &DensePolynomial<E::ScalarField>,
    domain: &Radix2EvaluationDomain<E::ScalarField>,
) -> (
    Vec<PartialEvalProof<E>>,
    Vec<univariate::DensePolynomial<E::ScalarField>>,
) {
    assert!(
        domain.size >= poly.deg_y as u64 * 2,
        "domain_size < degree * 2, consider double the domain_size then truncate the result"
    );

    let domain_size = domain.size as usize;
    let table = multi_partial_eval_precompute(pk, domain);

    // Interleaved RS encode and prepare all the partial_evals in the correct type
    let encoded = poly
        .coeffs
        .par_iter()
        .map(|row| domain.fft(row))
        .collect::<Vec<Vec<E::ScalarField>>>();
    let partial_evals = (0..domain_size)
        .into_par_iter()
        .map(|col_idx| {
            univariate::DensePolynomial::from_coefficients_vec(
                encoded
                    .par_iter()
                    .map(|row| row[col_idx])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    // get the reversed coefficients from encoded data in O(L*n) time
    let reversed_encoded = encoded
        .par_iter()
        .map(|fft_result| super::fft_rev(domain, poly.deg_y + 1, fft_result))
        .collect::<Vec<Vec<_>>>();

    // prod = FFT(v*F) \odot FFT([T])
    // but computed by rearranging the terms described in the paper, with the precomputed table
    let mut prod = (0..domain_size)
        .into_par_iter()
        .map(|col_idx| {
            let bases = table
                .par_iter()
                .map(|row| row[col_idx].clone())
                .collect::<Vec<_>>();
            let scalars = reversed_encoded
                .par_iter()
                .map(|row| row[col_idx])
                .collect::<Vec<_>>();
            <E::G1 as VariableBaseMSM>::msm(&bases, &scalars).expect("msm failed")
        })
        .collect::<Vec<E::G1>>();
    drop(reversed_encoded);
    domain.ifft_in_place(&mut prod);
    let mut h = prod; // no memcpy, only rename
    h.truncate(poly.deg_y);
    h.reverse();

    // finally compute fft(H) to get all the proofs
    domain.fft_in_place(&mut h);
    let proofs_proj = h; // no memcpy, only rename
    let proofs = proofs_proj
        .into_par_iter()
        .map(|proof| proof.into_affine())
        .collect::<Vec<_>>();

    (proofs, partial_evals)
}

/// Precompute FFT([v[i] * T]_1)
/// where v = (tau_x^0, tau_x^1, ..., tau_x^{L-1}), T = (tau_y^0, ..., tau_y^{k-1})
/// [*]_1 means * computed in the exponent in G1 with trapdoors
///
/// Visually, think of bKZG's SRS as Lxk matrix of [tau_x^i * tau_y^j]_1, i\in[L], j\in[k]
/// This precomputation is doing a FFT over group elements for every row, into a Lxn matrix
pub fn multi_partial_eval_precompute<E: Pairing>(
    pk: &BivariateProverParam<E>,
    domain: &Radix2EvaluationDomain<E::ScalarField>,
) -> Vec<Vec<E::G1Affine>> {
    pk.powers_of_g
        .par_iter()
        .map(|row| {
            let row_proj: Vec<E::G1> = row.par_iter().map(|c| c.into_group()).collect();
            E::G1::normalize_batch(&domain.fft(&row_proj))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bkzg::*, test_utils::test_rng};
    use ark_bn254::{Bn254, Fr};
    use ark_poly::EvaluationDomain;
    use ark_std::rand::Rng;
    use jf_pcs::prelude::*;

    #[test]
    fn test_bv_multi_partial_eval() {
        let rng = &mut test_rng();
        let max_deg_x = 4;
        let max_deg_y = 2u32.pow(5);
        let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
        let pp =
            BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
                .unwrap();

        let deg_x = max_deg_x;
        let deg_y = max_deg_y;
        let supported_degree = combine_u32(deg_x, deg_y);
        let (pk, vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
        let domain =
            Radix2EvaluationDomain::<Fr>::new(rng.gen_range(2 * deg_y + 1..3 * deg_y) as usize)
                .unwrap();

        let poly = DensePolynomial::rand(deg_x as usize, deg_y as usize, rng);
        let cm = BivariateKzgPCS::commit(&pk, &poly).unwrap();

        let (proofs, partial_evals) = super::multi_partial_eval(&pk, &poly, &domain);
        let elements = domain.elements().collect::<Vec<_>>();
        let at_x = false;
        proofs
            .par_iter()
            .zip(partial_evals.par_iter())
            .zip(elements.par_iter())
            .for_each(|((proof, partial_eval), y)| {
                let (expected_proof, expected_partial_eval) =
                    BivariateKzgPCS::partial_eval(&pk, &poly, y, at_x).unwrap();
                assert_eq!(proof, &expected_proof);
                assert_eq!(partial_eval, &expected_partial_eval);
                assert!(
                    BivariateKzgPCS::verify_partial(&vk, &cm, y, at_x, &partial_eval, proof)
                        .unwrap()
                );
            });
    }
}
