//! Multi-evaluations for Univariate Polynomials
//!
//! # References:
//! - Algorithm 2 in https://www.usenix.org/conference/usenixsecurity22/presentation/zhang-jiaheng
//! - https://eprint.iacr.org/2023/033
//! - https://eprint.iacr.org/2020/1516

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain};
use jf_pcs::prelude::{UnivariateKzgProof, UnivariateProverParam};
use p3_maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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
    let c = poly.coeffs.clone().into_iter().rev().collect::<Vec<_>>();
    // NOTE: arkwork's EC-NNT only works over Projective
    let t = pk.powers_of_g[..=poly.degree()]
        .par_iter()
        .map(|affine| affine.into_group())
        .collect::<Vec<_>>();

    let fft_c: Vec<E::ScalarField> = domain.fft(&c);
    let fft_t: Vec<E::G1> = domain.fft(&t);

    let prod: Vec<E::G1> = fft_c
        .par_iter()
        .zip(fft_t.par_iter())
        .map(|(c, t)| *t * c)
        .collect();

    let mut h = domain.ifft(&prod);
    h.truncate(poly.degree());
    h.reverse();

    let proofs = domain
        .fft(&h)
        .par_iter()
        .map(|proof| UnivariateKzgProof {
            proof: proof.into_affine(),
        })
        .collect();
    proofs
}
