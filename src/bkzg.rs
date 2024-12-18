//! Bivariate KZG, inspired from [PST13][https://eprint.iacr.org/2011/587]

use crate::bivariate::DensePolynomial;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, Group, VariableBaseMSM};
use ark_ff::{Field, One, Zero};
use ark_poly::{univariate, DenseUVPolynomial, Polynomial};
use ark_serialize::*;
use ark_std::{
    borrow::Borrow,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    UniformRand,
};
use derivative::Derivative;
use jf_pcs::{PCSError, PolynomialCommitmentScheme, StructuredReferenceString};
use p3_maybe_rayon::prelude::*;

/// KZG on bivariate polynomial
pub struct BivariateKzgPCS<E>(PhantomData<E>);

/// SRS for bKZG.
#[derive(Debug, Clone, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize, Default)]
pub struct BivariateKzgSRS<E: Pairing> {
    /// row-wise matrix for exponent values of (\tau_x^i * \tau_y^j)
    /// where i \in [0, d_x], j \in [0, d_y], and \tau_x, \tau_y are two trapdoors
    /// for the two variables X and Y.
    ///
    /// [  X^0 * Y^0   X^0 * Y^1   X^0 * Y^2   ...   X^0 * Y^d_y  ]
    /// [  X^1 * Y^0   X^1 * Y^1   X^1 * Y^2   ...   X^1 * Y^d_y  ]
    /// [  X^2 * Y^0   X^2 * Y^1   X^2 * Y^2   ...   X^2 * Y^d_y  ]
    /// [     ...          ...          ...    ...       ...      ]
    /// [ X^d_x * Y^0  X^d_x * Y^1  X^d_x * Y^2  ...  X^d_x * Y^d_y ]
    /// Replace X with \tau_x, Y with \tau_y
    pub powers_of_g: Vec<Vec<E::G1Affine>>,
    /// G2 generator
    pub h: E::G2Affine,
    /// tau_x * h
    pub tau_x_h: E::G2Affine,
    /// tau_y * h
    pub tau_y_h: E::G2Affine,
}

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
#[derivative(Hash)]
/// Evaluation proof
pub struct BivariateKzgProof<E: Pairing> {
    /// proof of correct g(Y) = f(x, Y) partial evaluated at x
    pub partial_eval_x_proof: E::G1Affine,
    /// proof of correct g(y) = f(x, y) at y
    pub eval_y_proof: E::G1Affine,
}

impl<E: Pairing> PolynomialCommitmentScheme for BivariateKzgPCS<E> {
    type SRS = BivariateKzgSRS<E>;
    type Polynomial = DensePolynomial<E::ScalarField>;
    type Point = (E::ScalarField, E::ScalarField);
    type Evaluation = E::ScalarField;
    type Commitment = E::G1Affine;
    type BatchCommitment = Vec<Self::Commitment>;
    type Proof = BivariateKzgProof<E>;
    type BatchProof = Vec<Self::Proof>;

    fn trim(
        srs: impl Borrow<BivariateKzgSRS<E>>,
        supported_degree: usize,
        _supported_num_vars: Option<usize>,
    ) -> Result<(BivariateProverParam<E>, BivariateVerifierParam<E>), PCSError> {
        BivariateKzgSRS::trim(srs.borrow(), supported_degree)
    }

    // "X^i * Y^j in the exponent" value evaluated at (\tau_x, \tau_y) are in SRS;
    // to compute f(\tau_x, \tau_y) in the exponent as the commitment, we walk through
    // SRS and coefficients and compute an MSM over them.
    fn commit(
        pk: impl Borrow<BivariateProverParam<E>>,
        poly: &DensePolynomial<E::ScalarField>,
    ) -> Result<Self::Commitment, PCSError> {
        let pk = pk.borrow();
        let cm = pk
            .powers_of_g
            .par_iter()
            .zip(poly.coeffs.par_iter())
            .map(|(bases, coeffs)| {
                <E::G1 as VariableBaseMSM>::msm(bases, coeffs).expect("msm during commit fail")
            })
            .reduce(
                || E::G1Affine::zero().into_group(),
                |acc, row_res| acc + row_res,
            );
        Ok(cm.into_affine())
    }

    fn open(
        pk: impl Borrow<BivariateProverParam<E>>,
        poly: &DensePolynomial<E::ScalarField>,
        (x, y): &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pk = pk.borrow();
        // witness poly for partial eval at x:
        //   w1(X, Y) = (f(X, Y) - f(x, Y)) / (X - x)
        // pi_1 = g^w1(\tau_x, \tau_y)
        let partial_eval = poly.partial_evaluate_x(&x);
        let div_poly =
            univariate::DensePolynomial::from_coefficients_slice(&[-*x, E::ScalarField::ONE]);
        let mut witness_poly = poly.clone();
        witness_poly.sub_assign_uv_poly(&partial_eval, false);
        let (witness_poly, remainder) = witness_poly.divide_with_q_and_r(&div_poly, true).unwrap();
        assert!(remainder.is_zero());
        let pi_1 = Self::commit(pk, &witness_poly)?;

        // second witness poly for full eval on the partial evaluated univariate poly
        //   w2(Y) = (f(x, Y) - f(x, y)) / (Y - y) = (g(Y) - g(y)) / (Y-y)
        // pi_2 = g^w2(\tau_y)
        let eval = partial_eval.evaluate(&y);
        let div_poly =
            univariate::DensePolynomial::from_coefficients_slice(&[-*y, E::ScalarField::ONE]);
        // Observe that this quotient does not change with $g(y)$ because $g(y)$ is the remainder term.
        // We can therefore omit $g(y)$ when computing the quotient.
        let witness_poly = &partial_eval / &div_poly;
        let pi_2 = <E::G1 as VariableBaseMSM>::msm(
            &pk.powers_of_g[0][..=witness_poly.degree()],
            &witness_poly.coeffs,
        )
        .expect("msm during commit fail")
        .into_affine();

        let proof = BivariateKzgProof {
            partial_eval_x_proof: pi_1,
            eval_y_proof: pi_2,
        };
        Ok((proof, eval))
    }

    fn verify(
        vk: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        commitment: &Self::Commitment,
        (x, y): &Self::Point,
        value: &Self::Evaluation,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        // for f(x, y) = z, with commitment C, proof (pi_1, pi_2), G1 generator G, G2 generator H
        // checking e(C-zG, H) =?= e(pi_1, (\tau_x-x)H) * e(pi_2, (\tau_y-y)H)
        // we use multi-pairing, thus moving the LHS to the right (and negate the term "C-zG")
        // so that the product of 3 pairings = 1
        let a = [
            proof.partial_eval_x_proof.into_group(),
            proof.eval_y_proof.into_group(),
            vk.g * value - commitment,
        ];
        let b = [
            vk.tau_x_h.into_group() - vk.h * x,
            vk.tau_y_h.into_group() - vk.h * y,
            vk.h.into_group(),
        ];
        let verified = E::multi_pairing(&a, &b).0.is_one();
        Ok(verified)
    }

    // TODO: implement following later
    fn batch_commit(
        _prover_param: impl Borrow<<Self::SRS as StructuredReferenceString>::ProverParam>,
        _polys: &[Self::Polynomial],
    ) -> Result<Self::BatchCommitment, PCSError> {
        todo!();
    }

    fn batch_open(
        _prover_param: impl Borrow<<Self::SRS as StructuredReferenceString>::ProverParam>,
        _batch_commitment: &Self::BatchCommitment,
        _polynomials: &[Self::Polynomial],
        _points: &[Self::Point],
    ) -> Result<(Self::BatchProof, Vec<Self::Evaluation>), PCSError> {
        todo!()
    }

    fn batch_verify<R: RngCore + CryptoRng>(
        _verifier_param: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        _multi_commitment: &Self::BatchCommitment,
        _points: &[Self::Point],
        _values: &[Self::Evaluation],
        _batch_proof: &Self::BatchProof,
        _rng: &mut R,
    ) -> Result<bool, PCSError> {
        todo!()
    }
}

impl<E: Pairing> BivariateKzgSRS<E> {
    /// The maximum supported degree in X
    pub fn max_deg_x(&self) -> usize {
        self.powers_of_g.len() - 1
    }
    /// The maximum supported degree in Y
    pub fn max_deg_y(&self) -> usize {
        self.powers_of_g[0].len() - 1
    }
}

/// Prover key for proof generation
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, Eq, PartialEq, Default)]
pub struct BivariateProverParam<E: Pairing> {
    /// see `BivariateKzgSRS.powers_of_g` for documentation
    pub powers_of_g: Vec<Vec<E::G1Affine>>,
}

/// Verifier key for proof verification
pub struct BivariateVerifierParam<E: Pairing> {
    /// G1 generator
    pub g: E::G1Affine,
    /// G2 generator
    pub h: E::G2Affine,
    /// tau_x * h
    pub tau_x_h: E::G2Affine,
    /// tau_y * h
    pub tau_y_h: E::G2Affine,
}

impl<E: Pairing> StructuredReferenceString for BivariateKzgSRS<E> {
    type ProverParam = BivariateProverParam<E>;
    type VerifierParam = BivariateVerifierParam<E>;

    // NOTE: the upstream trait didn't consider multivariate degrees,
    // for now, use a hacky trick, treat `supported_degree` as 2*u32 values
    // higher half is supported degree in X, the lower half in Y.
    fn extract_prover_param(&self, supported_degree: usize) -> Self::ProverParam {
        let (deg_x, deg_y) = split_u64(supported_degree as u64);
        let (deg_x, deg_y) = (deg_x as usize, deg_y as usize);
        assert!(self.max_deg_x() >= deg_x && self.max_deg_y() >= deg_y);
        BivariateProverParam {
            powers_of_g: self
                .powers_of_g
                .par_iter()
                .take(deg_x + 1)
                .map(|row| {
                    row.par_iter()
                        .take(deg_y + 1)
                        .map(|&g| g)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        }
    }
    fn extract_verifier_param(&self, _supported_degree: usize) -> Self::VerifierParam {
        BivariateVerifierParam {
            g: self.powers_of_g[0][0],
            h: self.h,
            tau_x_h: self.tau_x_h,
            tau_y_h: self.tau_y_h,
        }
    }

    /// We use a hacky trick, treat `supported_degree` as 2*u32 values
    /// higher half is supported degree in X, the lower half in Y.
    fn trim(
        &self,
        supported_degree: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let pk = self.extract_prover_param(supported_degree);
        let vk = self.extract_verifier_param(supported_degree);
        Ok((pk, vk))
    }

    /// We use a hacky trick, treat `supported_degree` as 2*u32 values
    /// higher half is supported degree in X, the lower half in Y.
    fn gen_srs_for_testing<R: RngCore + CryptoRng>(
        rng: &mut R,
        supported_degree: usize,
    ) -> Result<Self, PCSError> {
        let (deg_x, deg_y) = split_u64(supported_degree as u64);
        let (deg_x, deg_y) = (deg_x as usize, deg_y as usize);

        let g = E::G1::generator();
        let h = E::G2::generator();

        // !! two Trapdoors !!
        let tau_x = E::ScalarField::rand(rng);
        let tau_y = E::ScalarField::rand(rng);

        let powers_of_g = (0..deg_x + 1)
            .into_par_iter()
            .map(|i| {
                let row = (0..deg_y + 1)
                    .into_par_iter()
                    .map(|j| {
                        let exp = tau_x.pow([i as u64]) * tau_y.pow([j as u64]);
                        g * exp
                    })
                    .collect::<Vec<_>>();
                E::G1::normalize_batch(&row)
            })
            .collect::<Vec<_>>();
        assert_eq!(powers_of_g[0][0], g.into_affine());

        let tau_x_h = (h * tau_x).into_affine();
        let tau_y_h = (h * tau_y).into_affine();
        let h = h.into_affine();

        Ok(BivariateKzgSRS {
            powers_of_g,
            h,
            tau_x_h,
            tau_y_h,
        })
    }

    // these are not needed for now, thus left unimplememted
    fn trim_with_verifier_degree(
        &self,
        _prover_supported_degree: usize,
        _verifier_supported_degree: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        unimplemented!();
    }
    fn gen_srs_for_testing_with_verifier_degree<R: RngCore + CryptoRng>(
        _rng: &mut R,
        _prover_supported_degree: usize,
        _verifier_supported_degree: usize,
    ) -> Result<Self, PCSError> {
        unimplemented!()
    }
}

// returns (higher, lower) halves of u64
#[inline(always)]
pub(crate) fn split_u64(x: u64) -> (u32, u32) {
    ((x >> 32) as u32, x as u32)
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) fn combine_u32(high: u32, low: u32) -> u64 {
    ((high as u64) << 32) | (low as u64)
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use ark_std::rand::Rng;

    use super::*;
    use crate::test_utils::test_rng;

    #[test]
    fn split_combine_u64() {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let a = rng.next_u64();
            let (high, low) = super::split_u64(a);
            assert_eq!(a, super::combine_u32(high, low));
        }
    }

    #[test]
    fn bkzg() {
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

            let poly = DensePolynomial::rand(deg_x as usize, deg_y as usize, rng);
            let cm = BivariateKzgPCS::commit(&pk, &poly).unwrap();

            for _ in 0..10 {
                let point = (Fr::rand(rng), Fr::rand(rng));
                let (proof, eval) = BivariateKzgPCS::open(&pk, &poly, &point).unwrap();

                assert_eq!(eval, poly.evaluate(&point));
                assert!(BivariateKzgPCS::verify(&vk, &cm, &point, &eval, &proof).unwrap());
            }
        }
    }
}
