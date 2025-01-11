//! Univariate FRI Polynomial Commitment implementations
//!
//! Reference:
//! - https://eprint.iacr.org/2019/1020.pdf

use crate::{
    iopp::fri::{self, FriConfig, FriProof, TranscriptData},
    merkle_tree::{Path, SymbolMerkleTree},
};
use ark_ff::{FftField, Field};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial,
    Radix2EvaluationDomain,
};
use ark_serialize::*;
use ark_std::marker::PhantomData;
use p3_maybe_rayon::prelude::*;

/// Univariate FRI PCS
pub struct UniFriPCS<F>(PhantomData<F>);

impl<F: FftField> UniFriPCS<F> {
    /// Transparent setup
    /// - max_degree: degree upper bound of the polynomial to be committed
    /// - log_blowup: log of blowup factor
    pub fn setup(max_degree: usize, log_blowup: usize) -> UniFriURS<F> {
        let fri_config =
            FriConfig::new_conjectured::<F>(max_degree + 1, log_blowup, None, None, None);
        let domain = Radix2EvaluationDomain::new(fri_config.init_domain_size).unwrap();
        UniFriURS { fri_config, domain }
    }

    /// Commit to a polynomial, returns the merkle root of all evaluations in domain D_0, and prover data
    pub fn commit(pp: &UniFriURS<F>, poly: &DensePolynomial<F>) -> (Vec<u8>, UniFriProverData<F>) {
        let evals = pp.domain.fft(&poly.coeffs);
        let mt = SymbolMerkleTree::<F>::from_single_elem_leaves(&evals);

        (mt.root(), UniFriProverData { evals, mt })
    }

    /// Open/Evaluate `poly` at `point` and return the evaluation/opening proof and the evaluation
    /// p(y) = z where y is `point`, and `z` is eval being returned with a proof
    pub fn open(
        pp: &UniFriURS<F>,
        pd: &UniFriProverData<F>,
        poly: &DensePolynomial<F>,
        point: &F,
    ) -> (F, UniFriProof<F>) {
        // Compute and commit to the witness polynomial
        let eval = poly.evaluate(point);
        // w(X) = (p(X) - y) / (X - z), since remainder is zero anyway, we save `-y` before division
        let witness_poly = {
            let divisor = DensePolynomial::from_coefficients_slice(&[-*point, F::ONE]);
            poly / &divisor
        };
        let wit_evals = witness_poly.evaluate_over_domain_by_ref(pp.domain);
        // low-degree test on the witness polynomial
        let fri_proof = fri::prove(&pp.fri_config, wit_evals);

        let evals_and_proofs = TranscriptData::<F>::parse(&pp.fri_config, &fri_proof.transcript)
            .query_indices
            .par_iter()
            .map(|&idx| (pd.evals[idx], pd.mt.generate_proof(idx)))
            .collect();

        (
            eval,
            UniFriProof {
                evals_and_proofs,
                fri_proof,
            },
        )
    }

    /// Verify an opening proof
    pub fn verify(
        pp: &UniFriURS<F>,
        commitment: &Vec<u8>,
        point: &F,
        value: &F,
        proof: &UniFriProof<F>,
    ) -> bool {
        let query_indices =
            TranscriptData::<F>::parse(&pp.fri_config, &proof.fri_proof.transcript).query_indices;
        assert_eq!(query_indices.len(), proof.evals_and_proofs.len());
        assert_eq!(query_indices.len(), proof.fri_proof.query_proofs.len());

        // 1. check f(x) = y + (x - z) * w(x)
        let mut verified = query_indices
            .par_iter()
            .zip(proof.evals_and_proofs.par_iter())
            .zip(proof.fri_proof.query_proofs.par_iter())
            .all(|((&idx, (eval, proof)), query_proof)| {
                let x = pp.domain.element(idx);
                *eval == *value + (x - point) * query_proof.query_eval
                    && proof.verify(commitment, idx, [*eval])
            });
        // 2. verifiy low-degreeness of the witness poly
        verified &= fri::verify(&pp.fri_config, &proof.fri_proof);
        verified
    }
}

/// Prover data to enable opening of committed polynomial
#[derive(Clone)]
pub struct UniFriProverData<F: Field> {
    /// Evaluations of the polynomial committed over the eval domain in FriConfig
    pub evals: Vec<F>,
    /// The merkle tree for
    pub mt: SymbolMerkleTree<F>,
}

/// Evaluation/Opening proof for `UniFriPCS`
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct UniFriProof<F: Field> {
    /// Evaluation proofs of queried points on the original f, each pair is (eval, merkle_proof)
    pub evals_and_proofs: Vec<(F, Path<F>)>,
    /// IOPP proof that witness polynomial is a low-degree poly
    pub fri_proof: FriProof<F>,
}

/// Uniform Random String (URS) of Univariate FRI PCS, totally transparent
#[derive(Debug, Clone)]
pub struct UniFriURS<F: FftField> {
    /// FRI config, affect security level including conjectured soundness
    pub fri_config: FriConfig,
    /// evaluation domain D_0
    pub domain: Radix2EvaluationDomain<F>,
}

#[cfg(test)]
mod tests {
    use crate::test_utils::test_rng;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, UniformRand};

    #[test]
    fn test_uni_fri_pcs() {
        let rng = &mut test_rng();
        let max_degree = 64;
        let log_blowup = 1;

        let pp = UniFriPCS::<Fr>::setup(max_degree, log_blowup);
        for _ in 0..5 {
            let degree = rng.gen_range(0..max_degree);
            let poly = DensePolynomial::rand(degree, rng);
            let (comm, prover_data) = UniFriPCS::commit(&pp, &poly);

            for _ in 0..10 {
                let point = Fr::rand(rng);

                let (eval, proof) = UniFriPCS::open(&pp, &prover_data, &poly, &point);
                assert!(UniFriPCS::verify(&pp, &comm, &point, &eval, &proof));
                assert_eq!(eval, poly.evaluate(&point));
            }
        }
    }
}
