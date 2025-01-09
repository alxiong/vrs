//! Univariate FRI Polynomial Commitment implementations

use crate::{
    iopp::fri::{self, FriConfig, FriProof},
    merkle_tree::{Path, SymbolMerkleTree},
};
use ark_ff::{FftField, Field};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial,
    Radix2EvaluationDomain,
};
use ark_serialize::*;
use ark_std::marker::PhantomData;
use nimue::*;
use p3_maybe_rayon::prelude::*;

/// Univariate FRI PCS
pub struct UniFriPCS<F>(PhantomData<F>);

impl<F: FftField> UniFriPCS<F> {
    /// Transparent setup
    /// - max_degree: degree upper bound of the polynomial to be committed
    /// - log_blowup: log of blowup factor
    pub fn setup(max_degree: usize, log_blowup: usize) -> UniFriURS<F> {
        let config = FriConfig::new_conjectured::<F>(max_degree + 1, log_blowup, None, None);
        let domain = Radix2EvaluationDomain::new(config.init_domain_size).unwrap();
        let io = IOPattern::<DefaultHash>::new("UniFriEval")
            .absorb(32, "cm_p")
            .absorb(32, "cm_w")
            .squeeze(8, "x_idx");
        UniFriURS { config, domain, io }
    }

    /// Commit to a polynomial, returns the merkle root of all evaluations in domain D_0
    pub fn commit(pp: &UniFriURS<F>, poly: &DensePolynomial<F>) -> Vec<u8> {
        let evals: Vec<[F; 1]> = pp
            .domain
            .fft(&poly.coeffs)
            .into_par_iter()
            .map(|f| [f]) // SymbolMerkleTree only accept [F] as leaf
            .collect();
        let mt = SymbolMerkleTree::<F>::new(evals);
        mt.root()
    }

    /// Open/Evaluate `poly` at `point` and return the evaluation/opening proof and the evaluation
    /// p(y) = z where y is `point`, and `z` is eval being returned with a proof
    pub fn open(pp: &UniFriURS<F>, poly: &DensePolynomial<F>, point: &F) -> (F, UniFriProof<F>) {
        let mut merlin = pp.io.to_merlin();

        // Commit the original polynomial
        let evals: Vec<[F; 1]> = pp
            .domain
            .fft(&poly.coeffs)
            .into_par_iter()
            .map(|f| [f])
            .collect();
        let mt = SymbolMerkleTree::<F>::new(evals);

        // Compute and commit to the witness polynomial
        let eval = poly.evaluate(point);
        // w(X) = (p(X) - y) / (X - z), since remainder is zero anyway, we save `-y` part
        let witness_poly = {
            let divisor = DensePolynomial::from_coefficients_slice(&[-*point, F::ONE]);
            poly / &divisor
        };
        let wit_evals = witness_poly.evaluate_over_domain_by_ref(pp.domain);
        let wit_mt = SymbolMerkleTree::<F>::new(
            wit_evals
                .evals
                .par_iter()
                .map(|f| [*f])
                .collect::<Vec<[F; 1]>>(),
        );

        // Append to Transcript
        merlin.add_bytes(&mt.root()).unwrap();
        merlin.add_bytes(&wit_mt.root()).unwrap();

        // derive a in-domain element x (via its index)
        let x_idx =
            (u64::from_le_bytes(merlin.challenge_bytes::<8>().unwrap()) % pp.domain.size) as usize;
        let x = pp.domain.element(x_idx);

        // get proofs of evaluation at L_0[x_idx] on both p(X) and w(X)
        let in_domain_eval = poly.evaluate(&x);
        let eval_proof = mt.generate_proof(x_idx);
        let in_domain_wit_eval = witness_poly.evaluate(&x);
        let witness_eval_proof = wit_mt.generate_proof(x_idx);

        // low-degree test on the witness polynomial
        let fri_proof = fri::prove(&pp.config, wit_evals);

        (
            eval,
            UniFriProof {
                transcript: merlin.transcript().to_vec(),
                in_domain_eval,
                eval_proof,
                in_domain_wit_eval,
                witness_eval_proof,
                fri_proof,
            },
        )
    }

    /// Verify an opening proof
    pub fn verify(
        pp: &UniFriURS<F>,
        commitment: &[u8],
        point: &F,
        value: &F,
        proof: &UniFriProof<F>,
    ) -> bool {
        let mut arthur = pp.io.to_arthur(&proof.transcript);
        let comm = arthur.next_bytes::<32>().unwrap().to_vec();
        let wit_comm = arthur.next_bytes::<32>().unwrap().to_vec();
        assert_eq!(&comm, commitment);

        // FS-derive x (via its index)
        let x_idx =
            (u64::from_le_bytes(arthur.challenge_bytes::<8>().unwrap()) % pp.domain.size) as usize;
        let x = pp.domain.element(x_idx);

        // verify all evaluation proof (merkle proof of evaluation tree)
        if !proof
            .eval_proof
            .verify(&comm, x_idx, [proof.in_domain_eval])
            || !proof
                .witness_eval_proof
                .verify(&wit_comm, x_idx, [proof.in_domain_wit_eval])
        {
            return false;
        }

        // verify p(x) - y = (x - z)* w(x)
        proof.in_domain_eval - value == (x - point) * proof.in_domain_wit_eval
    }
}

/// Evaluation/Opening proof for `UniFriPCS`
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct UniFriProof<F: Field> {
    /// Transcript containing
    /// - the commitments to the original polynomial and witness polynomial
    pub transcript: Vec<u8>,
    /// p(x) for an x sampled from L_0 domain (instead of the overall field)
    pub in_domain_eval: F,
    /// evaluation proof of p(x) via a merkle proof against `comm` for a FS-derived x
    pub eval_proof: Path<F>,
    /// w(x) for an x sampled from L_0 domain (instead of the overall field)
    pub in_domain_wit_eval: F,
    /// evaluation proof of w(x) via a merkle proof against `witness_comm` for the same x
    pub witness_eval_proof: Path<F>,
    /// IOPP proof that witness polynomial is a low-degree poly
    pub fri_proof: FriProof<F>,
}

/// Uniform Random String (URS) of Univariate FRI PCS, totally transparent
#[derive(Debug, Clone)]
pub struct UniFriURS<F: FftField> {
    /// FRI config, affect security level including conjectured soundness
    pub config: FriConfig,
    /// evaluation domain D_0
    pub domain: Radix2EvaluationDomain<F>,
    /// Prover-verifier interaction pattern in open/eval protocol
    pub io: IOPattern,
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
        println!("config: {:?}", pp.config);
        for _ in 0..5 {
            let degree = rng.gen_range(0..max_degree);
            let poly = DensePolynomial::rand(degree, rng);
            let comm = UniFriPCS::commit(&pp, &poly);

            for _ in 0..10 {
                let point = Fr::rand(rng);

                let (eval, proof) = UniFriPCS::open(&pp, &poly, &point);
                assert!(UniFriPCS::verify(&pp, &comm, &point, &eval, &proof));
                assert_eq!(eval, poly.evaluate(&point));
            }
        }
    }
}
