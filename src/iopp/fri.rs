//! FRI low-degree test

use ark_ff::{batch_inversion_and_mul, FftField, Field};
use ark_poly::{EvaluationDomain, Evaluations, Radix2EvaluationDomain};
use ark_serialize::*;
use itertools::izip;
use nimue::{plugins::ark::*, *};
use p3_maybe_rayon::prelude::*;

use crate::{
    matrix::Matrix,
    merkle_tree::{Path, SymbolMerkleTree},
};

/// Parameters for FRI
#[derive(Debug, Clone)]
pub struct FriConfig {
    /// blowup factor for linear code
    pub log_blowup: usize,
    /// number of queries
    pub num_queries: usize,
    // TODO: allow more flexible per-step reduction
    // // reduction factor: a_i = |D_i-1| / |D_i| >= 2
    // pub red_factors: Vec<usize>,
    /// Proof-of-work for grinding
    pub pow_bits: usize,
    /// IOP transcript io pattern
    io: IOPattern,
}

impl FriConfig {
    /// init a config
    pub fn new<F>(
        log_blowup: usize,
        num_queries: usize,
        pow_bits: usize,
        init_domain_size: usize,
    ) -> Self
    where
        F: Field,
    {
        let mut io = IOPattern::<DefaultHash>::new("FRI");
        let num_rounds = init_domain_size.ilog2() as usize - log_blowup;
        for round in 0..num_rounds {
            io = io.absorb(32, &format!("root_{}", round));
            io = FieldIOPattern::<F>::challenge_scalars(io, 1, &format!("beta_{}", round));
        }
        io = FieldIOPattern::<F>::add_scalars(io, 1, "final_poly");
        io = io
            .absorb(8, "pow_wit") // u64 = [u8; 8]
            .squeeze((pow_bits + 7) / 8, "grind_res");
        for query in 0..num_queries {
            io = io.challenge_bytes(usize::BITS as usize / 8, &format!("query_{}", query));
        }

        Self {
            log_blowup,
            num_queries,
            pow_bits,
            io,
        }
    }

    /// the blowup factor
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
}

/// A FRI-proof
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct FriProof<F: Field> {
    /// Fiat-Shamir transcript, containing only prover messages
    /// containing merkle_root from all rounds, the final_poly (constant poly), and proof of work witness
    pub transcript: Vec<u8>,
    /// Query proofs for all queries
    pub query_proofs: Vec<QueryProof<F>>,
}

/// Proof for a single FRI query for all rounds
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct QueryProof<F: Field> {
    /// Top-most queried point, i.e. s_0 \in F
    pub query_point: F,
    /// answering (sibling_value, mt_proof) for per-round query,
    /// each leaf contains (folded_value, sibling_value)
    /// in-order: large domain to smaller reduced domain
    pub openings: Vec<(F, Path<F>)>,
}

impl<F: Field> QueryProof<F> {
    /// Returns the number of rounds this query proof contains
    pub fn num_rounds(&self) -> usize {
        self.openings.len()
    }
    /// Returns the top-most initial domain size
    pub fn init_domain_size(&self) -> usize {
        assert!(!self.openings.is_empty());
        self.openings[0].1.capacity() * 2
    }
}

/// Prover's grinding to solve proof-of-work with difficulty bits
/// and append the solution to the transcript when found.
fn pow_grind(merlin: &mut Merlin, pow_bits: usize) {
    assert!((pow_bits as u32) < u64::BITS);
    let pow_witness = (0..u64::MAX)
        .into_par_iter()
        .find_any(|witness| {
            let mut forked_merlin = merlin.clone();
            forked_merlin.add_bytes(&witness.to_le_bytes()).unwrap();
            let mut grinding_result = vec![0u8; (pow_bits + 7) / 8];
            forked_merlin
                .fill_challenge_bytes(&mut grinding_result)
                .unwrap();

            // succeed if the first `pow_bits` bits are all zero
            pow_verify(&grinding_result, pow_bits)
        })
        .expect("failed to find witness");

    // append actual pow solution/witness
    merlin.add_bytes(&pow_witness.to_le_bytes()).unwrap();
    let mut grinding_result = vec![0u8; (pow_bits + 7) / 8];
    merlin.fill_challenge_bytes(&mut grinding_result).unwrap();
}

/// Verify proof-of-work result, the first `pow_bits` are zeros
fn pow_verify(bytes: &[u8], pow_bits: usize) -> bool {
    bytes
        .iter()
        .take((pow_bits + 7) / 8)
        .enumerate()
        .all(|(i, &b)| {
            if (i + 1) * 8 > pow_bits {
                // Check the last, partially-filled byte
                (b >> (8 - pow_bits % 8)) == 0
            } else {
                b == 0
            }
        })
}

/// Proving low-degree of a polynomial given its evaluation on D
pub fn prove<F>(config: &FriConfig, evals: Evaluations<F, Radix2EvaluationDomain<F>>) -> FriProof<F>
where
    F: FftField,
{
    let mut merlin = config.io.to_merlin();
    let num_rounds = evals.evals.len().ilog2() as usize - config.log_blowup;
    let elems: Vec<F> = evals.domain().elements().collect();
    let domain_0_size = elems.len();

    // Commit Phase
    let mut folded = evals.evals.clone();
    let mut mts = Vec::with_capacity(num_rounds);
    let mut mts_leaves = Vec::with_capacity(num_rounds);

    // each fold is 2-to-1 where domains are halved and x and -x are mapped to x^2.
    // for rounds with reduction factor >2, we do multiple halving.
    while folded.len() > config.blowup() {
        let prev_domain_size = folded.len();
        let next_domain_size = prev_domain_size / 2;
        // orgnaize into 2*(n/2) matrix where each column is a leaf (size = 2 fields)
        // values in a leaf corresponds to x and -x, with index i and n/2+i for i \in [0,n/2)
        let folded_matrix = Matrix::new(folded, next_domain_size, 2).unwrap();
        let leaves: Vec<Vec<_>> = folded_matrix.par_col().collect();
        let mt = SymbolMerkleTree::<F>::from_slice(&leaves);

        // merkle-commit the evaluation vector (in 2-row matrix form), derive the verifier challenge
        merlin.add_bytes(&mt.root()).unwrap();
        assert_eq!(mt.root().len(), 32);
        let [beta]: [F; 1] = merlin.challenge_scalars().unwrap();

        // TODO: move this to a individual function and add tests for correctness
        // the original domain is D_{i-1} = (e_0, ..., e_{n-1}) from round i-1
        // each leaf at col j contains f_{i-1}(e_j) and f_{i-1}(-e_j)
        // the new folded domain is D_i = (e'_0, ... , e'_{n/2-1}) where e'_j = e_j^2
        //
        // Assume f_{i-1}(X) = f_even(X^2) + X * f_odd(X^2), with the combiner beta
        // folded: f_i(Y) = f_even(Y) + beta * f_odd(Y)
        //
        // the evaluation f_i(e_j) can be computed from evaluations of f_{i-1} as:
        //   f_i(e'_j) = (beta - e_j)(-2e_j) * f_{i-1}(-e_j) + (beta + e_j)(2e_j) * f_{i-1}(e_j)
        // where e'_j = e_j^2
        //
        // Rearranging the terms, we got:
        // let sum = f_{i-1}(e_j) + f_{i-1}(-e_j), diff = f_{i-1}(e_j) - f_{i-1}(-e_j)
        //   f_i(e'_j) = 1/2 * (sum + beta * diff / e_j)
        // we compute beta/e_j using batch inversion
        let mut all_beta_elem_inv = elems
            .iter()
            .step_by(domain_0_size / prev_domain_size)
            .take(next_domain_size)
            .cloned()
            .collect::<Vec<_>>();
        batch_inversion_and_mul(&mut all_beta_elem_inv, &beta);
        let half = F::from(2u64).inverse().unwrap();

        folded = leaves
            .par_iter()
            .zip(all_beta_elem_inv.par_iter())
            .map(|(evals, beta_elem_inv)| {
                let eval_e_j = evals[0];
                let eval_minus_e_j = evals[1];
                half * (eval_e_j + eval_minus_e_j + *beta_elem_inv * (eval_e_j - eval_minus_e_j))
            })
            .collect();

        mts.push(mt);
        mts_leaves.push(leaves);
    }
    // send over the final folded polynomial, which is a constant polynomial
    assert_eq!(folded.len(), config.blowup());
    let final_poly = folded[0];
    assert!(folded.par_iter().all(|f| f == &final_poly));
    merlin.add_scalars(&[final_poly]).unwrap();

    // Perform Proof-of-work grinding
    pow_grind(&mut merlin, config.pow_bits);

    // Query Phase
    // randomly sampled s_0 for each query, must be sequential squeezing
    let s_0_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| usize::from_le_bytes(merlin.challenge_bytes().unwrap()) % domain_0_size)
        .collect();

    let query_proofs = s_0_indices
        .into_par_iter()
        .map(|s_0_idx| {
            // s_{i+1} = s_i^2 value mapping corresponds to the index mapping as follows:
            // j-th element in i-th round is s_i, then s_{i+1} is the j%(n/2^{i+1})-th element in the next folded domain
            // our merkle tree at round i has size n/2^i, where n is the original domain size
            let openings = mts
                .iter()
                .zip(mts_leaves.iter())
                .scan(
                    (s_0_idx, domain_0_size / 2),
                    |(idx, tree_size), (mt, leaves)| {
                        // the index (0/1) of the sibling inside the 2-element leaf
                        let sibling_idx = 1 - *idx / *tree_size;

                        *idx %= *tree_size;
                        *tree_size /= 2;
                        Some((leaves[*idx][sibling_idx], mt.generate_proof(*idx)))
                    },
                )
                .collect::<Vec<(F, Path<F>)>>();
            QueryProof {
                query_point: evals.evals[s_0_idx],
                openings,
            }
        })
        .collect();

    FriProof {
        transcript: merlin.transcript().to_owned(),
        query_proofs,
    }
}

/// Verifying a FRI proof of low-degreeness
pub fn verify<F>(config: &FriConfig, proof: &FriProof<F>) -> bool
where
    F: FftField,
{
    let mut arthur = config.io.to_arthur(&proof.transcript);
    assert_eq!(proof.query_proofs.len(), config.num_queries);
    let num_rounds = proof.query_proofs[0].num_rounds();
    let domain_0_size = proof.query_proofs[0].init_domain_size();
    let elems: Vec<F> = Radix2EvaluationDomain::new(domain_0_size)
        .unwrap()
        .elements()
        .collect();

    // merkle roots
    let mut commits = Vec::with_capacity(num_rounds);
    // random combiners to fold poly in each round
    let mut betas = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        commits.push(arthur.next_bytes::<32>().unwrap().to_vec());
        let [beta]: [F; 1] = arthur.challenge_scalars().unwrap();
        betas.push(beta);
    }
    let [final_poly]: [F; 1] = arthur.next_scalars().unwrap();

    // verify proof-of-work grind
    let _pow_witness = arthur.next_bytes::<8>().unwrap();
    let mut grinding_result = vec![0u8; (config.pow_bits + 7) / 8];
    arthur.fill_challenge_bytes(&mut grinding_result).unwrap();
    if !pow_verify(&grinding_result, config.pow_bits) {
        return false;
    }

    let s_0_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| usize::from_le_bytes(arthur.challenge_bytes().unwrap()) % domain_0_size)
        .collect();

    s_0_indices
        .into_par_iter()
        .zip(proof.query_proofs.par_iter())
        .all(|(s_0_idx, query_proof)| {
            let mut idx = s_0_idx;
            let mut tree_size = domain_0_size / 2;
            let mut folded = query_proof.query_point;

            for (round, (sibling, mt_proof), root, beta) in
                izip!(0..num_rounds, &query_proof.openings, &commits, &betas)
            {
                let sibling_idx = 1 - idx / tree_size;
                let mut leaf = [folded; 2];
                leaf[sibling_idx] = *sibling;

                idx %= tree_size;
                tree_size /= 2;
                if mt_proof.index() != idx || !mt_proof.verify(root, leaf.clone()) {
                    return false;
                }

                // Compute the following (same from commit phase logic):
                // let sum = f_{i-1}(e_j) + f_{i-1}(-e_j), diff = f_{i-1}(e_j) - f_{i-1}(-e_j)
                //   f_i(e'_j) = 1/2 * (sum + beta * diff / e_j)
                // a key mapping relationship is that for a halving domains,
                //   j-th element in i-th round is the j*2^i-th element in the original domain
                let half = F::from(2u64).inverse().unwrap();
                folded = half
                    * (leaf[0]
                        + leaf[1]
                        + *beta * (leaf[0] - leaf[1]) / elems[idx * 2usize.pow(round as u32)]);
            }
            // check final matching constant poly
            folded == final_poly
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_rng;
    use ark_bn254::Fr;
    use ark_std::UniformRand;

    #[test]
    fn test_fri() {
        let rng = &mut test_rng();
        let log_blowup = 1;
        let num_queries = 5;
        let pow_bits = 3;
        let init_domain_size = 64;
        let domain = Radix2EvaluationDomain::new(init_domain_size).unwrap();
        let coeffs = (0..init_domain_size / (1 << log_blowup))
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let evals = domain.fft(&coeffs);

        let config = FriConfig::new::<Fr>(log_blowup, num_queries, pow_bits, init_domain_size);
        let fri_proof = super::prove(&config, Evaluations::from_vec_and_domain(evals, domain));
        assert!(super::verify(&config, &fri_proof));

        let mut bad_proof = fri_proof.clone();
        bad_proof.query_proofs[0].openings[0].0 = Fr::rand(rng);
        assert!(!super::verify(&config, &bad_proof));
    }

    #[test]
    #[should_panic]
    fn test_fri_on_bad_inputs() {
        let rng = &mut test_rng();
        let log_blowup = 1;
        let num_queries = 5;
        let pow_bits = 2;
        let init_domain_size = 64;
        // these don't match the evaluation of a polynomial of degree `init_domain_size / (1 << log_blowup)`
        let bad_evals = (0..init_domain_size)
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let domain = Radix2EvaluationDomain::new(init_domain_size).unwrap();

        let config = FriConfig::new::<Fr>(log_blowup, num_queries, pow_bits, init_domain_size);
        // this will fail since the final folded constant poly shouldn't match
        super::prove(&config, Evaluations::from_vec_and_domain(bad_evals, domain));
    }

    #[test]
    fn test_pow_grind() {
        assert!(pow_verify(&[0, 0], 3));
        assert!(!pow_verify(&[128], 3));
        assert!(pow_verify(&[0, 64], 9));

        for pow_bits in 2..6 {
            let io = IOPattern::<DefaultHash>::new("protocol")
                .absorb(u64::BITS as usize / 8, "pow_witness")
                .squeeze((pow_bits + 7) / 8, "grind_result");
            let mut merlin = io.to_merlin();
            pow_grind(&mut merlin, pow_bits);

            let mut arthur = io.to_arthur(&merlin.transcript());
            let _witness = arthur.next_bytes::<8>().unwrap();
            let mut grinding_result = vec![0u8; (pow_bits + 7) / 8];
            arthur.fill_challenge_bytes(&mut grinding_result).unwrap();
            assert!(pow_verify(&grinding_result, pow_bits));
        }
    }

    #[ignore]
    #[test]
    fn test_fri_fold() {
        /// input: leaves where each leaf consists of 2 value to be folded with combiner `beta`
        /// output: folded vector
        /// used for testing mostly
        #[allow(dead_code)]
        pub(crate) fn fold<F: FftField>(leaves: &[Vec<F>], beta: &F) -> Vec<F> {
            let domain_size = leaves.len() * 2;
            assert!(leaves.iter().all(|leaf| leaf.len() == 2));
            let mut all_beta_elem_inv = Radix2EvaluationDomain::new(domain_size)
                .unwrap()
                .elements()
                .collect::<Vec<_>>();
            all_beta_elem_inv.truncate(domain_size / 2);
            batch_inversion_and_mul(&mut all_beta_elem_inv, beta);
            let half = F::from(2u64).inverse().unwrap();

            let folded = leaves
                .par_iter()
                .zip(all_beta_elem_inv.par_iter())
                .map(|(evals, beta_elem_inv)| {
                    let eval_e_j = evals[0];
                    let eval_minus_e_j = evals[1];
                    half * (eval_e_j
                        + eval_minus_e_j
                        + *beta_elem_inv * (eval_e_j - eval_minus_e_j))
                })
                .collect();
            folded
        }

        let rng = &mut test_rng();
        let init_domain_size: usize = 64;
        let evals: Vec<Fr> = (0..init_domain_size).map(|_| Fr::rand(rng)).collect();
        let beta = Fr::rand(rng);

        assert!(init_domain_size.is_power_of_two());
        let domain = Radix2EvaluationDomain::<Fr>::new(init_domain_size).unwrap();
        let reduced_domain = Radix2EvaluationDomain::<Fr>::new(init_domain_size / 2).unwrap();
        for i in 0..init_domain_size / 2 {
            assert_eq!(reduced_domain.element(i), domain.element(2 * i));
        }

        let coeffs = domain.ifft(&evals);
        let (even_terms, odd_terms): (Vec<_>, Vec<_>) = coeffs
            .iter()
            .enumerate()
            .partition(|(index, _)| index % 2 == 0);
        let even_coeffs: Vec<_> = even_terms.into_iter().map(|(_, value)| *value).collect();
        let odd_coeffs: Vec<_> = odd_terms.into_iter().map(|(_, value)| *value).collect();
        let folded_coeffs: Vec<_> = even_coeffs
            .par_iter()
            .zip(odd_coeffs.par_iter())
            .map(|(e, o)| *e + beta * o)
            .collect();
        assert_eq!(folded_coeffs.len(), coeffs.len() / 2);
        assert_eq!(even_coeffs.len(), odd_coeffs.len());
        let folded_evals = reduced_domain.fft(&folded_coeffs);

        let matrix = Matrix::new(evals.clone(), init_domain_size / 2, 2).unwrap();
        let leaves: Vec<_> = matrix.par_col().collect();
        let folded = fold(&leaves, &beta);
        assert_eq!(folded, folded_evals);
    }
}
