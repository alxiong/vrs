//! FRI low-degree test

use ark_ff::{batch_inversion_and_mul, FftField, Field};
use ark_poly::{EvaluationDomain, Evaluations, Radix2EvaluationDomain};
use ark_serialize::*;
use itertools::izip;
use nimue::{plugins::ark::*, *};
use nimue_pow::{blake3::Blake3PoW, PoWChallenge, PoWIOPattern};
use p3_maybe_rayon::prelude::*;

use crate::{
    iopp::fri_params,
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
    /// Proof-of-work for grinding
    pub pow_bits: usize,
    /// Input message length, equivalently degree bound
    pub msg_len: usize,
    /// Number of rounds of interaction, derived
    pub num_rounds: usize,
    /// Size of D_0 or L_0, the initial evaluation domain, derived = msg_len * blowup
    pub init_domain_size: usize,
    /// IOP transcript io pattern
    io: IOPattern,
}

impl FriConfig {
    /// Using conjectured security of FRI: stronger assumption, more fragile, shorter proof size.
    /// See Conjecture 1 of https://eprint.iacr.org/2024/1161.pdf
    ///
    /// - `pow_bits` is optional and default to 16, bigger value means slower prover and shorter proofs
    /// - `sec_bits`: only 80/100-bit security, default to 100
    pub fn new_conjectured<F: Field>(
        msg_len: usize,
        log_blowup: usize,
        pow_bits: Option<usize>,
        sec_bits: Option<usize>,
    ) -> Self {
        let sec_bits = sec_bits.unwrap_or(100);
        let pow_bits = pow_bits.unwrap_or(16);
        let blowup = 1 << log_blowup;
        let num_queries = fri_params::conjectured::num_queries::<F>(sec_bits, blowup, pow_bits);

        Self::new_unchecked::<F>(msg_len, log_blowup, num_queries, pow_bits)
    }

    /// Similar to [`Self::new_conjectured`] but with provable bounds.
    /// See Theorem 1 of https://eprint.iacr.org/2024/1161.pdf
    pub fn new_provable<F: Field>(
        msg_len: usize,
        log_blowup: usize,
        pow_bits: Option<usize>,
        sec_bits: Option<usize>,
    ) -> Self {
        let sec_bits = sec_bits.unwrap_or(100);
        let pow_bits = pow_bits.unwrap_or(16);
        let blowup = 1 << log_blowup;
        let num_queries =
            fri_params::provable::num_queries::<F>(sec_bits, msg_len, blowup, pow_bits);

        Self::new_unchecked::<F>(msg_len, log_blowup, num_queries, pow_bits)
    }

    /// init a config without checking security level
    pub fn new_unchecked<F>(
        msg_len: usize,
        log_blowup: usize,
        num_queries: usize,
        pow_bits: usize,
    ) -> Self
    where
        F: Field,
    {
        let mut io = IOPattern::<DefaultHash>::new("FRI");
        let init_domain_size = (msg_len * (1 << log_blowup)).next_power_of_two();
        let num_rounds = init_domain_size.ilog2() as usize - log_blowup;
        for round in 0..num_rounds {
            io = io.absorb(32, &format!("root_{}", round));
            io = FieldIOPattern::<F>::challenge_scalars(io, 1, &format!("beta_{}", round));
        }
        io = FieldIOPattern::<F>::add_scalars(io, 1, "final_poly");
        io = io.challenge_pow(&format!("pow_bits_{}", pow_bits));
        for query in 0..num_queries {
            io = io.challenge_bytes(usize::BITS as usize / 8, &format!("query_{}", query));
        }

        Self {
            log_blowup,
            num_queries,
            pow_bits,
            msg_len,
            num_rounds,
            init_domain_size,
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

/// Proving low-degree of a polynomial given its evaluation on D
pub fn prove<F>(config: &FriConfig, evals: Evaluations<F, Radix2EvaluationDomain<F>>) -> FriProof<F>
where
    F: FftField,
{
    let mut merlin = config.io.to_merlin();
    let num_rounds = config.num_rounds;
    let elems: Vec<F> = evals.domain().elements().collect();
    let domain_0_size = config.init_domain_size;

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
    merlin
        .challenge_pow::<Blake3PoW>(config.pow_bits as f64)
        .unwrap();

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
    let num_rounds = config.num_rounds;
    let domain_0_size = config.init_domain_size;
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
    arthur
        .challenge_pow::<Blake3PoW>(config.pow_bits as f64)
        .unwrap();

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
                if !mt_proof.verify(root, idx, leaf.clone()) {
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
        let msg_len = 32;
        let init_domain_size = msg_len * (1 << log_blowup);
        let domain = Radix2EvaluationDomain::new(init_domain_size).unwrap();
        let coeffs = (0..init_domain_size / (1 << log_blowup))
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let evals = domain.fft(&coeffs);

        let config = FriConfig::new_unchecked::<Fr>(msg_len, log_blowup, num_queries, pow_bits);
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

        let config = FriConfig::new_unchecked::<Fr>(
            init_domain_size / (1 << log_blowup),
            log_blowup,
            num_queries,
            pow_bits,
        );
        // this will fail since the final folded constant poly shouldn't match
        super::prove(&config, Evaluations::from_vec_and_domain(bad_evals, domain));
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
