//! FRI low-degree test
//!
//! # Reference
//! - batching: https://eprint.iacr.org/2020/654.pdf (Sec 8.2)

use ark_ff::{batch_inversion_and_mul, FftField, Field};
use ark_poly::{EvaluationDomain, Evaluations, Radix2EvaluationDomain};
use ark_serialize::*;
use ark_std::{iter::successors, log2};
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
    /// Batching number for batched FRI
    pub num_batches: usize,
    /// IOP transcript io pattern
    pub(crate) io: IOPattern,
}

impl FriConfig {
    /// Using conjectured security of FRI: stronger assumption, more fragile, shorter proof size.
    /// See Conjecture 1 of https://eprint.iacr.org/2024/1161.pdf
    ///
    /// - `pow_bits` is optional and default to 16, bigger value means slower prover and shorter proofs
    /// - `sec_bits`: only 80/100-bit security, default to 100
    /// - `num_batches`: Proving a batch of polynomial using simple batched FRI
    pub fn new_conjectured<F: Field>(
        msg_len: usize,
        log_blowup: usize,
        pow_bits: Option<usize>,
        sec_bits: Option<usize>,
        num_batches: Option<usize>,
    ) -> Self {
        let sec_bits = sec_bits.unwrap_or(100);
        let pow_bits = pow_bits.unwrap_or(16);
        let num_batches = num_batches.unwrap_or(1);
        let blowup = 1 << log_blowup;
        let num_queries = fri_params::conjectured::num_queries::<F>(sec_bits, blowup, pow_bits);

        Self::new_unchecked::<F>(msg_len, log_blowup, num_queries, pow_bits, num_batches)
    }

    /// Similar to [`Self::new_conjectured`] but with provable bounds.
    /// See Theorem 1 of https://eprint.iacr.org/2024/1161.pdf
    pub fn new_provable<F: Field>(
        msg_len: usize,
        log_blowup: usize,
        pow_bits: Option<usize>,
        sec_bits: Option<usize>,
        num_batches: Option<usize>,
    ) -> Self {
        let sec_bits = sec_bits.unwrap_or(100);
        let pow_bits = pow_bits.unwrap_or(16);
        let num_batches = num_batches.unwrap_or(1);
        let blowup = 1 << log_blowup;
        let num_queries =
            fri_params::provable::num_queries::<F>(sec_bits, msg_len, blowup, pow_bits);

        Self::new_unchecked::<F>(msg_len, log_blowup, num_queries, pow_bits, num_batches)
    }

    /// init a config without checking security level
    pub fn new_unchecked<F>(
        msg_len: usize,
        log_blowup: usize,
        num_queries: usize,
        pow_bits: usize,
        num_batches: usize,
    ) -> Self
    where
        F: Field,
    {
        let mut io = IOPattern::<DefaultHash>::new("FRI");
        if num_batches > 1 {
            io = io.absorb(32, "batch_root"); // commit to the entire matrix/batch of evaluations
            io = FieldIOPattern::<F>::challenge_scalars(io, 1, "batch_alpha");
        }
        let init_domain_size = (msg_len * (1 << log_blowup)).next_power_of_two();
        let num_rounds = log2(init_domain_size) as usize - log_blowup;
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
            num_batches,
            io,
        }
    }

    /// the blowup factor
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
}

/// Messages passed between prover and verifier, namely the transcript
/// Corresponding to the `IOPattern` created during `FriConfig::new_unchecked()`
///
/// NOTE: reason why we don't directly include `TranscriptData` in `FriProof`
/// is the entire transcript construction spans over multiple API boundaries:
/// from commit phase to grinding to queries sampling, if we pass a partial
/// transcript around would makes the types ugly and unintutively. Instead, we
/// only let `merlin` be the state-carrying variable, which can snapshot to a
/// transcript bytes and parsed here to a full transcript.
#[derive(Clone)]
pub(crate) struct TranscriptData<F: Field> {
    /// commitment to the batched FRI instances
    pub batch_commit: Option<Vec<u8>>,
    /// random combiner for a batch of instances
    pub alpha: Option<F>,
    /// commitments from all rounds, shape: [[u8; 32]; num_rounds]
    pub round_commits: Vec<Vec<u8>>,
    /// folding challenge from all rounds
    pub round_chals: Vec<F>,
    /// final constant poly
    pub final_poly: F,
    /// queried indices during the query phase
    pub query_indices: Vec<usize>,
}

impl<F: Field> TranscriptData<F> {
    /// Parse the transcript data
    pub fn parse(config: &FriConfig, transcript: &[u8]) -> Self {
        let mut arthur = config.io.to_arthur(&transcript);

        let mut batch_commit = vec![0u8; 32];
        let mut alpha = F::ZERO;
        if config.num_batches > 1 {
            batch_commit = arthur.next_bytes::<32>().unwrap().to_vec();
            [alpha] = arthur.challenge_scalars().unwrap();
        }

        let num_rounds = config.num_rounds;
        let domain_0_size = config.init_domain_size;

        // merkle roots
        let mut round_commits = Vec::with_capacity(num_rounds);
        // random combiners to fold poly in each round
        let mut round_chals = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            round_commits.push(arthur.next_bytes::<32>().unwrap().to_vec());
            let [beta]: [F; 1] = arthur.challenge_scalars().unwrap();
            round_chals.push(beta);
        }
        let [final_poly]: [F; 1] = arthur.next_scalars().unwrap();

        // verify proof-of-work grind
        arthur
            .challenge_pow::<Blake3PoW>(config.pow_bits as f64)
            .unwrap();

        let query_indices: Vec<usize> = (0..config.num_queries)
            .map(|_| usize::from_le_bytes(arthur.challenge_bytes().unwrap()) % domain_0_size)
            .collect();

        TranscriptData {
            batch_commit: Some(batch_commit),
            alpha: Some(alpha),
            round_commits,
            round_chals,
            final_poly,
            query_indices,
        }
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
    /// Proof of correct batching, optional for batched FRI
    pub batching_proofs: Option<Vec<BatchedColProof<F>>>,
}

/// The query proof on the interleaved matrix.
/// NOTE: it's not a batched proof, it's the query proof of correct batching as part of the batched FRI
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedColProof<F: Field> {
    /// Evaluations of the queried column (query index is the top-most query on the batched eval)
    pub query_col_evals: Vec<F>,
    /// Merkle proof of each column
    pub opening_proof: Path<F>,
}

impl<F: Field> BatchedColProof<F> {
    /// Verify the correct batching proof
    /// - `commit`: commitment/merkle_root of the evaluation matrix
    /// - `alpha`: random combiner to squash the matrix into a single eval row (i.e. single FRI instance)
    /// - `query_eval`: the batched/combined evaluation at `query_idx` on batched instance
    pub fn verify(
        &self,
        config: &FriConfig,
        commit: &Vec<u8>,
        alpha: &F,
        query_eval: &F,
        query_idx: usize,
    ) -> bool {
        let alpha_powers: Vec<F> = successors(Some(F::ONE), |&prev| Some(prev * alpha))
            .take(config.num_batches)
            .collect();

        // check correct column batching via linear combination
        let mut verified = alpha_powers
            .par_iter()
            .zip(self.query_col_evals.par_iter())
            .map(|(&alpha_pow, eval)| alpha_pow * eval)
            .sum::<F>()
            == *query_eval;

        // check correct column opening
        verified &= self
            .opening_proof
            .verify(commit, query_idx, self.query_col_evals.clone());

        verified
    }
}

/// Proof for a single FRI query for all rounds
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct QueryProof<F: Field> {
    /// Top-most queried evaluation, i.e. f(s_0) where s_0 \in D_0
    pub query_eval: F,
    /// answering (sibling_value, mt_proof) for per-round query,
    /// each leaf contains (folded_value, sibling_value)
    /// in-order: large domain to smaller reduced domain
    pub opening_proof: Vec<(F, Path<F>)>,
}

impl<F: FftField> QueryProof<F> {
    /// Veriyf the query proof
    /// - `commits`: commitment/merkle_root of per round evaluations
    /// - `betas`: all the beta combiner in each round
    pub fn verify(
        &self,
        config: &FriConfig,
        commits: &[Vec<u8>],
        betas: &[F],
        query_idx: usize,
        final_poly: &F,
    ) -> bool {
        let elems: Vec<F> = Radix2EvaluationDomain::new(config.init_domain_size)
            .unwrap()
            .elements()
            .collect();

        let mut idx = query_idx;
        let mut tree_size = config.init_domain_size / 2;
        let mut folded = self.query_eval;
        let half = F::from(2u64).inverse().unwrap();

        for (round, (sibling, mt_proof), root, beta) in
            izip!(0..config.num_rounds, &self.opening_proof, commits, betas)
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
            folded = half
                * (leaf[0]
                    + leaf[1]
                    + *beta * (leaf[0] - leaf[1]) / elems[idx * 2usize.pow(round as u32)]);
        }
        // check final matching constant poly
        folded == *final_poly
    }
}

/// Output from the `commit_phase()`
pub struct CommitPhaseResult<F: Field> {
    /// Committed oracle for each round (i.e. Merkle tree)
    pub commits: Vec<SymbolMerkleTree<F>>,
    /// Values being committed in each commit, shape: [[[F]; domain_i_size / 2]; num_rounds]
    /// each leaf may contains multiple leaves (2 for now)
    pub openings: Vec<Vec<Vec<F>>>,
}

/// The commit phase of the FRI protocol
#[inline(always)]
pub fn commit_phase<F: FftField>(
    config: &FriConfig,
    evals: Evaluations<F, Radix2EvaluationDomain<F>>,
    merlin: &mut Merlin,
) -> CommitPhaseResult<F> {
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

    CommitPhaseResult {
        commits: mts,
        openings: mts_leaves,
    }
}

/// Responding a single query in the query phase of FRI protocol, given the prover data `pd` from the `commit_phase()`
#[inline(always)]
pub fn answer_query<F: FftField>(
    config: &FriConfig,
    pd: &CommitPhaseResult<F>,
    query_idx: usize,
) -> QueryProof<F> {
    // s_{i+1} = s_i^2 value mapping corresponds to the index mapping as follows:
    // j-th element in i-th round is s_i, then s_{i+1} is the j%(n/2^{i+1})-th element in the next folded domain
    // our merkle tree at round i has size n/2^i, where n is the original domain size
    let opening_proof = pd
        .commits
        .iter()
        .zip(pd.openings.iter())
        .scan(
            (query_idx, config.init_domain_size / 2),
            |(idx, tree_size), (mt, leaves)| {
                // the index (0/1) of the sibling inside the 2-element leaf
                let sibling_idx = 1 - *idx / *tree_size;

                *idx %= *tree_size;
                *tree_size /= 2;
                Some((leaves[*idx][sibling_idx], mt.generate_proof(*idx)))
            },
        )
        .collect::<Vec<(F, Path<F>)>>();

    let tree_0_size = config.init_domain_size / 2;
    let query_eval = if query_idx < tree_0_size {
        pd.openings[0][query_idx % tree_0_size][0]
    } else {
        pd.openings[0][query_idx % tree_0_size][1]
    };

    QueryProof {
        query_eval,
        opening_proof,
    }
}

/// Proving low-degree of a polynomial given its evaluation on D
pub fn prove<F>(config: &FriConfig, evals: Evaluations<F, Radix2EvaluationDomain<F>>) -> FriProof<F>
where
    F: FftField,
{
    let mut merlin = config.io.to_merlin();
    let (proof, _extra) = prove_internal(&mut merlin, config, evals, &[]);
    proof
}

/// Core prover logic, shared between single-instance and batched prover
/// We support extra queries on the same oracles: copy over if they were already queried in FRI proof
pub(crate) fn prove_internal<F: FftField>(
    merlin: &mut Merlin,
    config: &FriConfig,
    evals: Evaluations<F, Radix2EvaluationDomain<F>>,
    extra_query_indices: &[usize],
) -> (FriProof<F>, Vec<QueryProof<F>>) {
    let domain_0_size = config.init_domain_size;

    // Commit phase
    let pd = commit_phase(config, evals, merlin);

    // Perform Proof-of-work grinding
    merlin
        .challenge_pow::<Blake3PoW>(config.pow_bits as f64)
        .unwrap();

    // Query Phase
    // randomly sampled s_0 for each query, must be sequential squeezing
    let s_0_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| usize::from_le_bytes(merlin.challenge_bytes().unwrap()) % domain_0_size)
        .collect();

    // for each query, generate the query proof
    let query_proofs = s_0_indices
        .par_iter()
        .map(|&s_0_idx| answer_query(config, &pd, s_0_idx))
        .collect::<Vec<_>>();

    // Producing more query proofs for extra query points
    let extra_query_proofs = extra_query_indices
        .par_iter()
        .map(|&idx| {
            if let Some(pos) = s_0_indices.par_iter().position_first(|&i| i == idx) {
                query_proofs[pos].clone()
            } else {
                answer_query(config, &pd, idx)
            }
        })
        .collect();

    (
        FriProof {
            transcript: merlin.transcript().to_owned(),
            query_proofs,
            batching_proofs: None,
        },
        extra_query_proofs,
    )
}

/// Batched FRI, receiving a matrix of evaluations (each row corresponds to evals of a separate polynomial)
/// Sec 8.2 of https://eprint.iacr.org/2020/654.pdf
pub fn batch_prove<F: FftField>(config: &FriConfig, evals: &Matrix<F>) -> FriProof<F> {
    let mut merlin = config.io.to_merlin();

    let (proof, _extra_query_proofs, _extra_batching_proofs) =
        batch_prove_internal(&mut merlin, config, evals, &[]);
    proof
}

/// Core logic of batched FRI, optionally accept querying extra indices, copy over if they were already queried in FRI proof
pub(crate) fn batch_prove_internal<F: FftField>(
    merlin: &mut Merlin,
    config: &FriConfig,
    evals: &Matrix<F>,
    extra_query_indices: &[usize],
) -> (FriProof<F>, Vec<QueryProof<F>>, Vec<BatchedColProof<F>>) {
    // first commit all batches column wise
    let leaves = evals.par_col().collect::<Vec<Vec<_>>>();
    let mt = SymbolMerkleTree::<F>::from_slice(&leaves);
    merlin.add_bytes(&mt.root()).unwrap();

    // derive the random combiner alpha, and linear combine all rows
    let [alpha]: [F; 1] = merlin.challenge_scalars().unwrap();
    let alpha_powers: Vec<F> = successors(Some(F::ONE), |&prev| Some(prev * alpha))
        .take(evals.height())
        .collect();
    let batched_eval: Vec<F> = alpha_powers
        .par_iter()
        .zip(evals.par_row_enumerate())
        .map(|(&alpha_pow, (_, evals))| {
            evals
                .par_iter()
                .map(|eval| alpha_pow * eval)
                .collect::<Vec<F>>()
        })
        .reduce(
            || Vec::with_capacity(evals.width()),
            |mut acc, v| {
                if acc.is_empty() {
                    acc = v;
                } else {
                    acc.par_iter_mut()
                        .zip(v.par_iter())
                        .for_each(|(a, v)| *a += v);
                }
                acc
            },
        );

    // run the single-instance FRI on the batched/aggregated evals
    let evals = Evaluations::from_vec_and_domain(
        batched_eval,
        Radix2EvaluationDomain::new(config.init_domain_size).unwrap(),
    );
    let (mut proof, extra_query_proofs) =
        prove_internal(merlin, config, evals, extra_query_indices);

    // Update all the query opening proof on batched matrix to prove correct batching
    let fri_query_indices = TranscriptData::<F>::parse(config, &proof.transcript).query_indices;
    let batching_proofs = fri_query_indices
        .par_iter()
        .map(|&idx| BatchedColProof {
            query_col_evals: leaves[idx].clone(),
            opening_proof: mt.generate_proof(idx),
        })
        .collect::<Vec<_>>();
    proof.batching_proofs = Some(batching_proofs.clone());

    // producing correct batching proof for the extra query indices (if not included already)
    let extra_batching_proofs = extra_query_indices
        .par_iter()
        .map(|&idx| {
            if let Some(pos) = fri_query_indices.par_iter().position_first(|&i| i == idx) {
                batching_proofs[pos].clone()
            } else {
                BatchedColProof {
                    query_col_evals: leaves[idx].clone(),
                    opening_proof: mt.generate_proof(idx),
                }
            }
        })
        .collect();

    (proof, extra_query_proofs, extra_batching_proofs)
}

/// Verifying a FRI proof of low-degreeness
pub fn verify<F>(config: &FriConfig, proof: &FriProof<F>) -> bool
where
    F: FftField,
{
    // Parse messages being exchanged
    let TranscriptData {
        batch_commit,
        alpha,
        round_commits,
        round_chals,
        final_poly,
        query_indices,
    } = TranscriptData::parse(config, &proof.transcript);

    // Verify each query proof in parallel
    let mut verified = query_indices
        .par_iter()
        .zip(proof.query_proofs.par_iter())
        .all(|(&s_0_idx, query_proof)| {
            query_proof.verify(config, &round_commits, &round_chals, s_0_idx, &final_poly)
        });

    // for batched FRI, verify the batching on queried columns
    if let Some(batching_proofs) = &proof.batching_proofs {
        let batch_commit = batch_commit.unwrap_or(vec![]);
        let alpha = alpha.unwrap_or(F::ONE);

        verified &= query_indices
            .par_iter()
            .zip(batching_proofs.par_iter())
            .zip(proof.query_proofs.par_iter())
            .all(|((&idx, col_proof), query_proof)| {
                col_proof.verify(config, &batch_commit, &alpha, &query_proof.query_eval, idx)
            });
    }
    verified
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_rng;
    use ark_bn254::Fr;
    use ark_std::UniformRand;

    #[test]
    fn test_batched_fri() {
        let rng = &mut test_rng();
        let log_blowup = 2;
        let num_queries = 20;
        let pow_bits = 3;
        let msg_len = 32;
        let num_batches = 5;
        let init_domain_size = msg_len * (1 << log_blowup);
        let domain = Radix2EvaluationDomain::<Fr>::new(init_domain_size).unwrap();
        let evals = (0..num_batches)
            .map(|_| {
                let coeffs = (0..init_domain_size / (1 << log_blowup))
                    .map(|_| Fr::rand(rng))
                    .collect::<Vec<_>>();
                domain.fft(&coeffs)
            })
            .flatten()
            .collect();
        let matrix = Matrix::new(evals, init_domain_size, num_batches).unwrap();

        let config =
            FriConfig::new_unchecked::<Fr>(msg_len, log_blowup, num_queries, pow_bits, num_batches);
        let fri_proof = super::batch_prove(&config, &matrix);
        assert!(super::verify(&config, &fri_proof));

        let mut bad_proof = fri_proof.clone();
        let mut bad_batching_proof = fri_proof.batching_proofs.clone().unwrap();
        bad_batching_proof[0].query_col_evals[0] += Fr::rand(rng);
        bad_proof.batching_proofs = Some(bad_batching_proof);
        assert!(!super::verify(&config, &bad_proof));
    }

    #[test]
    fn test_fri() {
        let rng = &mut test_rng();
        let log_blowup = 1;
        let num_queries = 5;
        let pow_bits = 3;
        let msg_len = 32;
        let num_batches = 1;
        let init_domain_size = msg_len * (1 << log_blowup);
        let domain = Radix2EvaluationDomain::new(init_domain_size).unwrap();
        let coeffs = (0..init_domain_size / (1 << log_blowup))
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let evals = domain.fft(&coeffs);

        let config =
            FriConfig::new_unchecked::<Fr>(msg_len, log_blowup, num_queries, pow_bits, num_batches);
        let fri_proof = super::prove(&config, Evaluations::from_vec_and_domain(evals, domain));
        assert!(super::verify(&config, &fri_proof));

        let mut bad_proof = fri_proof.clone();
        bad_proof.query_proofs[0].opening_proof[0].0 = Fr::rand(rng);
        assert!(!super::verify(&config, &bad_proof));
    }

    #[test]
    #[should_panic]
    fn test_fri_on_bad_inputs() {
        let rng = &mut test_rng();
        let log_blowup = 1;
        let num_queries = 5;
        let pow_bits = 2;
        let num_batches = 1;
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
            num_batches,
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
