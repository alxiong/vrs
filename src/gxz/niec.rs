//! Non-interactive Evaluation Consolidation (NIEC) Protocol in GXZ'25

use ark_crypto_primitives::crh::CRHScheme;
use ark_ff::{FftField, Field};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial,
    Radix2EvaluationDomain,
};
use ark_serialize::*;
use ark_std::{borrow::Borrow, iter::successors};
use nimue::{
    plugins::ark::{FieldChallenges, FieldIOPattern},
    ByteReader, ByteWriter, DefaultHash, IOPattern,
};
use p3_maybe_rayon::prelude::*;

use crate::merkle_tree::{Path, Sha256OnFields, SymbolMerkleTree};

/// alias for a point in the domain of a Multilinear Polynomial, len = nv
type Point<F> = Vec<F>;

/// Configuration/Public parameter of consolidation
#[derive(Clone, Debug)]
pub struct ConsolidationConfig<F: FftField> {
    /// the evaluation domain
    pub domain: Radix2EvaluationDomain<F>,
    /// Number of variable of the multilinear poly, 2^nv = k (the degree of the twin univariate poly)
    pub nv: usize,
    /// Step size: the number of variable randomized at each step/round,
    /// the univariate poly sent each round is of degree 2^s-1.
    pub s: usize,
    /// The prover-verifier interaction
    pub(crate) io: IOPattern,
}

impl<F: FftField> ConsolidationConfig<F> {
    /// construct a new config
    pub fn new(num_vars: usize, domain_size: usize, step_size: usize) -> Self {
        assert!(
            num_vars % step_size == 0,
            "nv should be a multiple of step_size"
        );
        let num_rounds = num_vars / step_size;
        // NOTE: since `IOPattern` already domain-separate each message, thus many public parameters
        // are implicitly enforced by the pattern description (incl all other fields of the config),
        // thuse we don't explicitly add them again to the transcript, and would still be secure
        // against weak-FS vulnerability.
        let mut io = IOPattern::<DefaultHash>::new("EvalConsolidation");
        for i in 1..=num_rounds {
            io = io.absorb(32, &format!("root{}", i));
            io = FieldIOPattern::<F>::challenge_scalars(io, 1, &format!("r_{}", i));
        }

        Self {
            domain: Radix2EvaluationDomain::new(domain_size).unwrap(),
            nv: num_vars,
            s: step_size,
            io,
        }
    }
}

/// Non-interactive Proof of correct consolidation
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize, Default)]
pub struct ConsolidationProof<F: Field> {
    /// Fiat-Shamir transcript containing one merkle_root per round
    pub transcript: Vec<u8>,
    /// One univariate polynomial (in coeff form) per round sent by the prover
    pub round_polys: Vec<Vec<F>>,
    /// One merkle proof per round for the consolidation
    pub round_proofs: Vec<Path<F>>,
}

/// Consolidating statements about evaluations at n points to a single statement of a random point
///
/// - `evals`: evaluations of f(z_j) where z_j = (omega^j, omega^2j, omega^4j, ..., omega^{2^{nv-1}j}) and f is the MLP
///
/// Returns the consolidated evaluation point r, and proofs for all j \in [n]
pub fn consolidate<F: FftField>(
    pp: &ConsolidationConfig<F>,
    evals: &[F],
) -> (Point<F>, Vec<ConsolidationProof<F>>) {
    let domain_size = pp.domain.size as usize;
    assert_eq!(evals.len(), domain_size);
    // degree of the polynomial sent in each round = num_evals - 1
    let num_evals = 1 << pp.s;

    let mut merlin = pp.io.to_merlin();
    let mut proofs = vec![ConsolidationProof::default(); domain_size];
    let mut chals: Vec<F> = vec![]; // all challenge scalars

    // Round 1:
    // there are n/2^s unique univariate polynomials sent to n verifiers in the first round
    let num_uv_polys = domain_size / (1 << pp.s);
    let dist_between_coset_elems = num_uv_polys; // same value, better name for readability

    // prepare each univariate by interpolate from evaluations,
    // with small s, no need to run full-blown IFFT
    let round_1_polys = (0..num_uv_polys)
        .into_par_iter()
        .map(|j| {
            let offset = pp.domain.element(j);
            let evals: Vec<F> = (0..num_evals)
                .map(|l| evals[j + l * dist_between_coset_elems])
                .collect();
            interpolate(&evals, offset)
        })
        .collect::<Vec<Vec<F>>>();

    let round_1_mt = SymbolMerkleTree::<F>::from_slice(&round_1_polys);
    // append the merkle root to the transcript
    merlin.add_bytes(&round_1_mt.root()).unwrap();

    // there are only `num_uv_polys` unique merkle paths, generate them first,
    let round_1_mt_proofs = (0..num_uv_polys)
        .into_par_iter()
        .map(|j| round_1_mt.generate_proof(j))
        .collect::<Vec<_>>();
    // then update the consolidation proof for all nodes
    proofs.par_iter_mut().enumerate().for_each(|(j, proof)| {
        proof
            .round_polys
            .push(round_1_polys[j % num_uv_polys].clone());
        proof
            .round_proofs
            .push(round_1_mt_proofs[j % num_uv_polys].clone());
    });
    drop(round_1_mt_proofs);

    // Round 2 to nv/s
    let num_rounds = pp.nv / pp.s;
    let mut last_round_polys = round_1_polys;
    for i in 1..num_rounds {
        let [r]: [F; 1] = merlin.challenge_scalars().unwrap();
        chals.push(r);

        let num_uv_polys = domain_size / (1 << (pp.s * (i + 1)));
        let dist_between_coset_elems = num_uv_polys;

        let round_i_polys = (0..num_uv_polys)
            .into_par_iter()
            .map(|j| {
                // offset is omega^(2^i * j)
                let offset = pp.domain.element(j * (1 << (pp.s * i)));
                let evals = (0..num_evals)
                    .map(|l| {
                        let poly = DensePolynomial::from_coefficients_slice(
                            &last_round_polys[j + l * dist_between_coset_elems],
                        );
                        poly.evaluate(&r)
                    })
                    .collect::<Vec<F>>();
                interpolate(&evals, offset)
            })
            .collect::<Vec<Vec<F>>>();

        if num_uv_polys > 1 {
            let round_i_mt = SymbolMerkleTree::from_slice(&round_i_polys);
            merlin.add_bytes(&round_i_mt.root()).unwrap();

            let round_i_proofs = (0..num_uv_polys)
                .into_par_iter()
                .map(|j| round_i_mt.generate_proof(j))
                .collect::<Vec<_>>();

            proofs.par_iter_mut().enumerate().for_each(|(j, proof)| {
                proof
                    .round_polys
                    .push(round_i_polys[j % num_uv_polys].clone());
                proof
                    .round_proofs
                    .push(round_i_proofs[j % num_uv_polys].clone());
            });
            last_round_polys = round_i_polys;
        } else {
            // final round is a single poly, thus we don't use merkle tree any more, simply hash it
            assert_eq!(i, num_rounds);
            assert_eq!(num_uv_polys, 1);
            let hash = Sha256OnFields::evaluate(&(), round_i_polys[0].borrow()).unwrap();
            merlin.add_bytes(&hash).unwrap();
        }
    }

    // Derive the last challenge and construct the consolidated point
    let [r_last]: [F; 1] = merlin.challenge_scalars().unwrap();
    chals.push(r_last);
    assert_eq!(chals.len(), num_rounds);

    // r = (r1, r1^2, r1^4, ..., r1^{2^{s-1}}, r2, r2^2, ..., r2^{2^{s-1}}, ...)
    let consolidated_point = derive_consolidated_point(&chals, pp.s);

    // prepare full consolidation proof for all nodes
    let transcript = merlin.transcript().to_vec();
    proofs
        .par_iter_mut()
        .for_each(|proof| proof.transcript = transcript.clone());

    (consolidated_point, proofs)
}

/// Verify the consolidation for `idx`-th party
/// - `eval` is the original claimed evaluation (NOT the evaluation on the consolidated point)
pub fn verify<F: FftField>(
    pp: &ConsolidationConfig<F>,
    idx: usize,
    eval: &F,
    consolidated_point: &Point<F>,
    proof: &ConsolidationProof<F>,
) -> bool {
    assert_eq!(proof.round_polys.len(), pp.nv / pp.s);
    assert_eq!(proof.round_polys.len(), proof.round_proofs.len());

    let mut arthur = pp.io.to_arthur(&proof.transcript);
    let mut chals: Vec<F> = vec![]; // all challenge scalars

    // z_j = (omega^j, omega^2j, omega^4j, .. , omega^{2^{nv-1}*j})
    let eval_point: Point<F> =
        successors(Some(pp.domain.element(idx)), |&prev| Some(prev.square()))
            .take(pp.nv)
            .collect();

    let domain_size = pp.domain.size as usize;
    let num_uv_polys = domain_size / (1 << pp.s);

    // First round verification
    let round_1_poly = DensePolynomial::from_coefficients_slice(&proof.round_polys[0]);
    let mt_root = arthur.next_bytes::<32>().unwrap().to_vec();
    let mut verified = round_1_poly.evaluate(&eval_point[0]) == *eval;
    verified &= proof.round_proofs[0].verify(&mt_root, idx % num_uv_polys, round_1_poly.coeffs());

    // Round 2 to nv/s
    let num_rounds = pp.nv / pp.s;
    let mut last_round_poly = round_1_poly;

    for i in 1..num_rounds {
        let num_uv_polys = domain_size / (1 << (pp.s * (i + 1)));

        let [r]: [F; 1] = arthur.challenge_scalars().unwrap();
        chals.push(r);
        let round_i_poly = DensePolynomial::from_coefficients_slice(&proof.round_polys[i]);
        let round_i_root = arthur.next_bytes::<32>().unwrap().to_vec();

        verified &= last_round_poly.evaluate(&r) == round_i_poly.evaluate(&eval_point[pp.s * i]);
        if num_uv_polys > 1 {
            verified &= proof.round_proofs[i].verify(
                &round_i_root,
                idx % num_uv_polys,
                round_i_poly.coeffs(),
            );
            last_round_poly = round_i_poly;
        } else {
            // last round only have a single poly, no merkle tree is used, just check hash
            let hash = Sha256OnFields::evaluate(&(), round_i_poly.coeffs()).unwrap();
            verified &= round_i_root == hash;
        }
    }
    let [r_last]: [F; 1] = arthur.challenge_scalars().unwrap();
    chals.push(r_last);

    verified &= derive_consolidated_point(&chals, pp.s) == *consolidated_point;

    verified
}

// given r1, r2, ... r_{nv/s}, returns
// r = (r1, r1^2, r1^4, ..., r1^{2^{s-1}}, r2, r2^2, ..., r2^{2^{s-1}}, ...)
fn derive_consolidated_point<F: FftField>(chals: &[F], s: usize) -> Point<F> {
    chals
        .into_par_iter()
        .flat_map(|r| {
            successors(Some(*r), |&prev| Some(prev.square()))
                .take(s)
                .collect::<Vec<F>>()
        })
        .collect()
}

/// Interpolate evaluations of FFT points over cosets, return the coefficients
/// For small dimension, we use hand-calculated closed-form formula,
/// for larger dim (n>=8), we use generic IFFT from arkworks.
fn interpolate<F: FftField>(evals: &[F], offset: F) -> Vec<F> {
    let n = evals.len();
    assert!(n.is_power_of_two(), "eval len should be power_of_two");
    assert!(n > 1, "don't interpolate constant poly");

    match n {
        2 => {
            let two = F::from(2);
            let c0 = (evals[0] + evals[1]) / two;
            let c1 = (evals[0] - evals[1]) / (two * offset);
            vec![c0, c1]
        },
        4 => {
            let four_inv = F::from(4).inverse().unwrap();
            let offset_inv = offset.inverse().unwrap();
            let offset_inv_square = offset_inv.square();
            // omega^4 = 1, omega^2 = -1, omega^3=-omega
            let omega = F::get_root_of_unity(4).unwrap();
            let eval_0_and_2 = evals[0] + evals[2];
            let eval_1_and_3 = evals[1] + evals[3];
            let eval_0_minus_2 = evals[0] - evals[2];
            let eval_1_minus_3 = evals[1] - evals[3];

            let c0 = (eval_0_and_2 + eval_1_and_3) * four_inv;
            let c1 = (eval_0_minus_2 - omega * eval_1_minus_3) * four_inv * offset_inv;
            let c2 = (eval_0_and_2 - eval_1_and_3) * four_inv * offset_inv_square;
            let c3 = (eval_0_minus_2 + omega * eval_1_minus_3)
                * four_inv
                * offset_inv
                * offset_inv_square;
            vec![c0, c1, c2, c3]
        },
        _ => {
            let coset = Radix2EvaluationDomain::new_coset(n, offset).unwrap();
            coset.ifft(&evals)
        },
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};

    use crate::test_utils::test_rng;

    use super::*;

    #[test]
    fn test_interpolation() {
        let rng = &mut test_rng();
        let n = 32;
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let elems: Vec<Fr> = domain.elements().collect();

        for s in 1..10 {
            let num_coeffs = 1 << s;
            let p = DensePolynomial::rand(num_coeffs - 1, rng);

            let step_size = n / num_coeffs;
            for j in 0..step_size {
                let mut evals = vec![];
                for l in 0..(1 << s) {
                    evals.push(p.evaluate(&elems[j + l * step_size]));
                }

                let coeffs = interpolate(&evals, elems[j]);
                assert_eq!(coeffs, p.coeffs);
            }
        }
    }

    #[test]
    fn test_consolidation() {
        let rng = &mut test_rng();

        let nv = 12;
        let k = 1 << nv;
        let poly = DensePolynomial::<Fr>::rand(k, rng);

        let domain_size = 1 << (nv + 1);
        let domain = Radix2EvaluationDomain::<Fr>::new(domain_size).unwrap();
        let evals = domain.fft(&poly.coeffs);

        for step_size in [2, 3] {
            let pp = ConsolidationConfig::new(nv, domain_size, step_size);
            let (consolidated_point, proofs) = consolidate(&pp, &evals);
            assert_eq!(proofs.len(), domain_size);

            evals
                .iter()
                .zip(proofs.iter())
                .enumerate()
                .for_each(|(j, (eval, proof))| {
                    assert!(verify(&pp, j, eval, &consolidated_point, proof))
                });
        }
    }
}
