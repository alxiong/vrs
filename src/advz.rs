//! Univariate-KZG based VRS
//! Each row is a polynomial of degree (k-1), encoded by evaluating at n points;
//! All L rows are aggregated into a single polynomial using random linearly combination;
//! Run a single multi-evaluation on the aggregated polynomial to generate n proofs.
//!
//! References:
//! - Appendix A.2 in https://eprint.iacr.org/2024/1189
//! - Algorithm 1 in https://eprint.iacr.org/2021/1500

use crate::{
    matrix::Matrix,
    merkle_tree::{self, SymbolMerkleTree, SymbolMerkleTreeParams},
    multi_evals::univariate::multi_eval,
    VerifiableReedSolomon, VrsError, VrsShare,
};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{
    end_timer,
    fmt::Debug,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
};
use jf_pcs::{prelude::*, univariate_kzg::UnivariateKzgPCS, PolynomialCommitmentScheme};
use p3_maybe_rayon::prelude::*;

/// A univariate-KZG based VRS, see module doc.
#[derive(Debug, Clone)]
pub struct AdvzVRS<E> {
    _pairing: PhantomData<E>,
}

impl<F, E> VerifiableReedSolomon<F> for AdvzVRS<E>
where
    F: FftField,
    E: Pairing<ScalarField = F>,
{
    type PublicParams = UnivariateUniversalParams<E>;
    type ProverKey = AdvzVRSProverKey<F, E>;
    type VerifierKey = AdvzVRSVerifierKey<E>;
    // (merkle_root, [row_poly_cms])
    type Commitment = (
        <SymbolMerkleTreeParams<F> as merkle_tree::Config>::InnerDigest,
        Vec<Commitment<E>>,
    );
    // proof consists of (merkle_path, evaluation_proof)
    type Proof = (merkle_tree::Path<F>, UnivariateKzgProof<E>);

    fn setup<R>(
        max_y_degree: usize,
        _max_x_degree: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        UnivariateUniversalParams::gen_srs_for_testing(rng, max_y_degree).map_err(VrsError::from)
    }

    fn preprocess(
        pp: &Self::PublicParams,
        y_degree: usize,
        x_degree: usize,
        eval_domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        let (pk, vk) = UnivariateUniversalParams::trim(pp, y_degree).map_err(VrsError::from)?;
        Ok((
            AdvzVRSProverKey {
                width: y_degree + 1,
                height: x_degree + 1,
                pcs_pk: pk,
                domain: eval_domain.to_owned(),
            },
            AdvzVRSVerifierKey {
                width: y_degree + 1,
                height: x_degree + 1,
                pcs_vk: vk,
                domain_size: eval_domain.size as usize,
            },
        ))
    }

    fn compute_shares(
        pk: &AdvzVRSProverKey<F, E>,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
        if data.height() != pk.height || data.width() != pk.width {
            return Err(VrsError::InvalidParam("mismatched data size".to_string()));
        }
        let row_polys = data.to_row_uv_polys();
        assert_eq!(row_polys.len(), pk.height);

        let total_time = start_timer!(|| ark_std::format!(
            "AdvzVRS::compute shares (k={}, n={}, L={})",
            pk.width,
            pk.domain.size,
            pk.height
        ));

        // 1. encode kxL into nxL matrix (row-wise FFT)
        let encode_time = start_timer!(|| "encode data");
        let encoded = Self::interleaved_rs_encode(data, &pk.domain)?;
        end_timer!(encode_time);

        // 2. construct a merkle tree from the encoded matrix, each leaf is a L-size vector
        // TODO: directly pass in an iterator to MerkleTree constructor instead of collect it first
        let leaves = encoded.par_col().collect::<Vec<_>>();
        let mt = SymbolMerkleTree::<F>::new(leaves);

        // 3. from the merkle root, derive a random scalar to linearly combine k row-wise poly into 1 aggregated poly
        let r = mt.root_as_field();
        // aggregated_poly = r^0 * poly_0 + r^1 * poly_1 + ... + r^{L-1} * poly_{L-1}
        let mut r_i = F::ONE;
        let agg_poly = row_polys
            .iter()
            .skip(1)
            .fold(row_polys[0].clone(), |acc, poly| {
                r_i *= r;
                acc + poly * r_i
            });

        // 4 commit each row using pcs
        let commit_time = start_timer!(|| "commit row_poly");
        let row_poly_cms = row_polys
            .par_iter()
            .map(|poly| {
                UnivariateKzgPCS::commit(&pk.pcs_pk, poly).expect("fail to commit row_poly")
            })
            .collect();
        end_timer!(commit_time);
        let cm = (mt.root(), row_poly_cms);

        // 5. run multi-evaluation on the aggregated poly
        let multi_eval_time = start_timer!(|| "multi-eval on agg_poly");
        let eval_proofs = multi_eval(&pk.pcs_pk, &agg_poly, &pk.domain);
        let shares = eval_proofs
            .into_par_iter()
            .zip(encoded.par_col_enumerate())
            .map(|(eval_proof, (col_idx, data))| {
                let mt_proof = mt.generate_proof(col_idx);

                VrsShare {
                    data,
                    proof: (mt_proof, eval_proof),
                }
            })
            .collect::<Vec<_>>();

        end_timer!(multi_eval_time);
        end_timer!(total_time);
        Ok((cm, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        if idx >= vk.domain_size {
            return Err(VrsError::InvalidParam("share idx out-of-bound".to_string()));
        }
        if share.data.len() != vk.height {
            return Err(VrsError::InvalidParam(
                "share data len mismatch".to_string(),
            ));
        }

        let verify_time = start_timer!(|| "verify share");
        let path = &share.proof.0;
        let eval_proof = &share.proof.1;
        let root = &comm.0;
        let row_poly_cms: Vec<_> = comm.1.par_iter().map(|x| x.0).collect();

        // 1. verify merkle proof
        if !path.verify(root, idx, share.data.as_slice()) {
            return Ok(false);
        }

        // 2. compute scalar and randomly combine row_poly_cms to derive commitment to the agg_poly homomorphically
        let r = F::from_base_prime_field(
            <F::BasePrimeField as PrimeField>::from_le_bytes_mod_order(root),
        );
        let mut r_i = F::ONE;
        let mut r_pows = Vec::with_capacity(vk.height);
        r_pows.push(r_i);
        for _ in 1..vk.height {
            r_i *= r;
            r_pows.push(r_i);
        }
        let agg_poly_cm = <E::G1 as VariableBaseMSM>::msm(&row_poly_cms, &r_pows)
            .expect("fail to compute agg_poly_cm")
            .into_affine();

        // 3. verify evaluation proof against agg_poly_cm
        let point = Radix2EvaluationDomain::new(vk.domain_size)
            .unwrap()
            .element(idx);
        let agg_eval = r_pows
            .iter()
            .zip(share.data.iter())
            .map(|(&a, &b)| a * b)
            .fold(F::ZERO, |acc, x| acc + x);
        if !UnivariateKzgPCS::verify(
            &vk.pcs_vk,
            &Commitment(agg_poly_cm),
            &point,
            &agg_eval,
            eval_proof,
        )? {
            return Ok(false);
        }
        end_timer!(verify_time);
        Ok(true)
    }
}

/// Prover key for `AdvzVRS`
#[derive(Debug, Clone)]
pub struct AdvzVRSProverKey<F: FftField, E: Pairing<ScalarField = F>> {
    /// k: width of original payload
    pub width: usize,
    /// L: height of original payload
    pub height: usize,
    /// PCS proving key supporting degree <= k
    pub pcs_pk: UnivariateProverParam<E>,
    /// Evaluation domain, with domain_size = n
    pub domain: Radix2EvaluationDomain<F>,
}

/// Verifier key for `AdvzVRS`
#[derive(Debug, Clone)]
pub struct AdvzVRSVerifierKey<E: Pairing> {
    /// k: width of original payload
    pub width: usize,
    /// L: height of original payload
    pub height: usize,
    /// PCS verification key supporting degree <= k
    pub pcs_vk: UnivariateVerifierParam<E>,
    /// n: total evaluation domain size (= num_nodes)
    pub domain_size: usize,
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};

    use super::*;
    use crate::test_utils::test_rng;
    use ark_std::UniformRand;

    #[test]
    fn test_advz_vrs() {
        let rng = &mut test_rng();
        let k = 2usize.pow(10);
        let l = 8;
        let n = 2usize.pow(11);

        let pp = AdvzVRS::<Bn254>::setup(k - 1, l - 1, rng).unwrap();
        let domain = Radix2EvaluationDomain::new(n).unwrap();
        let (pk, vk) = AdvzVRS::preprocess(&pp, k - 1, l - 1, &domain).unwrap();

        let data = (0..k * l).map(|_| Fr::rand(rng)).collect();
        let data = Matrix::new(data, k, l).unwrap();
        let (cm, shares) = AdvzVRS::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(AdvzVRS::verify_share(&vk, &cm, idx, share).unwrap());
        }
    }
}
