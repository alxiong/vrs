//! FRIDA: IOPP-based DAS
//!
//! Reference:
//! - https://eprint.iacr.org/2024/248

use crate::{
    iopp::fri::{self, BatchedColProof, FriConfig, FriProof, QueryProof, TranscriptData},
    matrix::Matrix,
    VerifiableReedSolomon, VrsError, VrsShare,
};
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::*;
use ark_std::{
    end_timer,
    fmt::Debug,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
};

/// FRIDA-based VRS
#[derive(Debug, Clone)]
pub struct FridaVRS<F: FftField> {
    _field: PhantomData<F>,
}

impl<F: FftField> VerifiableReedSolomon<F> for FridaVRS<F> {
    type PublicParams = FriConfig;
    // (fri_config, domain)
    type ProverKey = (FriConfig, Radix2EvaluationDomain<F>);
    type VerifierKey = FriConfig;
    type Commitment = FriProof<F>;
    type Proof = FridaProof<F>;

    fn setup<R>(
        max_row_degree: usize,
        max_col_degree: usize,
        _rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        let log_blowup = 2; // P26 of FRIDA choose blowup = 4
        let fri_config = FriConfig::new_conjectured::<F>(
            max_row_degree + 1,
            log_blowup,
            None,
            None,
            Some(max_col_degree + 1),
        );
        Ok(fri_config)
    }

    fn preprocess(
        pp: &Self::PublicParams,
        row_degree: usize,
        col_degree: usize,
        eval_domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        assert_eq!(
            (eval_domain.size as usize) << (pp.log_blowup - 1),
            pp.init_domain_size
        );
        assert_eq!(pp.msg_len, row_degree + 1);
        assert_eq!(pp.num_batches, col_degree + 1);

        Ok(((pp.clone(), eval_domain.to_owned()), pp.clone()))
    }

    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
        assert_eq!(data.width(), pk.0.msg_len);
        assert_eq!(data.height(), pk.0.num_batches);

        let total_time = start_timer!(|| ark_std::format!(
            "Frida::compute shares (k={}, n={}, L={}, blowup={})",
            data.width(),
            pk.1.size,
            data.height(),
            pk.0.blowup(),
        ));
        let init_domain_size = pk.0.init_domain_size;

        // 1. encode kxL into nxL matrix (row-wise FFT)
        let encode_time = start_timer!(|| "encode data (and prepare FRI evals)");
        let evals = if init_domain_size == pk.1.size as usize {
            Self::interleaved_rs_encode(data, &pk.1)?
        } else {
            let domain = Radix2EvaluationDomain::new(init_domain_size).unwrap();
            Self::interleaved_rs_encode(data, &domain)?
        };
        end_timer!(encode_time);

        // 2. compute proofs for all columns (one per node/replica)
        let node_indices = (0..init_domain_size).collect::<Vec<_>>();

        let prover_time = start_timer!(|| "batched FRI prove");
        let mut merlin = pk.0.io.to_merlin();
        let (fri_proof, extra_query_proofs, extra_batching_proofs) =
            fri::batch_prove_internal(&mut merlin, &pk.0, &evals, &node_indices);
        end_timer!(prover_time);

        // 3. finally, assemble all shares
        let shares = evals
            .col()
            .step_by(init_domain_size / pk.1.size as usize)
            .zip(
                extra_query_proofs
                    .into_iter()
                    .zip(extra_batching_proofs.into_iter()),
            )
            .map(
                |(col_data, (selected_query_proof, selected_batching_proof))| VrsShare {
                    data: col_data,
                    proof: FridaProof {
                        selected_query_proof,
                        selected_batching_proof,
                    },
                },
            )
            .collect();

        end_timer!(total_time);

        Ok((fri_proof, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        // verify the commitment (which itself is a FRI proof)
        let mut verified = fri::verify(vk, comm);

        // verify the frida proof on particular queried column
        let TranscriptData {
            batch_commit,
            alpha,
            round_commits,
            round_chals,
            final_poly,
            query_indices: _indices,
        } = TranscriptData::parse(vk, &comm.transcript);

        verified &= share.proof.selected_query_proof.verify(
            vk,
            &round_commits,
            &round_chals,
            idx,
            &final_poly,
        );
        verified &= share.proof.selected_batching_proof.verify(
            vk,
            &batch_commit.unwrap_or(vec![]),
            &alpha.unwrap_or(F::ONE),
            &share.proof.selected_query_proof.query_eval,
            idx,
        );

        Ok(verified)
    }
}

/// Opening proof in FRIDA.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct FridaProof<F: FftField> {
    /// The specific QSelect query proof for all rounds.
    pub selected_query_proof: QueryProof<F>,
    /// The specific QSelect batching proof.
    pub selected_batching_proof: BatchedColProof<F>,
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use super::*;
    use crate::test_utils::test_rng;
    use ark_std::UniformRand;

    #[test]
    fn test_frida_vrs() {
        let rng = &mut test_rng();
        let k = 2usize.pow(10);
        let l = 8;
        let n = 2usize.pow(11);

        let pp = FridaVRS::<Fr>::setup(k - 1, l - 1, rng).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let (pk, vk) = FridaVRS::preprocess(&pp, k - 1, l - 1, &domain).unwrap();

        let data = (0..k * l).map(|_| Fr::rand(rng)).collect();
        let data = Matrix::new(data, k, l).unwrap();
        let (cm, shares) = FridaVRS::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(FridaVRS::verify_share(&vk, &cm, idx, share).unwrap());
        }
    }
}
