//! Transparent VRS based on evaluation consolidation technique

use crate::{
    gxz::{niec, twin::MultilinearTwin},
    matrix::Matrix,
    merkle_tree::{Path, SymbolMerkleTree},
    poly::bivariate,
    VerifiableReedSolomon, VrsError, VrsShare,
};
use ark_crypto_primitives::crh::sha256::{digest::Digest, Sha256};
use ark_ff::{FftField, Field};
use ark_poly::{univariate, DenseUVPolynomial, Polynomial, Radix2EvaluationDomain};
use ark_serialize::*;
use ark_std::{
    end_timer,
    iter::successors,
    log2,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
};
use derivative::Derivative;
use jf_pcs::{prelude::MLE, PolynomialCommitmentScheme, StructuredReferenceString};
use nimue::{
    plugins::ark::{FieldChallenges, FieldIOPattern},
    ByteReader, ByteWriter, DefaultHash, IOPattern,
};
use p3_maybe_rayon::prelude::*;

use super::niec::{ConsolidationConfig, ConsolidationProof};

/// Transparent GXZ-based VRS pairing evaluation consolidation and a multilinear PCS
#[derive(Derivative)]
#[derivative(Debug, Clone)]
pub struct GxzVRS<F, PCS>
where
    F: FftField,
    PCS: PolynomialCommitmentScheme<Polynomial = MLE<F>>,
{
    _field: PhantomData<F>,
    _pcs: PhantomData<PCS>,
}

impl<F, PCS> VerifiableReedSolomon<F> for GxzVRS<F, PCS>
where
    F: FftField,
    PCS: PolynomialCommitmentScheme<Polynomial = MLE<F>, Point = Vec<F>>,
{
    // only MLPC.pp because need `domain_size` to generate `ConsolidationConfig` which is known at `preprocess()`
    type PublicParams = PCS::SRS;
    /// (NIEC.pp, MLPC.pk, io)
    /// io only contains the single round interaction to derive the partial eval point
    type ProverKey = (
        ConsolidationConfig<F>,
        <PCS::SRS as StructuredReferenceString>::ProverParam,
        IOPattern,
    );
    /// (NIEC.pp, MLPC.vk, io)
    type VerifierKey = (
        ConsolidationConfig<F>,
        <PCS::SRS as StructuredReferenceString>::VerifierParam,
        IOPattern,
    );
    type Commitment = PCS::Commitment;
    type Proof = GxzProof<F, PCS>;

    fn setup<R>(
        max_y_degree: usize,
        max_x_degree: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        let nv_x = log2(max_x_degree + 1) as usize;
        let nv_y = log2(max_y_degree + 1) as usize;
        let nv = nv_x + nv_y;

        let srs = PCS::gen_srs_for_testing(rng, nv)?;
        Ok(srs)
    }

    fn preprocess(
        pp: &Self::PublicParams,
        y_degree: usize,
        x_degree: usize,
        domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        let nv_x = log2(x_degree + 1) as usize;
        let nv_y = log2(y_degree + 1) as usize;
        let nv = nv_x + nv_y;

        // NOTE: hardcode a step size of 2, adjustable.
        let config = ConsolidationConfig::new(nv_y, domain.size as usize, 2);
        let (pcs_pk, pcs_vk) = <PCS::SRS as StructuredReferenceString>::trim(&pp, nv)?;

        let mut io = IOPattern::<DefaultHash>::new("GxzVRS")
            .absorb(32, "mlpc_commit")
            .absorb(32, "mt_root");
        io = FieldIOPattern::<F>::challenge_scalars(io, 1, "beta");

        Ok(((config.clone(), pcs_pk, io.clone()), (config, pcs_vk, io)))
    }

    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
        let domain = &pk.0.domain;
        let domain_size = domain.size as usize;
        let nv_x = log2(data.height()) as usize;
        let mut merlin = pk.2.to_merlin();

        let total_time = start_timer!(|| ark_std::format!(
            "GxzVRS::compute shares (k={}, n={}, L={})",
            data.width(),
            domain_size,
            data.height()
        ));

        // 1. interpret data matrix as bivariate poly (in coeff form), and commit its twin MLE
        let commit_time = start_timer!(|| "commit twin MLE");
        let data_bv_poly = bivariate::DensePolynomial::new(
            data.into_par_row().collect::<Vec<_>>(),
            data.height() - 1,
            data.width() - 1,
        );
        let twin_mle = MLE::from(data_bv_poly.twin()); // simply put inside Arc<_>
        let mle_commit = PCS::commit(&pk.1, &twin_mle)?;
        merlin.add_bytes(&hash_mlpc_commit(&mle_commit)).unwrap();
        end_timer!(commit_time);

        // 2. interleaved encoding
        let encode_time = start_timer!(|| "encode data");
        let encoded = Self::interleaved_rs_encode(data, domain)?;
        end_timer!(encode_time);

        // 3. commit all column-symbols
        let mt_commit_time = start_timer!(|| "merkle commit symbols");
        let leaves = encoded.par_col().collect::<Vec<_>>();
        let mt = SymbolMerkleTree::<F>::new(leaves);
        merlin.add_bytes(&mt.root()).unwrap();
        end_timer!(mt_commit_time);

        // 4. compute all the partial evaluations
        let [beta]: [F; 1] = merlin.challenge_scalars().unwrap();
        let evals: Vec<F> = encoded
            .par_col()
            .map(|col| {
                let col_poly = univariate::DensePolynomial::from_coefficients_vec(col);
                col_poly.evaluate(&beta)
            })
            .collect();
        assert_eq!(evals.len(), domain_size);

        // 4.5 run evaluation consolidation
        let consolidate_time = start_timer!(|| "eval consolidate");
        let (consolidated_point_y, consolidation_proofs) = niec::consolidate(&pk.0, &evals);
        let consolidated_point_x = successors(Some(beta), |&prev| Some(prev.square()))
            .take(nv_x)
            .collect();
        let consolidated_point = [consolidated_point_x, consolidated_point_y].concat();
        end_timer!(consolidate_time);

        // 5. Run MLPC evaluation on the consolidated point
        let mlpc_eval_time = start_timer!(|| "mlpc eval");
        let (mlpc_eval_proof, consolidated_eval) =
            PCS::open(&pk.1, &twin_mle, &consolidated_point)?;
        end_timer!(mlpc_eval_time);

        // prepare all VRS share
        let transcript = merlin.transcript().to_vec();
        let shares = encoded
            .col()
            .zip(consolidation_proofs.into_iter())
            .enumerate()
            .map(|(j, (data, consolidation_proof))| {
                let symbol_mt_proof = mt.generate_proof(j);
                VrsShare {
                    data,
                    proof: GxzProof {
                        transcript: transcript.clone(),
                        consolidation_proof,
                        symbol_mt_proof,
                        mlpc_eval_proof: mlpc_eval_proof.clone(),
                        consolidated_point: consolidated_point.clone(),
                        consolidated_eval: consolidated_eval.clone(),
                    },
                }
            })
            .collect::<Vec<_>>();
        end_timer!(total_time);

        Ok((mle_commit, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        let nv_x = log2(share.data.len()) as usize;
        let mut arthur = vk.2.to_arthur(&share.proof.transcript);

        // 1. verify the symbol merkle proof
        let commit_hash = arthur.next_bytes::<32>().unwrap().to_vec();
        let root = arthur.next_bytes::<32>().unwrap().to_vec();

        let mut verified = hash_mlpc_commit(comm) == commit_hash;
        verified &= share
            .proof
            .symbol_mt_proof
            .verify(&root, idx, share.data.as_slice());

        // 2. derive the partial evaluation point
        let [beta]: [F; 1] = arthur.challenge_scalars().unwrap();
        let partial_eval =
            univariate::DensePolynomial::from_coefficients_slice(&share.data).evaluate(&beta);

        // 3. verify consolidation
        let consolidated_point_x: Vec<F> = successors(Some(beta), |&prev| Some(prev.square()))
            .take(nv_x)
            .collect();
        verified &= share.proof.consolidated_point[..nv_x] == consolidated_point_x[..];

        verified &= niec::verify(
            &vk.0,
            idx,
            &partial_eval,
            &share.proof.consolidated_point[nv_x..],
            &share.proof.consolidation_proof,
        );

        // 4. verify the MLPC evaluation/opening
        verified &= PCS::verify(
            &vk.1,
            comm,
            &share.proof.consolidated_point,
            &share.proof.consolidated_eval,
            &share.proof.mlpc_eval_proof,
        )?;
        Ok(verified)
    }
}

#[inline(always)]
fn hash_mlpc_commit<T: CanonicalSerialize>(commit: &T) -> Vec<u8> {
    let mut bytes = vec![];
    commit.serialize_compressed(&mut bytes).unwrap();
    Sha256::digest(&bytes).to_vec()
}

/// Per-replica Proof in transparent GXZ
#[derive(CanonicalSerialize, CanonicalDeserialize, Derivative)]
#[derivative(Debug, Clone)]
pub struct GxzProof<F: Field, PCS>
where
    F: Field,
    PCS: PolynomialCommitmentScheme<Polynomial = MLE<F>>,
{
    /// transcript containing the MLPC commitment and symbol merkle root
    pub transcript: Vec<u8>,
    /// evaluation consolidation proof
    pub consolidation_proof: ConsolidationProof<F>,
    /// merkle proof for the column/symbol, used to derive partial evaluation point
    pub symbol_mt_proof: Path<F>,
    /// Evaluation proof of the MLP commitment on `consolidated_point`
    pub mlpc_eval_proof: PCS::Proof,
    /// the consolidated evaluation point
    pub consolidated_point: PCS::Point,
    /// the evaluation on `consolidated_point`
    pub consolidated_eval: PCS::Evaluation,
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};

    use super::*;
    use crate::test_utils::test_rng;
    use ark_poly::EvaluationDomain;
    use ark_std::UniformRand;
    use jf_pcs::prelude::MultilinearKzgPCS;

    #[test]
    fn test_gxz_vrs_with_pst() {
        let rng = &mut test_rng();
        let k = 2usize.pow(10);
        let l = 8;
        let n = 2usize.pow(11);

        // using PST multilinear PCS
        let pp = GxzVRS::<Fr, MultilinearKzgPCS<Bn254>>::setup(k - 1, l - 1, rng).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let (pk, vk) =
            GxzVRS::<Fr, MultilinearKzgPCS<Bn254>>::preprocess(&pp, k - 1, l - 1, &domain).unwrap();

        let data = (0..k * l).map(|_| Fr::rand(rng)).collect();
        let data = Matrix::new(data, k, l).unwrap();
        let (cm, shares) =
            GxzVRS::<Fr, MultilinearKzgPCS<Bn254>>::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(
                GxzVRS::<Fr, MultilinearKzgPCS<Bn254>>::verify_share(&vk, &cm, idx, share).unwrap()
            );
        }
    }
}
