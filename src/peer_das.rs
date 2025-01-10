//! PeerDAS: a step towards Ethereum's full DAS
//! PeerDAS is highly similar to ADVZ by treating each row as polynomial and commit them, the main difference is that PeerDAS treats each row polynomial separately
//! and run the multi-evals on all row polynomials instead of one aggregated ones.
//!
//! Reference:
//! - https://eprint.iacr.org/2024/1362.pdf

use crate::{
    matrix::Matrix, multi_evals::univariate::multi_eval, VerifiableReedSolomon, VrsError, VrsShare,
};
use ark_crypto_primitives::crh::sha256::{digest::Digest, Sha256};
use ark_ec::{pairing::Pairing, AffineRepr, VariableBaseMSM};
use ark_ff::{FftField, One, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{
    end_timer,
    fmt::Debug,
    iter::successors,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
};
use jf_pcs::{prelude::*, univariate_kzg::UnivariateKzgPCS, PolynomialCommitmentScheme};
use p3_maybe_rayon::prelude::*;

/// A KZG-based PeerDAS
#[derive(Debug, Clone)]
pub struct PeerDasVRS<E> {
    _pairing: PhantomData<E>,
}

impl<F, E> VerifiableReedSolomon<F> for PeerDasVRS<E>
where
    F: FftField,
    E: Pairing<ScalarField = F>,
{
    type PublicParams = UnivariateUniversalParams<E>;
    // pcs_pk, domain
    type ProverKey = (UnivariateProverParam<E>, Radix2EvaluationDomain<F>);
    // pcs_vk, domain_size
    type VerifierKey = (UnivariateVerifierParam<E>, usize);
    type Commitment = Vec<Commitment<E>>;
    type Proof = Vec<UnivariateKzgProof<E>>;

    fn setup<R>(
        max_row_degree: usize,
        _max_col_degree: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        UnivariateUniversalParams::gen_srs_for_testing(rng, max_row_degree).map_err(VrsError::from)
    }

    fn preprocess(
        pp: &Self::PublicParams,
        row_degree: usize,
        _col_degree: usize,
        eval_domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        let (pk, vk) = UnivariateUniversalParams::trim(pp, row_degree).map_err(VrsError::from)?;
        Ok((
            (pk, eval_domain.to_owned()),
            (vk, eval_domain.size as usize),
        ))
    }

    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
        let total_time = start_timer!(|| ark_std::format!(
            "PeerDAS::compute shares (k={}, n={}, L={})",
            data.width(),
            pk.1.size,
            data.height(),
        ));

        // 1. encode kxL into nxL matrix (row-wise FFT)
        let encode_time = start_timer!(|| "encode data");
        let encoded = Self::interleaved_rs_encode(data, &pk.1)?;
        end_timer!(encode_time);

        // 2 commit each row using pcs
        let row_polys = data.to_row_uv_polys();
        let commit_time = start_timer!(|| "commit row_poly");
        let row_poly_cms = row_polys
            .par_iter()
            .map(|poly| UnivariateKzgPCS::commit(&pk.0, poly).expect("fail to commit row_poly"))
            .collect();
        end_timer!(commit_time);

        // 3. run multi-evaluation on all row poly
        let multi_eval_time = start_timer!(|| "multi-eval on all row_poly");
        let eval_proofs: Vec<Vec<_>> = row_polys
            .par_iter()
            .map(|poly| multi_eval(&pk.0, poly, &pk.1))
            .collect();
        let shares = encoded
            .par_col_enumerate()
            .map(|(col_idx, data)| {
                let proof: Vec<UnivariateKzgProof<E>> = eval_proofs
                    .par_iter()
                    .map(|proof_per_row| proof_per_row[col_idx].clone())
                    .collect();
                VrsShare { data, proof }
            })
            .collect::<Vec<_>>();
        end_timer!(multi_eval_time);

        end_timer!(total_time);
        Ok((row_poly_cms, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        let verify_time = start_timer!(|| "verify share");

        let point: F = Radix2EvaluationDomain::new(vk.1).unwrap().element(idx);

        // get randomizer
        let r = {
            let mut hasher = Sha256::new();
            let mut bytes = vec![];
            comm.serialize_uncompressed(&mut bytes)?;
            idx.serialize_uncompressed(&mut bytes)?;
            share.data.serialize_uncompressed(&mut bytes)?;
            share.proof.serialize_uncompressed(&mut bytes)?;
            hasher.update(bytes);
            let res = hasher.finalize();

            let modulus_bytes =
                (<F::BasePrimeField as PrimeField>::MODULUS_BIT_SIZE as usize - 1) / 8;

            F::from_random_bytes(&res[..modulus_bytes]).unwrap()
        };

        // e(w(tau), h^{tau - point}) =?= e(cm - g^y, h)
        // e( \sum r^i * w(tau), h^{tau - point}) * e(\sum r^i * g^y_i - \sum r^i * cm_i, h) == 1
        let l = comm.len();
        let r_powers: Vec<F> = successors(Some(F::ONE), |&prev| Some(prev * r))
            .take(l)
            .collect();

        let proofs: Vec<E::G1Affine> = share.proof.par_iter().map(|p| p.proof.clone()).collect();
        let batched_proof = <E::G1 as VariableBaseMSM>::msm(&proofs, &r_powers).unwrap();

        let eval_exp: F = r_powers
            .par_iter()
            .zip(share.data.par_iter())
            .map(|(r_pow, y_i)| *r_pow * y_i)
            .sum();
        let batched_eval = vk.0.g * eval_exp;

        let cms: Vec<E::G1Affine> = comm.par_iter().map(|cm| cm.0).collect();
        let batched_cm = <E::G1 as VariableBaseMSM>::msm(&cms, &r_powers).unwrap();

        let verified = E::multi_pairing(
            &[batched_proof, batched_eval - batched_cm],
            &[
                vk.0.beta_h.into_group() - vk.0.h * point,
                vk.0.h.into_group(),
            ],
        )
        .0
        .is_one();

        // NOTE: this is the non-batched verification with 2L Pairings
        // let verified = comm
        //     .par_iter()
        //     .zip(share.data.par_iter().zip(share.proof.par_iter()))
        //     .all(|(cm, (eval, eval_proof))| {
        //         UnivariateKzgPCS::verify(&vk.0, cm, &point, eval, eval_proof).unwrap()
        //     });

        end_timer!(verify_time);
        Ok(verified)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};

    use super::*;
    use crate::test_utils::test_rng;
    use ark_std::UniformRand;

    #[test]
    fn test_peer_das() {
        let rng = &mut test_rng();
        let k = 2usize.pow(10);
        let l = 8;
        let n = 2usize.pow(11);

        let pp = PeerDasVRS::<Bn254>::setup(k - 1, l - 1, rng).unwrap();
        let domain = Radix2EvaluationDomain::new(n).unwrap();
        let (pk, vk) = PeerDasVRS::preprocess(&pp, k - 1, l - 1, &domain).unwrap();

        let data = (0..k * l).map(|_| Fr::rand(rng)).collect();
        let data = Matrix::new(data, k, l).unwrap();
        let (cm, shares) = PeerDasVRS::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(PeerDasVRS::verify_share(&vk, &cm, idx, share).unwrap());
        }
    }
}
