//! Pedersen-based NNT'22 VID scheme

use ark_ec::CurveGroup;
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{
    end_timer,
    iter::successors,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
};
use p3_maybe_rayon::prelude::*;

use crate::{matrix::Matrix, VerifiableReedSolomon, VrsError, VrsShare};

/// A DLog-based Pedersen-based NNT VID
pub struct PedersenNntVRS<C: CurveGroup> {
    _group: PhantomData<C>,
}

impl<F, C> VerifiableReedSolomon<F> for PedersenNntVRS<C>
where
    F: FftField,
    C: CurveGroup<ScalarField = F>,
{
    // a list of base group elements g_i
    type PublicParams = Vec<C::Affine>;
    // (pp, domain)
    type ProverKey = (Self::PublicParams, Radix2EvaluationDomain<F>);
    // (pp, domain_size)
    type VerifierKey = (Self::PublicParams, usize);
    // each column vector commit to a group
    type Commitment = Vec<C::Affine>;
    // no proof needed
    type Proof = ();

    fn setup<R>(
        _max_row_degree: usize,
        max_col_degree: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        let mut pp = Vec::with_capacity(max_col_degree + 1);
        let mut base = C::rand(rng);
        for _ in 0..max_col_degree + 1 {
            pp.push(base);
            base.double_in_place();
        }
        Ok(C::normalize_batch(&pp))
    }

    fn preprocess(
        pp: &Self::PublicParams,
        _row_degree: usize,
        col_degree: usize,
        eval_domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        let mut trimmed = pp.clone();
        trimmed.truncate(col_degree + 1);
        Ok((
            (trimmed.clone(), eval_domain.to_owned()),
            (trimmed, eval_domain.size as usize),
        ))
    }

    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
        let total_time = start_timer!(|| ark_std::format!(
            "PedersenNNT::compute shares (k={}, n={}, L={})",
            data.width(),
            pk.1.size,
            data.height()
        ));

        // 1. encode Lxk into Lxn matrix (row-wise FFT)
        let encode_time = start_timer!(|| "encode data");
        let encoded = Self::interleaved_rs_encode(data, &pk.1)?;
        end_timer!(encode_time);

        // put each col to a share
        let shares = encoded
            .par_col()
            .map(|data| VrsShare { data, proof: () })
            .collect();

        // 2. commit each columns
        let commit_time = start_timer!(|| "commit col_poly");
        let commits = data
            .par_col()
            .map(|col| {
                C::msm(&pk.0[..col.len()], &col)
                    .expect("msm failed")
                    .into_affine()
            })
            .collect();
        end_timer!(commit_time);
        end_timer!(total_time);
        Ok((commits, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        // compute the vector commitment of the column received
        let cm = C::msm(&vk.0[..share.data.len()], &share.data).expect("col-commit failed");

        // homomorphically deduce the expected commitment
        let point = Radix2EvaluationDomain::<F>::new(vk.1).unwrap().element(idx);
        // (1, x, x^2, ... , x^{k-1})
        let scalars: Vec<F> = successors(Some(F::ONE), |&prev| Some(prev * point))
            .take(comm.len())
            .collect();

        let expected_cm = C::msm(&comm, &scalars).expect("msm failed");
        Ok(cm == expected_cm)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective};
    use ark_std::UniformRand;

    use super::*;
    use crate::test_utils::test_rng;

    #[test]
    fn test_pedersen_nnt() {
        let rng = &mut test_rng();
        let row_degree = 64;
        let col_degree = 64;
        let domain_size = row_degree * 2;

        let pp = PedersenNntVRS::<G1Projective>::setup(row_degree, col_degree, rng).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(domain_size).unwrap();
        let (pk, vk) =
            PedersenNntVRS::<G1Projective>::preprocess(&pp, row_degree, col_degree, &domain)
                .unwrap();

        let data = (0..(row_degree + 1) * (col_degree + 1))
            .map(|_| Fr::rand(rng))
            .collect();
        let data = Matrix::new(data, row_degree + 1, col_degree + 1).unwrap();
        let (cm, shares) = PedersenNntVRS::<G1Projective>::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(PedersenNntVRS::verify_share(&vk, &cm, idx, share).unwrap());
        }
    }
}
