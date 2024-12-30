//! KZG-based NNT'22 VID scheme

use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::FftField;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use ark_std::{
    end_timer,
    iter::successors,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
};
use jf_pcs::prelude::*;
use p3_maybe_rayon::prelude::*;

use crate::{matrix::Matrix, VerifiableReedSolomon, VrsError, VrsShare};

/// A univariate-KZG based NNT VID
#[derive(Debug, Clone)]
pub struct KzgNntVRS<E> {
    _pairing: PhantomData<E>,
}

impl<F, E> VerifiableReedSolomon<F> for KzgNntVRS<E>
where
    F: FftField,
    E: Pairing<ScalarField = F>,
{
    type PublicParams = UnivariateUniversalParams<E>;
    // (pcs_pk, domain)
    type ProverKey = (UnivariateProverParam<E>, Radix2EvaluationDomain<F>);
    // (pcs_pk, domain_size)
    type VerifierKey = (UnivariateProverParam<E>, usize);
    // each column vector commit to a G1
    type Commitment = Vec<E::G1Affine>;
    // no proof needed
    type Proof = ();

    fn setup<R>(
        max_row_degree: usize,
        max_col_degree: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        let max_degree = max_row_degree.max(max_col_degree);
        UnivariateUniversalParams::gen_srs_for_testing(rng, max_degree).map_err(VrsError::from)
    }

    fn preprocess(
        pp: &Self::PublicParams,
        row_degree: usize,
        col_degree: usize,
        domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        let max_degree = row_degree.max(col_degree);
        let (pk, _vk) = UnivariateUniversalParams::trim(pp, max_degree).map_err(VrsError::from)?;
        Ok(((pk.clone(), domain.to_owned()), (pk, domain.size as usize)))
    }

    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
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
        let col_polys = data.to_col_uv_polys();
        let commit_time = start_timer!(|| "commit col_poly");
        let commits = col_polys
            .par_iter()
            .map(|poly| {
                UnivariateKzgPCS::commit(&pk.0, poly)
                    .expect("fail to commit col_poly")
                    .0
            })
            .collect();
        end_timer!(commit_time);
        Ok((commits, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        // compute the vector commitment of the column received
        let cm = UnivariateKzgPCS::commit(
            &vk.0,
            &DensePolynomial::from_coefficients_slice(&share.data),
        )?
        .0;

        // homomorphically deduce the expected commitment
        let point = Radix2EvaluationDomain::<F>::new(vk.1).unwrap().element(idx);
        // (1, x, x^2, ... , x^{k-1})
        let scalars: Vec<F> = successors(Some(F::ONE), |&prev| Some(prev * point))
            .take(comm.len())
            .collect();
        let expected_cm = <E::G1 as VariableBaseMSM>::msm(&comm, &scalars).expect("msm failed");

        Ok(cm == expected_cm.into_affine())
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use ark_std::UniformRand;

    use super::*;
    use crate::test_utils::test_rng;

    #[test]
    fn test_kzg_nnt() {
        let rng = &mut test_rng();
        let row_degree = 64;
        let col_degree = 64;
        let domain_size = row_degree * 2;

        let pp = KzgNntVRS::<Bn254>::setup(row_degree, col_degree, rng).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(domain_size).unwrap();
        let (pk, vk) = KzgNntVRS::preprocess(&pp, row_degree, col_degree, &domain).unwrap();

        let data = (0..(row_degree + 1) * (col_degree + 1))
            .map(|_| Fr::rand(rng))
            .collect();
        let data = Matrix::new(data, row_degree + 1, col_degree + 1).unwrap();
        let (cm, shares) = KzgNntVRS::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(KzgNntVRS::verify_share(&vk, &cm, idx, share).unwrap());
        }
    }
}
