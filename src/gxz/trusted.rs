//! Trusted VRS using bivariate KZG and fast bivariate multi-partial-eval

use ark_ec::pairing::Pairing;
use ark_ff::FftField;
use ark_poly::{
    univariate::{self},
    DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use ark_std::{
    end_timer,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
};
use jf_pcs::prelude::*;
use p3_maybe_rayon::prelude::*;

use crate::{
    matrix::Matrix,
    multi_evals::{self, bivariate::PrecomputeTable},
    pcs::bkzg::{
        combine_u32, BivariateKzgPCS, BivariateKzgSRS, BivariateProverParam,
        BivariateVerifierParam, PartialEvalProof,
    },
    poly::bivariate,
    VerifiableReedSolomon, VrsError, VrsShare,
};

/// Trusted VRS based on bivariate KZG and bivariate multi-evaluation algorithm
#[derive(Debug, Clone)]
pub struct BkzgGxzVRS<E: Pairing> {
    _pairing: PhantomData<E>,
}

impl<F, E> VerifiableReedSolomon<F> for BkzgGxzVRS<E>
where
    F: FftField,
    E: Pairing<ScalarField = F>,
{
    type PublicParams = BivariateKzgSRS<E>;
    /// (pcs.pk, domain, precompute_table)
    type ProverKey = (
        BivariateProverParam<E>,
        Radix2EvaluationDomain<F>,
        PrecomputeTable<E>,
    );
    /// (pcs.vk, domain)
    type VerifierKey = (BivariateVerifierParam<E>, Radix2EvaluationDomain<F>);
    type Commitment = E::G1Affine;
    type Proof = PartialEvalProof<E>;

    fn name() -> &'static str {
        "bkzgGxz"
    }

    fn setup<R>(
        max_width: usize,
        max_height: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        let hacky_supported_degree = combine_u32(max_height as u32 - 1, max_width as u32 - 1);
        let pp = BivariateKzgPCS::<E>::gen_srs_for_testing(rng, hacky_supported_degree as usize)?;
        Ok(pp)
    }

    fn preprocess(
        pp: &Self::PublicParams,
        width: usize,
        height: usize,
        domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        let supported_degree = combine_u32(height as u32 - 1, width as u32 - 1);
        let (pk, vk) = BivariateKzgPCS::trim(pp, supported_degree as usize, None)?;
        let table = multi_evals::bivariate::multi_partial_eval_precompute(&pk, domain);

        Ok(((pk, domain.clone(), table), (vk, domain.clone())))
    }

    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
        let pcs_pk = &pk.0;
        let domain = &pk.1;
        let total_time = start_timer!(|| ark_std::format!(
            "BkzgGxzVRS::compute shares (k={}, n={}, L={})",
            data.width(),
            pk.1.size,
            data.height()
        ));

        let commit_time = start_timer!(|| "commit bi-poly");
        let bv_poly = bivariate::DensePolynomial::new(
            data.into_par_row().collect::<Vec<_>>(),
            data.height() - 1,
            data.width() - 1,
        );
        let commit = BivariateKzgPCS::commit(pcs_pk, &bv_poly)?;
        end_timer!(commit_time);

        let multi_eval_time = start_timer!(|| "MultiPartialEval");
        let (proofs, partial_evals) =
            multi_evals::bivariate::multi_partial_eval(pcs_pk, &bv_poly, domain);
        end_timer!(multi_eval_time);

        let shares = proofs
            .into_par_iter()
            .zip(partial_evals.into_par_iter())
            .map(|(proof, partial_eval)| VrsShare {
                data: partial_eval.coeffs.clone(),
                proof,
            })
            .collect::<Vec<_>>();

        end_timer!(total_time);
        Ok((commit, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        let pcs_vk = &vk.0;
        let domain = &vk.1;
        let at_x = false;
        let y = domain.element(idx);

        let verified = BivariateKzgPCS::verify_partial(
            pcs_vk,
            comm,
            &y,
            at_x,
            &univariate::DensePolynomial::from_coefficients_slice(&share.data),
            &share.proof,
        )?;
        Ok(verified)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};

    use super::*;
    use crate::test_utils::test_rng;

    #[test]
    fn test_kzg_nnt() {
        let rng = &mut test_rng();
        let width: usize = 64;
        let height = 64;
        let domain_size = width.next_power_of_two() * 2;

        let pp = BkzgGxzVRS::<Bn254>::setup(width, height, rng).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(domain_size).unwrap();
        let (pk, vk) = BkzgGxzVRS::<Bn254>::preprocess(&pp, width, height, &domain).unwrap();

        let data = Matrix::rand(rng, width, height);
        let (cm, shares) = BkzgGxzVRS::<Bn254>::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(BkzgGxzVRS::verify_share(&vk, &cm, idx, share).unwrap());
        }
    }
}
