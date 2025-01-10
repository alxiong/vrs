#![warn(
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    rust_2021_compatibility
)]
#![deny(unsafe_code)]

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{
    fmt::Debug,
    rand::{CryptoRng, RngCore},
};
use matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use thiserror::Error;

pub mod advz;
pub mod bivariate;
pub mod gxz;
pub mod iopp;
pub mod matrix;
pub mod merkle_tree;
pub mod multi_evals;
pub mod nnt;
pub mod pcs;
pub mod peer_das;
#[cfg(test)]
pub mod test_utils;

/// Interface for a verifiable (interleaved) Reed-Solomon scheme.
pub trait VerifiableReedSolomon<F: FftField>: Sized {
    /// Public paramters for the scheme, generated during setup
    type PublicParams: Clone + Debug;
    /// Proving key for the prover to generate shares and opening proofs
    type ProverKey: Clone + Debug;
    /// Verifier key for nodes/replicas to verify their opening proofs
    type VerifierKey: Clone + Debug;
    /// Commitment to the data
    type Commitment: Clone + Debug;
    /// Proof for a valid share w.r.t the commitment
    type Proof: Clone + Debug;

    // TODO: (alex) name max_row_dgree is little confusing, change it
    /// Construct public parameters given the degree upper bounds on X and Y dimension
    /// max_row_degree = width-1, max_col_degree = height - 1
    fn setup<R>(
        max_row_degree: usize,
        max_col_degree: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng;

    /// Instance-specific preprocessing
    /// - trim the public parameters to concrete degrees for specific instances
    /// - instance-independent precomputation
    fn preprocess(
        pp: &Self::PublicParams,
        row_degree: usize,
        col_degree: usize,
        eval_domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError>;

    /// Computing all shares (data chunk + proof) for every replica
    /// Implictly involve the following inside this operation
    /// - interleaved RS encoding
    /// - data blob commitment
    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError>;

    /// As the `idx`-th party/replica, verify the share received
    /// Returns Err(_) if input are malformed, else Ok(result) on the verification result.
    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError>;

    /// Row-wise (k,n)-RS encode the data matrix: (F^L)^k -> (F^L)^n
    /// `domain.size = n` and `data.width() = k`
    /// Returns the encoded matrix of size (F^L)^n
    #[inline]
    fn interleaved_rs_encode(
        data: &Matrix<F>,
        domain: &Radix2EvaluationDomain<F>,
    ) -> Result<Matrix<F>, VrsError> {
        // flatten encoded data
        let encoded: Vec<F> = data.par_row().flat_map(|row| domain.fft(row)).collect();
        let encoded_matrix = Matrix::new(encoded, domain.size as usize, data.height())?;
        Ok(encoded_matrix)
    }
}

/// A share for a node/replica in VerifiableRS scheme
#[derive(Debug, Default, Clone)]
pub struct VrsShare<F: FftField, T: VerifiableReedSolomon<F>> {
    /// data chunk for a share
    pub data: Vec<F>,
    /// proof that `data` is consistent with `comm`
    pub proof: T::Proof,
}

/// Custom error type for VRS-related error
#[derive(Debug, Error)]
pub enum VrsError {
    #[error("PCS operation failed: {0}")]
    PcsErr(#[from] jf_pcs::PCSError),
    #[error("Arkworks primitives failed: {0}")]
    ArkPrimitivesErr(#[from] ark_crypto_primitives::Error),
    #[error("Arkworks serde failed: {0}")]
    SerdeErr(#[from] ark_serialize::SerializationError),
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
    #[error("Uncategorized: {0}")]
    Anyhow(#[from] anyhow::Error),
}
