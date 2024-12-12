#![warn(
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    rust_2021_compatibility
)]
#![deny(unsafe_code)]

use anyhow::Result;
use ark_ff::Field;
use ark_std::fmt::Debug;
use ark_std::rand::{CryptoRng, RngCore};

pub mod matrix;
mod utils;

/// Interface for a verifiable (interleaved) Reed-Solomon scheme.
pub trait VerifiableReedSolomon<F: Field>: Sized {
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

    /// Construct public parameters given the degree upper bounds on X and Y dimension
    fn setup<R>(max_row_degree: usize, max_col_degree: usize, rng: &mut R) -> Self::PublicParams
    where
        R: RngCore + CryptoRng;

    /// Trim the public parameters to concrete degrees for specific instances
    fn trim(
        pp: &Self::PublicParams,
        row_degree: usize,
        col_degree: usize,
    ) -> Result<(Self::ProverKey, Self::VerifierKey)>;

    /// Computing all shares (data chunk + proof) for every replica
    /// Implictly involve the following inside this operation
    /// - interleaved RS encoding
    /// - data blob commitment
    fn compute_shares(pk: &Self::ProverKey) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>)>;

    /// As the `idx`-th party/replica, verify the share received
    /// Returns Err(_) if input are malformed, else Ok(result) on the verification result.
    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool>;
}

/// A share for a node/replica in VerifiableRS scheme
#[derive(Debug, Default, Clone)]
pub struct VrsShare<F: Field, T: VerifiableReedSolomon<F>> {
    /// data chunk for a share
    pub data: Vec<F>,
    /// proof that `data` is consistent with `comm`
    pub proof: T::Proof,
}
