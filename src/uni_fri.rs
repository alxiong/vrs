//! Univariate FRI Polynomial Commitment implementations

use crate::iopp::{
    fri,
    fri::{FriConfig, FriProof},
};
use ark_ff::FftField;
use ark_poly::{univariate::DensePolynomial, Radix2EvaluationDomain};
use ark_std::{
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    UniformRand,
};
use p3_maybe_rayon::prelude::*;

/// Univariate FRI PCS
pub struct UniFriPCS<F>(PhantomData<F>);

impl<F: FftField> UniFriPCS<F> {
    /// Transparent setup
    /// - sec_bit: security level, at least >= 112, preferably >=128
    /// - log_blowup: log of blowup factor
    /// - max_degree: degree upper bound of the polynomial to be committed
    pub fn setup(sec_bit: usize, log_blowup: usize, max_degree: usize) -> UniFriPCS<F> {
        unimplemented!()
    }

    /// Commit to a polynomial, returns the merkle root of all evaluations in domain D_0
    pub fn commit(pp: &UniFriPCS<F>, poly: &DensePolynomial<F>) -> Vec<u8> {
        unimplemented!()
    }

    /// Open/Evaluate `poly` at `point` and return the evaluation/opening proof and the evaluation
    pub fn open(pp: &UniFriPCS<F>, poly: &DensePolynomial<F>, point: &F) -> (Vec<u8>, F) {
        unimplemented!()
    }

    /// Verify an opening proof
    pub fn verify(
        pp: &UniFriPCS<F>,
        commitment: &[u8],
        point: &F,
        value: &F,
        proof: &FriProof<F>,
    ) -> bool {
        false
    }
}
/// Uniform Random String (URS) of Univariate FRI PCS, totally transparent
#[derive(Debug, Clone)]
pub struct UniFriURS<F: FftField> {
    /// FRI config, affect security level including conjectured soundness
    pub config: FriConfig,
    /// evaluation domain D_0
    pub domain: Radix2EvaluationDomain<F>,
}
