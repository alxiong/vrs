//! Interactive Oracle Proof of Proximity (IOPP) implementations

pub mod fri;
pub mod fri_params;

/// Interactive Oracle Proof of Proximity (IOPP) made non-interactive via Fiat-Shamir transcript
/// We only focus on linear code and proximity to codewords (polynomial over fields)
pub trait IOPP<F> {
    // TODO:  come back later
}
