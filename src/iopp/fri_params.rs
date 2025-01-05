//! Recommended parameters based on concrete security of non-interactive FRI
//! Sec 4.3 of <https://eprint.iacr.org/2024/1161>

use ark_ff::{Field, PrimeField};

/// Oracle query bound log(Q) for 80-bit security
const Q_BITS_80_SEC: usize = 60;
/// Oracle query bound log(Q) for 100-bit security
const Q_BITS_100_SEC: usize = 80;
/// Non-interactive soundness error 2^-v
const V_BITS: usize = 20;

/// bounds and parameters derived from provable soundness
pub mod provable {
    use super::*;

    /// Compute the num_of_queries (\ell) required to achieve `sec_bit`-bit security
    /// Assuming fixed oracle query Q=2^60, non-interactive soundness error v=20 (FRI with soundness error 2^-20 with Q-query adversary)
    /// Assuming 256-bit output for the random oracle/transcript challenger
    ///
    /// # Parameters
    /// - security bit: currently only supports 80 or 100
    /// - generic `F` gives field size |F| (with extension)
    /// - message size d_0 (unlike Tab.3, we don't force it to be powers_of_two or 2^k)
    /// - blowup factor = \rho^-1
    /// - init domain size (implied by last two) |L_0| = d_0 * blowup_factor
    /// - grinding bit: z (bring 2^-z factor to LHS of Eq.14)
    ///
    /// # Panics
    /// - if Johnson upper bound m<3, then infeasible
    #[inline(always)]
    pub fn num_queries<F: Field>(
        sec_bits: usize,
        msg_len: usize,
        blowup: usize,
        grinding_bits: usize,
    ) -> usize {
        let q_bits = match sec_bits {
            80 => Q_BITS_80_SEC,
            100 => Q_BITS_100_SEC,
            _ => panic!("only support 80-bit and 100-bit security for now"),
        };
        let l0 = msg_len * blowup;
        let rho = (blowup as f64).recip();

        // Satisfy Eq.13
        let max_m = {
            // the upper bound of the Johnson bound m (>=3) based on Eq.13
            let field_bit_size = <F::BasePrimeField as PrimeField>::MODULUS_BIT_SIZE as usize
                * F::extension_degree() as usize;

            // first compute |F|/(Q*2^{v+1}* |L_0|^2) in bits/logscale as it may still be too big for u64
            let log_l0 = (l0 as f64).log2().floor() as usize + 1;
            let log_tmp = field_bit_size - (q_bits + V_BITS + 1 + 2 * log_l0);

            // sqrt7(3*rho^{3/2} * (1<<log_tmp)) - 1/2
            ((3f64.log2() + rho.powf(1.5) + log_tmp as f64) / 7.0).exp2() - 0.5
        };
        if max_m < 3.0 {
            panic!("infeasible FRI param");
        }
        let m = max_m.floor();

        // choose maximal proximity \delta = 1-sqrt(rho) * (1+1/2m)
        let delta = 1.0 - rho.sqrt() * (1.0 + (2.0 * m).recip());

        // compute \ell
        // RHS (including 2^-z on LHS) are computed directly in log values
        let log_rhs = grinding_bits as isize - (q_bits + V_BITS + 1) as isize;
        let log_lhs_base = (1.0 - delta).log2();
        let num_queries = (log_rhs as f64 / log_lhs_base).ceil() as usize;
        num_queries
    }
}

/// bounds and parameters derived from more aggressive soundness conjectures
pub mod conjectured {
    use super::*;
    /// Compute the num_of_queries (\ell) required to achieve `sec_bit`-bit security
    /// Assuming fixed oracle query Q=2^60, non-interactive soundness error v=20 (FRI with soundness error 2^-20 with Q-query adversary)
    /// Assuming 256-bit output for the random oracle/transcript challenger
    ///
    /// # Parameters
    /// - security bit: currently only supports 80 or 100
    /// - generic `F` gives field size |F| (with extension)
    /// - blowup factor = \rho^-1
    /// - grinding bit: z (bring 2^-z factor to LHS of Eq.14)
    ///
    /// # Panics
    /// - if field size (extension included) is too small, then infeasible
    #[inline(always)]
    pub fn num_queries<F: Field>(sec_bits: usize, blowup: usize, grinding_bits: usize) -> usize {
        let q_bits = match sec_bits {
            80 => Q_BITS_80_SEC,
            100 => Q_BITS_100_SEC,
            _ => panic!("only support 80-bit and 100-bit security for now"),
        };
        let rho = (blowup as f64).recip();

        // Satisfy Eq.15
        let field_bit_size = <F::BasePrimeField as PrimeField>::MODULUS_BIT_SIZE as usize
            * F::extension_degree() as usize;
        if field_bit_size < q_bits + V_BITS + 1 {
            panic!("field size too small");
        }

        // choose maximal proximity \delta = 1-rho
        // let delta = 1.0 - rho;

        // compute \ell
        // RHS (including 2^-z on LHS) are computed directly in log values
        let log_rhs = grinding_bits as isize - (q_bits + V_BITS + 1) as isize;
        let num_queries = (log_rhs as f64 / rho.log2()).ceil() as usize;
        num_queries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Fp192, Fp256, MontBackend, MontConfig};

    #[derive(MontConfig)]
    #[modulus = "6277101735386680763835789423207666416083908700390324961279"] // 192-bit prime
    #[generator = "3"] // Example generator
    #[small_subgroup_base = "2"] // Commonly used base
    #[small_subgroup_power = "96"] // Choose such that 2^96 divides p - 1    pub struct Fp192Config;
    pub struct Fp192Config;
    pub type F192 = Fp192<MontBackend<Fp192Config, 3>>;

    #[derive(MontConfig)]
    #[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"] // bls12-381 scalar
    #[generator = "7"]
    #[small_subgroup_base = "3"]
    #[small_subgroup_power = "1"]
    pub struct FrConfig;
    pub type F256 = Fp256<MontBackend<FrConfig, 4>>;

    #[test]
    fn test_provable_fri_params() {
        for k in [10, 15, 20] {
            assert_eq!(provable::num_queries::<F192>(80, 1 << k, 2, 0), 163);
            assert_eq!(provable::num_queries::<F192>(80, 1 << k, 4, 0), 82);
            assert_eq!(provable::num_queries::<F192>(80, 1 << k, 8, 0), 55);
            assert_eq!(provable::num_queries::<F192>(80, 1 << k, 16, 0), 41);
            assert_eq!(provable::num_queries::<F256>(80, 1 << k, 2, 0), 163);
            assert_eq!(provable::num_queries::<F256>(80, 1 << k, 4, 0), 82);
            assert_eq!(provable::num_queries::<F256>(80, 1 << k, 8, 0), 55);
            assert_eq!(provable::num_queries::<F256>(80, 1 << k, 16, 0), 41);

            assert_eq!(provable::num_queries::<F256>(100, 1 << k, 2, 0), 203);
            assert_eq!(provable::num_queries::<F256>(100, 1 << k, 4, 0), 102);
            assert_eq!(provable::num_queries::<F256>(100, 1 << k, 8, 0), 68);
            assert_eq!(provable::num_queries::<F256>(100, 1 << k, 16, 0), 51);
        }
        assert_eq!(provable::num_queries::<F192>(100, 1 << 25, 2, 0), 209);
        assert_eq!(provable::num_queries::<F192>(100, 1 << 25, 16, 0), 52);
    }

    #[test]
    fn test_conjectured_fri_params() {
        assert_eq!(conjectured::num_queries::<F192>(80, 2, 0), 81);
        assert_eq!(conjectured::num_queries::<F192>(80, 4, 0), 41);
        assert_eq!(conjectured::num_queries::<F192>(80, 8, 0), 27);
        assert_eq!(conjectured::num_queries::<F192>(80, 16, 0), 21);
        assert_eq!(conjectured::num_queries::<F256>(80, 2, 0), 81);
        assert_eq!(conjectured::num_queries::<F256>(80, 4, 0), 41);
        assert_eq!(conjectured::num_queries::<F256>(80, 8, 0), 27);
        assert_eq!(conjectured::num_queries::<F256>(80, 16, 0), 21);

        assert_eq!(conjectured::num_queries::<F256>(100, 2, 0), 101);
        assert_eq!(conjectured::num_queries::<F256>(100, 4, 0), 51);
        assert_eq!(conjectured::num_queries::<F256>(100, 8, 0), 34);
        assert_eq!(conjectured::num_queries::<F256>(100, 16, 0), 26);
    }
}
