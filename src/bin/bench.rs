//! Simple benchmark, run `cargo run --bin bench --release`, optionally with `RAYON_NUM_THREADS=1` prefix if single-threaded.
//! This is not as robust as criterion, but avoid the complexity and long delay of running many iterations and averaging them.
use ark_bn254::{Bn254, Fr};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use jf_pcs::prelude::*;
use vrs::{bivariate, bkzg::*};

use std::time::Instant;

/// a CryptoRng
pub fn test_rng() -> StdRng {
    // arbitrary seed
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

fn main() {
    let rng = &mut test_rng();
    let max_deg_x = 64;
    let max_deg_y = 2u32.pow(16);
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);

    print!("TrustedSetup");
    let start = Instant::now();
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();
    println!(" takes {} ms", start.elapsed().as_millis());

    println!("MultiPartialEval/Fast");
    println!("deg_x,deg_y,domain_size,prover(ms),communication(byte)");
    for log_deg_y in 10..=16 {
        let deg_x = max_deg_x;
        let deg_y = 2u32.pow(log_deg_y);
        let supported_degree = combine_u32(deg_x, deg_y);
        let (pk, _vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(2 * deg_y as usize).unwrap();

        let poly = bivariate::DensePolynomial::<Fr>::rand(deg_x as usize, deg_y as usize, rng);
        let cm = BivariateKzgPCS::commit(&pk, &poly).unwrap();

        let table = vrs::multi_evals::bivariate::multi_partial_eval_precompute(&pk, &domain);

        let start = Instant::now();
        let (proofs, _partial_evals) = vrs::multi_evals::bivariate::multi_partial_eval_with_table::<
            Bn254,
        >(&poly, &domain, &table);
        println!(
            "{},{},{},{},{}",
            deg_x,
            deg_y,
            2 * deg_y,
            start.elapsed().as_millis(),
            proofs[0].serialized_size(Compress::No) + cm.serialized_size(Compress::No)
        );
    }
}
