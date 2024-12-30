//! Simple benchmark, run `cargo run --bin bench --release`, optionally with `RAYON_NUM_THREADS=1` prefix if single-threaded.
//! This is not as robust as criterion, but avoid the complexity and long delay of running many iterations and averaging them.
use ark_bn254::{Bn254, Fr};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    UniformRand,
};
use itertools::Itertools;
use jf_pcs::prelude::*;
use vrs::{
    advz::AdvzVRS, bivariate, bkzg::*, matrix::Matrix, nnt::kzg::KzgNntVRS, VerifiableReedSolomon,
};

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
    let max_deg_x = 2u32.pow(13);
    let max_deg_y = 256;
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);

    // NOTE: only need to change these (and the max_deg above) for different regimes
    let deg_x_choices = [256, 512, 1024, 2048, 4096, 8192];
    let deg_y_choices = [256];

    print!("TrustedSetup");
    let start = Instant::now();
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();
    println!(" takes {} ms", start.elapsed().as_millis());

    println!("MultiPartialEval/Fast");
    println!("deg_x, deg_y, domain_size, commit(ms), prover(ms), communication(byte)");
    for (deg_x, deg_y) in deg_x_choices
        .into_iter()
        .cartesian_product(deg_y_choices.into_iter())
    {
        let supported_degree = combine_u32(deg_x, deg_y);
        let (pk, _vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(2 * deg_y as usize).unwrap();

        let poly = bivariate::DensePolynomial::<Fr>::rand(deg_x as usize, deg_y as usize, rng);

        let start = Instant::now();
        let cm = BivariateKzgPCS::commit(&pk, &poly).unwrap();
        let commit_time = start.elapsed().as_millis();

        let table = vrs::multi_evals::bivariate::multi_partial_eval_precompute(&pk, &domain);

        let start = Instant::now();
        let (proofs, _partial_evals) = vrs::multi_evals::bivariate::multi_partial_eval_with_table::<
            Bn254,
        >(&poly, &domain, &table);
        println!(
            "{}, {}, {}, {}, {}, {}",
            deg_x,
            deg_y,
            2 * deg_y,
            commit_time,
            start.elapsed().as_millis(),
            proofs[0].serialized_size(Compress::No) + cm.serialized_size(Compress::No)
        );
    }

    let pp = AdvzVRS::<Bn254>::setup(max_deg_y as usize, max_deg_x as usize, rng).unwrap();

    println!("\nMultiEval/Advz");
    println!("deg_x, deg_y, domain_size, commit+prover(ms), communication(byte)");
    for (deg_x, deg_y) in deg_x_choices
        .into_iter()
        .cartesian_product(deg_y_choices.into_iter())
    {
        let deg_x = deg_x as usize;
        let deg_y = deg_y as usize;
        let domain_size = 2 * deg_y as usize;

        let domain = Radix2EvaluationDomain::new(domain_size).unwrap();
        let (pk, _vk) = AdvzVRS::preprocess(&pp, deg_y, deg_x, &domain).unwrap();

        let data = (0..(deg_y + 1) * (deg_x + 1))
            .map(|_| Fr::rand(rng))
            .collect();
        let data = Matrix::new(data, deg_y + 1, deg_x + 1).unwrap();

        // we shall remove encode time from prover time
        let start = Instant::now();
        let _ = AdvzVRS::<Bn254>::interleaved_rs_encode(&data, &domain).unwrap();
        let encode_time = start.elapsed().as_millis();

        let start = Instant::now();
        let (cm, shares) = AdvzVRS::compute_shares(&pk, &data).unwrap();

        println!(
            "{}, {}, {}, {}, {}",
            deg_x,
            deg_y,
            domain_size,
            start.elapsed().as_millis() - encode_time,
            shares[0].proof.serialized_size(Compress::No) + cm.serialized_size(Compress::No)
        );
    }

    println!("\nKZG-NNT");
    println!("deg_x, deg_y, domain_size, commit(ms), verify(ms), communication(byte)");

    let pp = KzgNntVRS::<Bn254>::setup(max_deg_y as usize, max_deg_x as usize, rng).unwrap();
    for (deg_x, deg_y) in deg_x_choices
        .into_iter()
        .cartesian_product(deg_y_choices.into_iter())
    {
        let deg_x = deg_x as usize;
        let deg_y = deg_y as usize;
        let domain_size = 2 * deg_y;

        let domain = Radix2EvaluationDomain::new(domain_size).unwrap();
        let (pk, vk) = KzgNntVRS::preprocess(&pp, deg_y, deg_x, &domain).unwrap();

        let data = (0..(deg_y + 1) * (deg_x + 1))
            .map(|_| Fr::rand(rng))
            .collect();
        let data = Matrix::new(data, deg_y + 1, deg_x + 1).unwrap();

        // we shall remove encode time from prover time
        let start = Instant::now();
        let _ = KzgNntVRS::<Bn254>::interleaved_rs_encode(&data, &domain).unwrap();
        let encode_time = start.elapsed().as_millis();

        let start = Instant::now();
        let (cm, shares) = KzgNntVRS::compute_shares(&pk, &data).unwrap();
        let commit_time = start.elapsed().as_millis() - encode_time;

        let start = Instant::now();
        let idx = rng.gen_range(0..domain_size);
        KzgNntVRS::verify_share(&vk, &cm, idx, &shares[idx]).unwrap();
        let verify_time = start.elapsed().as_millis();

        println!(
            "{}, {}, {}, {}, {}, {}",
            deg_x,
            deg_y,
            domain_size,
            commit_time,
            verify_time,
            shares[0].proof.serialized_size(Compress::No) + cm.serialized_size(Compress::No)
        );
    }
}
