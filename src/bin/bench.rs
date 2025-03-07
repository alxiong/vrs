use std::{env, time::Instant};

use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::*;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use itertools::Itertools;
use jf_pcs::prelude::MultilinearKzgPCS;
use vrs::{
    advz::AdvzVRS,
    frida::FridaVRS,
    gxz::{transparent::GxzVRS, trusted::BkzgGxzVRS},
    matrix::Matrix,
    nnt::{kzg::KzgNntVRS, pedersen::PedersenNntVRS},
    peer_das::PeerDasVRS,
    VerifiableReedSolomon,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.len() {
        1 => {
            bench_vid();
            bench_das();
        },
        2 => match args[1].as_str() {
            "vid" => bench_vid(),
            "das" => bench_das(),
            _ => {
                eprintln!("Unknown option: {}", args[1]);
                std::process::exit(1);
            },
        },
        _ => {
            eprintln!("Usage: {} [vid|das]", args[0]);
            std::process::exit(1);
        },
    }
}

fn bench_das() {
    println!("ℹ️ DAS Benchmark");
    let log_l_choices = vec![6]; // fixed a L=64, 2KB symbol size
    let log_k_choices = (12..=16).collect::<Vec<_>>();

    println!("\n🔔 FRIDA");
    bench_helper::<Fr, FridaVRS<Fr>>(&log_l_choices, &log_k_choices, Some(ExtraOpt::frida()));

    println!("\n🔔 Advz");
    bench_helper::<Fr, AdvzVRS<Bn254>>(&log_l_choices, &log_k_choices, None);

    println!("\n🔔 PeerDAS");
    bench_helper::<Fr, PeerDasVRS<Bn254>>(&log_l_choices, &log_k_choices, None);

    println!("\n🔔 KzgNNT");
    bench_helper::<Fr, KzgNntVRS<Bn254>>(&log_l_choices, &log_k_choices, None);

    println!("\n🔔 PedersenNNT");
    bench_helper::<Fr, PedersenNntVRS<G1Projective>>(&log_l_choices, &log_k_choices, None);

    println!("\n🔔 Gxz + PST");
    bench_helper::<Fr, GxzVRS<Fr, MultilinearKzgPCS<Bn254>>>(
        &log_l_choices,
        &log_k_choices,
        Some(ExtraOpt::gxz()),
    );

    println!("\n🔔 BkzgGxz");
    bench_helper::<Fr, BkzgGxzVRS<Bn254>>(&log_l_choices, &log_k_choices, None);
}

fn bench_vid() {
    println!("ℹ️ VID Benchmark");
    let log_l_choices = (11..=15).collect::<Vec<_>>();
    let log_k_choices = vec![6]; // fixed k=128, total 256 nodes

    println!("\n🔔 FRIDA");
    bench_helper::<Fr, FridaVRS<Fr>>(&log_l_choices, &log_k_choices, Some(ExtraOpt::frida()));

    println!("\n🔔 Advz");
    bench_helper::<Fr, AdvzVRS<Bn254>>(&log_l_choices, &log_k_choices, None);

    println!("\n🔔 KzgNNT");
    bench_helper::<Fr, KzgNntVRS<Bn254>>(&log_l_choices, &log_k_choices, None);

    println!("\n🔔 PedersenNNT");
    bench_helper::<Fr, PedersenNntVRS<G1Projective>>(&log_l_choices, &log_k_choices, None);

    println!("\n🔔 Gxz + PST");
    bench_helper::<Fr, GxzVRS<Fr, MultilinearKzgPCS<Bn254>>>(
        &log_l_choices,
        &log_k_choices,
        Some(ExtraOpt::gxz()),
    );

    println!("\n🔔 BkzgGxz");
    bench_helper::<Fr, BkzgGxzVRS<Bn254>>(&log_l_choices, &log_k_choices, None);
}

/// extra options to instruct the `bench_helper()`, since there are some unique requirement for different schemes
#[derive(Debug, Clone, Default)]
struct ExtraOpt {
    /// specific to Frida, as it doesn't need max_x_degree etc. just redo the setup for each degree
    per_degree_setup: bool,
    /// involve evaluation consolidation which has restriction on num_vars being multiple of two
    niec: bool,
}

impl ExtraOpt {
    fn frida() -> Self {
        let mut opt = Self::default();
        opt.per_degree_setup = true;
        opt
    }

    fn gxz() -> Self {
        let mut opt = Self::default();
        opt.niec = true;
        opt
    }
}

fn bench_helper<F: FftField, S: VerifiableReedSolomon<F>>(
    log_l_choices: &[usize],
    log_k_choices: &[usize],
    extra_opt: Option<ExtraOpt>,
) {
    assert!(!log_l_choices.is_empty() && !log_k_choices.is_empty());
    let extra_opt = extra_opt.unwrap_or(ExtraOpt::default());

    let rng = &mut StdRng::from_seed([42; 32]);
    let mut log_l_choices = log_l_choices.to_vec();
    let mut log_k_choices = log_k_choices.to_vec();
    log_l_choices.sort();
    log_k_choices.sort();
    let max_x_degree = 1 << log_l_choices.last().unwrap();
    let max_y_degree = 1 << log_k_choices.last().unwrap();

    let mut pp = S::setup(max_y_degree, max_x_degree, rng).unwrap();

    println!("l, k, n, prover (ms), communication (byte), verifier (ms)");
    for (log_l, log_k) in log_l_choices
        .into_iter()
        .cartesian_product(log_k_choices.into_iter())
    {
        if extra_opt.niec && log_k % 2 != 0 {
            // skip cases where num_var is not a multiple of 2 (step_size)
            continue;
        }

        let l = 1 << log_l;
        let k = 1 << log_k;

        if extra_opt.per_degree_setup {
            pp = S::setup(k - 1, l - 1, rng).unwrap();
        }
        // fixed RS rate of 1/2
        let domain = Radix2EvaluationDomain::new(2 * k).unwrap();

        let (pk, vk) = S::preprocess(&pp, k - 1, l - 1, &domain).unwrap();

        let data = (0..k * l).map(|_| F::rand(rng)).collect();
        let data = Matrix::new(data, k, l).unwrap();

        // Benchmarking prover time by computing all shares and minus the interleaved RS encoding time
        let start = Instant::now();
        let (cm, shares) = S::compute_shares(&pk, &data).unwrap();
        let total_prepare_time = start.elapsed().as_millis();

        let start = Instant::now();
        assert!(S::verify_share(&vk, &cm, 0, &shares[0]).unwrap());
        let verifier_time = start.elapsed().as_millis();

        let start = Instant::now();
        let _ = S::interleaved_rs_encode(&data, &domain).unwrap();
        let encode_time = start.elapsed().as_millis();

        let prover_time = total_prepare_time - encode_time;

        // Communication consists of commitment size and opening proof per replica/share
        let communication =
            shares[0].proof.serialized_size(Compress::No) + cm.serialized_size(Compress::No);

        println!(
            "{}, {}, {}, {}, {}, {}",
            l, k, domain.size, prover_time, communication, verifier_time
        );
    }
}
