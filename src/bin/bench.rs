use std::{env, time::Instant};

use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::*;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use itertools::Itertools;
use jf_pcs::prelude::MultilinearKzgPCS;
use prettytable::{row, Table};
use vrs::{
    advz::AdvzVRS,
    frida::FridaVRS,
    gxz::{transparent::GxzVRS, trusted::BkzgGxzVRS},
    matrix::Matrix,
    nnt::{kzg::KzgNntVRS, pedersen::PedersenNntVRS},
    pcs::lightligero_test::LightLigeroPCS,
    peer_das::PeerDasVRS,
    zoda::ZodaVRS,
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
            "test" => bench_test(),
            "ndss-base" => bench_ndss_base(),
            "ndss-all" => bench_ndss_all(),
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

// only test frida and ours
fn bench_ndss_base() {
    let header = row![
        "Scheme",
        "N",
        "l",
        "k",
        "n",
        "|M| (MB)",
        "prover (ms)",
        "per-node overhead. (KB)",
        "per-node (KB)",
        "verifier (ms)"
    ];

    let mut frida_table = Table::new();
    frida_table.add_row(header.clone());
    let mut conda_pst_table = Table::new();
    conda_pst_table.add_row(header.clone());
    let mut conda_lightligero_table = Table::new();
    conda_lightligero_table.add_row(header);

    for num_nodes in [512, 1024, 2048, 4096] {
        // for num_nodes in [512] {
        // size means number of fields, not in bytes
        for block_log_size in [19, 20, 21, 22] {
            // for block_log_size in [19] {
            let (log_k, log_l) = frida_shape_heuristic(block_log_size);
            bench_ndss_helper::<FridaVRS<Fr>>(&mut frida_table, num_nodes, log_k, log_l);

            let (log_k, log_l) = conda_shape_heuristic(block_log_size, num_nodes);
            bench_ndss_helper::<GxzVRS<Fr, MultilinearKzgPCS<Bn254>>>(
                &mut conda_pst_table,
                num_nodes,
                log_k,
                log_l,
            );

            bench_ndss_helper::<GxzVRS<Fr, LightLigeroPCS<Bn254>>>(
                &mut conda_lightligero_table,
                num_nodes,
                log_k,
                log_l,
            );
        }
    }

    println!("\n🔔 Frida");
    frida_table.printstd();
    println!("\n🔔 Conda+PST");
    conda_pst_table.printstd();
    println!("\n🔔 Conda+lightligero");
    conda_lightligero_table.printstd();
}

fn bench_ndss_helper<S: VerifiableReedSolomon<Fr>>(
    table: &mut Table,
    num_nodes: usize,
    log_k: usize,
    log_l: usize,
) {
    let rng = &mut StdRng::from_seed([42; 32]);
    let blowup = 4;
    let block_log_size = log_k + log_l;
    let k = 1 << log_k;
    let l = 1 << log_l;
    let n = k * blowup;

    let pp = S::setup(k, l, rng).unwrap();
    let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    let (pk, vk) = S::preprocess(&pp, k, l, &domain).unwrap();
    let data = Matrix::rand(rng, k, l);

    let start = Instant::now();
    let (cm, shares) = S::compute_shares(&pk, &data).unwrap();
    let prover_time = start.elapsed().as_millis();

    let start = Instant::now();
    assert!(S::verify_share(&vk, &cm, 0, &shares[0]).unwrap());
    let verifier_time = start.elapsed().as_millis();

    let per_node_comm_overhead =
        shares[0].proof.serialized_size(Compress::No) + cm.serialized_size(Compress::No);

    let per_node_comm =
        data.serialized_size(Compress::No) * blowup / num_nodes + per_node_comm_overhead;

    table.add_row(row![
        S::name(),
        num_nodes,
        l,
        k,
        n,
        (1 << block_log_size) * Fr::MODULUS_BIT_SIZE / (8 * 2u32.pow(20)),
        prover_time,
        per_node_comm_overhead as f64 / 1024.0,
        per_node_comm as f64 / 1024.0,
        verifier_time
    ]);
}

/// give the log number of fields, figure the balanced/optimal shape for FRIDA
/// Returns (log_k, log_L), namely log_width, log_height
#[inline]
fn frida_shape_heuristic(log_size: usize) -> (usize, usize) {
    // for FRIDA, communication cost is lambda * (log^2(k) + L), thus we balance log^2(k) and L
    // instead of closed-form form, we start with large L and slowly decrease until <= log^2(k)
    // let mut log_l = log_size;
    // let mut log_k = 0usize;
    // while (1 << log_l) > log_k.pow(2u32) {
    //     log_l -= 1;
    //     log_k += 1;
    // }
    // assert_eq!(log_l + log_k, log_size);

    let log_l = 5;
    let log_k = log_size - log_l;
    (log_k, log_l)
}

/// Returns (log_k, log_L), namely log_width, log_height
#[inline]
fn conda_shape_heuristic(block_log_size: usize, num_nodes: usize) -> (usize, usize) {
    // TODO: double-check this! // same col as number of nodes after 4x blowup
    let mut log_k = num_nodes.ilog2() as usize - 2;
    if log_k % 2 != 0 {
        log_k -= 1;
    }
    let log_l = block_log_size - log_k;
    (log_k, log_l)
}

#[inline]
fn nnt_shape_heuristic(block_log_size: usize, num_nodes: usize) -> (usize, usize) {
    let log_k = num_nodes.ilog2() as usize - 2; // same col as number of nodes after 4x blowup
    let log_l = block_log_size - log_k;
    (log_k, log_l)
}

#[inline]
fn zoda_shape_heuristic(block_log_size: usize) -> (usize, usize) {
    let log_k = block_log_size / 2; // strictly square
    let log_l = block_log_size - log_k;
    (log_k, log_l)
}

#[inline]
fn fast_advz_shape_heuristic(block_log_size: usize, num_nodes: usize) -> (usize, usize) {
    nnt_shape_heuristic(block_log_size, num_nodes)
}

#[inline]
fn short_advz_shape_heuristic(block_log_size: usize, num_nodes: usize) -> (usize, usize) {
    let (log_k, log_l) = nnt_shape_heuristic(block_log_size, num_nodes);
    (log_k + 2, log_l - 2)
}

fn bench_ndss_all() {
    let header = row![
        "Scheme",
        "N",
        "l",
        "k",
        "n",
        "|M| (MB)",
        "prover (ms)",
        "per-node overhead. (KB)",
        "per-node (KB)",
        "verifier (ms)"
    ];

    let mut table = Table::new();
    table.add_row(header.clone());

    let block_log_size = 22usize;
    let num_nodes = 1024;

    let (log_k, log_l) = frida_shape_heuristic(block_log_size);
    bench_ndss_helper::<FridaVRS<Fr>>(&mut table, num_nodes, log_k, log_l);

    let (log_k, log_l) = zoda_shape_heuristic(block_log_size);
    bench_ndss_helper::<ZodaVRS<Fr>>(&mut table, num_nodes, log_k, log_l);

    let (log_k, log_l) = conda_shape_heuristic(block_log_size, num_nodes);
    bench_ndss_helper::<GxzVRS<Fr, MultilinearKzgPCS<Bn254>>>(&mut table, num_nodes, log_k, log_l);
    bench_ndss_helper::<GxzVRS<Fr, LightLigeroPCS<Bn254>>>(&mut table, num_nodes, log_k, log_l);

    let (log_k, log_l) = fast_advz_shape_heuristic(block_log_size, num_nodes);
    bench_ndss_helper::<AdvzVRS<Bn254>>(&mut table, num_nodes, log_k, log_l);

    let (log_k, log_l) = short_advz_shape_heuristic(block_log_size, num_nodes);
    bench_ndss_helper::<AdvzVRS<Bn254>>(&mut table, num_nodes, log_k, log_l);

    let (log_k, log_l) = nnt_shape_heuristic(block_log_size, num_nodes);
    bench_ndss_helper::<KzgNntVRS<Bn254>>(&mut table, num_nodes, log_k, log_l);
    bench_ndss_helper::<PedersenNntVRS<G1Projective>>(&mut table, num_nodes, log_k, log_l);

    table.printstd();
}

// handy temporary bench
fn bench_test() {
    let rng = &mut StdRng::from_seed([42; 32]);
    let k = 2usize.pow(12);
    let l = 2usize.pow(8);
    let n = 2usize.pow(14);

    // let pp = GxzVRS::<Fr, MultilinearKzgPCS<Bn254>>::setup(k, l, rng).unwrap();
    let pp = FridaVRS::<Fr>::setup(k, l, rng).unwrap();
    // let pp = AdvzVRS::<Bn254>::setup(k, l, rng).unwrap();

    let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    // let (pk, _vk) = GxzVRS::<Fr, MultilinearKzgPCS<Bn254>>::preprocess(&pp, k, l, &domain).unwrap();
    let (pk, _vk) = FridaVRS::preprocess(&pp, k, l, &domain).unwrap();
    // let (pk, _vk) = AdvzVRS::<Bn254>::preprocess(&pp, k, l, &domain).unwrap();

    let data = Matrix::rand(rng, k, l);

    let start = Instant::now();
    // let (cm, shares) = GxzVRS::<Fr, MultilinearKzgPCS<Bn254>>::compute_shares(&pk, &data).unwrap();
    let (cm, shares) = FridaVRS::compute_shares(&pk, &data).unwrap();
    // let (cm, shares) = AdvzVRS::<Bn254>::compute_shares(&pk, &data).unwrap();
    let total_prepare_time = start.elapsed().as_millis();

    let prover_time = total_prepare_time;

    // Communication consists of commitment size and opening proof per replica/share
    let communication =
        shares[0].proof.serialized_size(Compress::No) + cm.serialized_size(Compress::No);

    println!("l, k, n, prover (ms), communication (byte)");
    println!(
        "{}, {}, {}, {}, {}",
        l, k, domain.size, prover_time, communication
    );
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
    let max_height = 1 << log_l_choices.last().unwrap();
    let max_width = 1 << log_k_choices.last().unwrap();

    let mut pp = S::setup(max_width, max_height, rng).unwrap();

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
            pp = S::setup(k, l, rng).unwrap();
        }
        // fixed RS rate of 1/2
        let domain = Radix2EvaluationDomain::new(4 * k).unwrap();

        let (pk, vk) = S::preprocess(&pp, k, l, &domain).unwrap();

        let data = Matrix::rand(rng, k, l);

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
