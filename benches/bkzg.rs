//! Bivariate KZG related benchmark
//! `cargo bench --bench bkzg`

#[macro_use]
extern crate criterion;

use ark_bn254::{Bn254, Fr};
use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
    UniformRand,
};
use criterion::Criterion;
use itertools::Itertools;
use jf_pcs::prelude::*;
use vrs::{pcs::bkzg::*, poly::bivariate::DensePolynomial};

/// a CryptoRng
pub fn test_rng() -> StdRng {
    // arbitrary seed
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

fn partial_eval_x_vs_y(c: &mut Criterion) {
    let mut group = c.benchmark_group("bKZG::PartialEval");
    group.sample_size(10);

    let rng = &mut test_rng();
    let max_deg_x = 2u32.pow(10);
    let max_deg_y = 2u32.pow(10);
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();

    let (pk, _vk) = BivariateKzgPCS::trim(&pp, hacky_supported_degree as usize, None).unwrap();
    let poly = DensePolynomial::rand(max_deg_x as usize, max_deg_y as usize, rng);
    let (x, y) = (Fr::rand(rng), Fr::rand(rng));

    println!(
        "ℹ️ PartialEval of bivariate poly, deg_x={}, deg_y={}",
        poly.deg_x, poly.deg_y
    );
    let at_x = true;
    group.bench_function("at X", |b| {
        b.iter(|| BivariateKzgPCS::partial_eval(&pk, &poly, &x, at_x).unwrap())
    });
    group.bench_function("at Y", |b| {
        b.iter(|| BivariateKzgPCS::partial_eval(&pk, &poly, &y, !at_x).unwrap())
    });

    group.finish();
}

fn e2e(c: &mut Criterion) {
    let rng = &mut test_rng();
    let max_deg_x = 32;
    let max_deg_y = 2u32.pow(16);
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();

    let deg_x_choices = [8, 16, 32];
    let deg_y_choices = [2u32.pow(14), 2u32.pow(15), 2u32.pow(16)];
    for (deg_x, deg_y) in deg_x_choices
        .into_iter()
        .cartesian_product(deg_y_choices.into_iter())
    {
        let mut group = c.benchmark_group(format!("bKZG::e2e::deg_x={},deg_y={}", deg_x, deg_y));
        group.sample_size(10);

        let supported_degree = combine_u32(deg_x, deg_y);
        let (pk, vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();

        let poly = DensePolynomial::rand(deg_x as usize, deg_y as usize, rng);
        let cm = BivariateKzgPCS::commit(&pk, &poly).unwrap();
        group.bench_function("commit", |b| {
            b.iter(|| BivariateKzgPCS::commit(&pk, &poly).unwrap())
        });

        let point = (Fr::rand(rng), Fr::rand(rng));
        let (proof, eval) = BivariateKzgPCS::open(&pk, &poly, &point).unwrap();
        group.bench_function("eval", |b| {
            b.iter(|| BivariateKzgPCS::open(&pk, &poly, &point).unwrap())
        });

        group.bench_function("verify", |b| {
            b.iter(|| BivariateKzgPCS::verify(&vk, &cm, &point, &eval, &proof).unwrap())
        });

        let at_x = true;
        let (proof, partial_eval) =
            BivariateKzgPCS::partial_eval(&pk, &poly, &point.0, at_x).unwrap();
        group.bench_function("parital_eval", |b| {
            b.iter(|| BivariateKzgPCS::partial_eval(&pk, &poly, &point.0, at_x).unwrap())
        });
        group.bench_function("verify_parital", |b| {
            b.iter(|| {
                BivariateKzgPCS::verify_partial(&vk, &cm, &point.0, at_x, &partial_eval, &proof)
                    .unwrap()
            })
        });

        group.finish();
    }
}

criterion_group!(benches, partial_eval_x_vs_y, e2e);

criterion_main!(benches);
