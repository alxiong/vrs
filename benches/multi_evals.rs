//! Benchmark for fast multi-evaluations proof generation
//! `cargo bench --bench multi_evals`

use ark_bn254::{Bn254, Fr};
use ark_ec::pairing::Pairing;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use jf_pcs::prelude::*;
use once_cell::sync::Lazy;
use p3_maybe_rayon::prelude::*;
use std::hint::black_box;
use vrs::{pcs::bkzg::*, poly::bivariate};

use gungraun::{library_benchmark, library_benchmark_group, main};

/// a CryptoRng
pub fn test_rng() -> StdRng {
    // arbitrary seed
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

// Expensive setup for univariate benchmarks - computed once
static UV_SETUP: Lazy<(UnivariateProverParam<Bn254>, DensePolynomial<Fr>)> = Lazy::new(|| {
    let rng = &mut test_rng();
    let max_degree = 2usize.pow(10);
    let pp = UnivariateUniversalParams::<Bn254>::gen_srs_for_testing(rng, max_degree).unwrap();
    let degree = 2usize.pow(10);
    let (pk, _vk) = UnivariateUniversalParams::trim(&pp, degree).unwrap();
    let poly = DensePolynomial::rand(degree, rng);
    (pk, poly)
});

struct UvBenchInput<'a> {
    pk: &'a UnivariateProverParam<Bn254>,
    poly: &'a DensePolynomial<Fr>,
    domain: Radix2EvaluationDomain<Fr>,
}

fn uv_setup<'a>(log_size: u32) -> UvBenchInput<'a> {
    let domain = Radix2EvaluationDomain::new(2usize.pow(log_size)).unwrap();
    UvBenchInput {
        pk: &UV_SETUP.0,
        poly: &UV_SETUP.1,
        domain,
    }
}

// evaluate all domain elements one-by-one naively
fn uv_multi_eval_naive<E: Pairing>(
    pk: &UnivariateProverParam<E>,
    poly: &DensePolynomial<E::ScalarField>,
    domain: &Radix2EvaluationDomain<E::ScalarField>,
) -> Vec<UnivariateKzgProof<E>> {
    let elements = domain.elements().collect::<Vec<_>>();
    elements
        .par_iter()
        .map(|point| {
            let (proof, _eval) = UnivariateKzgPCS::open(pk, poly, point).unwrap();
            proof
        })
        .collect()
}

#[library_benchmark]
#[bench::size_12(uv_setup(12))]
#[bench::size_15(uv_setup(15))]
#[bench::size_20(uv_setup(20))]
fn univariate_fast(input: UvBenchInput) -> Vec<UnivariateKzgProof<Bn254>> {
    black_box(vrs::multi_evals::univariate::multi_eval(
        &input.pk,
        &input.poly,
        &input.domain,
    ))
}

#[library_benchmark]
#[bench::size_12(uv_setup(12))]
#[bench::size_15(uv_setup(15))]
#[bench::size_20(uv_setup(20))]
fn univariate_naive(input: UvBenchInput) -> Vec<UnivariateKzgProof<Bn254>> {
    black_box(uv_multi_eval_naive(&input.pk, &input.poly, &input.domain))
}

// Expensive setup for bivariate benchmarks - computed once
static BV_SETUP: Lazy<BivariateProverParam<Bn254>> = Lazy::new(|| {
    let rng = &mut test_rng();
    let max_deg_x = 8;
    let max_deg_y = 2u32.pow(10);
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();
    let supported_degree = combine_u32(max_deg_x, max_deg_y);
    let (pk, _vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
    pk
});

struct BvBenchInput {
    pk: BivariateProverParam<Bn254>,
    poly: bivariate::DensePolynomial<Fr>,
    domain: Radix2EvaluationDomain<Fr>,
}

struct BvBenchInputWithTable {
    poly: bivariate::DensePolynomial<Fr>,
    domain: Radix2EvaluationDomain<Fr>,
    table: Vec<Vec<PartialEvalProof<Bn254>>>,
}

fn bv_setup(log_deg_y: u32) -> BvBenchInput {
    let rng = &mut test_rng();
    let deg_x = 8;
    let deg_y = 2u32.pow(log_deg_y);
    let domain = Radix2EvaluationDomain::<Fr>::new(2 * deg_y as usize).unwrap();
    let poly = bivariate::DensePolynomial::rand(deg_x, deg_y as usize, rng);
    BvBenchInput {
        pk: BV_SETUP.clone(),
        poly,
        domain,
    }
}

fn bv_setup_with_table(log_deg_y: u32) -> BvBenchInputWithTable {
    let rng = &mut test_rng();
    let deg_x = 8;
    let deg_y = 2u32.pow(log_deg_y);
    let domain = Radix2EvaluationDomain::<Fr>::new(2 * deg_y as usize).unwrap();
    let poly = bivariate::DensePolynomial::rand(deg_x, deg_y as usize, rng);
    let table = vrs::multi_evals::bivariate::multi_partial_eval_precompute(&BV_SETUP, &domain);
    BvBenchInputWithTable {
        poly,
        domain,
        table,
    }
}

// evaluate all domain elements (along Y) one-by-one naively
fn bv_multi_partial_eval_naive<E: Pairing>(
    pk: &BivariateProverParam<E>,
    poly: &bivariate::DensePolynomial<E::ScalarField>,
    domain: &Radix2EvaluationDomain<E::ScalarField>,
) -> Vec<PartialEvalProof<E>> {
    let elements = domain.elements().collect::<Vec<_>>();
    elements
        .par_iter()
        .map(|point| {
            let (proof, _eval) = BivariateKzgPCS::partial_eval(pk, poly, point, false).unwrap();
            proof
        })
        .collect()
}

#[library_benchmark]
#[bench::deg_y_10(bv_setup_with_table(10))]
fn bivariate_fast(
    input: BvBenchInputWithTable,
) -> (Vec<PartialEvalProof<Bn254>>, Vec<DensePolynomial<Fr>>) {
    black_box(
        vrs::multi_evals::bivariate::multi_partial_eval_with_table::<Bn254>(
            &input.poly,
            &input.domain,
            &input.table,
        ),
    )
}

#[library_benchmark]
#[bench::deg_y_10(bv_setup(10))]
fn bivariate_naive(input: BvBenchInput) -> Vec<PartialEvalProof<Bn254>> {
    black_box(bv_multi_partial_eval_naive(
        &input.pk,
        &input.poly,
        &input.domain,
    ))
}

library_benchmark_group!(
    name = uv_multi_eval;
    benchmarks = univariate_fast, univariate_naive
);
library_benchmark_group!(
    name = bv_multi_eval;
    benchmarks = bivariate_fast, bivariate_naive
);

main!(library_benchmark_groups = uv_multi_eval, bv_multi_eval);
