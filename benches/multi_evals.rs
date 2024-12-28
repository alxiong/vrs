//! Benchmark for fast multi-evaluations proof generation
//! `cargo bench --bench multi_evals`
#[macro_use]
extern crate criterion;

use ark_bn254::{Bn254, Fr};
use ark_ec::pairing::Pairing;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use jf_pcs::prelude::*;
use p3_maybe_rayon::prelude::*;
use vrs::{bivariate, bkzg::*};

use criterion::{BenchmarkId, Criterion};

/// a CryptoRng
pub fn test_rng() -> StdRng {
    // arbitrary seed
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
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

fn univariate(c: &mut Criterion) {
    let mut group = c.benchmark_group("Univariate MultiEval");
    group.sample_size(10);

    let rng = &mut test_rng();
    let max_degree = 2usize.pow(20);
    let pp = UnivariateUniversalParams::<Bn254>::gen_srs_for_testing(rng, max_degree).unwrap();

    for log_deg in 10..=20 {
        let domain = Radix2EvaluationDomain::new(2usize.pow(log_deg + 1)).unwrap();
        let poly = DensePolynomial::rand(1 << log_deg, rng);
        let (pk, _vk) = UnivariateUniversalParams::trim(&pp, 1 << log_deg).unwrap();

        group.bench_function(
            format!("Fast: Deg=2^{}, |Domain|=2^{}", log_deg, log_deg + 1),
            |b| b.iter(|| vrs::multi_evals::univariate::multi_eval(&pk, &poly, &domain)),
        );
    }

    group.finish();
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

fn bivariate(c: &mut Criterion) {
    let mut group = c.benchmark_group("BivariateMultiPartialEval");
    group.sample_size(10);

    let rng = &mut test_rng();
    let max_deg_x = 64;
    let max_deg_y = 2u32.pow(16);
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();

    for log_deg_y in 10..=16 {
        let deg_x = max_deg_x;
        let deg_y = 2u32.pow(log_deg_y);
        let supported_degree = combine_u32(deg_x, deg_y);
        let (pk, _vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(2 * deg_y as usize).unwrap();

        let poly = bivariate::DensePolynomial::<Fr>::rand(deg_x as usize, deg_y as usize, rng);

        let table = vrs::multi_evals::bivariate::multi_partial_eval_precompute(&pk, &domain);
        group.bench_with_input(
            BenchmarkId::new(
                "FastDAS",
                format!(
                    "deg_x={}, deg_y=2^{}, |Domain|=2^{}",
                    deg_x,
                    log_deg_y,
                    log_deg_y + 1
                ),
            ),
            &(poly, domain, table),
            |b, (poly, domain, table)| {
                b.iter(|| {
                    vrs::multi_evals::bivariate::multi_partial_eval_with_table::<Bn254>(
                        &poly, &domain, &table,
                    )
                })
            },
        );
        // group.bench_function(
        //     format!(
        //         "Naive: deg_x={}, deg_y=2^{}, |Domain|=2^{}",
        //         deg_x,
        //         log_deg_y,
        //         log_deg_y + 1
        //     ),
        //     |b| b.iter(|| bv_multi_partial_eval_naive(&pk, &poly, &domain)),
        // );
    }
    group.finish();
}

fn bivariate_vid(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bivariate MultiPartialEval VID");
    group.sample_size(10);

    let rng = &mut test_rng();
    let max_deg_x = 1 << 14;
    let max_deg_y = 256;
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();

    for log_deg_x in 8..15 {
        let log_deg_y = 8;
        let deg_x = 1 << log_deg_x;
        let deg_y = max_deg_y;
        let supported_degree = combine_u32(deg_x, deg_y);
        let (pk, _vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(2 * deg_y as usize).unwrap();

        let poly = bivariate::DensePolynomial::<Fr>::rand(deg_x as usize, deg_y as usize, rng);

        let table = vrs::multi_evals::bivariate::multi_partial_eval_precompute(&pk, &domain);
        group.bench_function(
            format!(
                "Fast VID: deg_x={}, deg_y=2^{}, |Domain|=2^{}",
                deg_x,
                log_deg_y,
                log_deg_y + 1
            ),
            |b| {
                b.iter(|| {
                    vrs::multi_evals::bivariate::multi_partial_eval_with_table::<Bn254>(
                        &poly, &domain, &table,
                    )
                })
            },
        );
        // group.bench_function(
        //     format!(
        //         "Naive: deg_x={}, deg_y=2^{}, |Domain|=2^{}",
        //         deg_x,
        //         log_deg_y,
        //         log_deg_y + 1
        //     ),
        //     |b| b.iter(|| bv_multi_partial_eval_naive(&pk, &poly, &domain)),
        // );
    }
    group.finish();
}

criterion_group!(benches, univariate);
// criterion_group!(benches, bivariate, bivariate_vid);

criterion_main!(benches);
