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

use criterion::Criterion;

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
    let max_degree = 2usize.pow(15);
    let pp = UnivariateUniversalParams::<Bn254>::gen_srs_for_testing(rng, max_degree).unwrap();

    let degree = 2usize.pow(10);
    let (pk, _vk) = UnivariateUniversalParams::trim(&pp, degree).unwrap();

    for log_size in [12, 15, 20] {
        let domain = Radix2EvaluationDomain::new(2usize.pow(log_size)).unwrap();
        let poly = DensePolynomial::rand(degree, rng);

        group.bench_function(format!("Fast: Deg=2^10, |Domain|=2^{}", log_size), |b| {
            b.iter(|| vrs::multi_evals::univariate::multi_eval(&pk, &poly, &domain))
        });
        group.bench_function(format!("Naive: Deg=2^10, |Domain|=2^{}", log_size), |b| {
            b.iter(|| uv_multi_eval_naive(&pk, &poly, &domain))
        });
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
    let mut group = c.benchmark_group("Bivariate MultiPartialEval");
    group.sample_size(10);

    let rng = &mut test_rng();
    let max_deg_x = 8;
    let max_deg_y = 2u32.pow(10);
    let hacky_supported_degree = combine_u32(max_deg_x, max_deg_y);
    let pp = BivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, hacky_supported_degree as usize)
        .unwrap();

    for log_deg_y in [10] {
        let deg_x = max_deg_x;
        let deg_y = 2u32.pow(log_deg_y);
        let supported_degree = combine_u32(deg_x, deg_y);
        let (pk, _vk) = BivariateKzgPCS::trim(&pp, supported_degree as usize, None).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(2 * deg_y as usize).unwrap();

        let poly = bivariate::DensePolynomial::rand(deg_x as usize, deg_y as usize, rng);

        group.bench_function(
            format!(
                "Fast: deg_x={}, deg_y=2^{}, |Domain|=2^{}",
                deg_x,
                log_deg_y,
                log_deg_y + 1
            ),
            |b| b.iter(|| vrs::multi_evals::bivariate::multi_partial_eval(&pk, &poly, &domain)),
        );
        group.bench_function(
            format!(
                "Naive: deg_x={}, deg_y=2^{}, |Domain|=2^{}",
                deg_x,
                log_deg_y,
                log_deg_y + 1
            ),
            |b| b.iter(|| bv_multi_partial_eval_naive(&pk, &poly, &domain)),
        );
    }
    group.finish();
}

// criterion_group!(benches, univariate, bivariate);
criterion_group!(benches, bivariate);

criterion_main!(benches);
