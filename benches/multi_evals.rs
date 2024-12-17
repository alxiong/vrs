//! Benchmark for fast multi-evaluations proof generation
//! `cargo bench --bench multi_evals`
#[macro_use]
extern crate criterion;

use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use jf_pcs::prelude::*;
use p3_maybe_rayon::prelude::*;

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
    let mut group = c.benchmark_group("Univariate Multi-evaluations");
    group.sample_size(10);

    let rng = &mut test_rng();
    let max_degree = 2usize.pow(15);
    let pp = UnivariateUniversalParams::<Bn254>::gen_srs_for_testing(rng, max_degree).unwrap();

    let degree = 2usize.pow(10);
    let (pk, _vk) = UnivariateUniversalParams::trim(&pp, degree).unwrap();

    for log_size in [12, 15, 20] {
        let domain = Radix2EvaluationDomain::new(2usize.pow(log_size)).unwrap();
        let poly = DensePolynomial::rand(degree, rng);

        group.bench_function(format!("::Fast: Deg=2^10, |Domain|=2^{}", log_size), |b| {
            b.iter(|| vrs::multi_evals::univariate::multi_eval(&pk, &poly, &domain))
        });
        group.bench_function(format!("::Naive: Deg=2^10, |Domain|=2^{}", log_size), |b| {
            b.iter(|| uv_multi_eval_naive(&pk, &poly, &domain))
        });
    }

    group.finish();
}

criterion_group!(benches, univariate);

criterion_main!(benches);
