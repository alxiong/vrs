//! Testing O(n) FFT(v.rev()) algorithm vs O(n*log n) naive impl
//! `cargo bench --bench fft_rev`

#[macro_use]
extern crate criterion;

use ark_bn254::Fr;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
    UniformRand,
};
use itertools::Itertools;

use criterion::Criterion;
use vrs::multi_evals::fft_rev;

/// a CryptoRng
pub fn test_rng() -> StdRng {
    // arbitrary seed
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

fn fast_fft_rev(c: &mut Criterion) {
    let rng = &mut test_rng();

    let degrees = [2usize.pow(12), 2usize.pow(13)];
    let domain_sizes = [2usize.pow(14), 2usize.pow(15), 2usize.pow(16)];
    for (deg, domain_size) in degrees
        .into_iter()
        .cartesian_product(domain_sizes.into_iter())
    {
        let mut group =
            c.benchmark_group(format!("fft_rev::deg={},domain_size={}", deg, domain_size));
        group.sample_size(10);

        let domain = Radix2EvaluationDomain::<Fr>::new(domain_size).unwrap();
        let mut v = vec![];
        for _ in 0..deg + 1 {
            v.push(Fr::rand(rng));
        }
        let fft_result = domain.fft(&v);
        group.bench_function("fast", |b| {
            b.iter(|| fft_rev(&domain, deg + 1, &fft_result))
        });
        group.bench_function("naive", |b| {
            b.iter(|| {
                v.reverse();
                domain.fft(&v)
            })
        });

        group.finish();
    }
}

criterion_group!(benches, fast_fft_rev);

criterion_main!(benches);
