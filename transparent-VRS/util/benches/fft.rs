use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::UniformRand;
use rand::thread_rng;
use util::mul_group::Radix2Group;

fn main() {
    fft(20, 1, 1);
}

fn fft(log_d: usize, code_rate: usize, repetion: usize) {
    let mut rng = thread_rng();
    let v = (0..(1 << log_d))
        .map(|_| <Fr as UniformRand>::rand(&mut rng))
        .collect::<Vec<_>>();
    let mul_group = Radix2Group::new(log_d + code_rate);
    let start = Instant::now();
    for _ in 0..repetion {
        mul_group.fft(v.clone());
    }
    println!("{}", start.elapsed().as_millis());
}
