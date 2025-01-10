use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::UniformRand;
use csv::Writer;
use frida::{Prover, Verifier};
use rand::{thread_rng, RngCore};
use util::mul_group::Radix2Group;

fn frida() {
    let mut rng = thread_rng();
    let mut wtr = Writer::from_path("frida.csv").unwrap();
    wtr.write_record(&["log_blob_size", "prover_time", "proof_size"])
        .unwrap();
    let poly_num = 32;
    for log_degree in 11..18 {
        let polies = (0..poly_num)
            .map(|_| {
                (0..(1 << log_degree))
                    .map(|_| <Fr as UniformRand>::rand(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coderate = 2;
        let groups = (0..log_degree)
            .rev()
            .map(|x| Radix2Group::new(x + 1 + coderate))
            .collect::<Vec<_>>();
        let challenges = {
            (
                <Fr as UniformRand>::rand(&mut rng),
                (0..log_degree)
                    .map(|_| <Fr as UniformRand>::rand(&mut rng))
                    .collect::<Vec<_>>(),
            )
        };
        let leaf_indices = (0..((100.0 / (2.0 / (1.0 + 0.5_f32.powi(coderate as i32))).log2())
            .ceil() as usize)
            - 20)
            .map(|_| rng.next_u32() as usize)
            .collect::<Vec<_>>();
        let now = Instant::now();
        for _ in 0..9 {
            let prover = Prover::new(&polies, &groups[0]);
            let (prover_state, _) = prover.commit_phase(&groups, &challenges);
            let _ = prover.sample(
                &prover_state,
                leaf_indices.clone(),
                1 << (log_degree + coderate),
            );
        }
        let prover = Prover::new(&polies, &groups[0]);
        let (prover_state, iopp_commits) = prover.commit_phase(&groups, &challenges);
        let query_results = prover.sample(
            &prover_state,
            leaf_indices.clone(),
            1 << (log_degree + coderate),
        );
        let prover_time = now.elapsed().as_micros() as usize / 10;
        let proof_size =
            iopp_commits.proof_size() + query_results.iter().map(|x| x.proof_size()).sum::<usize>();
        wtr.write_record(&[log_degree + 5, prover_time, proof_size].map(|x| x.to_string()))
            .unwrap();

        let commit = prover.commit();
        let verifier = Verifier::new(commit, poly_num, 1 << (log_degree + coderate - 1));
        verifier.verify(
            &groups,
            &challenges,
            leaf_indices,
            iopp_commits,
            query_results,
        );
    }
}

fn main() {
    frida();
}
