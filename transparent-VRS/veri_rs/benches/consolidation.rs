use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::UniformRand;
use csv::Writer;
use rand::thread_rng;
use veri_rs::consolidation::{VeriRsProver, VeriRsVerifier};

fn vid() {
    let mut rng = thread_rng();
    let mut wtr = Writer::from_path("vid_consolidation.csv").unwrap();
    wtr.write_record(&["log_blob_size", "encode_time", "prover_time", "proof_size"])
        .unwrap();
    for log_blob_size in 16..23 {
        let step = 3;
        let log_layer_num = log_blob_size - 8;
        let code_rate = 1;
        let prover = VeriRsProver::setup(log_blob_size, log_layer_num, code_rate, step);
        let data = (0..(1 << log_blob_size))
            .map(|_| <Fr as UniformRand>::rand(&mut rng))
            .collect::<Vec<_>>();
        let now = Instant::now();
        for _ in 0..9 {
            let _ = prover.encode(data.clone());
        }
        let codewords = prover.encode(data.clone());
        let encode_time = now.elapsed().as_micros() as usize / 10;
        let now = Instant::now();
        for _ in 0..9 {
            let _ = prover.prove(data.clone(), codewords.clone());
        }
        let proofs = prover.prove(data, codewords);
        let prover_time = now.elapsed().as_micros() as usize / 10;
        let symbol = proofs.n_th_replica(103);
        let proof_size = symbol.proof_size();
        wtr.write_record(
            &[log_blob_size, encode_time, prover_time, proof_size].map(|x| x.to_string()),
        )
        .unwrap();
        let verifier = VeriRsVerifier::setup(
            103,
            step,
            log_blob_size - log_layer_num + code_rate,
            code_rate,
        );
        assert!(verifier.verify(symbol));
    }
}

fn das() {
    let mut rng = thread_rng();
    let mut wtr = Writer::from_path("das_consolidation.csv").unwrap();
    wtr.write_record(&["log_blob_size", "encode_time", "prover_time", "proof_size"])
        .unwrap();
    for log_blob_size in 16..23 {
        let step = 3;
        let log_layer_num = 8;
        let code_rate = 1;
        let prover = VeriRsProver::setup(log_blob_size, log_layer_num, code_rate, step);
        let data = (0..(1 << log_blob_size))
            .map(|_| <Fr as UniformRand>::rand(&mut rng))
            .collect::<Vec<_>>();
        let now = Instant::now();
        for _ in 0..9 {
            let _ = prover.encode(data.clone());
        }
        let codewords = prover.encode(data.clone());
        let encode_time = now.elapsed().as_micros() as usize / 10;
        let now = Instant::now();
        for _ in 0..9 {
            let _ = prover.prove(data.clone(), codewords.clone());
        }
        let proofs = prover.prove(data, codewords);
        let prover_time = now.elapsed().as_micros() as usize / 10;
        let symbol = proofs.n_th_replica(103);
        let proof_size = symbol.proof_size();
        wtr.write_record(
            &[log_blob_size, encode_time, prover_time, proof_size].map(|x| x.to_string()),
        )
        .unwrap();
        let verifier = VeriRsVerifier::setup(
            103,
            step,
            log_blob_size - log_layer_num + code_rate,
            code_rate,
        );
        assert!(verifier.verify(symbol));
    }
}

fn vss() {
    let mut rng = thread_rng();
    let mut wtr = Writer::from_path("vss_consolidation.csv").unwrap();
    wtr.write_record(&["log_blob_size", "encode_time", "prover_time", "proof_size"])
        .unwrap();
    for log_blob_size in 16..21 {
        let step = 3;
        let log_layer_num = 1;
        let code_rate = 1;
        let prover = VeriRsProver::setup(log_blob_size, log_layer_num, code_rate, step);
        let data = (0..(1 << log_blob_size))
            .map(|_| <Fr as UniformRand>::rand(&mut rng))
            .collect::<Vec<_>>();
        let now = Instant::now();
        for _ in 0..9 {
            let _ = prover.encode(data.clone());
        }
        let codewords = prover.encode(data.clone());
        let encode_time = now.elapsed().as_micros() as usize / 10;
        let now = Instant::now();
        for _ in 0..9 {
            let _ = prover.prove(data.clone(), codewords.clone());
        }
        let proofs = prover.prove(data, codewords);
        let prover_time = now.elapsed().as_micros() as usize / 10;
        let symbol = proofs.n_th_replica(103);
        let proof_size = symbol.proof_size();
        wtr.write_record(
            &[log_blob_size, encode_time, prover_time, proof_size].map(|x| x.to_string()),
        )
        .unwrap();
        let verifier = VeriRsVerifier::setup(
            103,
            step,
            log_blob_size - log_layer_num + code_rate,
            code_rate,
        );
        assert!(verifier.verify(symbol));
    }
}

fn main() {
    // vid();
    // das();
    vss();
}
