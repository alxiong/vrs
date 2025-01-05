# Verifiable Reed-Solomon Encoding (VRS)

This repository contains implementations and benchmarks of various verifiable RS encoding schemes, which are commonly used in [Verifiable Information Dispersal (VID)](https://decentralizedthoughts.github.io/2024-08-08-vid/), [Data Availability Sampling](https://www.paradigm.xyz/2022/08/das) (DAS), and sometimes [Verifiable Secret Sharing](https://en.wikipedia.org/wiki/Verifiable_secret_sharing) (VSS).

## Get Started

Install [Rust](https://www.rust-lang.org/) and [`just` runner](https://just.systems/).

``` sh
cargo test --release
cargo bench
```

## Toolkit

_note: we only support `arkworks` backends for now, `zkcrypto` and `plonky3` support are left as future work._

- `bivariate::DensePolynomial`: the missing _dense_ polynomial implementation from arkworks, with basic arithmetic operations, including partial evaluation (at X or Y), mixed-sub and mixed-div by a univariate poly.
- `bkzg`: a KZG variant for bivariate polynomials, implementd `jf_pcs::PolynomialCommitmentScheme` with benchmarks. Additionally, proof and verification of `PartialEval` (at X or Y) are available.
- `multi_evals`: fast algorithm for computing n evaluation/opening proofs, both for the univariate (a.k.a. [FK23](https://eprint.iacr.org/2023/033)) and the bivariate case, with benchmarks.
- `gxz`: Efficient VRS in [GXZ'25]()
  - transparent: leveraging _evaluation consolidation_ techniques and any multilinear PCS
  - trusted: leveraging bKZG and fast `MultiPartialEval`
- `advz`: VRS scheme for VID using only univariate multi-evals.
- `nnt`: VRS scheme in [NNT'22](https://arxiv.org/pdf/2111.12323) for VID, instantiated with KZG and Pedersen.
- `fri`: FRI protocol (with grinding)
  - parameter selection suggested by [BT'24](https://eprint.iacr.org/2024/1161), flexible 80/100-bit security based on provable or conjectured soundness
- [] `frida`: VRS scheme in [FRIDA](https://eprint.iacr.org/2024/248) for DAS

## Benchmarks

View all the benchmark codes in [`./benches`](./benches) and some early results in [`./BENCH.md`](./BENCH.md)
    
