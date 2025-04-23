default:
    just --list

# Benchmark VID or DAS, run `just bench [vid|das]`
bench *args:
    cargo run --bin bench --release -- {{args}}

# Print trace of certain tests
trace *test:
    cargo test --release {{test}} --features print-trace -- --nocapture

# Benchmark for NDSS against baseline
ndss:
    @# RAYON_NUM_THREADS=1 cargo run --bin bench --release --features print-trace -- ndss-base
    RAYON_NUM_THREADS=1 cargo run --bin bench --release -- ndss-base

# Benchmark for NDSS among all
ndss-all:
    RAYON_NUM_THREADS=1 cargo run --bin bench --release -- ndss-all
