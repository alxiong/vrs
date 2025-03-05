default:
    just --list

# Benchmark VID or DAS, run `just bench [vid|das]`
bench *args:
    cargo run --bin bench --release -- {{args}}

# Print trace of certain tests
trace *test:
    cargo test --release {{test}} --features print-trace -- --nocapture
