default:
    just --list

# Benchmark VID or DAS, run `just bench [vid|das]`
bench *args:
    cargo run --bin bench --release -- {{args}}

