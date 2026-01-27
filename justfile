default:
    just --list

# Benchmark VID or DAS, run `just bench [vid|das]`
bench *args:
    cargo run --bin bench --release -- {{args}}

# Print trace of certain tests
trace *test:
    cargo test --release {{test}} --features print-trace -- --nocapture

_nixery_meta := if arch() == 'x86_64' { "shell" } else { "arm64/shell" }

# Run Gungraun benchmark
gg *ARGS:
    @just gg-{{os()}} {{ARGS}}

gg-macos *ARGS:
    {{justfile_directory()}}/scripts/dockerized-bench.sh {{ARGS}}

gg-linux *ARGS:
    RUSTFLAGS='--cfg gungraun' cargo bench --workspace {{ARGS}}
