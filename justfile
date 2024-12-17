default:
    just --list

benchmark:
    @echo "Benchmarking ..."
    cargo test --release --features print-trace advz -- --nocapture
    @echo "Done!"

# Run a test named "tmp"
tmp:
    cargo test tmp -- --nocapture
