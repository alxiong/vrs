default:
    just --list

benchmark:
    @echo "Benchmarking ..."
    cargo bench
    @echo "Done!"

# Run a test named "tmp"
tmp:
    cargo test tmp -- --nocapture
