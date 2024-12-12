default:
    just --list

benchmark:
    @echo "Benchmarking ..."
    @echo "Done!"

# Run a test named "tmp"
test-tmp:
    cargo test tmp -- --nocapture
