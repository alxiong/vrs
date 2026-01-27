#!/usr/bin/env bash

set -euo pipefail

cleanup() {
    echo ""
    echo "Stopping container..."
    # Try graceful stop with 5s timeout, then force kill
    if ! docker stop -t 5 "$container" 2>/dev/null; then
        echo "Force killing container..."
        docker kill "$container" 2>/dev/null || true
    fi
    docker rm "$container" >/dev/null 2>&1 || true
    echo "Container cleaned up"
}
# Set up signal handlers
trap cleanup EXIT INT TERM

# Determine platform-specific nixery path
if [ "$(uname -m)" = "x86_64" ]; then
    NIXERY_META="shell"
else
    NIXERY_META="arm64/shell"
fi

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Start container in detached mode
container=$(docker run -d --init \
    -v "$WORKSPACE_DIR:/workspace:ro" \
    -v vrs-cargo:/.cargo \
    -v vrs-rustup:/.rustup \
    -v "$WORKSPACE_DIR/target:/workspace/target" \
    -e CARGO_HOME=/.cargo \
    -w /workspace \
    --security-opt seccomp=unconfined \
    "nixery.dev/$NIXERY_META/gcc/just/rustup/valgrind" \
    sh -c "
        export PATH=/.cargo/bin:\$PATH
        # Install gungraun-runner if not present
        if ! command -v gungraun-runner >/dev/null 2>&1; then
            echo 'Installing gungraun-runner...'
            cargo install --version 0.17.0 gungraun-runner
        fi
        exec just gg $*
    ")

echo "Started container: $container"

# Follow logs (this blocks until container exits or we get interrupted)
docker logs -f "$container"

# Wait for container to finish
docker wait "$container" >/dev/null 2>&1 || true
