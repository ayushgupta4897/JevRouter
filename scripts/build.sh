#!/usr/bin/env bash
# Build the complete router from this one clone: vendored switchyard-server, its Python
# bindings, and jevjudge, all into a project-local .venv. No external `cargo install`,
# no network access beyond crates.io/PyPI for dependencies.
#
#   scripts/build.sh              # build everything
#   scripts/build.sh --skip-rust  # rebuild jevjudge only (fast iteration on the sidecar)
#
# Requires: Rust via rustup (vendor/switchyard/rust-toolchain.toml pins the version and
# rustup picks it up automatically), Python >= 3.11, and `uv` (or pip) for the venv.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENDOR="$ROOT/vendor/switchyard"
VENV="$ROOT/.venv"
SKIP_RUST="${1:-}"

if [[ "$SKIP_RUST" != "--skip-rust" ]]; then
  echo "== [1/3] building switchyard-server (release; ~2-3 min on first build)"
  ( cd "$VENDOR" && cargo build --release -p switchyard-server )
  echo "   binary: $VENDOR/target/release/switchyard-server"
fi

echo "== [2/3] project venv + switchyard Python bindings"
if [[ ! -d "$VENV" ]]; then
  if command -v uv >/dev/null; then uv venv "$VENV" -q; else python3 -m venv "$VENV"; fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PIP="uv pip"; command -v uv >/dev/null || PIP="pip"
$PIP install -q maturin
if [[ "$SKIP_RUST" != "--skip-rust" ]]; then
  ( cd "$VENDOR" && CARGO_TARGET_DIR="$VENDOR/target" maturin develop --release --quiet )
fi

echo "== [3/3] jevjudge (editable) + its dev/test deps"
$PIP install -q -e "$ROOT/jevjudge[dev]"

echo
echo "Built. Verify with:  scripts/e2e.sh"
echo "  switchyard-server : $VENDOR/target/release/switchyard-server"
echo "  python venv       : $VENV  (activate: source .venv/bin/activate)"
python3 -c "import switchyard, jevjudge; print('  switchyard pkg   :', switchyard.__file__); print('  jevjudge pkg     :', jevjudge.__file__)" 2>/dev/null || true
