# Development task runner. Run `just` to list every recipe.
#
# Requires: cargo, uv (https://docs.astral.sh/uv/).

# The Python package lives in its own workspace member; uv drives it from there.
python_dir := "python"

# List available recipes.
default:
    @just --list

# Format Rust sources in place.
[group('lint')]
fmt:
    cargo fmt --all

# Check formatting without writing changes (as CI does).
[group('lint')]
fmt-check:
    cargo fmt --all -- --check

# Lint the core crate and its tests, denying warnings.
[group('lint')]
lint:
    cargo clippy -p zarr-cast-value --tests -- -D warnings

# Build the core crate.
[group('build')]
build:
    cargo build -p zarr-cast-value

# Build the Python extension module into the `python` project's venv.
[group('build')]
build-python:
    uv run --directory {{ python_dir }} --with maturin maturin develop

# Build a release wheel into `python/target/wheels`.
[group('build')]
wheel:
    uv run --directory {{ python_dir }} --with maturin maturin build --release

# Run the Rust test suite.
[group('test')]
test-rust:
    cargo test -p zarr-cast-value

# Run the Python test suite against a freshly built extension module.
[group('test')]
test-python: build-python
    uv run --directory {{ python_dir }} --group test pytest

# Run every test suite.
[group('test')]
test: test-rust test-python

# Run the Criterion benchmarks.
[group('test')]
bench:
    cargo bench -p zarr-cast-value

# Build the documentation site into `site/`.
[group('docs')]
docs:
    uvx --with mkdocstrings-python --with numpy zensical build --clean

# Serve the documentation site with live reload.
[group('docs')]
docs-serve:
    uvx --with mkdocstrings-python --with numpy zensical serve

# Build the Rust API docs and open them in a browser.
[group('docs')]
docs-rust:
    cargo doc -p zarr-cast-value --no-deps --open

# Run every check CI runs.
ci: fmt-check lint test

# Remove build artifacts.
clean:
    cargo clean
    rm -rf site
