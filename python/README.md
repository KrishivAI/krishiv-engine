# krishiv (Python)

Python bindings for Krishiv. The native module is `krishiv._native`, built by
the `krishiv-python` crate (PyO3 0.30) with maturin.

## Supported Python

CPython 3.11 and newer. The default wheel is one `abi3` wheel (crate feature `abi3-py311`) covering 3.11+; the
free-threaded `cp314t` wheel is built separately. CI builds both and imports them (`wheels` job).

## Develop

```sh
cd python
uv venv && uv pip install maturin pytest
. .venv/bin/activate
maturin develop            # default abi3-py311 build
python -c "import krishiv; print(krishiv.__version__)"
pytest
```

## Free-threaded build (cp314t)

abi3 does not support free-threaded CPython, so that wheel is a separate,
non-abi3 build (disable the default `abi3-py311` crate feature):

```sh
cd python
uv venv --python 3.14t .venv-314t && uv pip install --python .venv-314t/bin/python maturin pytest
. .venv-314t/bin/activate
maturin develop --no-default-features     # or: maturin build --no-default-features
```

Verified on CPython 3.14.8t: imports and runs with the GIL disabled.

Publishing wheels (`maturin publish`, `uv publish`) is a hard stop; see
`/RELEASING.md`.
