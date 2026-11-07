# Deployment

| Mode | Status |
|---|---|
| Embedded: `pip install krishiv`, the engine runs inside your Python process | Runs today |
| Single server speaking Spark Connect | In development |
| Distributed: a scheduler and workers | In development |

## Embedded

Nothing to deploy: import the package. Supported CPUs are x86-64-v3 (Intel Haswell or AMD Zen and newer) and Armv8.2
(AWS Graviton2 or newer, Apple silicon). On an older x86-64 CPU install `krishiv-compat` instead (the same package,
imports as `krishiv`; install one of the two, not both). An unsupported CPU is detected at import time and reported by
name instead of crashing.

Memory use is bounded by `krishiv.*` settings (see [configuration](configuration.md)); operators spill to disk when the
limit is reached.

Nothing has been published to PyPI or crates.io yet. Until the first release, build the wheel from the repository
(`maturin build` in `python/`).
