# Krishiv

> **Status: Pre-alpha. Under active construction; not usable yet.** There is no query engine, no release and no
> stable API. Everything may change without notice.

**Krishiv** is a Rust-native, Spark-compatible batch and streaming engine (Spark 4.x semantics), with incremental
materialized views and queryable streaming state.

- Runs embedded, single-node or distributed.
- Planned APIs: Python (PySpark-compatible), Rust, SQL, Spark Connect, Flight SQL and MCP.

## Current state

Phase P2a is in progress. The repository has a layered multi-crate workspace with layering, dependency, size and
lint gates in CI, the Spark error-class and config registries, a test kit with Spark golden fixtures, a benchmark
harness, and Python package and UI skeletons.

## Design records

References such as "ADR 0034" or "spec chapter" point to design records that are kept in a private repository for
now. See [CONTRIBUTING.md](CONTRIBUTING.md) for how this affects contributors.

## Contributing and community

- [Contributing guide](CONTRIBUTING.md) (build, test, DCO sign-off, PR rules)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)

## License

Licensed under the [Apache License, Version 2.0](LICENSE) (see also [NOTICE](NOTICE)). Contributions require a DCO
sign-off (`git commit -s`).

## Trademarks

Apache Spark is a trademark of the Apache Software Foundation. Krishiv is an independent project and is not
affiliated with or endorsed by the Apache Software Foundation.
