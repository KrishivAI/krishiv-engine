---
description: "Install Krishiv: the Python package, the Docker image, the command-line binaries or the Rust crates, and the platforms and Python versions each supports."
---

# Install

Pick the form that matches how you will use Krishiv.

| You want | Install | Page |
|---|---|---|
| The engine inside your Python program | `pip install krishiv` | [Quickstart](quickstart.md) |
| A server for PySpark, JDBC or ADBC clients | the Docker image (Spark Connect, Flight SQL, Delta, Iceberg), or the `krishiv` binary (Spark Connect only) | [Deployment](../guides/deploy.md) |
| The `krishiv sql` shell | the `krishiv` binary | [Run SQL from the command line](../guides/run-sql-from-the-command-line.md) |
| Krishiv from Rust | `cargo add krishiv` | [Rust reference](../reference/rust/index.md) |

<!-- RELEASE-DAY: every command below is checked on a clean machine against the published 0.3.0 artifacts before the tag is announced. -->

## Python

```bash
pip install krishiv
```

Python 3.11 to 3.14 is supported, including the free-threaded 3.14t build. `pyarrow` is installed with the package.
`toPandas` also needs pandas and numpy: `pip install "krishiv[pandas]"`.

To connect to a running Krishiv server from stock PySpark you need only the thin client, not Krishiv:

```bash
pip install pyspark-client
```

## Docker

<!-- DOCKER-IMAGE: link the Docker guide here once `run-krishiv-in-docker.md` is on main. -->

The official image runs `krishiv server`: Spark Connect and Arrow Flight SQL on port `15002`, the admin API and web console
on `8080`. It runs as a non-root user and has a health check.

```bash
docker pull ghcr.io/krishivai/krishiv:0.3.0
```

Images exist for `linux/amd64` and `linux/arm64`. A server on a non-loopback address needs authentication and TLS before
it starts; see [Securing a single-node server](../guides/single-node-server-security.md).

## Command-line binaries

Each release attaches the `krishiv` binary to its [GitHub release](https://github.com/KrishivAI/krishiv-engine/releases),
with a `.sha256` file for each, an SBOM and a build provenance attestation.

The quickest way is the install script. It picks the build for your machine (Linux x86-64 and aarch64, macOS arm64; on
x86-64 Linux the `krishiv-compat` build when the CPU lacks AVX2), downloads it and `SHA256SUMS` from the release, checks the
checksum and only then installs `krishiv` into `~/.local/bin`:

```bash
curl -fsSL https://krishiv.ai/install.sh | sh
```

| Variable | Effect |
|---|---|
| `KRISHIV_VERSION` | Release to install, for example `0.3.1` (default: the latest) |
| `KRISHIV_INSTALL_DIR` | Directory for the binary (default: `$HOME/.local/bin`) |
| `KRISHIV_COMPAT=1` | Install the `krishiv-compat` build even on a CPU that could run the fast one |

To uninstall, delete the file: `rm ~/.local/bin/krishiv`.

To download by hand instead:

<!-- BINARIES: replace with the final asset names and the install one-liners from the binaries workflow (sonnet-8): Linux x86-64 and aarch64 (gnu and musl), macOS arm64 and x86-64, Windows x86-64. Verify the checksum and the attestation command. -->

```bash
# Replace <asset> with the file for your platform from the release page.
curl -fsSLO https://github.com/KrishivAI/krishiv-engine/releases/download/v0.3.0/<asset>
curl -fsSLO https://github.com/KrishivAI/krishiv-engine/releases/download/v0.3.0/<asset>.sha256
sha256sum -c <asset>.sha256
chmod +x <asset> && ./<asset> --version
```

The x86-64 Linux build needs a CPU with AVX2 (x86-64-v3) and checks this on start; a `krishiv-compat` build for older CPUs
is also attached.

## Rust

```bash
cargo add krishiv                 # the library
cargo install krishiv-cli         # the `krishiv` command, from source
```

Building needs Rust 1.97 or newer. Features are additive; the [Rust reference](../reference/rust/index.md) lists them.

## Platforms and versions

<!-- PLATFORMS: the table states what the RC acceptance run passed; Windows x86-64 and macOS x86-64 are in only if their CI jobs pass. -->

| Platform | Python wheel | CLI binary | Docker |
|---|---|---|---|
| Linux x86-64 (glibc and musl) | yes | yes | yes |
| Linux aarch64 | yes | yes | yes |
| macOS arm64 | yes | yes | no |
| macOS x86-64 | pending | pending | no |
| Windows x86-64 | pending | pending | no |

Python 3.11 to 3.14 on every platform with a wheel. A platform marked pending ships only if its release checks pass.

## What each artifact includes

| Artifact | Delta tables | Iceberg tables | Arrow Flight SQL | Cloud stores (`s3://` ...) |
|---|---|---|---|---|
| Docker image | yes | yes | yes | yes |
| `krishiv` release binary (`server`, `ui`) | no | no | no | no |
| Python wheel | yes | no | no | yes |

The release binary refuses a Delta table with `DATA_SOURCE_NOT_FOUND`; `pip install krishiv` reads and writes Delta
tables. For Iceberg tables or ADBC and JDBC clients, use the Docker image or build the `krishiv` binary from source with
the `iceberg`, `catalog-iceberg-rest` (Iceberg REST catalogs in the server) and `flight-sql` features, plus `delta` for
Delta tables.

## Check the install

With the Python package and CLI on the host (the pip or cargo routes):

```bash
python -c "import krishiv; print(krishiv.session().sql('SELECT 1 AS ok').collect())"
krishiv --version
```

With only the Docker image, start it on loopback and check its health endpoint (the image has no host CLI or Python package):

```bash
docker run -d --name krishiv -p 127.0.0.1:15002:15002 -p 127.0.0.1:8080:8080 ghcr.io/krishivai/krishiv:0.3.0
# The image serves TLS with a self-signed certificate (-k); retry while the server starts.
curl -fsSk --retry 30 --retry-delay 1 --retry-all-errors https://127.0.0.1:8080/healthz
```
