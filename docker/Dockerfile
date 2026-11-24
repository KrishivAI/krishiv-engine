# SPDX-License-Identifier: Apache-2.0
# The official Krishiv image: `krishiv` with the server, Spark Connect, Arrow Flight SQL, the web
# console, Delta and Iceberg tables and the cloud object stores (S3, GCS, Azure).
#
#   docker buildx build -f docker/Dockerfile -t krishiv .          (context: the repository root)
#   docker run --rm -p 15002:15002 -p 8080:8080 krishiv ...         (see docker/README.md)
#
# Ports: 15002 Spark Connect and Arrow Flight SQL (one listener), 8080 admin (/healthz, /readyz,
# /metrics, the web console). The server refuses 0.0.0.0 without authentication and TLS (ADR 0046), so
# docker/entrypoint.sh supplies both: the token from KRISHIV_AUTH_TOKEN (or a secret file, or one it
# generates) and a self-signed certificate it generates on first start. See docker/README.md.

ARG RUST_VERSION=1.99.1

# Base images are pinned by digest; refresh with `docker pull <tag>` + `docker image inspect --format "{{index .RepoDigests 0}}" <tag>`.
FROM rust:${RUST_VERSION}-slim-bookworm@sha256:f2c004e3d2d8b547e8cf7ffbc29205385f4e71d23e53e4e383065db5e9f822cd AS build
ARG KRISHIV_GIT_SHA=unknown
ENV KRISHIV_GIT_SHA=${KRISHIV_GIT_SHA} \
    CARGO_INCREMENTAL=0 \
    CARGO_TERM_COLOR=never
WORKDIR /src
COPY . .
# The generic x86-64 / armv8 baseline (no target-cpu), so the image runs on any host of its arch.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --profile dist --locked -p krishiv-cli \
        --features server,connect,flight-sql,cloud,ui,delta,iceberg \
    && install -D target/dist/krishiv /out/krishiv

FROM debian:bookworm-slim@sha256:e67eeead736ff7bbe258ef088bff683635c561b1fcfefad0806c1df4ab49ce0e AS runtime
# ca-certificates: TLS to object stores; curl: the health check; openssl: the first-start certificate.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl openssl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 krishiv \
    && useradd --system --uid 10001 --gid krishiv --home-dir /var/lib/krishiv --create-home --shell /usr/sbin/nologin krishiv \
    && install -d -o 10001 -g 10001 /data /var/lib/krishiv/tls
VOLUME /data
COPY --from=build /out/krishiv /usr/local/bin/krishiv
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
USER 10001:10001
WORKDIR /var/lib/krishiv
EXPOSE 15002 8080
# `/healthz` answers over plain HTTP, or over HTTPS when TLS is configured (a self-signed
# certificate is fine from inside the container).
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz >/dev/null || curl -fskS https://127.0.0.1:8080/healthz >/dev/null || exit 1
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["server"]
