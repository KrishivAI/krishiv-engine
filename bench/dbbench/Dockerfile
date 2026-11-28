# SPDX-License-Identifier: Apache-2.0
# The image of db-benchmark dry runs (bench/dbbench/dryrun.sh): Ubuntu 24.04 with what the benchmark's
# _setup_utils/repro.sh installs that its launcher, its data generators and the duckdb and krishiv solutions need.
# It is not the benchmark's machine (c6id.metal); a dry run shows that every question completes and that the
# answers agree, not the published time.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && apt-get -y -qq install --no-install-recommends \
      r-base-core r-cran-data.table r-cran-bit64 r-cran-jsonlite \
      python3 python3-pip python3-venv virtualenv git curl ca-certificates time procps sudo \
    && rm -rf /var/lib/apt/lists/*
