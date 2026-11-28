#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Runs tests/scripts/flight_sql_jdbc/FlightSqlJdbc.java with the Arrow Flight SQL JDBC driver
# (arrow-java's flight-sql-jdbc-driver) against a running `krishiv server` (design P3-T101 §11).
# Called by `jdbc_flight_sql_tpch_q1` in crates/krishiv-cli/tests/flight_sql.rs; the arguments
# are passed to the program:
#
#   sh tests/scripts/flight_sql_jdbc.sh <jdbc url> <q1> <q1 with ?> <date for ?> \
#     [<url with user= and password=> <url with another user=>]
#
# The driver jar is downloaded once from Maven Central into
# ${XDG_CACHE_HOME:-~/.cache}/krishiv/flight-sql-jdbc and checked against its pinned SHA-256.
# The program runs on the host's `java` when it is 21 or later (the single-file source launcher),
# else in the digest-pinned eclipse-temurin:21-jdk container on the host network.
set -eu

VERSION=19.0.0
# Maven Central's flight-sql-jdbc-driver-19.0.0.jar (its published SHA-1 300f1bc6d866aebe730d493605b82cf600cc1d3d).
JAR_SHA256=d3beee43c613c457789825343368f652d570d76c08799dad38a43a10e569b57f
# eclipse-temurin:21-jdk, pinned by digest (multi-arch index), used only without a host java.
IMAGE=eclipse-temurin:21-jdk@sha256:db5cc6341a1d7bc40cdb51e1f0d9824f3bb2cc8107d29da421dc4df7d11f05f1
URL="https://repo1.maven.org/maven2/org/apache/arrow/flight-sql-jdbc-driver/$VERSION/flight-sql-jdbc-driver-$VERSION.jar"

cache="${XDG_CACHE_HOME:-$HOME/.cache}/krishiv/flight-sql-jdbc"
jar="$cache/flight-sql-jdbc-driver-$VERSION.jar"
if [ ! -f "$jar" ]; then
  mkdir -p "$cache"
  curl -sSfL -o "$jar.part" "$URL"
  mv "$jar.part" "$jar"
fi
if ! echo "$JAR_SHA256  $jar" | sha256sum -c --quiet -; then
  echo "flight_sql_jdbc.sh: $jar does not match its pinned SHA-256" >&2
  exit 1
fi

here=$(cd "$(dirname "$0")" && pwd)
program="$here/flight_sql_jdbc/FlightSqlJdbc.java"
# Arrow's memory module reads java.nio buffers' internals.
opens="--add-opens=java.base/java.nio=ALL-UNNAMED"

# A host java older than 21 (a CI image's default 17, say) is skipped for the container.
major=0
if command -v java >/dev/null 2>&1; then
  major=$(java -XshowSettings:properties -version 2>&1 | sed -n 's/^ *java.specification.version = //p')
fi
if [ "${major:-0}" -ge 21 ] 2>/dev/null; then
  # Java 24 and later refuse the `sun.misc.Unsafe` memory access Netty (inside the driver) uses for its
  # allocator ("ExceptionInInitializerError ... EmptyByteBuf.memoryAddress"); the runner image moved to
  # Java 25 on 2026-11-25. Java 21 does not know the option.
  if [ "$major" -ge 24 ]; then opens="$opens --sun-misc-unsafe-memory-access=allow"; fi
  # shellcheck disable=SC2086 # two options in one variable
  exec java $opens -cp "$jar" "$program" "$@"
fi
# `tpch URL QUERIES DATA` reads QUERIES in the program: mount it; DATA stays the host path,
# since the server (not this program) opens it.
if [ "${1:-}" = "tpch" ] && [ "$#" -eq 4 ]; then
  queries="$3"
  exec docker run --rm --network host \
    -v "$jar:/driver.jar:ro" -v "$program:/FlightSqlJdbc.java:ro" -v "$queries:/queries:ro" \
    "$IMAGE" java "$opens" -cp /driver.jar /FlightSqlJdbc.java "$1" "$2" /queries "$4"
fi
exec docker run --rm --network host \
  -v "$jar:/driver.jar:ro" -v "$program:/FlightSqlJdbc.java:ro" \
  "$IMAGE" java "$opens" -cp /driver.jar /FlightSqlJdbc.java "$@"
