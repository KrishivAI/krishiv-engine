#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Runs tests/scripts/flight_sql_jdbc/FlightSqlJdbc.java with the Arrow Flight SQL JDBC driver
# (arrow-java's flight-sql-jdbc-driver) against a running `krishiv server` (design P3-T101 §11).
# Called by `jdbc_flight_sql_tpch_q1` in crates/krishiv-cli/tests/flight_sql.rs; the arguments
# are passed to the program:
#
#   sh tests/scripts/flight_sql_jdbc.sh <jdbc url> <q1> <q1 with ?> <date for ?>
#
# The driver jar is downloaded once from Maven Central into
# ${XDG_CACHE_HOME:-~/.cache}/krishiv/flight-sql-jdbc and checked against its pinned SHA-256.
# The program runs on the host's `java` (21 or later: the single-file source launcher) when
# there is one, else in the digest-pinned eclipse-temurin:21-jdk container on the host network.
set -eu

VERSION=19.0.0
# Fails closed until the hash of the downloaded jar is pinned here.
JAR_SHA256=UNPINNED
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

if command -v java >/dev/null 2>&1; then
  exec java "$opens" -cp "$jar" "$program" "$@"
fi
exec docker run --rm --network host \
  -v "$jar:/driver.jar:ro" -v "$program:/FlightSqlJdbc.java:ro" \
  "$IMAGE" java "$opens" -cp /driver.jar /FlightSqlJdbc.java "$@"
