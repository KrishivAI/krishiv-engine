#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# The image's entrypoint. `krishiv server` refuses to bind 0.0.0.0 without authentication and TLS
# (ADR 0046), so for the server this script supplies both, then runs `krishiv server --host 0.0.0.0`:
#
#   token: $KRISHIV_AUTH_TOKEN, else the file $KRISHIV_AUTH_TOKEN_FILE (default /run/secrets/krishiv_token),
#          else a random one, generated and printed once. It authenticates as an admin principal.
#   TLS:   $KRISHIV_TLS_CERT and $KRISHIV_TLS_KEY (default /etc/krishiv/tls/{cert,key}.pem) if both
#          exist, else a self-signed certificate (localhost, 127.0.0.1, the container's host name),
#          generated on first start under $KRISHIV_TLS_DIR (default /var/lib/krishiv/tls); its path is
#          printed. Mount that directory to keep the certificate across containers.
#
# Any other command (`sql`, `--version`, ...), and `server` with its own --conf for authentication
# or TLS settings, runs as given.
set -eu

case "${1:-server}" in
  server) ;;
  *) exec krishiv "$@" ;;
esac
[ "$#" -eq 0 ] || shift

say() { echo "krishiv-image: $*" >&2; }
die() { say "$*"; exit 1; }

# The caller configures authentication or TLS itself: add nothing but the bind address.
host_args="--host 0.0.0.0"
own=
for a in "$@"; do
  case "$a" in
    krishiv.auth.*|krishiv.server.tls.*|--conf=krishiv.auth.*|--conf=krishiv.server.tls.*) own=1 ;;
    spark.connect.authenticate.*|--conf=spark.connect.authenticate.*) own=1 ;;
    --host|--host=*) host_args= ;;
  esac
done
# shellcheck disable=SC2086 # host_args is empty or two words
[ -z "$own" ] || exec krishiv server $host_args "$@"

state=${KRISHIV_TLS_DIR:-/var/lib/krishiv/tls}

# Token. A secret file is copied to a private file we own: Docker and compose mount secrets as
# root-owned 0444 or 0400, which the server refuses (group/other bits) or this user cannot read.
if [ -n "${KRISHIV_AUTH_TOKEN:-}" ]; then
  secret="env:KRISHIV_AUTH_TOKEN"
else
  tokenfile=${KRISHIV_AUTH_TOKEN_FILE:-/run/secrets/krishiv_token}
  if [ -r "$tokenfile" ] && [ -s "$tokenfile" ]; then
    mkdir -p "$state"
    (umask 077; cat "$tokenfile" >"$state/token.secret")
    secret="file:$state/token.secret"
  elif [ -n "${KRISHIV_AUTH_TOKEN_FILE:-}" ] || [ -e "$tokenfile" ]; then
    die "the token file $tokenfile is not readable by uid $(id -u) or is empty (mount it readable, e.g. compose secret uid: \"10001\", mode: 0400)"
  else
    mkdir -p "$state"
    tokenfile=$state/token
    if [ ! -s "$tokenfile" ]; then
      (umask 077; head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' >"$tokenfile")
      say "no KRISHIV_AUTH_TOKEN and no token file: generated a token in $tokenfile: $(cat "$tokenfile")"
    fi
    chmod 600 "$tokenfile"
    secret="file:$tokenfile"
  fi
fi

# TLS.
cert=${KRISHIV_TLS_CERT:-/etc/krishiv/tls/cert.pem}
key=${KRISHIV_TLS_KEY:-/etc/krishiv/tls/key.pem}
if [ ! -r "$cert" ] || [ ! -r "$key" ]; then
  [ -z "${KRISHIV_TLS_CERT:-}${KRISHIV_TLS_KEY:-}" ] || die "KRISHIV_TLS_CERT/KRISHIV_TLS_KEY are set but $cert or $key is not readable"
  cert=$state/cert.pem
  key=$state/key.pem
  if [ ! -r "$cert" ] || [ ! -r "$key" ]; then
    mkdir -p "$state"
    host=$(hostname)
    (umask 077; openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=localhost" \
      -addext "subjectAltName=DNS:localhost,DNS:$host,IP:127.0.0.1" -keyout "$key" -out "$cert" 2>/dev/null)
    chmod 644 "$cert"
    say "generated a self-signed certificate: $cert (copy it out to trust it: docker cp <container>:$cert .)"
  fi
fi

# A static service token must say when it stops working: default one year, like the certificate.
expires="${KRISHIV_AUTH_TOKEN_EXPIRES:-$(date -u -d '+365 days' +%F)}"

exec krishiv server $host_args \
  --conf "krishiv.server.tls.certFile=$cert" \
  --conf "krishiv.server.tls.keyFile=$key" \
  --conf "krishiv.auth.tokens.docker.secret=$secret" \
  --conf "krishiv.auth.tokens.docker.principal=docker" \
  --conf "krishiv.auth.tokens.docker.kind=service" \
  --conf "krishiv.auth.tokens.docker.role=admin" \
  --conf "krishiv.auth.principals.docker.expires=$expires" \
  --conf "krishiv.io.local.allowedRoots=${KRISHIV_ALLOWED_ROOTS:-/data}" \
  "$@"
