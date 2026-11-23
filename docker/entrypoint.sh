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

for a in "$@"; do
  case "$a" in
    krishiv.auth.*|krishiv.server.tls.*|--host|--host=*)
      exec krishiv server "$@" ;; # the caller configures authentication and TLS itself
  esac
done

say() { echo "krishiv-image: $*" >&2; }
state=${KRISHIV_TLS_DIR:-/var/lib/krishiv/tls}

# Token.
if [ -n "${KRISHIV_AUTH_TOKEN:-}" ]; then
  secret="env:KRISHIV_AUTH_TOKEN"
else
  tokenfile=${KRISHIV_AUTH_TOKEN_FILE:-/run/secrets/krishiv_token}
  if [ -r "$tokenfile" ]; then
    secret="file:$tokenfile"
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

exec krishiv server --host 0.0.0.0 \
  --conf "krishiv.server.tls.certFile=$cert" \
  --conf "krishiv.server.tls.keyFile=$key" \
  --conf "krishiv.auth.tokens.docker.secret=$secret" \
  --conf "krishiv.auth.tokens.docker.principal=docker" \
  --conf "krishiv.auth.tokens.docker.kind=service" \
  --conf "krishiv.auth.tokens.docker.role=admin" \
  "$@"
