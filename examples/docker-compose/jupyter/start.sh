#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Fetches the Krishiv container's self-signed certificate, then starts JupyterLab.
#
# The certificate is read from the server itself on this compose network (trust on first use),
# which is fine for a local quickstart and nothing else. compose.yaml points gRPC at the file
# (GRPC_DEFAULT_SSL_ROOTS_FILE_PATH) and sets SPARK_REMOTE, so a notebook connects with
# `SparkSession.builder.getOrCreate()`.
set -eu
python - "${GRPC_DEFAULT_SSL_ROOTS_FILE_PATH:-/tmp/krishiv-cert.pem}" <<'PY'
import os
import ssl
import sys
from urllib.parse import urlparse

remote = urlparse(os.environ.get("SPARK_REMOTE", "sc://krishiv:15002"))
with open(sys.argv[1], "w") as out:
    out.write(ssl.get_server_certificate((remote.hostname, remote.port or 15002)))
PY
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
  --IdentityProvider.token="${JUPYTER_TOKEN:-krishiv}" "$@"
