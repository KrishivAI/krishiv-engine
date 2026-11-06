# SPDX-License-Identifier: Apache-2.0
"""Serves the built UI and a fixture `/api/internal/*`, for the page snapshot check and for looking at the pages.

    python ui/tests/fixture_server.py [--port 0]    # prints the port; the token is `fixture`

The routes and JSON shapes are those of `krishiv_server::internal`; the data is `fixtures.json`. The event stream
sends the live list once a second, as the real one does.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).parent
ASSETS = HERE.parent.parent / "crates" / "krishiv-ui-assets" / "assets"
DATA = json.loads((HERE / "fixtures.json").read_text())
TOKEN = "fixture"
# The same policy `krishiv_server::internal` sends: a page that needs more than this fails the snapshot check.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'"
)


def route(path: str) -> object | None:
    d = DATA
    if path == "/api/internal/health":
        return d["health"]
    if path == "/api/internal/queries":
        return {"queries": d["live"]}
    if path == "/api/internal/history":
        return {"queries": d["history"]}
    if path == "/api/internal/resources":
        return {"memory": d["resources"]["memory"], "caches": d["health"]["caches"]}
    if path.startswith("/api/internal/queries/"):
        return d["detail"]
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", CSP)
        super().end_headers()

    def send_json(self, status: int, body: object) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith("/api/internal/"):
            authorised = (
                self.headers.get("Authorization") == f"Bearer {TOKEN}"
                or f"token={TOKEN}" in url.query
            )
            if not authorised:
                return self.send_json(
                    401,
                    {
                        "error": "unauthorized",
                        "message": "a valid UI token is required",
                    },
                )
            if url.path == "/api/internal/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(
                            f"event: queries\ndata: {json.dumps({'queries': DATA['live']})}\n\n".encode()
                        )
                        self.wfile.flush()
                        time.sleep(1)
                except (BrokenPipeError, ConnectionResetError):
                    return None
            body = route(url.path)
            return (
                self.send_json(200, body)
                if body is not None
                else self.send_json(404, {"error": "not_found"})
            )
        rel = url.path.lstrip("/") or "index.html"
        f = ASSETS / rel
        if not f.is_file():
            return self.send_json(404, {"error": "not_found"})
        raw = f.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(f.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        self.send_json(200, {"cancelled": True})


def serve(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    server = serve(ap.parse_args().port)
    print(f"http://127.0.0.1:{server.server_address[1]}/?token={TOKEN}", flush=True)
    server.serve_forever()
