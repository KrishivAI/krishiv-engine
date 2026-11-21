# SPDX-License-Identifier: Apache-2.0
"""Serves the built UI and a fixture `/api/internal/*`, for the page snapshot check and for looking at the pages.

    python ui/tests/fixture_server.py [--port 0]    # prints the port; the token is `fixture`

The routes and JSON shapes are those of `krishiv_server::internal`; the data is `fixtures.json`. The event stream
sends the live list once a second, as the real one does.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).parent
ASSETS = HERE.parent.parent / "crates" / "krishiv-ui-assets" / "assets"
DATA = json.loads((HERE / "fixtures.json").read_text())
TOKEN = "fixture"
# The content type comes from the file's suffix in this table, never from the request.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
}
# Relative name -> (file, content type), for every file in the asset folder now.
FILES = {
    f.relative_to(ASSETS).as_posix(): (
        f,
        CONTENT_TYPES.get(f.suffix, "application/octet-stream"),
    )
    for f in ASSETS.rglob("*")
    if f.is_file()
}
# The same policy `krishiv_server::internal` sends: a page that needs more than this fails the snapshot check.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'"
)


# The embedding console the embed check frames the profile from (its origin is the one allowed origin).
EMBED_PARENT = "http://parent.test"
# Framed by the page's CSP but absent from the allowed origins: its theme messages must be ignored.
EMBED_OTHER = "http://other.test"
# The paths the page serves as deep links (routes.json); the page itself is `index.html`.
DEEP_LINKS = (
    "health",
    "queries",
    "history",
    "resources",
    "catalog",
    "sql",
    "environment",
)


XSS = '<img src=x onerror="window.__xss=1"><script>window.__xss=1</script>'
DATA["live"].append(
    {
        "query_id": "01932a40-1a00-7a01-8001-aaaaaaaaaa99",
        "state": "running",
        "principal": XSS,
        "query_text": f"SELECT '{XSS}'",
        "submitted_at": 1,
    }
)
# Settings as the real server sends them: a sensitive key shows `***` (the planted values never leave the fixture).
ENVIRONMENT = {
    "version": "fixture",
    "settings": [
        {"key": "krishiv.auth.token", "value": "***", "redacted": True},
        {"key": "spark.sql.shuffle.partitions", "value": "200", "redacted": False},
        {"key": "db.password", "value": "***", "redacted": True},
        {"key": "spark.hadoop.fs.s3a.secret.key", "value": "***", "redacted": True},
    ],
}
CATALOG = {
    "catalogs": [{"name": "spark_catalog"}, {"name": XSS}],
    "spark_catalog": {"schemas": [{"name": "default"}]},
    "default": {"tables": [{"name": "orders"}, {"name": "lineitem"}]},
    "orders": {
        "columns": [
            {"name": "o_orderkey", "data_type": "bigint"},
            {"name": "o_comment", "data_type": "string", "comment": XSS},
        ]
    },
}
# Editor runs: a statement containing `sleep` stays running until it is cancelled.
RUNS: dict[str, dict] = {}


def route(path: str) -> object | None:
    d = DATA
    parts = path.split("/")
    if path == "/api/internal/environment":
        return ENVIRONMENT
    if path == "/api/internal/catalog/catalogs":
        return {"catalogs": CATALOG["catalogs"]}
    if len(parts) == 6 and parts[5] == "schemas":
        return {"schemas": CATALOG["spark_catalog"]["schemas"]}
    if len(parts) == 7 and parts[6] == "tables":
        return {"tables": CATALOG["default"]["tables"]}
    if len(parts) == 8 and parts[7] == "columns":
        return {"columns": CATALOG["orders"]["columns"]}
    if path.startswith("/api/internal/sql/") and parts[-1] in RUNS:
        return RUNS[parts[-1]]
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
    csp = CSP

    def log_message(self, *_: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", self.csp)
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
        self.csp = CSP
        if url.path == "/api/internal/embed/config":
            # Needs no token, as on the real server (the embedded page has no session).
            return self.send_json(
                200, {"allowed_origins": [EMBED_PARENT], "hide_attribution": False}
            )
        if url.path.startswith("/api/internal/embed/queries/"):
            if "sig=ok" not in url.query:
                return self.send_json(401, {"error": "unauthorized"})
            return self.send_json(200, DATA["detail"])
        if url.path.startswith("/api/internal/"):
            authorised = (
                self.headers.get("Authorization") == f"Bearer {TOKEN}"
                or f"token={TOKEN}" in url.query
                or f"krishiv_ui_token={TOKEN}" in self.headers.get("Cookie", "")
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
                            f"event: environment\ndata: {json.dumps(ENVIRONMENT)}\n\n".encode()
                        )
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
        # Only files that were in the asset folder at start-up are served, looked up by their relative name:
        # no path is built from the request, so nothing outside the folder can be named.
        name = url.path.lstrip("/") or "index.html"
        parts = name.split("/")
        framed = (
            len(parts) == 3
            and parts[0] == "queries"
            and parts[2] == "profile"
            and parse_qs(url.query).get("embed") == ["1"]
        )
        # The page's asset links are relative, so on a deep link they are asked for below the path: the root's files,
        # as the real server serves them.
        if "assets/" in name:
            name = name[name.rfind("assets/") :]
        elif name.endswith("/favicon.svg"):
            name = "favicon.svg"
        elif parts[0] in DEEP_LINKS or (parts[0] == "queries" and len(parts) <= 3):
            name = "index.html"
        served = FILES.get(name)
        if served is None:
            return self.send_json(404, {"error": "not_found"})
        file, content_type = served
        raw = file.read_bytes()
        if framed:
            # Only the embedded profile may be framed, and only from the allowed origin.
            self.csp = CSP.replace(
                "frame-ancestors 'none'",
                f"frame-ancestors {EMBED_PARENT} {EMBED_OTHER}",
            )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/internal/session":
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self.send_json(401, {"error": "unauthorized"})
            self.send_response(204)
            self.send_header(
                "Set-Cookie",
                f"krishiv_ui_token={TOKEN}; HttpOnly; SameSite=Strict; Path=/api/internal",
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if self.headers.get("X-Krishiv-Csrf") != "1":
            return self.send_json(403, {"error": "forbidden"})
        if path == "/api/internal/sql":
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            sql = json.loads(raw).get("sql", "")
            run_id = f"run{len(RUNS)}"
            if "sleep" in sql:
                RUNS[run_id] = {"state": "running"}
            elif "boom" in sql:
                RUNS[run_id] = {
                    "state": "failed",
                    "error": f"cannot parse {XSS}",
                    "error_class": "PARSE_SYNTAX_ERROR",
                }
            else:
                RUNS[run_id] = {
                    "state": "finished",
                    "columns": ["a", XSS],
                    "rows": [[1, XSS], [2, None]],
                    "truncated": False,
                }
            return self.send_json(202, {"run_id": run_id})
        if path.startswith("/api/internal/sql/") and path.endswith("/cancel"):
            run_id = path.split("/")[-2]
            if RUNS.get(run_id, {}).get("state") == "running":
                RUNS[run_id] = {"state": "cancelled", "error": "cancelled"}
                return self.send_json(200, {"cancelled": True})
            return self.send_json(404, {"error": "not_running"})
        self.send_json(200, {"cancelled": True})


def serve(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    server = serve(ap.parse_args().port)
    print(f"http://127.0.0.1:{server.server_address[1]}/?token={TOKEN}", flush=True)
    server.serve_forever()
