# SPDX-License-Identifier: Apache-2.0
"""Serves the built UI and a fixture `/api/internal/*`, for the page snapshot check and for looking at the pages.

    python ui/tests/fixture_server.py [--port 0] [--mode server|embedded|profiles] [--role operator|admin]
    # prints the address; the token is `fixture`

The routes and JSON shapes are those of `krishiv_server::internal`; the data is `fixtures.json`. The event stream
sends the live list once a second, as the real one does. `--mode` chooses which console the fixture is: `embedded`
reports one session and lists only its queries; `profiles` has the history and a query's page and refuses the rest,
as the real sources do. One more query has a page, in every mode: the stored profile of TPC-H Q9 (`Q9_ID`).
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
MODES = ("server", "embedded", "profiles")
SESSION_ID = "01932a40-1a00-7a01-8001-5e5510000001"
# What a console over stored profiles answers: the history, a query's page and the health the page reads its mode from.
# TPC-H Q9 as a stored profile: the document the Python viewer's tests read, at SF 0.01.
Q9_ID = "01932a40-1a00-7a01-8001-000000000009"
# A query whose plan arrives with the third answer; the count of answers so far.
LATE_PLAN_ID = "01932a40-1a00-7a01-8001-00000000000b"
LATE_PLAN_ASKED = [0]
# A query the fixture answers with no plan tree.
NO_PLAN_ID = "01932a40-1a00-7a01-8001-00000000000a"
NO_PLAN_ASKED = [0]
Q9_PROFILE = json.loads(
    (
        HERE.parent.parent / "python" / "tests" / "fixtures" / "q09-profile.json"
    ).read_text()
)
# A profile's metric -> the operator column the page reads, as `ProfilesSource::from_profiles` names them.
OPERATOR_METRICS = {
    "drivers": "drivers",
    "rows_in": "rows_in",
    "rows_out": "rows_out",
    "compute_ns": "compute_time_ns",
    "blocked_io_ns": "blocked_on_io_ns",
    "blocked_memory_ns": "blocked_on_memory_ns",
    "blocked_sink_ns": "blocked_on_sink_ns",
    "peak_reserved_bytes": "peak_reserved_bytes",
    "spilled_bytes": "spilled_bytes",
}
PROFILES_ROUTES = (
    "/api/internal/health",
    "/api/internal/history",
    "/api/internal/queries/",
    "/api/internal/compare",
)


def history(query: str) -> list[dict]:
    """The history under the page's filters, as `krishiv_server::internal` applies them: any part of the client's
    name, a tag by its key or as `key=value`."""
    asked = {k: v[0] for k, v in parse_qs(query).items()}
    client, tag = asked.get("client", "").lower(), asked.get("tag", "")
    return [
        q
        for q in DATA["history"]
        if client in (q.get("client_type") or "").lower()
        and (
            not tag
            or any(tag in (k, f"{k}={v}") for k, v in (q.get("tags") or {}).items())
        )
    ]


def stored_detail(query_id: str, profile: dict, sql: str) -> dict:
    """A query's page from its stored profile, as the console over stored profiles answers it: the profile's
    operators in id order (each with its plan node and role), the one with the most compute time and the
    profile's plan tree. A stored profile without a `usage` member has
    no CPU time, peak memory or row count, and the page says so. This run has no warning: nothing spilled, no
    driver did most of an operator's work and nothing waited for memory."""
    operators = [
        {
            "operator_id": o["id"],
            "node_id": o["node"],
            "name": o["name"],
            "role": o["role"],
        }
        | {column: o["metrics"][metric] for column, metric in OPERATOR_METRICS.items()}
        for o in sorted(profile["operators"], key=lambda o: o["id"])
    ]
    hot = max(operators, key=lambda o: o["compute_ns"])
    return {
        "query": {
            "query_id": query_id,
            "state": "finished",
            "principal": "ann",
            "query_text": sql,
            "submitted_at": 1793988000000000,
            "wall_ns": profile["query"]["wall_time_ns"],
            "cpu_ns": None,
            "peak_memory_bytes": None,
            "rows_produced": None,
            "spill_bytes": None,
            "bytes_read": None,
        },
        "operators": operators,
        "hot_operator": {
            "operator_id": hot["operator_id"],
            "operator": hot["name"],
            "compute_ns": hot["compute_ns"],
        },
        # The server sends a node's id, name and children, never its detail text.
        "plan": [
            {"id": n["id"], "name": n["name"], "children": n["children"]}
            for n in profile["plan"]
        ],
    }


Q9_DETAIL = stored_detail(
    Q9_ID,
    Q9_PROFILE,
    (HERE.parent.parent / "bench" / "queries" / "tpch" / "q09.sql").read_text().strip(),
)


def route(
    path: str, mode: str = "server", role: str = "admin", query: str = ""
) -> object | None:
    d = DATA
    parts = path.split("/")
    if mode == "profiles" and not path.startswith(PROFILES_ROUTES):
        return None
    if path == "/api/internal/environment":
        return ENVIRONMENT if role == "admin" else None
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
        return d["health"] | {
            "mode": mode,
            "session_id": SESSION_ID if mode == "embedded" else None,
            "role": role,
        }
    if path == "/api/internal/queries":
        return {"queries": d["live"]}
    if path == "/api/internal/history":
        return {"queries": history(query)}
    if path == "/api/internal/resources":
        return {"memory": d["resources"]["memory"], "caches": d["health"]["caches"]}
    if path == f"/api/internal/queries/{Q9_ID}":
        return Q9_DETAIL
    if path == f"/api/internal/queries/{NO_PLAN_ID}":
        # An ended query whose plan the source will never have (a server with no history store, a query
        # trimmed out of it): the page is told not to wait.
        NO_PLAN_ASKED[0] += 1
        return d["detail"] | {"plan": None, "plan_pending": False}
    if path == f"/api/internal/queries/{LATE_PLAN_ID}":
        # A query that has just ended: its profile is written a moment later, so the first answers carry no
        # plan and a later one does.
        LATE_PLAN_ASKED[0] += 1
        return d["detail"] | (
            {} if LATE_PLAN_ASKED[0] > 2 else {"plan": None, "plan_pending": True}
        )
    if path.startswith("/api/internal/queries/"):
        return d["detail"]
    return None


class Handler(BaseHTTPRequestHandler):
    csp = CSP

    def log_message(self, *_: object) -> None:
        pass

    @property
    def mode(self) -> str:
        return self.server.mode  # type: ignore[attr-defined]

    @property
    def role(self) -> str:
        return self.server.role  # type: ignore[attr-defined]

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

    def cookie_name(self) -> str:
        """The session cookie's name carries the port, as the server's does: cookies are not kept per port.

        The port is the listener's own, never text from the request (the real server takes it from a `Host`
        header it has already checked against its allowlist).
        """
        return f"krishiv_ui_token_{int(self.server.server_address[1])}"

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
                or f"{self.cookie_name()}={TOKEN}" in self.headers.get("Cookie", "")
            )
            if not authorised:
                return self.send_json(
                    401,
                    {
                        "error": "unauthorized",
                        "message": "a valid UI token is required",
                    },
                )
            self.server.asked.append(url.path)  # type: ignore[attr-defined]
            if url.path == "/api/internal/events" and self.mode != "profiles":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        if self.role == "admin":
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
            if url.path == "/api/internal/environment" and self.role != "admin":
                return self.send_json(
                    403,
                    {
                        "error": "forbidden",
                        "message": "the principal's role does not allow this",
                    },
                )
            body = route(url.path, self.mode, self.role, url.query)
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
                f"{self.cookie_name()}={TOKEN}; HttpOnly; SameSite=Strict; Path=/api/internal",
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
            elif "wide" in sql:
                # A result wider and longer than a phone's screen, with one value far longer than a line.
                RUNS[run_id] = {
                    "state": "finished",
                    "columns": [f"a_long_column_name_{i}" for i in range(12)],
                    "rows": [
                        [
                            "lorem ipsum " * 50 if (r, c) == (0, 1) else r * 1000 + c
                            for c in range(12)
                        ]
                        for r in range(40)
                    ],
                    "truncated": False,
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


def serve(
    port: int = 0, mode: str = "server", role: str = "admin"
) -> ThreadingHTTPServer:
    """The fixture server as a console of `mode` for a caller of `role` (the settings are the administrator's);
    `asked` collects the `/api/internal/*` paths pages requested."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.mode = mode  # type: ignore[attr-defined]
    server.role = role  # type: ignore[attr-defined]
    server.asked = []  # type: ignore[attr-defined]
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default="server")
    ap.add_argument("--role", choices=("operator", "admin"), default="admin")
    args = ap.parse_args()
    server = serve(args.port, args.mode, args.role)
    print(f"http://127.0.0.1:{server.server_address[1]}/?token={TOKEN}", flush=True)
    server.serve_forever()
