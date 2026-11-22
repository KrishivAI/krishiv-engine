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
# The session a signed link of the embed check names (a space: the page must encode it).
EMBED_SESSION = "s 1"
# The paths the page serves as deep links (routes.json); the page itself is `index.html`.
DEEP_LINKS = (
    "health",
    "queries",
    "history",
    "sessions",
    "workload",
    "advisor",
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
    "default": {
        "tables": [
            {"name": "orders", "is_temporary": False},
            {"name": "lineitem", "is_temporary": False},
            {"name": "recent_orders", "is_temporary": True},
        ]
    },
    "orders": {
        "columns": [
            {"name": "o_orderkey", "data_type": "bigint"},
            {"name": "o_comment", "data_type": "string", "comment": XSS},
        ]
    },
}
# The sessions of a server's clients (`system.sessions`), as `/api/internal/sessions` orders them: the open ones
# first, each part by its last activity. The ids are the `session_id` of the fixture's queries.
SESSION_A = "01932a40-2b00-7b01-8001-5e55100000a1"
SESSION_B = "01932a40-2b00-7b01-8001-5e55100000b2"
SESSION_C = "01932a40-2b00-7b01-8001-5e55100000c3"
SESSION_XSS = "01932a40-2b00-7b01-8001-5e55100000d4"
MINUTE = 60_000_000


def fresh_sessions() -> list[dict]:
    """The sessions as they are before any is ended; times count back from now, so "ago" texts are short."""
    now = int(time.time() * 1_000_000)
    base = {
        "transport": "connect",
        "current_catalog": "spark_catalog",
        "current_namespace": "default",
        "tags": {},
        "queries_running": 0,
        "queries_started": 0,
        "closed_at": None,
        "close_reason": None,
        "delegated_by": None,
    }
    return [
        base
        | {
            "session_id": SESSION_A,
            "principal": "ann",
            "client_type": "_SPARK_CONNECT_PYTHON spark/4.2.0 os/linux python/3.12",
            "state": "active",
            "created_at": now - 90 * MINUTE,
            "last_active_at": now - 5_000_000,
            "tags": {"team": "risk", "nightly": ""},
            "queries_running": 1,
            "queries_started": 14,
        },
        base
        | {
            "session_id": SESSION_B,
            "principal": "bob",
            "client_type": "ADBC Flight SQL",
            "transport": "flight_sql",
            "state": "active",
            "created_at": now - 30 * MINUTE,
            "last_active_at": now - 2 * MINUTE,
            "current_namespace": "sales",
            "queries_running": 1,
            "queries_started": 3,
            "delegated_by": "bi-gateway",
        },
        base
        | {
            "session_id": SESSION_XSS,
            "principal": XSS,
            "client_type": XSS,
            "state": "active",
            "created_at": now - 10 * MINUTE,
            "last_active_at": now - 9 * MINUTE,
            "tags": {XSS: XSS},
            # The session the console's editor runs in: the console cannot end it.
            "endable": False,
        },
        base
        | {
            "session_id": SESSION_C,
            "principal": "cy",
            "client_type": "krishiv-cli/0.2.1",
            "state": "expired",
            "created_at": now - 300 * MINUTE,
            "last_active_at": now - 200 * MINUTE,
            "closed_at": now - 140 * MINUTE,
            "close_reason": "idle_timeout",
            "queries_started": 1,
        },
    ]


# A server's resource groups (`system.resource_groups`), a parent before its children. `root.etl` is at its
# concurrency limit and `root.etl.nightly` at its queue limit; `root.adhoc` was removed and still runs a query.
def group(group_id: str, parent: str | None, **over: object) -> dict:
    return {
        "group_id": group_id,
        "parent_id": parent,
        "state": "active",
        "scheduling_policy": "weighted_fair",
        "scheduling_weight": 1,
        "priority": 0,
        "workload_class": "interactive",
        "hard_concurrency_limit": 16,
        "max_queued": 100,
        "soft_memory_limit_bytes": None,
        "per_query_memory_limit_bytes": None,
        "scan_quota_bytes": None,
        "scan_quota_period_s": None,
        "running": 0,
        "queued": 0,
        "memory_reserved_bytes": 0,
        "scan_bytes_in_window": None,
        "scan_window_resets_at": None,
    } | over


# A server's partition advice (`/api/internal/advice`): one entry a table, its suggestions by rank.
ADVICE = [
    {
        "table_name": "spark_catalog.default.events",
        "table_queries": 120,
        "columns": [
            {
                "rank": 1,
                "column_name": "event_date",
                "column_queries": 100,
                "partitions": 365,
                "pruned_fraction": 0.8310502283105022,
            },
            {
                "rank": 2,
                "column_name": "region",
                "column_queries": 20,
                "partitions": 3,
                "pruned_fraction": 0.1111111111111111,
            },
        ],
    },
    {
        "table_name": "spark_catalog.sales.orders",
        "table_queries": 2140,
        "columns": [
            {
                "rank": 1,
                "column_name": "o_orderstatus",
                "column_queries": 1070,
                "partitions": 3,
                "pruned_fraction": 0.3333333333333333,
            },
            {
                "rank": 2,
                "column_name": XSS,
                "column_queries": 4,
                "partitions": 2,
                "pruned_fraction": 0.0009345794392523364,
            },
        ],
    },
    {
        "table_name": "iceberg.web.page_views_by_country_and_device_class",
        "table_queries": 7,
        "columns": [
            {
                "rank": 1,
                "column_name": "country",
                "column_queries": 7,
                "partitions": 190,
                "pruned_fraction": 0.9947368421052631,
            }
        ],
    },
]

GROUPS = [
    group(
        "root",
        None,
        soft_memory_limit_bytes=16 << 30,
        running=7,
        queued=5,
        memory_reserved_bytes=9 << 30,
    ),
    group(
        "root.interactive",
        "root",
        scheduling_weight=3,
        hard_concurrency_limit=8,
        per_query_memory_limit_bytes=2 << 30,
        running=2,
        memory_reserved_bytes=1 << 30,
    ),
    group(
        "root.etl",
        "root",
        scheduling_policy="query_priority",
        priority=2,
        workload_class="batch",
        hard_concurrency_limit=4,
        max_queued=20,
        soft_memory_limit_bytes=12 << 30,
        scan_quota_bytes=500 << 30,
        scan_quota_period_s=86400,
        scan_bytes_in_window=120 << 30,
        scan_window_resets_at=1_795_300_000_000_000,
        running=4,
        queued=5,
        memory_reserved_bytes=7 << 30,
    ),
    group(
        "root.etl.nightly",
        "root.etl",
        workload_class="batch",
        hard_concurrency_limit=2,
        max_queued=5,
        running=2,
        queued=5,
        memory_reserved_bytes=3 << 30,
    ),
    group(
        "root.adhoc",
        "root",
        state="removed",
        hard_concurrency_limit=2,
        max_queued=0,
        running=1,
        memory_reserved_bytes=1 << 30,
    ),
]
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
    name, a tag by its key or as `key=value`, any part of the statement (either case)."""
    asked = {k: v[0] for k, v in parse_qs(query).items()}
    client, tag = asked.get("client", "").lower(), asked.get("tag", "")
    session = asked.get("session", "")
    statement = asked.get("text", "").lower()
    return [
        q
        for q in DATA["history"]
        if client in (q.get("client_type") or "").lower()
        and (not session or q.get("session_id") == session)
        and statement in (q.get("query_text") or "").lower()
        and (
            not tag
            or any(tag in (k, f"{k}={v}") for k, v in (q.get("tags") or {}).items())
        )
    ]


COMPARED = (
    "wall_ns",
    "cpu_ns",
    "peak_memory_bytes",
    "rows_produced",
    "bytes_read",
    "spill_bytes",
)


def compare(query: str) -> tuple[int, dict]:
    """Two runs of the history side by side, as `krishiv_server::internal` answers `/api/internal/compare`: each
    run is picked by its id and, when the request names one, its session; the change is listed for the numbers
    both runs have. A session that is named but empty is refused, and a run that is not there is a 404."""
    asked = parse_qs(query, keep_blank_values=True)
    runs = []
    for side in ("a", "b"):
        session = asked.get(f"{side}_session")
        if session == [""]:
            return 400, {"error": "bad_request"}
        run = next(
            (
                q
                for q in DATA["history"]
                if [q["query_id"]] == asked.get(side)
                and (session is None or [q.get("session_id")] == session)
            ),
            None,
        )
        if run is None:
            return 404, {
                "error": "no_such_query",
                "message": "one of the queries is not in the history",
            }
        runs.append(run)
    delta = {
        k: runs[1][k] - runs[0][k]
        for k in COMPARED
        if isinstance(runs[0].get(k), int) and isinstance(runs[1].get(k), int)
    }
    return 200, {"a": runs[0], "b": runs[1], "delta": delta}


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
        "warnings": [],
    }


Q9_DETAIL = stored_detail(
    Q9_ID,
    Q9_PROFILE,
    (HERE.parent.parent / "bench" / "queries" / "tpch" / "q09.sql").read_text().strip(),
)


def route(
    path: str,
    mode: str = "server",
    role: str = "admin",
    query: str = "",
    sessions: list[dict] | None = None,
    groups: list[dict] | None = None,
    advice: list[dict] | None = None,
) -> object | None:
    d = DATA
    parts = path.split("/")
    if mode == "profiles" and not path.startswith(PROFILES_ROUTES):
        return None
    # Every client's sessions, the resource groups and the partition advice are a server's: the other consoles
    # have none of those pages.
    if path == "/api/internal/sessions" and mode == "server":
        return {"sessions": sessions or [], "truncated": False}
    if path == "/api/internal/groups" and mode == "server":
        return {"groups": groups or []}
    if path == "/api/internal/advice" and mode == "server":
        return {"tables": advice or [], "truncated": False}
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
            # Blank values are kept: `session=` names a session (an empty one), which no link was signed for.
            asked = parse_qs(url.query, keep_blank_values=True)
            # `ok` is a link for the id alone; `ok-session` was signed for EMBED_SESSION, as the server signs a
            # link that names a session, so the read must name it too.
            valid = asked.get("sig") == ["ok"] and "session" not in asked
            valid = valid or (
                asked.get("sig") == ["ok-session"]
                and asked.get("session") == [EMBED_SESSION]
            )
            if not valid:
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
            if url.path.startswith("/api/internal/queries/"):
                self.server.query_reads.append(self.path)  # type: ignore[attr-defined]
            if url.path == "/api/internal/health" and self.server.health_failures > 0:  # type: ignore[attr-defined]
                self.server.health_failures -= 1  # type: ignore[attr-defined]
                return self.send_json(
                    503,
                    {"error": "unavailable", "message": "the console is starting"},
                )
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
            if url.path == "/api/internal/compare":
                self.server.compares.append(url.query)  # type: ignore[attr-defined]
                return self.send_json(*compare(url.query))
            body = route(
                url.path,
                self.mode,
                self.role,
                url.query,
                self.server.sessions,  # type: ignore[attr-defined]
                self.server.groups,  # type: ignore[attr-defined]
                self.server.advice,  # type: ignore[attr-defined]
            )
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
        if path.startswith("/api/internal/sessions/") and path.endswith("/close"):
            # Ends an open session as the real server does: its row stays, as a killed one.
            session_id = path.split("/")[-2]
            for s in self.server.sessions:  # type: ignore[attr-defined]
                if s["session_id"] == session_id and s["state"] == "active":
                    cancelled = s["queries_running"]
                    s |= {
                        "state": "killed",
                        "close_reason": "admin_kill",
                        "closed_at": int(time.time() * 1_000_000),
                        "queries_running": 0,
                    }
                    self.server.closed.append(session_id)  # type: ignore[attr-defined]
                    return self.send_json(
                        200, {"closed": True, "queries_cancelled": cancelled}
                    )
            return self.send_json(
                404,
                {
                    "error": "not_open",
                    "message": "no open session with that id can be ended here",
                },
            )
        if path.startswith("/api/internal/sql/") and path.endswith("/cancel"):
            run_id = path.split("/")[-2]
            if RUNS.get(run_id, {}).get("state") == "running":
                RUNS[run_id] = {"state": "cancelled", "error": "cancelled"}
                return self.send_json(200, {"cancelled": True})
            return self.send_json(404, {"error": "not_running"})
        if path.startswith("/api/internal/queries/") and path.endswith("/cancel"):
            self.server.cancels.append(self.path)  # type: ignore[attr-defined]
        self.send_json(200, {"cancelled": True})


def serve(
    port: int = 0,
    mode: str = "server",
    role: str = "admin",
    health_failures: int = 0,
    groups: bool = True,
    advice: bool = True,
) -> ThreadingHTTPServer:
    """The fixture server as a console of `mode` for a caller of `role` (the settings are the administrator's);
    `asked` collects the `/api/internal/*` paths pages requested. The first `health_failures` requests for the
    health answer 503, as a console that is let in before its server can answer. Each server has its own
    sessions (`closed` collects the ids of the ones a page ended); without `groups` it is a server started
    with no resource-group file, without `advice` one whose history has none. `query_reads`, `compares` and `cancels`
    collect the addresses, parameters included, a page read a query's detail from and cancelled a query at."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.mode = mode  # type: ignore[attr-defined]
    server.role = role  # type: ignore[attr-defined]
    server.asked = []  # type: ignore[attr-defined]
    server.health_failures = health_failures  # type: ignore[attr-defined]
    server.sessions = fresh_sessions()  # type: ignore[attr-defined]
    server.closed = []  # type: ignore[attr-defined]
    server.groups = GROUPS if groups else []  # type: ignore[attr-defined]
    server.advice = ADVICE if advice else []  # type: ignore[attr-defined]
    server.query_reads = []  # type: ignore[attr-defined]
    server.compares = []  # type: ignore[attr-defined]
    server.cancels = []  # type: ignore[attr-defined]
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
