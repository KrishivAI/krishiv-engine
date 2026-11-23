# SPDX-License-Identifier: Apache-2.0
"""Catalogs declared on the session builder (``krishiv.catalog.<name>.*``, the lakehouse guide's "Declare a
catalog"): an embedded session takes them as declarations of its catalog registry, not as configuration (they
were ``KRISHIV_UNKNOWN_CONFIG`` before).

The offline tests run a small Iceberg REST stub in this process (``/v1/config`` and the namespace and table
listings). The smoke runs when ``ICEBERG_REST_URI`` names a real Iceberg REST catalog
(``tests/scripts/iceberg_rest_fixture.sh`` starts the Apache fixture in Docker)."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from krishiv.errors import PySparkException
from krishiv.sql import SparkSession

REST_URI = os.environ.get("ICEBERG_REST_URI")

# What the stub answers, by path: one namespace `rc` holding one table `t`.
ANSWERS: dict[str, object] = {
    "/v1/config": {"defaults": {}, "overrides": {}},
    "/v1/namespaces": {"namespaces": [["rc"]]},
    "/v1/namespaces/rc/tables": {"identifiers": [{"namespace": ["rc"], "name": "t"}]},
}


class _Stub(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = ANSWERS.get(self.path.split("?", 1)[0])
        data = json.dumps(body if body is not None else {"error": {"message": "no", "type": "x", "code": 404}})
        self.send_response(200 if body is not None else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def stub_uri() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _declared(uri: str, name: str = "lake") -> SparkSession.Builder:
    b = SparkSession.builder
    for k, v in {"type": "iceberg-rest", "uri": uri, "storage": "allowed-prefixes"}.items():
        b = b.config(f"krishiv.catalog.{name}.{k}", v)
    return b


def test_a_builder_declares_an_iceberg_rest_catalog(stub_uri: str) -> None:
    s = _declared(stub_uri).create()
    try:
        assert [r[0] for r in s.sql("SHOW NAMESPACES IN lake").collect()] == ["rc"]
        assert [r["tableName"] for r in s.sql("SHOW TABLES IN lake.rc").collect()] == ["t"]
        assert "lake" in [c.name for c in s.catalog.listCatalogs()]
        # A declaration is not configuration: it is neither listed nor settable afterwards.
        assert s.conf.get("krishiv.catalog.lake.type", None) is None
    finally:
        s.stop()


def test_the_default_catalog_and_spark_s_form(stub_uri: str) -> None:
    s = (
        SparkSession.builder.config("spark.sql.catalog.ice", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.ice.type", "rest")
        .config("spark.sql.catalog.ice.uri", stub_uri)
        .config("krishiv.catalog.ice.storage", "allowed-prefixes")
        .config("krishiv.catalog.default", "ice")
        .create()
    )
    try:
        assert s.catalog.currentCatalog() == "ice"
        assert [r[0] for r in s.sql("SHOW NAMESPACES").collect()] == ["rc"]
    finally:
        s.stop()


def test_a_refused_declaration_fails_the_build_by_its_own_class(stub_uri: str) -> None:
    with pytest.raises(PySparkException) as e:
        SparkSession.builder.config("krishiv.catalog.lake.type", "iceberg-rest").create()  # no uri
    assert e.value.getCondition() == "KRISHIV_INVALID_CONFIG_VALUE", e.value
    assert "needs a uri" in str(e.value)
    with pytest.raises(PySparkException) as e:
        _declared(stub_uri).config("krishiv.catalog.lake.nonsense", "1").create()
    assert e.value.getCondition() == "KRISHIV_INVALID_CONFIG_VALUE", e.value
    assert "has no setting `nonsense`" in str(e.value)


def test_a_session_without_declarations_is_unchanged() -> None:
    s = SparkSession.builder.config("spark.sql.ansi.enabled", "false").create()
    try:
        assert s.catalog.listCatalogs()[0].name == "spark_catalog"
        assert s.conf.get("spark.sql.ansi.enabled") == "false"
    finally:
        s.stop()


_DELTA_CATALOG = ("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")


def test_the_session_catalog_keys_stay_configuration(stub_uri: str) -> None:
    # delta-spark's documented builder setup configures `spark_catalog`, which is not a declaration.
    s = SparkSession.builder.config(*_DELTA_CATALOG).create()
    try:
        assert s.sql("SELECT 1 AS one").collect()[0]["one"] == 1
    finally:
        s.stop()
    s = _declared(stub_uri).config(*_DELTA_CATALOG).config("spark.sql.defaultcatalog", "lake").create()
    try:
        assert s.sql("SELECT current_catalog() AS c").collect()[0]["c"] == "lake"
        assert s.conf.get("spark.sql.defaultCatalog") == "lake"
        # Configuration keeps the key under its one spelling, as `krishiv server` does.
        listed = [r[0] for r in s.sql("SET").collect() if r[0].lower() == "spark.sql.defaultcatalog"]
        assert listed == ["spark.sql.defaultCatalog"]
        assert s.sql("SELECT 1 AS one").collect()[0]["one"] == 1
    finally:
        s.stop()


@pytest.mark.skipif(REST_URI is None, reason="no ICEBERG_REST_URI (tests/scripts/iceberg_rest_fixture.sh starts one)")
def test_smoke_against_an_iceberg_rest_catalog() -> None:
    # The 0.3 RC smoke's embedded Iceberg part (SMOKE-iceberg-builder, SMOKE-iceberg-list-namespaces).
    assert REST_URI is not None
    s = _declared(REST_URI).create()
    try:
        names = [r[0] for r in s.sql("SHOW NAMESPACES IN lake").collect()]
        assert isinstance(names, list)
        assert "lake" in [c.name for c in s.catalog.listCatalogs()]
    finally:
        s.stop()


def test_new_session_keeps_the_declarations(stub_uri: str) -> None:
    s = _declared(stub_uri, f"c{uuid.uuid4().hex[:8]}").create()
    try:
        other = s.newSession()
        assert len(other.catalog.listCatalogs()) == len(s.catalog.listCatalogs())
    finally:
        s.stop()
