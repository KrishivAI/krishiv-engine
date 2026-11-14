# SPDX-License-Identifier: Apache-2.0
"""Cloud object stores are built into the wheel (0.2.0 shipped without them: ``s3://``, ``gs://`` and ``az://``
failed with KRISHIV_UNSUPPORTED_FEATURE).

The offline test runs everywhere: a store URL must get past "not built in" and fail on the network instead. The round
trip runs when ``KRISHIV_TEST_S3_ENDPOINT`` names an S3 emulator with a bucket ``krishiv-test`` (MinIO, credentials
``minioadmin``/``minioadmin``; ``tests/scripts/object_store_emulators.sh`` and the wheels workflow start one)."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from krishiv.sql import SparkSession

pa = pytest.importorskip("pyarrow")

ENDPOINT = os.environ.get("KRISHIV_TEST_S3_ENDPOINT")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.mark.parametrize(
    "url",
    [
        "s3://krishiv-test/x.parquet",
        "gs://krishiv-test/x.parquet",
        "az://krishiv-test/x.parquet",
        "https://127.0.0.1:9/x.parquet",
        "hf://datasets/krishiv-test/x/x.parquet",
    ],
)
def test_cloud_urls_are_built_into_the_wheel(spark: SparkSession, url: str) -> None:
    # Point every store at a closed local port so nothing leaves the machine and the failure is quick.
    for key, value in {
        "fs.s3a.endpoint": "http://127.0.0.1:9",
        "fs.s3a.access.key": "k",
        "fs.s3a.secret.key": "s",
        "fs.s3a.endpoint.region": "us-east-1",
        "fs.s3a.connection.maximum": "1",
    }.items():
        spark.conf.set(key, value)
    with pytest.raises(Exception) as err:  # noqa: PT011 - the class depends on the store; the text is what matters
        spark.read.parquet(url).collect()
    assert "KRISHIV_UNSUPPORTED_FEATURE" not in str(err.value), err.value
    assert "hadoop" not in str(err.value).lower(), err.value  # PY-031: https:// once failed naming a Hadoop class


@pytest.mark.skipif(not ENDPOINT, reason="KRISHIV_TEST_S3_ENDPOINT is not set")
def test_s3_round_trip_against_an_emulator(spark: SparkSession) -> None:
    options: dict[str, Any] = {
        "fs.s3a.endpoint": ENDPOINT,
        "fs.s3a.access.key": "minioadmin",
        "fs.s3a.secret.key": "minioadmin",
        "fs.s3a.path.style.access": "true",
        "fs.s3a.endpoint.region": "us-east-1",
    }
    for key, value in options.items():
        spark.conf.set(key, value)
    url = f"s3://krishiv-test/wheel-{uuid.uuid4().hex}"
    spark.createDataFrame(pa.table({"id": [3, 1, 2], "name": ["c", None, "b"]})).write.parquet(url)
    rows = sorted((r["id"], r["name"]) for r in spark.read.parquet(url).collect())
    assert rows == [(1, None), (2, "b"), (3, "c")]
