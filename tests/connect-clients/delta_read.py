# SPDX-License-Identifier: Apache-2.0
"""Spark-written Delta tables read through Spark Connect (P3 PR-53a leftover: the Delta read direction).

Report-only. ``tests/fixtures/delta-4.4.1/<table>`` were written by Spark 4.2.0 + delta-spark 4.4.1 and each has the
rows Spark read back (``EXPECTED.json``). A pyspark client connected to ``krishiv server`` reads every table three
ways (``SELECT .. FROM delta.`path```, ``spark.read.format("delta").load``, ``spark.read.load`` with the format
named in the options) and the rows must equal Spark's, compared as strings, as a set (Spark's order is not recorded
for every table). Tables Krishiv refuses on purpose (``refused`` in ``EXPECTED.json``) must fail with a Spark error
class rather than return rows. No Docker: the tables are committed.

    cd bench && uv run python ../tests/connect-clients/delta_read.py --remote sc://127.0.0.1:15002 --out /tmp/delta-read
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "delta-4.4.1"


FRACTION = re.compile(r"(\d{2}:\d{2}:\d{2})\.(\d*?)0+\b")


def norm(text: str) -> str:
    """Trailing zeros of a timestamp's fraction dropped (Spark's cast shows ``.5``, the recording ``.500000``)."""
    return FRACTION.sub(
        lambda m: m.group(1) + ("." + m.group(2) if m.group(2) else ""), text
    )


def render(value) -> str:
    """A JSON value of ``EXPECTED.json`` as the string Spark's ``CAST(.. AS STRING)`` shows.

    A JSON list is rendered as a struct (``{a, b}``). An array column would need ``[a, b]``; no
    fixture has one.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):  # a struct or an array of the recording
        return "{" + ", ".join(render(v) for v in value) + "}"
    return norm(str(value))


def expected_rows(expected: dict) -> tuple[list[str], list[tuple[str, ...]]]:
    """The column names and the rows (as strings) Spark recorded for a table."""
    cols = (
        sorted(expected["rows"][0]) if expected["rows"] else []
    )  # an empty record: nothing to compare
    return cols, sorted(tuple(render(r[c]) for c in cols) for r in expected["rows"])


def actual_rows(df, cols: list[str]) -> list[tuple[str, ...]]:
    from pyspark.sql import functions as F

    picked = df.select(
        *[F.coalesce(F.col(f"`{c}`").cast("string"), F.lit("NULL")) for c in cols]
    )
    return sorted(tuple(norm(v) for v in r) for r in picked.collect())


# The reader-feature gate (crates/krishiv-connectors-delta/src/features.rs).
REFUSAL_CLASS = "DELTA_UNSUPPORTED_FEATURES_FOR_READ"


def check_table(spark, name: str) -> dict:
    from pyspark.errors import PySparkException

    expected = json.loads((FIXTURES / name / "EXPECTED.json").read_text())
    path = str(FIXTURES / name)
    readers = {
        "sql": lambda: spark.sql(f"SELECT * FROM delta.`{path}`"),
        "format_load": lambda: spark.read.format("delta").load(path),
        "options_load": lambda: spark.read.load(path, format="delta"),
    }
    if expected.get("refused"):
        out = {}
        for k, read in readers.items():
            try:
                read().collect()
                out[k] = {
                    "status": "fail",
                    "detail": "read succeeded but Krishiv refuses this table",
                }
            except PySparkException as e:
                cls = e.getCondition() if hasattr(e, "getCondition") else None
                msg = str(e)
                if cls == REFUSAL_CLASS or REFUSAL_CLASS in msg:
                    out[k] = {"status": "pass", "detail": msg.splitlines()[0][:200]}
                else:
                    out[k] = {
                        "status": "fail",
                        "detail": f"refused with {cls or type(e).__name__}, expected {REFUSAL_CLASS}: "
                        + (msg.splitlines()[0][:150] if msg else ""),
                    }
        return out
    cols, want = expected_rows(expected)
    out = {}
    for k, read in readers.items():
        try:
            got = actual_rows(read(), cols)
            out[k] = (
                {"status": "pass", "rows": len(got)}
                if got == want
                else {
                    "status": "fail",
                    "detail": f"{len(got)} rows, Spark recorded {len(want)}; first difference "
                    f"{next(((g, w) for g, w in zip(got, want, strict=False) if g != w), None)}",
                }
            )
        except PySparkException as e:
            out[k] = {
                "status": "fail",
                "detail": str(e).splitlines()[0][:200] if str(e) else type(e).__name__,
            }
    return out


def tables() -> list[str]:
    docs = {
        p.parent.name: json.loads(p.read_text())
        for p in FIXTURES.glob("*/EXPECTED.json")
    }
    return sorted(t for t, d in docs.items() if d.get("rows") or d.get("refused"))


def summary(results: dict[str, dict]) -> str:
    cells = [(t, k, r) for t, rs in results.items() for k, r in rs.items()]
    passed = sum(1 for _, _, r in cells if r["status"] == "pass")
    lines = [f"Delta read through Connect: {passed}/{len(cells)} checks passed.", ""]
    lines += [
        f"- {t} / {k}: {r['detail']}" for t, k, r in cells if r["status"] != "pass"
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--out", default="target/delta-read")
    ap.add_argument(
        "--tables", help="comma list (default: every fixture table with recorded rows)"
    )
    a = ap.parse_args(argv)
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(a.remote).getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    names = a.tables.split(",") if a.tables else tables()
    results = {t: check_table(spark, t) for t in names}
    spark.stop()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "delta-read.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    text = summary(results)
    (out / "delta-read-summary.md").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
