#!/usr/bin/env python3
"""Generate the golden-runner fixtures (P2-T152): shared test tables, HiveResult probes, sortedness cases.

Every expectation comes from a REAL Spark 4.2 run (the `krishiv-bench` container), never from docs or
memory. One `spark-shell` run (`goldentables/dump_fixtures.scala`) does three things.

1. It builds the tables `SQLQueryTestSuite.createTestTables` builds (`testdata`, `arraydata`, `mapdata`,
   `aggtest`, `onek`, `tenk1`), from the suite's own `test-data/postgresql/*.data` files, and the TPC-DS
   tables `TPCDSSchema` declares (empty, as the suite creates them for `pipe-operators.sql`). Tables are
   dumped with the schema Spark reports (`catalogString` types, nullability) and the rows it stores.
2. It runs the probe queries below through `HiveResult.hiveResultString` and records Spark's text next to
   the typed values, so the Rust cell formatter is checked against Spark's output.
3. It runs the sortedness queries and records `isSemanticallySorted` on their analysed plans.

The Spark test-data and schema sources are fetched from the pinned Spark commit (Apache-2.0, see NOTICE).
Output is deterministic (no timestamps, stable ordering) and holds no local paths or host names.

Usage (from repo root; take the bench lock, start Spark, stop it afterwards; see bench/README.md):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    flock "$KRISHIV_DATA/.bench.lock" nice -n 19 \
      python3 tests/scripts/gen_golden_fixtures.py --out tests/fixtures/spark-4.2.0
    docker compose -f bench/baselines/docker-compose.yml stop spark
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_function_matrix import container_image

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_golden_fixtures.py"
SPARK_COMMIT = "32f7299601108917fb01920a54e084595b7b3bf8"  # tag v4.2.0
RAW = f"https://raw.githubusercontent.com/apache/spark/{SPARK_COMMIT}"
TEST_DATA = ("agg.data", "onek.data", "tenk.data")
TEST_DATA_DIR = "sql/core/src/test/resources/test-data/postgresql"
TPCDS_SCHEMA = "sql/core/src/test/scala/org/apache/spark/sql/TPCDSSchema.scala"
REMOTE_DIR = "/tmp/krishiv_golden_data"
WAREHOUSE = "/tmp/krishiv_golden_warehouse"
TABLES = ("testdata", "arraydata", "mapdata", "aggtest", "onek", "tenk1")
DEFAULT_STYLE = "UTF-8"
STYLES = ("UTF-8", "BASIC", "BASE64", "HEX", "HEX_DISCRETE")
ZONES = (
    "UTC",
    "America/Los_Angeles",
    "Asia/Kolkata",
    "Europe/London",
    "Asia/Kathmandu",
    "Australia/Lord_Howe",
    "+10:00",
    "GMT+1",
)
# Zone IDs that Spark resolves in its own way: the Java short IDs, offsets in several forms, prefixes.
SPELLINGS = (
    "PST", "EST", "MST", "HST", "IST", "CST", "JST", "AET", "+5:30", "-9:00", "+0530", "-103045", "+10:00:30",
    "UTC+01:00", "GMT-8", "UT+2", "UTC", "GMT", "UT", "Z", "GMT+1:30", "Etc/GMT+5", "Asia/Calcutta",
)  # fmt: skip
BASE_ZONE = "America/Los_Angeles"  # the zone SQLQueryTestSuite runs in

# Probes whose text does not depend on the session time zone or the binary style.
PLAIN = [
    "select true, false, cast(null as boolean), cast(null as int), null",
    "select cast(-128 as tinyint), cast(127 as tinyint), cast(-32768 as smallint), cast(32767 as smallint)",
    "select 0, -1, 2147483647, -2147483648, 9223372036854775807L, -9223372036854775807L - 1L",
    # floating point: Java's Double.toString / Float.toString
    "select 0.0d, -0.0d, 1.0d, -1.0d, 100.0d, 123456.789d, 0.5d, 0.1d + 0.2d",
    "select double('NaN'), double('Infinity'), double('-Infinity')",
    "select 1.0E7d, 9999999.0d, 9999999.999999998d, 12345678.9d, 1.0E21d, 1.0E22d, 1.0E23d",
    "select 0.001d, 0.00099d, 9.999E-4d, 1.0E-4d, 1.0E-5d, 5.0E-5d, 1.2345E-7d",
    "select 1.7976931348623157E308d, 2.2250738585072014E-308d, 4.9E-324d, 1.0E-323d, 3.0E-323d, 2.0E-323d",
    "select 3.141592653589793d, 2.718281828459045d, 1.0d / 3.0d, 2.0d / 3.0d, 9007199254740993d",
    "select 100.0d * 100.0d, 1.0E6d, 123456789012345.0d, 0.30000000000000004d, 4.35d * 100",
    "select float(0), float(-0.0d), float(1), float(0.1), float(100), float(1.0E7), float(9999999)",
    "select float('NaN'), float('Infinity'), float('-Infinity'), float(0.001), float(0.0001), float(0.00099)",
    "select float(3.4028235E38d), float(1.17549435E-38d), float(1.4E-45d), float(2.8E-45d), float(1.0E-44d)",
    "select float(16777216), float(16777217), float(3.3333333), float(1) / float(3), float(123456789)",
    "select float(0.1) + float(0.2), float(1.0E10), float(1.5E-3), float(8388608.5)",
    # decimals keep their scale and never use an exponent
    "select 1.50bd, 0.000bd, 0bd, -0.5bd, 123.45bd, -123.45bd, 1.0bd, 0.00001bd",
    "select cast(0 as decimal(5,2)), cast(-0.5 as decimal(10,3)), cast(1E+3 as decimal(10,0)), 1e30bd",
    "select cast(12345678901234567890123456789012345678 as decimal(38,0)), "
    "cast(-12345678901234567890123456789012345678 as decimal(38,0))",
    "select cast(0.5 as decimal(38,38)), cast(-0.00000000000000000000000000000000000001 as decimal(38,38))",
    "select cast(1 as decimal(20,10)), cast(123456789.123456789 as decimal(18,9)), cast(5 as decimal(3,0))",
    # strings: quoted only when nested
    "select 'abc', '', ' ', 'a\\tb', 'x\\ny', 'na\\u00efve', '\\u20ac', '\\U0001F600', cast(null as string)",
    "select array('a', null, 'b', '', 'say \"hi\"'), array('x,y', '[1]', '{a:b}')",
    # dates (the year is the year of the era, never signed or padded beyond four digits)
    "select date'1970-01-01', date'2000-02-29', date'0001-01-01', date'9999-12-31', date'1582-10-15'",
    "select date_add(date'9999-12-31', 1), date_add(date'9999-12-31', 366), date_add(date'9999-12-31', 3000000)",
    "select date_sub(date'0001-01-01', 1), date_sub(date'0001-01-01', 366), date_sub(date'0001-01-01', 367)",
    "select date_sub(date'0001-01-01', 3000), date_sub(date'1970-01-01', 719528), date_sub(date'0001-01-01', 365 * 4)",
    "select date_sub(date'0001-01-01', 800000), date_add(date'9999-12-31', 36525000)",
    "select date'1970-01-01' + 1, date'1970-01-01' - 1, date'1900-03-01' - date'1900-02-28'",
    "select timestamp_ntz'2021-03-14 02:30:00', timestamp_ntz'2021-03-14 02:30:00.1', "
    "timestamp_ntz'2021-03-14 02:30:00.123456', timestamp_ntz'2021-03-14 02:30:00.120000'",
    "select timestamp_ntz'0001-01-01 00:00:00', timestamp_ntz'9999-12-31 23:59:59.999999', "
    "timestamp_ntz'1969-12-31 23:59:59.999999', timestamp_ntz'1970-01-01 00:00:00.000001'",
    "select timestamp_ntz'0001-01-01 00:00:00' - interval '1' second, "
    "timestamp_ntz'9999-12-31 23:59:59.999999' + interval '1' second",
    "select timestamp_ntz'0001-01-01 00:00:00' - interval '1' year, "
    "timestamp_ntz'0001-01-01 00:00:00' - interval '2' year, "
    "timestamp_ntz'0001-01-01 00:00:00' - interval '5000' year, "
    "timestamp_ntz'0001-01-01 00:00:00' - interval '200000' year",
    "select timestamp_ntz'9999-12-31 00:00:00' + interval '1' year, "
    "timestamp_ntz'9999-01-01 00:00:00' + interval '90000' year, "
    "timestamp_ntz'9999-01-01 00:00:00' + interval '200000' year",
    # binary: the style decides; nested values are not quoted
    "select array(x'0102', x'', x'ff')",
    "select named_struct('b', x'cafe')",
    "select map(1, x'01')",
    # nesting, nulls and quoting
    "select array(1, 2, 3), array(), array(null, 1), array(array(1, 2), array(), array(null))",
    "select array(1.5bd, null), array(double('NaN'), 1.0E10d, -0.0d), array(float(0.1), float(1.0E8))",
    "select array(date'2020-01-01', null), array(true, false, null), array(cast(1 as tinyint))",
    "select map(1, 'a'), map(), map(1, null), map('k', array(1, 2)), map(array(1), 'x')",
    "select map(3, 'c', 1, 'a', 2, 'b'), map('b', 1, 'a', 2, 'B', 3), map(10, 1, 9, 2, 100, 3, 1, 4)",
    "select map('a b', 1, 'a', 2, 'a,b', 3, 'a\"', 4, 'A', 5, '\\u00e9', 6, 'z', 7)",
    "select map(1.5bd, 2.5bd, 0.5bd, 1.5bd), map(true, 1, false, 2), map(double('NaN'), 1, 1.0d, 2, -1.0d, 3)",
    "select map(date'2020-01-02', 'b', date'2020-01-01', 'a'), map(1, map(2, map(3, 4)))",
    "select map('x', named_struct('a', 1, 'b', array('y'))), map(named_struct('a', 2), named_struct('a', 1))",
    "select named_struct('a', 1, 'b', 'x', 'c', cast(null as int)), named_struct('a', named_struct('b', array(1)))",
    "select array(named_struct('a', 1), named_struct('a', cast(null as int))), named_struct('a b', 1, 'c\"d', 2)",
    "select named_struct('s', 'he said \"hi\"', 'e', '', 'n', cast(null as string), 'd', 1.5bd, 'f', float(2.5))",
    "select named_struct('d', date'2020-01-01', 'm', map(1, 'a'), 'a', array(1), 'z', named_struct())",
    "select cast(null as array<int>), cast(null as map<int,int>), cast(null as struct<a:int>)",
    "select array(named_struct('a', map('k', array(1.5bd))))",
    "select * from values (1, 'a', 1.5d), (2, cast(null as string), cast(null as double)) as t(i, s, d)",
    "select 1 as a, 'x' as b, array(1) as c, map(1, 2) as d, named_struct('e', 1) as e",
    # intervals: Hive style, whatever the fields
    "select interval '1-2' year to month, interval '1' year, interval '14' month, interval -'1-2' year to month",
    "select interval '0' month, interval '178956970-7' year to month, interval -'178956970-8' year to month",
    "select interval '1 2:3:4.005' day to second, interval '1' day, interval '2:3' hour to minute",
    "select interval '3' minute, interval '4.5' second, interval -'1 2:3:4.5' day to second, interval '0' second",
    "select interval '106751991 04:00:54.775807' day to second, interval -'106751991 04:00:54.775808' day to second",
    "select interval '0 0:0:0.000001' day to second, interval -'0 0:0:0.000001' day to second",
    "select interval '100 10:20:30.123456' day to second, interval '1' hour, interval '36' hour",
    "select cast(null as interval day), array(interval '1' day, null), map(1, interval '1' month)",
    "select named_struct('a', interval '1-2' year to month, 'b', interval '1 2:3:4' day to second)",
    "select make_interval(), make_interval(1), make_interval(1, 2), make_interval(1, 2, 3, 4, 5, 6, 7.5)",
    "select make_interval(0, 1, 0, 2, 3, 0, 0), make_interval(0, 0, 0, 0, 0, 0, 0.000001)",
    "select make_interval(-1, -2, 0, -3, -4, -5, -6.5), make_interval(0, 13), make_interval(0, -13)",
    "select make_interval(0, 0, 0, 0, 25, 61, 61.000001), make_interval(0, 0, 0, 0, 0, 0, 3600)",
    "select make_interval(0, 0, 0, 0, 0, 0, -0.5), make_interval(0, 0, 0, 1, 0, 0, 0), make_interval(100)",
    "select array(make_interval(1)), map(1, make_interval(0, 1)), named_struct('a', make_interval(0, 0, 0, 1))",
    # empty result
    "select 1 where false",
]

# Probes rendered in each session time zone.
ZONED = [
    "select timestamp_micros(0), timestamp_micros(1609459200000000), timestamp_micros(1500000)",
    "select timestamp_micros(-1), timestamp_micros(1), timestamp_micros(-1000000), timestamp_micros(100000)",
    "select timestamp_micros(1615708799000000), timestamp_micros(1615708800000000)",  # LA spring-forward
    "select timestamp_micros(1636275599000000), timestamp_micros(1636279200000000)",  # LA fall-back
    "select timestamp_micros(-62135596800000000), timestamp_micros(-62135596800000001)",  # 0001-01-01 UTC
    "select timestamp_micros(-3000000000000000), timestamp_micros(-2717640000000000)",  # before 1884
    "select timestamp_micros(253402300799999999), timestamp_micros(253402300800000000)",  # 9999-12-31 UTC
    "select timestamp_micros(4102444800123456), timestamp_micros(2000000000000000000)",
    "select timestamp_micros(-9000000000000000000)",
    "select timestamp'2021-03-14 02:30:00', timestamp'2021-11-07 01:30:00', timestamp'2021-07-04 12:00:00.5'",
    "select timestamp'0001-01-01 00:00:00', timestamp'9999-12-31 23:59:59.999999'",
    "select timestamp'2021-01-01 00:00:00' + interval '1' second, timestamp'1969-12-31 23:59:59.999999'",
    "select array(timestamp_micros(0)), map(1, timestamp_micros(1500000))",
    "select named_struct('t', timestamp_micros(1609459200000001), 'd', date'2021-01-01')",
    "select cast(null as timestamp), timestamp_micros(0) - timestamp_micros(1)",
    "select timestamp_ntz'2021-03-14 02:30:00.5', array(timestamp_ntz'2021-03-14 02:30:00')",
]

SPELLING_PROBE = "select timestamp_micros(1609459200000000), timestamp_micros(-3000000000000000)"

# Probes rendered in each binary output style.
STYLED = [
    "select x'', x'00', x'0a0b', x'DEADBEEF', x'48656C6C6F', cast(null as binary)",
    "select x'e282ac', x'c3a9', x'f09f9880', x'7f', x'80', x'ff'",
    "select x'ff41c3', x'f0908d', x'c0af', x'eda080', x'f4908080', x'41e282', x'c328', x'e0a080', x'f0'",
    "select cast('abc' as binary), cast('na\\u00efve' as binary), encode('x', 'UTF-8'), unhex('4142')",
    "select array(x'0102', x'', x'ff', cast(null as binary)), named_struct('b', x'cafe'), map(1, x'01')",
    "select named_struct('a', x'41'), map(x'41', x'42')",
]

# Queries over the views t1(a, b) and t2(a, b): is the analysed plan "semantically sorted"?
SORTED = [
    # SQLQueryTestSuite's own cases
    "select * from t1",
    "select * from t1 order by a, b",
    "select * from t1 join t2 on t1.a = t2.a",
    "select * from t1 join t2 on t1.a = t2.a order by t1.a",
    "select a, max(b) from t1 group by a",
    "select a, max(b) from t1 group by a order by a",
    "select distinct a from t1",
    "select distinct a from t1 order by a",
    # sorts below, above and between the other operators
    "select a from t1 order by a",
    "select * from t1 where a > 1 order by a",
    "select a + 1 from (select * from t1 order by a) x",
    "select * from (select * from t1 order by a) x where a > 1",
    "select * from (select * from t1 order by a) x limit 1",
    "select * from t1 order by a limit 1",
    "select * from t1 limit 1",
    "select * from t1 sort by a",
    "select * from (select * from t1 sort by a) x",
    "select * from (select * from t1 order by a) x join t2 on x.a = t2.a",
    "select * from t1 join (select * from t2 order by a) y on t1.a = y.a",
    "select * from (select * from t1 order by a) x join t2 on x.a = t2.a order by x.a",
    "select count(*) from (select * from t1 order by a) x",
    "select a, count(*) from (select * from t1 order by a) x group by a",
    "select * from (select a, count(*) c from t1 group by a) x order by a",
    "select * from (select a, count(*) c from t1 group by a) x",
    "select * from (select * from t1 order by a) x order by b",
    "select * from (select * from t1 order by a) x where a in (select a from t2)",
    # subqueries inside expressions are not children of the plan
    "select * from t1 where a in (select a from t2 order by a)",
    "select * from t1 where exists (select 1 from t2 order by a)",
    "select a, (select b from t2 order by a limit 1) from t1",
    "select * from t1 where a = (select max(a) from t2)",
    "select * from t1 where a in (select a from t2) order by a",
    "select * from t1 where a in (select a from (select * from t2 order by a) y)",
    "select * from (select * from t1 where a in (select a from t2 order by a)) x",
    # CTEs: the definitions are children of the plan
    "with c as (select * from t1 order by a) select * from c",
    "with c as (select * from t1) select * from c order by a",
    "with c as (select * from t1) select * from c",
    "with c as (select * from t1 order by a) select * from t2",
    "with c as (select * from t1 order by a) select * from c join t2 on c.a = t2.a",
    "with c as (select a from t1 group by a), d as (select * from t2 order by a) select * from c",
    "select * from (with c as (select * from t1 order by a) select * from c) x",
]


def case_lines() -> list[dict]:
    cases: list[dict] = []
    for sql in PLAIN:
        cases.append({"kind": "probe", "sql": sql, "time_zone": BASE_ZONE, "binary_style": DEFAULT_STYLE})
    for sql in ZONED:
        cases.extend({"kind": "probe", "sql": sql, "time_zone": z, "binary_style": DEFAULT_STYLE} for z in ZONES)
    cases.extend(
        {"kind": "probe", "sql": SPELLING_PROBE, "time_zone": z, "binary_style": DEFAULT_STYLE} for z in SPELLINGS
    )
    for sql in STYLED:
        cases.extend({"kind": "probe", "sql": sql, "time_zone": BASE_ZONE, "binary_style": s} for s in STYLES)
    cases.extend({"kind": "sorted", "sql": sql} for sql in SORTED)
    return cases


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(f"{RAW}/{path}", timeout=120) as resp:
        return resp.read()


def tpcds_tables(scala: str) -> list[tuple[str, str]]:
    """`(table, column list)` of the `tableColumns` map of TPCDSSchema."""
    body = scala.split("tableColumns", 1)[1].split("tablePartitionColumns", 1)[0]
    tables = []
    for name, cols in re.findall(r'"(\w+)"\s*->\s*"""(.*?)"""\.stripMargin', body, re.DOTALL):
        lines = [ln.strip().removeprefix("|").strip() for ln in cols.strip().splitlines()]
        tables.append((name, " ".join(ln for ln in lines if ln)))
    return tables


def docker(container: str, *args: str, stdin: str | bytes | None = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "exec", *(["-i"] if stdin is not None else []), container, *args]
    return subprocess.run(cmd, input=stdin, capture_output=True, check=True, text=isinstance(stdin, str))


def put(container: str, remote: str, content: str | bytes) -> None:
    docker(container, "sh", "-c", f"cat > {remote}", stdin=content)


def run_dump(container: str, files: dict[str, str | bytes]) -> list[str]:
    docker(container, "sh", "-c", f"rm -rf {REMOTE_DIR} {WAREHOUSE} && mkdir -p {REMOTE_DIR}")
    try:
        for name, content in files.items():
            put(container, f"{REMOTE_DIR}/{name}", content)
        script = (Path(__file__).parent / "goldentables" / "dump_fixtures.scala").read_text(encoding="utf-8")
        put(container, f"{REMOTE_DIR}/dump_fixtures.scala", script)
        cmd = (
            f"cd {REMOTE_DIR} && /opt/spark/bin/spark-shell --master 'local[1]' --driver-memory 2g "
            f"--conf spark.sql.warehouse.dir={WAREHOUSE} --conf spark.ui.enabled=false "
            f"-I {REMOTE_DIR}/dump_fixtures.scala 2>/dev/null < /dev/null"
        )
        out = subprocess.run(
            ["docker", "exec", container, "sh", "-c", cmd], capture_output=True, text=True, timeout=3600, check=True
        ).stdout
    finally:
        docker(container, "sh", "-c", f"rm -rf {REMOTE_DIR} {WAREHOUSE}")
    lines = [ln for ln in out.splitlines() if ln.startswith("@@")]
    if "@@END" not in lines:
        raise SystemExit("spark-shell script did not complete:\n" + out[-2000:])
    return lines


def flat(v: object) -> bool:
    """Whether `v` prints on one line: scalars, lists of such, and objects of scalars."""
    if isinstance(v, dict):
        return not any(isinstance(x, dict | list) for x in v.values())
    return all(flat(x) for x in v) if isinstance(v, list) else True


def render(v: object, level: int = 0) -> str:
    """JSON with one row, column or probe value per line (a table of 10,000 rows stays readable)."""
    pad = " " * level
    if isinstance(v, dict):
        body = ",\n".join(f"{pad} {json.dumps(k)}: {render(x, level + 1)}" for k, x in v.items())
        return "{\n" + body + f"\n{pad}}}"
    if isinstance(v, list) and v and not all(not isinstance(x, dict | list) for x in v):
        body = ",\n".join(f"{pad} {json.dumps(x, ensure_ascii=False) if flat(x) else render(x, level + 1)}" for x in v)
        return "[\n" + body + f"\n{pad}]"
    return json.dumps(v, ensure_ascii=False)


def write_json(path: Path, doc: object) -> None:
    path.write_text(render(doc) + "\n", encoding="utf-8")


def build(lines: list[str], sources: dict[str, object], image: str) -> dict[str, dict]:
    version = next(ln.split(" ", 1)[1] for ln in lines if ln.startswith("@@VERSION "))
    if version != "4.2.0":
        raise SystemExit(f"fixtures under spark-4.2.0 need Spark 4.2.0, the container runs {version}")
    meta = {"format_version": FORMAT_VERSION, "spark_version": version, "generator": GENERATOR, "image": image}
    tables: dict[str, dict] = {}
    probes: list[dict] = []
    sorted_cases: list[dict] = []
    for ln in lines:
        tag, _, payload = ln[2:].partition(" ")
        if tag == "TABLE":
            t = json.loads(payload)
            tables[t["name"]] = {"name": t["name"], "columns": t["columns"], "rows": []}
        elif tag == "ROW":
            r = json.loads(payload)
            tables[r["table"]]["rows"].append(r["row"])
        elif tag == "PROBE":
            probes.append(json.loads(payload))
        elif tag == "SORTED":
            sorted_cases.append(json.loads(payload))
    failed = sorted({f"{p['sql']} -> {p['error']}" for p in probes + sorted_cases if "error" in p})
    if failed:
        raise SystemExit("queries Spark rejected:\n" + "\n".join(failed))
    docs = {}
    for name in (*TABLES, "tpcds-schemas"):
        if name == "tpcds-schemas":
            members = [t for n, t in tables.items() if n not in TABLES]
            doc = {"tables": sorted(members, key=lambda t: t["name"])}
        else:
            doc = tables[name]
        docs[f"golden-tables/{name}.json"] = {**meta, "sources": sources, **doc}
    docs["hive-result.json"] = {**meta, "probes": probes}
    docs["semantically-sorted.json"] = {**meta, "cases": sorted_cases}
    return docs


def self_check(out: Path) -> None:
    """Refuse to leave local paths or host/user names in the fixtures."""
    user = getpass.getuser()
    words = [w for w in (socket.gethostname(), user if len(user) >= 4 else "") if w]
    for f in sorted(out.rglob("*.json")):
        text = f.read_text(encoding="utf-8")
        if re.search(r"/home/|/Users/|[A-Za-z]:\\\\", text):
            raise SystemExit(f"{f}: contains a local path")
        hits = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text)]
        if hits:
            raise SystemExit(f"{f}: contains local host/user name {hits}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", default="krishiv-bench-spark")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0")
    args = ap.parse_args()

    files: dict[str, str | bytes] = {}
    sources = {}
    for name in TEST_DATA:
        data = fetch(f"{TEST_DATA_DIR}/{name}")
        files[name] = data
        sources[f"{TEST_DATA_DIR}/{name}"] = hashlib.sha256(data).hexdigest()
    scala = fetch(TPCDS_SCHEMA)
    sources[TPCDS_SCHEMA] = hashlib.sha256(scala).hexdigest()
    tpcds = tpcds_tables(scala.decode("utf-8"))
    files["tpcds_tables.tsv"] = "".join(f"{n}\t{c}\n" for n, c in tpcds)
    files["cases.jsonl"] = "".join(json.dumps(c) + "\n" for c in case_lines())

    lines = run_dump(args.container, files)
    docs = build(lines, {"spark_commit": SPARK_COMMIT, "sha256": sources}, container_image(args.container))
    if len(docs["golden-tables/tpcds-schemas.json"]["tables"]) != len(tpcds):
        raise SystemExit("TPC-DS table count differs from TPCDSSchema")
    out = Path(args.out)
    for rel, doc in sorted(docs.items()):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        write_json(out / rel, doc)
    self_check(out)
    print(f"wrote {len(docs)} fixtures under {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
