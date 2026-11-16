#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the input_file_name / input_file_block_start / input_file_block_length fixture.

Every case runs in a REAL Spark 4.2.0 spark-shell (`apache/spark:4.2.0`, throw-away `docker run --rm`) over files
written with fixed names under a mounted directory; rows are sorted strings, errors carry class, SQLSTATE and message.
The run is repeated on `local[1]` and `local[2]`; the fixture holds the `local[1]` result, and a case whose
`local[2]` result differs gets a `local2` entry. Feeds tests/scripts/gen_input_file_goldens.scala.

Usage (from repo root):

    python3 tests/scripts/gen_input_file_goldens.py [--out tests/fixtures/spark-4.2.0/input-file-goldens.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_input_file_goldens.py"
IMAGE = "apache/spark:4.2.0"
IFN = "input_file_name()"
NOBC = "spark.sql.autoBroadcastJoinThreshold=-1"
SPLIT = "spark.sql.files.maxPartitionBytes=100000"
V2 = "spark.sql.sources.useV1SourceList="
CACHE = "CACHE TABLE ca AS SELECT * FROM a"

# (conf "k=v|k=v", pre statements ";;"-separated, sql)
CASES: list[tuple[str, str, str]] = [
    ("", "", f"SELECT {IFN}, input_file_block_start(), input_file_block_length(), id FROM a"),
    ("", "", f"SELECT {IFN}, id FROM a WHERE {IFN} LIKE '%a1%'"),
    ("", "", f"SELECT id FROM a WHERE {IFN} LIKE '%a1%'"),
    ("", "", f"SELECT {IFN}, count(*) FROM a GROUP BY 1"),
    ("", "", f"SELECT max({IFN}) FROM a"),
    ("", "", f"SELECT {IFN}, n FROM (SELECT count(*) n FROM a)"),
    ("", "", f"SELECT DISTINCT {IFN} FROM a"),
    ("", "", f"SELECT {IFN} FROM (SELECT DISTINCT id FROM a)"),
    ("", "", f"SELECT {IFN}, id FROM a ORDER BY id"),
    ("", "", f"SELECT {IFN} FROM a ORDER BY id"),
    ("", "", f"SELECT {IFN} FROM (SELECT * FROM a ORDER BY id)"),
    ("", "", f"SELECT {IFN} FROM (SELECT * FROM a LIMIT 10)"),
    ("", "", f"SELECT {IFN} FROM a LIMIT 1"),
    ("", "", f"SELECT {IFN} FROM (SELECT /*+ REPARTITION(3) */ * FROM a)"),
    ("", "", f"SELECT {IFN} FROM a DISTRIBUTE BY id"),
    ("", "", f"SELECT {IFN}, row_number() OVER (ORDER BY id) FROM a"),
    ("", "", f"SELECT id, count(*) OVER (PARTITION BY {IFN}) FROM a"),
    ("", "", f"SELECT f FROM (SELECT {IFN} f, row_number() OVER (ORDER BY id) r FROM a)"),
    ("", "", f"SELECT {IFN} FROM (SELECT id, row_number() OVER (ORDER BY id) r FROM a)"),
    ("", "", f"SELECT {IFN} FROM a JOIN b USING (id)"),
    ("", "", f"SELECT id FROM a JOIN b USING (id) WHERE {IFN} LIKE '%a1%'"),
    ("", "", f"SELECT * FROM (SELECT {IFN} f, id FROM a) JOIN b USING (id)"),
    ("", "", f"SELECT {IFN} FROM a JOIN range(5) r ON a.id = r.id"),
    (NOBC, "", f"SELECT {IFN} FROM a JOIN range(5) r ON a.id = r.id"),
    ("", "", f"SELECT {IFN} FROM a CROSS JOIN (VALUES (1)) v"),
    (NOBC, "", f"SELECT {IFN} FROM a CROSS JOIN (VALUES (1)) v"),
    ("", "", f"SELECT {IFN}, id FROM (SELECT * FROM a UNION ALL SELECT * FROM b)"),
    ("", "", f"SELECT {IFN}, id FROM (SELECT * FROM a UNION SELECT * FROM b)"),
    ("", "", f"SELECT {IFN}, id FROM (SELECT * FROM a INTERSECT SELECT * FROM b)"),
    ("", "", f"SELECT {IFN} FROM a JOIN (SELECT id FROM b UNION ALL SELECT id FROM c) u USING (id)"),
    ("", "", f"SELECT {IFN}, id FROM (SELECT * FROM a UNION ALL SELECT * FROM range(2))"),
    ("", "", f"SELECT {IFN}, input_file_block_start(), input_file_block_length()"),
    ("", "", f"SELECT {IFN}, input_file_block_start(), input_file_block_length() FROM range(2)"),
    ("", "", f"SELECT {IFN}, input_file_block_start(), input_file_block_length() FROM VALUES (1) AS v(x)"),
    ("", CACHE, f"SELECT {IFN}, input_file_block_start(), input_file_block_length(), id FROM ca"),
    ("", CACHE, f"SELECT {IFN}, input_file_block_start(), input_file_block_length(), id FROM a"),
    (SPLIT, "", "SELECT input_file_block_start() s, input_file_block_length() l, count(*) FROM big GROUP BY 1, 2"),
    (SPLIT, "", f"SELECT {IFN}, count(*) FROM big GROUP BY 1"),
    ("", "", "SELECT input_file_block_start() s, input_file_block_length() l, count(*) FROM big GROUP BY 1, 2"),
    ("", "", f"SELECT {IFN}, input_file_block_start(), input_file_block_length(), id FROM c"),
    ("", "", f"SELECT {IFN}, input_file_block_start(), input_file_block_length(), id FROM d"),
    ("", "", f"SELECT {IFN}, input_file_block_start(), input_file_block_length(), id FROM t"),
    ("", "", f"SELECT (SELECT max({IFN}) FROM b) FROM a"),
    ("", "", f"SELECT {IFN}, id FROM a WHERE id IN (SELECT id FROM b)"),
    ("", "", f"SELECT {IFN}, id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.id = a.id)"),
    ("", "", f"SELECT id FROM a WHERE id IN (SELECT id FROM b WHERE {IFN} LIKE '%b1%')"),
    ("", "", f"SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE {IFN} LIKE '%a1%')"),
    ("", "", f"SELECT {IFN}, x FROM a LATERAL VIEW explode(array(1, 2)) AS x"),
    ("", "", "DF:select"),
    ("", "", "DF:filter"),
    ("", "", "DF:blocks"),
    ("", "", f"SELECT {IFN} = _metadata.file_path, {IFN}, _metadata.file_path, id FROM t"),
    ("", "", f"SELECT {IFN} = _metadata.file_path, count(*) FROM t GROUP BY 1"),
    ("", "", f"SELECT {IFN} = _metadata.file_path, {IFN}, id FROM parquet.`/w/data/a`"),
    ("", "", f"SELECT {IFN} = _metadata.file_path, id FROM a"),
    # DataSourceV2 file source (Iceberg-like: a DataSourceV2Relation, not a LogicalRelation)
    (V2, "CREATE OR REPLACE TEMP VIEW av2 AS SELECT * FROM parquet.`/w/data/a`",
     f"SELECT {IFN}, input_file_block_start(), input_file_block_length(), id FROM av2"),
    (V2, "CREATE OR REPLACE TEMP VIEW av2 AS SELECT * FROM parquet.`/w/data/a`",
     f"SELECT {IFN} FROM av2 JOIN b USING (id)"),
    (V2, "CREATE OR REPLACE TEMP VIEW av2 AS SELECT * FROM parquet.`/w/data/a`",
     f"SELECT {IFN} FROM av2 JOIN range(5) r ON av2.id = r.id"),
    (V2, "CREATE OR REPLACE TEMP VIEW av2 AS SELECT * FROM parquet.`/w/data/a`",
     f"SELECT {IFN} FROM av2 WHERE {IFN} LIKE '%a1%'"),
    # one more shape per remaining item
    ("", "", f"SELECT {IFN} FROM a WHERE id > 1 ORDER BY id LIMIT 1"),
    ("", "", f"SELECT {IFN}, id + 1 FROM a ORDER BY {IFN}, id"),
    ("", "", f"SELECT {IFN} f, count(*) OVER () FROM a"),
    ("", "", f"SELECT id, (SELECT {IFN} FROM b LIMIT 1) FROM a"),
    ("", "", f"SELECT id, (SELECT count(*) FROM b WHERE {IFN} LIKE '%b1%') FROM a"),
    ("", "", f"SELECT id FROM a WHERE id IN (SELECT a2.id FROM a a2 JOIN b ON a2.id = b.id WHERE {IFN} LIKE '%a1%')"),
    ("", "", f"SELECT {IFN} FROM a WHERE id IN (SELECT id FROM b)"),
    (NOBC, "", f"SELECT {IFN} FROM a WHERE id IN (SELECT id FROM b)"),
    (NOBC, "", f"SELECT * FROM (SELECT {IFN} f, id FROM a) JOIN b USING (id)"),
    ("", "", f"SELECT {IFN}, id FROM a GROUP BY id, {IFN}"),
    ("", "", f"SELECT {IFN} FROM a GROUP BY id"),
    ("", "", f"SELECT count(*) FROM a WHERE {IFN} = ''"),
    ("", "", "SELECT count(*), min(s), max(s), min(l), max(l) FROM (SELECT input_file_block_start() s, input_file_block_length() l FROM big)"),
    (SPLIT, "", "SELECT count(*), min(s), max(s), min(l), max(l) FROM (SELECT input_file_block_start() s, input_file_block_length() l FROM big)"),
]


def run(master: str, workdir: Path) -> tuple[str, list[str], list[dict]]:
    data = workdir / "data"
    data.mkdir()
    data.chmod(0o777)
    workdir.chmod(0o777)
    with (workdir / "cases.jsonl").open("w", encoding="utf-8") as f:
        for n, (conf, pre, sql) in enumerate(CASES):
            f.write(json.dumps({"n": n, "conf": conf, "pre": pre, "sql": sql}) + "\n")
    script = (Path(__file__).parent / "gen_input_file_goldens.scala").read_text(encoding="utf-8")
    cmd = ["docker", "run", "--rm", "-i", "-e", "HOME=/tmp", "-v", f"{workdir}:/w", IMAGE,
           "/opt/spark/bin/spark-shell", "--master", master, "--driver-memory", "1g",
           "--conf", "spark.sql.warehouse.dir=/w/wh", "--conf", "spark.ui.enabled=false"]
    out = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=3600, check=False).stdout
    lines = [m.group(1) for line in out.splitlines() if (m := re.match(r"^(?:scala> )*(@@.*)$", line))]
    if "@@END" not in lines:
        raise SystemExit("spark-shell did not complete:\n" + out[-3000:])
    version = next(x for x in lines if x.startswith("@@VERSION "))[10:]
    files = [x[7:] for x in lines if x.startswith("@@FILE ")]
    cases = [json.loads(x[7:]) for x in lines if x.startswith("@@CASE ")]
    subprocess.run(["docker", "run", "--rm", "-v", f"{workdir}:/w", "--entrypoint", "rm", IMAGE, "-rf", "/w/data", "/w/wh"],
                   check=False, capture_output=True)
    return version, files, cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/input-file-goldens.json")
    args = ap.parse_args()
    results = {}
    for master in ("local[1]", "local[2]"):
        with tempfile.TemporaryDirectory(prefix="krishiv-ifn-") as tmp:
            results[master] = run(master, Path(tmp))
    version, files, c1 = results["local[1]"]
    _, _, c2 = results["local[2]"]
    cases = []
    for (conf, pre, sql), a, b in zip(CASES, c1, c2, strict=True):
        case = {"sql": sql, "conf": dict(kv.split("=", 1) for kv in conf.split("|")) if conf else {}}
        if pre:
            case["pre"] = pre.split(";;")
        a = {k: v for k, v in a.items() if k != "n"}
        b = {k: v for k, v in b.items() if k != "n"}
        case.update(a)
        if a != b:
            case["local2"] = b
        cases.append(case)
    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": version.split()[0],
        "generator": GENERATOR,
        "master": "local[1]",
        "setup": {
            "files": [f.split(" ")[0].strip('"') + f" ({f.rsplit(' ', 1)[1]} bytes)" for f in files],
            "contents": {
                "a/a1.parquet": "id 1,2", "a/a2.parquet": "id 3", "b/b1.parquet": "id 1,10",
                "c/c1.csv": "header id; 1,7", "d/d1.json": "id 2,8",
                "big/big.parquet": "id 0..199999, h = sha2(id), parquet.block.size=65536",
            },
            "views": {"a": "parquet a", "b": "parquet b", "c": "csv c (header, inferSchema)", "d": "json d",
                      "big": "parquet big", "t": "CREATE TABLE t USING parquet LOCATION <root>/a (catalog table)"},
            "paths": "rows print <root> for the mounted data directory",
        },
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
