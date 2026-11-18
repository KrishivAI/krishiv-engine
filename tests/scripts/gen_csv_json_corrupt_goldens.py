#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Goldens for PY-076, PY-067 and SQL-30 from a real Spark 4.2.0 (docker `apache/spark:4.2.0`, local[4]).

Writes the small inputs, runs `gen_csv_json_corrupt_goldens.scala` in spark-shell, and writes the parsed probe
lines as JSON. PY-067's big file is not committed: it is generated from the parameters recorded in the JSON.

    python3 tests/scripts/gen_csv_json_corrupt_goldens.py tests/fixtures/spark-4.2.0/csv-json-corrupt-record
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

IMAGE = "apache/spark:4.2.0"
BIG_ROWS = 100_000  # row i: "<i>,name<i>,<pad>" with the pad making every row at least 80 bytes
BIG_PAD_TO = 80
BIG2_ROWS = (60_000, 20_000)  # two files in one directory; sizes differ so the length sort reorders


def big_lines(n: int, prefix: str = "") -> bytes:
    out = []
    for i in range(n):
        head = f"{prefix}{i},name{i},"
        out.append(head + "x" * max(1, BIG_PAD_TO - len(head) - 1))
    return ("\n".join(out) + "\n").encode()


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="krishiv-corrupt-"))
    inputs = {
        "bad.json": b'{"a":1}\n{"a":\nnot json\n{"a":4}\n',
        "bad.csv": b"1\nabc\n3\n",
        "utf16le.csv": b"\xff\xfe" + "a,b\n1,x\n2,y\n".encode("utf-16-le"),
        "utf16be.csv": b"\xfe\xff" + "a,b\n1,x\n2,y\n".encode("utf-16-be"),
    }
    for name, data in inputs.items():
        (work / name).write_bytes(data)
    (work / "big.csv").write_bytes(big_lines(BIG_ROWS))
    (work / "big2").mkdir()
    (work / "big2" / "f1.csv").write_bytes(big_lines(BIG2_ROWS[0], "a"))
    (work / "big2" / "f2.csv").write_bytes(big_lines(BIG2_ROWS[1], "b"))
    for p in [work, *work.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    script = (pathlib.Path(__file__).parent / "gen_csv_json_corrupt_goldens.scala").read_bytes()
    res = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "-e",
            "HOME=/tmp",
            "-v",
            f"{work}:/w:ro",
            IMAGE,
            "/opt/spark/bin/spark-shell",
            "--master",
            "local[4]",
            "--driver-memory",
            "2g",
        ],
        input=script,
        capture_output=True,
        check=False,
    )
    text = res.stdout.decode("utf-8", "replace")
    probes = {}
    for m in re.finditer(r"@@([^\t\n]+)\t(OK|ERR)\t([^\n]*)", text):
        label, status, rest = m.groups()
        if status == "OK":
            probes[label] = {"status": "ok", "value": rest}
        else:
            cls, cond, sqlstate, params, msg = rest.split("\t", 4)
            probes[label] = {
                "status": "error",
                "exception": cls,
                "condition": cond,
                "sqlstate": sqlstate,
                "parameters": params,
                "message": msg,
            }
    doc = {
        "format_version": 1,
        "spark_version": "4.2.0",
        "image": IMAGE,
        "generator": "tests/scripts/gen_csv_json_corrupt_goldens.py",
        "inputs_hex": {k: v.hex() for k, v in inputs.items()},
        "big_file": {
            "rows": BIG_ROWS,
            "pad_to": BIG_PAD_TO,
            "max_partition_bytes": 1048576,
            "two_file_rows": list(BIG2_ROWS),
            "two_file_prefixes": ["a", "b"],
        },
        "probes": probes,
    }
    (out_dir / "goldens.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(len(probes), "probes", file=sys.stderr, flush=True)
    if not probes:
        os.write(2, res.stderr[-2000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
