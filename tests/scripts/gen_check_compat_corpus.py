# SPDX-License-Identifier: Apache-2.0
"""Writes the check_compat corpus (T029): 30 PySpark scripts, 30 SQL files and
expected.json (file, line, name, status) into crates/krishiv-cli/tests/check_compat_corpus.
The planted names are chosen here, not by the scanner, so the expected findings
are independent of it."""
import json, pathlib, random

out = pathlib.Path("crates/krishiv-cli/tests/check_compat_corpus")
matrix = json.loads(pathlib.Path("crates/krishiv-cli/src/check_compat/functions.json").read_text())
ident = lambda n: n.isidentifier() and not n.startswith("_")
supported = [n for n, s in matrix.items() if s == "supported" and ident(n) and n not in ("if", "in", "not", "and", "or", "case", "when", "like", "between", "div", "mod")]
not_yet = [n for n, s in matrix.items() if s == "not yet" and ident(n)]
unknown = [f"legacy_fn_{i}" for i in range(40)]
rnd = random.Random(29)
expected = []

def pick(file, line):
    r = rnd.random()
    if r < 0.5:
        return rnd.choice(supported), None
    if r < 0.8:
        n = rnd.choice(not_yet)
        expected.append({"file": file, "line": line, "name": n, "status": "not yet"})
        return n, "not yet"
    n = rnd.choice(unknown)
    expected.append({"file": file, "line": line, "name": n, "status": "unknown"})
    return n, "unknown"

import shutil
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
for i in range(30):
    f = f"q{i:02}.sql"
    lines = [f"-- query {i}: nope(1)"]
    for _ in range(rnd.randint(3, 6)):
        n, _s = pick(f, len(lines) + 1)
        lines.append(f"SELECT {n}(a, b) AS c{len(lines)} FROM t{i} WHERE x IN (1, 2) AND y = 'x({n})';")
    (out / f).write_text("\n".join(lines) + "\n")
for i in range(30):
    f = f"job{i:02}.py"
    lines = ["import pyspark.sql.functions as F", "# F.nope('x')"]
    for _ in range(rnd.randint(3, 6)):
        n, _s = pick(f, len(lines) + 1)
        if rnd.random() < 0.5:
            lines.append(f"df = df.select(F.{n}('a'), F.col('b'))")
        else:
            lines.append(f'df = spark.sql("SELECT {n}(a) FROM t{i}")')
    (out / f).write_text("\n".join(lines) + "\n")
expected.sort(key=lambda e: (e["file"], e["line"], e["name"]))
(out / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
print(len(expected), "planted findings")
