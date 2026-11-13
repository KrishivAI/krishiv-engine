---
description: Scan a PySpark or SQL project for functions Krishiv does not run the way Spark does.
---

# Check a PySpark or SQL project with check-compat

`krishiv check-compat` reads your `.py`, `.sql` and `.ipynb` files and lists the functions that Krishiv does not run the way Spark does. It does not execute anything. Use it before you move a job, to see what to change.

## Run it

Give it files or directories. Directories are walked; virtual environments, `.git`, `node_modules` and similar folders are skipped, and directory symlinks are not followed.

```bash
krishiv check-compat .
```

Each finding is one line: file, line, the function, and its status from the [compatibility data](../concepts/compatibility.md). When the function has a workaround, the line ends with it.

```text
./job.py:4: sentences: not yet
./job.py:5: hll_sketch_agg: not yet (alternative: approx_count_distinct(expr) returns the estimate directly; serialized HyperLogLog sketches are not available)
./job.py:5: hll_sketch_estimate: not yet (alternative: approx_count_distinct(expr) over the raw column returns the estimate; serialized HyperLogLog sketches are not available)
./report.sql:1: mystery_fn: unknown, not supported
```

A name the data does not list is reported as `unknown, not supported`, as the compatibility page states. A supported function is never reported.

## What it reads

- SQL files: every function call, ignoring comments, string literals and table or column lists.
- Python files: calls into `pyspark.sql.functions` as you imported it (`F.name(..)`, `from pyspark.sql.functions import name as alias`), and the SQL inside `spark.sql(..)`, `expr(..)`, `selectExpr(..)` and `filter(..)`. SQL strings are read only in files that import PySpark or use `spark`.
- Notebooks: each code cell, with the line in the notebook JSON and the cell number.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | No findings |
| 1 | At least one finding |
| 2 | A path could not be read, or a named file is not `.py`, `.sql` or `.ipynb` |

## JSON report

`--format json` prints one object. `schema_version` changes only for a breaking change; new fields are added without a bump.

```bash
krishiv check-compat --format json report.sql
```

```json
{"errors":[],"files_scanned":1,"findings":[{"file":"report.sql","line":1,"name":"mystery_fn","status":"unknown"}],"schema_version":1}
```

Each finding has `file`, `line`, `name` and `status` (`supported` is never listed; `partial`, `not yet` or `unknown`). Notebook findings add `cell`. A finding adds `alternative` when a workaround is known.

## Limits

The scan reads text, not a running session. It does not follow a function name held in a variable, and it checks function calls only: DataFrame methods and options are not checked. A clean result means no listed gap was found, not that the job is proven compatible.
