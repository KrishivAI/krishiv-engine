# python-datasource-pushdown (Spark 4.3.0 oracle for T040 part 2)

Produced by executing Spark 4.3.0 (`apache/spark:4.3.0`, image `sha256:3dedc5c8665a6f47...`, Python 3.10.12,
pyarrow 21.0.0, JDK 21 of the image), never transcribed. Nothing here is hand-written.

- `classic.json`: classic PySpark, `local[2]`. `connect.json`: PySpark Connect client against a Spark Connect server
  (`start-connect-server.sh --master local[2]`) in the same container.
- `oracle.py.txt`: the generator (`python3 oracle.py classic|connect OUT.json`; stored as `.txt` so the Python gates do
  not lint a Spark-only script). Each data source is a recording source: every lifecycle step appends a line to a
  log (`init`, `schema`, `reader`, `pushFilters <repr(filters)>`, `partitions`, `read <i>`); the driver adds phase
  markers. `pushFilters` returns the input unchanged (nothing handled) unless a case says otherwise.
- Nondeterministic in the files: object ids inside the `DATA_SOURCE_EXTRANEOUS_FILTERS` text, `#NNN` expression ids,
  the order of `read N` lines (tasks run in parallel). Compare counts and shapes, not those.
- `spark.sql.python.filterPushdown.enabled` is **false by default** at 4.3.0 (`conf_default_filterPushdown_enabled`);
  every case except the disabled-conf ones sets it to true. ANSI is on by default.
- Rules cited by the generator come from `pyspark/sql/tests/test_python_datasource.py` at `v4.3.0` (the case lists are
  copied from `_check_filters` calls in `test_unsupported_filter`, `test_filter_value_type`, `test_filter_type`,
  `test_filter_nested_column`, and `test_filter_pushdown`'s four-conjunct case); the expected values of that test
  are NOT used, only what Spark sent.

## What the run shows (classic and Connect agree unless said)

1. **Filter lists** (`filter_cases[]`, `four_conjunct`): `pushFilters` is not called for `x = 1 or x > 2`,
   `(0 < x and x < 1) or x = 2`, `x = null`, and `x in (1.1, 2.1)` on `decimal` (those cases of the Spark test pass
   vacuously). Inferred `IsNotNull` precedes the conjunct that implies it and follows input order:
   `x.a = 1 and x.b = x.c` gives `[IsNotNull(x.a), IsNotNull(x.b), IsNotNull(x.c), EqualTo(x.a, 1)]`;
   the four-conjunct case gives `[IsNotNull(x), IsNotNull(y), EqualTo(x,1), EqualTo(y,2)]`. `x <=> 1`, `x is null` and
   `x in (1, 2)` carry no `IsNotNull`. Untranslatable or unserializable conjuncts (`x % 5 = 1`, `x[0] = 1`,
   `like 'a%a%'`, `ilike`, collated `=`, interval value) leave only `IsNotNull(x)`. `x like 'a%b'` is two filters
   (`StringStartsWith`, `StringEndsWith`). Python reprs: `EqualTo(attribute=('x',), value=1)`, decimal and
   datetime values arrive as Python values.
2. **Call counts** (`call_counts`; the scan has 3 partitions so one execution is `read 3`):
   - Classic, **no Filter over the scan**: `df.explain()` calls nothing (not `reader`, not `partitions`).
     An action runs `reader 1, partitions 1, read 3`; a second `collect()` on the same DataFrame only reads (3).
   - Classic, **with a Filter**: `explain()` runs `reader 1, pushFilters 1, partitions 1, read 0`;
     `explain()` + `collect()` + `collect()` is `1/1/1/0`, then read 3, read 3 (planning once, reads each time);
     `explain()` + `count()` + `show()` plans again for each (`count` and `show` each `reader 1, pushFilters 1,
     partitions 1, read 3`).
   - Connect: `load()` runs nothing (`init`/`schema` run at the first request: explain or action). Every request
     plans again: with a Filter, `collect()` twice is two full `reader/pushFilters/partitions` plus two reads, and
     `explain()` then `collect()` plans twice. With no Filter, a second `collect()` reads only (3).
3. **Residual scenario** (`residual`): `WHERE 10 / x > 1 AND (a = 1 OR b = 2)` over `(0,0,0)`, ANSI on or off,
   pushdown on, `pushFilters` handles nothing: no error, no rows. `pushFilters` receives `[IsNotNull(x)]` only. The
   Filter above the scan is `isnotnull(x) AND (a = 1 OR b = 2) AND (10.0 / cast(x as double) > 1.0)`, the same for
   either written order of the two conjuncts (serializable-unhandled, translated, then untranslated). `10 / x` is
   a double division. A row `(0,1,1)` selects the OR and raises `DIVIDE_BY_ZERO` (22012) under ANSI.
4. **Errors** (`errors`):
   - `pushFilters` raising: `AnalysisException`, condition `PYTHON_DATA_SOURCE_ERROR`, SQLSTATE `38000`, parameters
     `action=initialize`, `type=reader`, `msg=<python traceback>`; raised by **`explain()` and by the action**.
   - `DATA_SOURCE_EXTRANEOUS_FILTERS`: same wrapper (`initialize`/`reader`), the PySpark text is inside `msg`
     (see the file for the exact text); raised by `explain()` and by the action.
   - `DATA_SOURCE_PUSHDOWN_DISABLED` (conf false, reader overrides `pushFilters`): same wrapper; **raised by the
     action only, not by `explain()`** (explain succeeds), with or without a filter; a reader that does not override
     `pushFilters` is unaffected.
   - Bad column path: **not reproduced**. Seven schemas through SQL (dotted top-level name, backtick, struct with a
     dotted field, case mismatch, map value, array of struct) all resolved or were not pushed. The only source is
     `UserDefinedPythonDataSource.scala` at `v4.3.0` (`getField`: `pythonDataSourceError(action = "plan",
     tpe = "filter", msg = s"Cannot find field $columnPath in schema")`): **unverified-by-run**.
5. **EXPLAIN** (`explain_pushed`, `four_conjunct`): the pushed filters show only in `explain(mode="formatted")`, as
   `PushedFilters: [IsNotNull(x), IsNotNull(y), EqualTo(x,1)]` (the filters the reader handled), plus
   `ReadSchema: struct<x:int,y:int>`; the simple and extended physical plan line is
   `BatchScan <name>[x#N, y#N] (Python) RuntimeFilters: []` with no filter text. With nothing handled,
   formatted has no `PushedFilters` line. Handled conjuncts are absent from the `Filter` node above the scan.

## Contradictions with `design/P3-T040-part2-pushfilters-connect.md` revision 3

- §2.8 / §1: "with no `Filter` over the scan the planning call still runs `reader()` and `partitions()`" is false at
  4.3.0: `explain()` with no Filter calls neither (counts 0, 0, 0); they run at the action.
- §2.7 / §2.8: `DATA_SOURCE_PUSHDOWN_DISABLED` is not raised by `explain()`; only the action raises it.
- §6 asks for conf "on" scenario; the default is **off** (false) in Spark 4.3.0.
- Connect plans per request (no plan reuse), as §8.1 says; classic reuses the DataFrame's plan across `collect()`
  but not across `count()`/`show()`: confirmed.
