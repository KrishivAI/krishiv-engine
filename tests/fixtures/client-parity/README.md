# Client parity cases

`cases.json` lists queries over the repository's Parquet, CSV, JSON and Delta fixtures. Two clients run every case
and compare the Arrow result with the case's `types` and `expected`, so the Python package and the npm package's
embedded engine return the same rows:

- `python/tests/test_client_parity.py` (the wheel, `DataFrame.toArrow()`);
- `js/test/parity.test.ts` (the `krishiv` npm package over the native addon; runs when `KRISHIV_NATIVE_ENGINE` is set).

Each case runs in a fresh session with `conf`; `setup` statements run first; `{root}` in SQL is the repository root.
Expected values come from an independent source (`source`): pyarrow for Parquet, delta-spark's own
`EXPECTED.json` for Delta, and the hand-written fixture files for CSV and JSON.

## Canonical form
Both tests turn an Arrow result into the same JSON, column by column:

| Arrow type | type string | value |
| --- | --- | --- |
| int8, int16, int32, int64 | `tinyint`, `smallint`, `int`, `bigint` | decimal string (`"-1500"`) |
| float32, float64 | `float`, `double` | JSON number |
| decimal128(p, s) | `decimal(p,s)` | string with the scale's digits (`"-90.00"`) |
| utf8 (any width) | `string` | string |
| binary (any width) | `binary` | lowercase hex |
| bool | `boolean` | `true` / `false` |
| date32 | `date` | `YYYY-MM-DD` |
| timestamp with a time zone / without | `timestamp` / `timestamp_ntz` | UTC `YYYY-MM-DDTHH:MM:SS.ffffffZ` |
| list | `array<T>` | array |
| struct | `struct<a:T,b:U>` | object |

A null at any depth is `null`. Adding a case: write the fixture, take the expected rows from a reader other than
Krishiv, and run both tests.
