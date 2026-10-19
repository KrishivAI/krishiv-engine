# krishiv (Python)

Python bindings for Krishiv. The native module is `krishiv._native`, built by
the `krishiv-python` crate (PyO3 0.30) with maturin.

## Supported Python

CPython 3.11 and newer. The default wheel is one `abi3` wheel (crate feature `abi3-py311`) covering 3.11+; the
free-threaded `cp314t` wheel is built separately. CI builds both and imports them (`wheels` job).

## Use

```python
import krishiv
from krishiv.sql import functions as F

spark = krishiv.session()  # SparkSession.builder.getOrCreate()
df = spark.sql("SELECT 1 AS a")
df = df.withColumn("b", F.col("a") + 1)
print(df.columns, df.schema)  # analysis only, nothing runs
print(df.collect())  # runs the query in-process: [Row(a=1, b=2)]
df.show()
table = df.toArrow()  # a pyarrow.Table
```

Actions (`collect`, `show`, `toArrow`, `toPandas`, `count`) run the query in-process with the GIL released. A
DataFrame also implements the Arrow PyCapsule interface (`__arrow_c_stream__`), so `pa.table(df)`,
`polars.DataFrame(df)` and `duckdb.from_arrow(df)` read the result without copying; each read runs the query again,
and releasing the stream cancels it.

`krishiv.sql` mirrors `pyspark.sql` (signatures and import paths); errors are in `krishiv.errors`, shaped like
`pyspark.errors`. pyarrow is optional: install `krishiv[pyarrow]`; `toArrow`, `collect`, `show` and `toPandas` need it (`count` and
the PyCapsule interface do not). `toPandas` also needs pandas (`pip install krishiv[pandas]`). SQL expression strings
(`selectExpr`, `where("...")`, `expr`) are not available yet.

## Develop

```sh
cd python
uv venv && uv pip install maturin pytest
. .venv/bin/activate
maturin develop            # default abi3-py311 build
python -c "import krishiv; print(krishiv.__version__)"
pytest
```

Code quality (ruff lint and format, ty; versions pinned in `uv.lock`):

```sh
cd python
uv run ruff check && uv run ruff format --check && uv run ty check
```

## Free-threaded build (cp314t)

abi3 does not support free-threaded CPython, so that wheel is a separate,
non-abi3 build (disable the default `abi3-py311` crate feature but keep `mimalloc`):

```sh
cd python
uv venv --python 3.14t .venv-314t && uv pip install --python .venv-314t/bin/python maturin pytest
. .venv-314t/bin/activate
maturin develop --no-default-features --features mimalloc   # or: maturin build ...
# (mimalloc is a default feature; keep it when turning the defaults off)
```

Verified on CPython 3.14.8t: imports and runs with the GIL disabled.

Publishing wheels (`maturin publish`, `uv publish`) is a hard stop; see
`/RELEASING.md`.
