# SPDX-License-Identifier: Apache-2.0
"""The generated ``pyspark.sql.functions`` wrappers: in sync with the registry, PySpark's names and
parameters, the call each one builds, and a run of some of them on a small DataFrame."""

from __future__ import annotations

import ast
import datetime
import inspect
import math
import warnings
from pathlib import Path
from typing import Any

import pytest
from _pyspark_signatures import PYSPARK_SIGNATURES

from krishiv import _native, errors
from krishiv.sql import DataFrame, SparkSession
from krishiv.sql import _generated_functions as G
from krishiv.sql import functions as F

pa = pytest.importorskip("pyarrow")

PYTHON_DIR = Path(__file__).resolve().parent.parent
GENERATED = PYTHON_DIR / "krishiv" / "sql"

#: The registry function a PySpark name sends when it is not the name itself.
REGISTRY_NAME = {
    "bitwise_not": "~",
    "bitwiseNOT": "~",
    "mean": "avg",
    "pow": "power",
    "shiftLeft": "shiftleft",
    "shiftRight": "shiftright",
    "shiftRightUnsigned": "shiftrightunsigned",
    "toDegrees": "degrees",
    "toRadians": "radians",
}
#: Spark's `trim(chars, str)`: PySpark's `(col, trim)` is sent reversed.
REVERSED = {"ltrim", "rtrim", "trim"}
HAND_WRITTEN = set(F._HAND_WRITTEN)
#: Parameters that only take some values.
CHECKED_VALUES = {("sha2", "numBits"): 256}


def _params(fn: Any) -> str:
    out = []
    for p in inspect.signature(fn).parameters.values():
        assert p.kind in (p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL), f"{fn.__name__}.{p.name}"
        out.append(
            f"*{p.name}"
            if p.kind is p.VAR_POSITIONAL
            else p.name
            if p.default is p.empty
            else f"{p.name}={p.default!r}"
        )
    return ", ".join(out)


def _all_of(path: Path) -> list[str]:
    for node in ast.parse(path.read_text()).body:
        is_all = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        )
        if is_all and isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise AssertionError(f"no __all__ in {path}")


def test_the_export_list_has_one_source() -> None:
    """``F.__all__`` is the generated names plus the hand-written ones, in no other list: the three lists (the
    generated module, ``functions.py`` and PySpark's signatures) cannot drift apart without a failure here."""
    assert list(F.__all__) == [*G.__all__, *F._HAND_WRITTEN]
    assert len(set(F.__all__)) == len(F.__all__)
    assert set(F._HAND_WRITTEN).isdisjoint(G.__all__)
    for name in F._HAND_WRITTEN:
        assert getattr(F, name).__module__ == F.__name__, name
    for name in G.__all__:
        assert getattr(F, name) is getattr(G, name), name
    # PySpark's side: a name Krishiv exports and PySpark lacks, or the reverse, is a failure.
    assert set(PYSPARK_SIGNATURES) | {"column"} == set(F.__all__)


def test_names_and_parameters_are_pysparks() -> None:
    exported = set(F.__all__)
    assert exported == set(PYSPARK_SIGNATURES) | {"column"}
    assert set(G.__all__) <= exported
    assert exported - set(G.__all__) == HAND_WRITTEN
    for name, expected in PYSPARK_SIGNATURES.items():
        assert _params(getattr(F, name)) == expected, name
    assert len(G.__all__) > 150


def test_registered_functions_pyspark_lacks_are_not_exported() -> None:
    for name in ("power", "mod", "if"):
        assert not hasattr(F, name), name
    for name in ("count", "log", "ln"):
        assert callable(getattr(F, name))


def test_stub_declares_what_the_module_defines() -> None:
    stub = ast.parse((GENERATED / "_generated_functions.pyi").read_text())
    defs = {n.name: n for n in stub.body if isinstance(n, ast.FunctionDef)}
    assert sorted(defs) == sorted(G.__all__) == sorted(_all_of(GENERATED / "_generated_functions.pyi"))
    for name, node in defs.items():
        args = node.args
        rendered = [a.arg for a in args.args]
        defaults = [ast.unparse(d) for d in args.defaults]
        rendered = [
            f"{a}={defaults[i - (len(rendered) - len(defaults))]}" if i >= len(rendered) - len(defaults) else a
            for i, a in enumerate(rendered)
        ]
        if args.vararg:
            rendered.append(f"*{args.vararg.arg}")
        assert ", ".join(rendered) == PYSPARK_SIGNATURES[name], name
        assert node.returns is not None
        assert ast.unparse(node.returns) == "Column"


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native builders record what they are given, so a call's shape can be compared."""
    monkeypatch.setattr(_native, "col", lambda name: ("col", name))
    monkeypatch.setattr(_native, "lit", lambda value: ("lit", value))
    monkeypatch.setattr(_native, "call_function", lambda name, args: ("call", name, tuple(args)))


COLUMN_ANNOTATIONS = {"ColumnOrName", "Column", "ColumnOrName | float", "ColumnOrName | int"}
LITERALS = {"int": 7, "str": "s_x", "bool": True}


def _sample(name: str, p: inspect.Parameter) -> tuple[list[Any], list[tuple[str, Any]]]:
    """Values for the parameter, and the native arguments they must become."""
    if p.kind is p.VAR_POSITIONAL:
        names = [f"c_{p.name}0", f"c_{p.name}1"]
        return names, [("col", n) for n in names]
    annotation = str(p.annotation).removesuffix(" | None")
    if annotation in COLUMN_ANNOTATIONS:
        return [f"c_{p.name}"], [("col", f"c_{p.name}")]
    kind = next(k for k in annotation.split(" | ") if k != "Column")
    value = CHECKED_VALUES.get((name, p.name), LITERALS[kind])
    return [value], [("lit", value)]


@pytest.mark.parametrize("name", sorted(G.__all__))
def test_every_wrapper_builds_its_registry_call(name: str, recorded: None) -> None:
    fn = getattr(G, name)
    params = list(inspect.signature(fn).parameters.values())
    samples = [_sample(name, p) for p in params]
    values = [v for vs, _ in samples for v in vs]
    expected = [a for _, args in samples for a in args]
    if name in REVERSED:
        expected.reverse()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert fn(*values)._c == ("call", REGISTRY_NAME.get(name, name), tuple(expected))
        # Without the optional parameters: they are left out of the call, and a default that is not
        # None (`first`'s `ignorenulls`) is sent.
        required = [vs for (vs, _), p in zip(samples, params, strict=True) if p.default is p.empty]
        short = fn(*[v for vs in required for v in vs])._c
    sent_defaults = [p for p in params if p.default is not p.empty and p.default is not None]
    assert len(short[2]) == sum(len(vs) for vs in required) + len(sent_defaults)


PYSPARK_CALLS = [
    (lambda: F.abs("a"), ("call", "abs", (("col", "a"),))),
    (lambda: F.ceil("a", 2), ("call", "ceil", (("col", "a"), ("lit", 2)))),
    (lambda: F.ceil("a", F.col("s")), ("call", "ceil", (("col", "a"), ("col", "s")))),
    (lambda: F.atan2(1.0, "b"), ("call", "atan2", (("lit", 1.0), ("col", "b")))),
    (lambda: F.atan2("a", F.col("b")), ("call", "atan2", (("col", "a"), ("col", "b")))),
    (lambda: F.date_add("d", 1), ("call", "date_add", (("col", "d"), ("lit", 1)))),
    (lambda: F.date_add("d", "n"), ("call", "date_add", (("col", "d"), ("col", "n")))),
    (lambda: F.first("a"), ("call", "first", (("col", "a"), ("lit", False)))),
    (lambda: F.first("a", True), ("call", "first", (("col", "a"), ("lit", True)))),
    (lambda: F.first_value("a"), ("call", "first_value", (("col", "a"),))),
    (lambda: F.last_value("a", True), ("call", "last_value", (("col", "a"), ("lit", True)))),
    (lambda: F.lpad("s", 5, "-"), ("call", "lpad", (("col", "s"), ("lit", 5), ("lit", "-")))),
    (lambda: F.regexp_replace("s", "x", "y"), ("call", "regexp_replace", (("col", "s"), ("lit", "x"), ("lit", "y")))),
    (lambda: F.substring("s", 1, 2), ("call", "substring", (("col", "s"), ("lit", 1), ("lit", 2)))),
    (lambda: F.substring("s", "p", "n"), ("call", "substring", (("col", "s"), ("col", "p"), ("col", "n")))),
    (lambda: F.trim("s"), ("call", "trim", (("col", "s"),))),
    (lambda: F.trim("s", F.col("x")), ("call", "trim", (("col", "x"), ("col", "s")))),
    (lambda: F.btrim("s", "x"), ("call", "btrim", (("col", "s"), ("col", "x")))),
    (lambda: F.log("a"), ("call", "ln", (("col", "a"),))),
    (lambda: F.log(2.0, "a"), ("call", "log", (("lit", 2.0), ("col", "a")))),
    (lambda: F.mean("a"), ("call", "avg", (("col", "a"),))),
    (lambda: F.pow("a", 2.0), ("call", "power", (("col", "a"), ("lit", 2.0)))),
    (lambda: F.coalesce("a", F.col("b")), ("call", "coalesce", (("col", "a"), ("col", "b")))),
    (lambda: F.greatest("a", "b", "c"), ("call", "greatest", (("col", "a"), ("col", "b"), ("col", "c")))),
    (lambda: F.raise_error("boom"), ("call", "raise_error", (("lit", "boom"),))),
    (lambda: F.conv("s", 10, 2), ("call", "conv", (("col", "s"), ("lit", 10), ("lit", 2)))),
    (lambda: F.row_number(), ("call", "row_number", ())),
    (lambda: F.bitwise_not("a"), ("call", "~", (("col", "a"),))),
]


@pytest.mark.parametrize(
    ("build", "expected"), PYSPARK_CALLS, ids=[f"{call[1]}-{i}" for i, (_, call) in enumerate(PYSPARK_CALLS)]
)
def test_calls_are_built_as_pyspark_sends_them(build: Any, expected: Any, recorded: None) -> None:
    assert build()._c == expected


def test_argument_checks(recorded: None) -> None:
    with pytest.raises(errors.PySparkValueError, match="at least 2 columns"):
        F.greatest("a")
    with pytest.raises(errors.PySparkValueError, match="at least 2 columns"):
        F.least()
    with pytest.raises(errors.PySparkValueError, match="numBits"):
        F.sha2("a", 100)
    with pytest.raises(errors.PySparkTypeError, match="col"):
        F.abs(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="cols"):
        F.coalesce("a", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="pad"):
        F.lpad("s", 5, [1])  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="col1"):
        F.atan2(None, "b")  # ty: ignore[invalid-argument-type]
    # PySpark's `ColumnOrName | int` takes an int as a literal and resolves anything else as a column.
    with pytest.raises(errors.PySparkTypeError, match="days"):
        F.date_add("d", 1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="len"):
        F.substring("s", 1, 2.0)  # ty: ignore[invalid-argument-type]


def test_deprecated_names_warn_and_still_work(recorded: None) -> None:
    with pytest.warns(FutureWarning, match="use shiftleft instead") as w:
        assert F.shiftLeft("a", 2)._c == ("call", "shiftleft", (("col", "a"), ("lit", 2)))
    assert w[0].filename == __file__


@pytest.fixture
def df() -> DataFrame:
    spark = SparkSession.builder.create()
    return spark.createDataFrame(
        pa.table(
            {
                "a": [1.5, -2.5, 3.0],
                "i": [1, 2, 3],
                "n": pa.array([10, None, 30], pa.int64()),
                "s": ["  x ", "xxabxx", None],
                "d": ["2024-01-30"] * 3,
            }
        )
    )


def _rows(frame: DataFrame) -> list[list[Any]]:
    return [list(r) for r in frame.collect()]


def test_math_and_rounding_run(df: DataFrame) -> None:
    out = df.select(F.abs("a"), F.ceil("a"), F.round("a", 0), F.bround("a", 0), F.pow("i", 2.0), F.negative("i"))
    assert _rows(out) == [[1.5, 2, 2.0, 2.0, 1.0, -1], [2.5, -2, -3.0, -2.0, 4.0, -2], [3.0, 3, 3.0, 3.0, 9.0, -3]]
    out = df.select(F.log2("i"), F.log(2.0, "i"), F.log("i"), F.hypot("i", 4.0))
    rows = _rows(out)
    assert rows[1][:2] == [1.0, 1.0]
    assert rows[2][2] == pytest.approx(math.log(3))
    assert rows[0][3] == pytest.approx(math.sqrt(17))
    assert _rows(df.select(F.greatest("a", "i"), F.least("a", "i"))) == [[1.5, 1.0], [2.0, -2.5], [3.0, 3.0]]


def test_trims_take_pysparks_arguments(df: DataFrame) -> None:
    out = df.select(F.trim("s"), F.trim("s", F.lit("x")), F.ltrim("s", F.lit("x")), F.rtrim("s", F.lit("x")))
    assert _rows(out) == [["x", "  x ", "  x ", "  x "], ["xxabxx", "ab", "abxx", "xxab"], [None, None, None, None]]
    assert _rows(df.select(F.btrim("s", F.lit("x"))))[1] == ["ab"]


def test_strings_run(df: DataFrame) -> None:
    out = df.select(F.lpad("s", 8, "-"), F.substring("s", 1, 2), F.regexp_replace("s", "x", "y"), F.length("s"))
    assert _rows(out) == [["----  x ", "  ", "  y ", 4], ["--xxabxx", "xx", "yyabyy", 6], [None, None, None, None]]
    out = df.select(F.startswith("s", F.lit("xx")), F.endswith("s", F.lit("xx")), F.contains("s", F.lit("ab")))
    assert _rows(out) == [[False, False, False], [True, True, True], [None, None, None]]
    assert _rows(df.select(F.upper(F.trim("s"))))[0] == ["X"]


def test_conditionals_bits_and_dates_run(df: DataFrame) -> None:
    out = df.select(F.nvl("n", F.lit(0)), F.ifnull("n", F.col("i")), F.nullif("i", F.lit(2)), F.isnull("n"))
    assert _rows(out) == [[10, 10, 1, False], [0, 2, None, True], [30, 30, 3, False]]
    assert _rows(df.select(F.nvl2("n", F.lit(1), F.lit(0)), F.nanvl("a", F.col("a")))) == [
        [1, 1.5],
        [0, -2.5],
        [1, 3.0],
    ]
    out = df.select(F.bitwise_not("i"), F.shiftleft("i", 2), F.shiftright("i", 1), F.bit_count("i"))
    assert _rows(out) == [[-2, 4, 0, 1], [-3, 8, 1, 1], [-4, 12, 1, 2]]
    day = F.col("d").cast("date")
    out = df.select(F.date_add(day, 3), F.date_sub(day, 30), F.datediff(F.date_add(day, 3), day), F.dayofmonth(day))
    assert _rows(out)[0] == [datetime.date(2024, 2, 2), datetime.date(2023, 12, 31), 3, 30]


def test_aggregates_run(df: DataFrame) -> None:
    out = df.agg(F.first("a"), F.last("i"), F.sum("i"), F.mean("a"), F.avg("i"), F.min("a"), F.max("i"), F.count("*"))
    assert _rows(out) == [[1.5, 3, 6, pytest.approx(2.0 / 3), 2.0, -2.5, 3, 3]]
    grouped = df.groupBy(F.col("d")).agg(F.sum("i"))
    assert _rows(grouped) == [["2024-01-30", 6]]
