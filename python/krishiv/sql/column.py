# SPDX-License-Identifier: Apache-2.0
"""``Column``: a column expression, as ``pyspark.sql.Column`` (a thin wrapper on the native handle).

Every method builds the same engine expression the SQL path builds, resolved
by the analyzer when the query runs. Nothing is evaluated in Python: where
the engine has no equivalent yet (bitwise operators, ``isnan``, nested
field access), the method builds the node the SQL equivalent builds, so it
fails with the engine's own error, as the SQL does.
"""

from __future__ import annotations

from typing import Any

from krishiv import _native
from krishiv.errors import (
    PySparkAttributeError,
    PySparkTypeError,
    PySparkValueError,
    not_expected_type,
    unsupported_feature,
)
from krishiv.sql.types import DataType, _simple_string_of


def _literal(value: Any) -> _native.Column:
    """The native column for a Python value used as a literal."""
    # bool is an int subclass; the native side tells them apart.
    if value is None or isinstance(value, bool | int | float | str):
        try:
            return _native.lit(value)
        except OverflowError as e:
            raise PySparkValueError(f"integer literal out of the 64-bit range: {value}") from e
    raise not_expected_type("value", "None, bool, int, float or str", value)


def _operand(value: Any) -> _native.Column:
    """A Column stays one; any other value is a literal (``col == "x"`` compares with a string)."""
    return value._c if isinstance(value, Column) else _literal(value)


def _named(value: Any, arg: str = "cols") -> _native.Column:
    """A Column stays one; a ``str`` names a column (``"*"`` is all of them)."""
    if isinstance(value, Column):
        return value._c
    if isinstance(value, str):
        return _native.col(value)
    raise not_expected_type(arg, "Column or str", value)


def _call(name: str, *cols: Any) -> Column:
    """A call of the engine function ``name`` with column or literal arguments
    (a ``str`` is a literal here, not a column name), resolved when analyzed."""
    return Column(_native.call_function(name, [_operand(c) for c in cols]))


class Column:
    """A column expression. Build one with ``F.col``, ``df["name"]`` or an operator."""

    __slots__ = ("_c",)
    __hash__ = None  # __eq__ builds a Column, as in PySpark

    def __init__(self, native: _native.Column) -> None:
        self._c = native

    # Arithmetic.
    def __add__(self, other: Any) -> Column:
        return Column(self._c.add(_operand(other)))

    def __radd__(self, other: Any) -> Column:
        return Column(_operand(other).add(self._c))

    def __sub__(self, other: Any) -> Column:
        return Column(self._c.sub(_operand(other)))

    def __rsub__(self, other: Any) -> Column:
        return Column(_operand(other).sub(self._c))

    def __mul__(self, other: Any) -> Column:
        return Column(self._c.mul(_operand(other)))

    def __rmul__(self, other: Any) -> Column:
        return Column(_operand(other).mul(self._c))

    def __truediv__(self, other: Any) -> Column:
        return Column(self._c.div(_operand(other)))

    def __rtruediv__(self, other: Any) -> Column:
        return Column(_operand(other).div(self._c))

    def __mod__(self, other: Any) -> Column:
        return Column(self._c.rem(_operand(other)))

    def __rmod__(self, other: Any) -> Column:
        return Column(_operand(other).rem(self._c))

    def __neg__(self) -> Column:
        return Column(self._c.neg())

    # Comparison and logic.
    def __eq__(self, other: object) -> Column:  # ty: ignore[invalid-method-override]
        return Column(self._c.eq(_operand(other)))

    def __ne__(self, other: object) -> Column:  # ty: ignore[invalid-method-override]
        return Column(self._c.ne(_operand(other)))

    def __lt__(self, other: Any) -> Column:
        return Column(self._c.lt(_operand(other)))

    def __le__(self, other: Any) -> Column:
        return Column(self._c.le(_operand(other)))

    def __gt__(self, other: Any) -> Column:
        return Column(self._c.gt(_operand(other)))

    def __ge__(self, other: Any) -> Column:
        return Column(self._c.ge(_operand(other)))

    def __and__(self, other: Any) -> Column:
        return Column(self._c.and_(_operand(other)))

    def __rand__(self, other: Any) -> Column:
        return Column(_operand(other).and_(self._c))

    def __or__(self, other: Any) -> Column:
        return Column(self._c.or_(_operand(other)))

    def __ror__(self, other: Any) -> Column:
        return Column(_operand(other).or_(self._c))

    def __invert__(self) -> Column:
        return Column(self._c.invert())

    def __bool__(self) -> bool:
        raise PySparkValueError(errorClass="CANNOT_CONVERT_COLUMN_INTO_BOOL")

    def __iter__(self) -> Any:
        raise PySparkTypeError("Column is not iterable")

    def __repr__(self) -> str:
        return "Column<...>"

    def __getattr__(self, item: Any) -> Column:
        # Only missing attributes reach here, so methods and `_c` are safe.
        if item.startswith("__"):
            raise PySparkAttributeError(f"'Column' object has no attribute '{item}'")
        return self.getField(item)

    def __getitem__(self, k: Any) -> Column:
        if isinstance(k, slice):
            if k.step is not None:
                raise PySparkValueError(errorClass="SLICE_WITH_STEP")
            return self.substr(k.start, k.stop)
        return self.getItem(k)

    def eqNullSafe(self, other: Any) -> Column:
        return Column(self._c.eq_null_safe(_operand(other)))

    def isNull(self) -> Column:
        return Column(self._c.is_null())

    def isNotNull(self) -> Column:
        return Column(self._c.is_not_null())

    def isNaN(self) -> Column:
        """Whether the value is NaN, as ``pyspark.sql.Column.isNaN``.

        .. note:: The engine has no ``isnan`` function yet: this builds the
            node ``SELECT isnan(x)`` builds, so it fails with the engine's
            own ``KRISHIV_UNSUPPORTED_FEATURE`` error, as the SQL does.

        Examples
        --------
        >>> df = spark.createDataFrame([(1.0,), (float("nan"),)], ["x"])
        >>> df.select(df.x.isNaN()).collect()
        [Row(value=False), Row(value=True)]
        """
        return _call("isnan", self)

    def like(self, other: str) -> Column:
        if not isinstance(other, str):
            raise not_expected_type("other", "str", other)
        return Column(self._c.like(other))

    def rlike(self, other: str) -> Column:
        """Whether the string matches the regex, as ``pyspark.sql.Column.rlike``.

        .. note:: The engine resolves ``rlike`` but cannot run it yet:
            collecting fails with ``KRISHIV_UNSUPPORTED_FEATURE``.

        Examples
        --------
        >>> df = spark.createDataFrame([("Alice",), ("Bob",)], ["name"])
        >>> df.filter(df.name.rlike("ice$")).collect()
        [Row(name='Alice')]
        """
        if not isinstance(other, str):
            raise not_expected_type("other", "str", other)
        return _call("rlike", self, other)

    def ilike(self, other: str) -> Column:
        """Whether the string matches the pattern ignoring case, as
        ``pyspark.sql.Column.ilike``.

        .. note:: The engine resolves ``ilike`` but cannot run it yet:
            collecting fails with ``KRISHIV_UNSUPPORTED_FEATURE``.

        Examples
        --------
        >>> df = spark.createDataFrame([("Alice",), ("Bob",)], ["name"])
        >>> df.filter(df.name.ilike("%ICE")).collect()
        [Row(name='Alice')]
        """
        if not isinstance(other, str):
            raise not_expected_type("other", "str", other)
        return _call("ilike", self, other)

    def contains(self, other: Any) -> Column:
        """Whether the string holds ``other``, as ``pyspark.sql.Column.contains``.

        Examples
        --------
        >>> df = spark.createDataFrame([("Alice",), ("Bob",)], ["name"])
        >>> df.filter(df.name.contains("o")).collect()
        [Row(name='Bob')]
        """
        return _call("contains", self, other)

    def startswith(self, other: Any) -> Column:
        """Whether the string starts with ``other``, as ``pyspark.sql.Column.startswith``.

        Examples
        --------
        >>> df = spark.createDataFrame([("Alice",), ("Bob",)], ["name"])
        >>> df.filter(df.name.startswith("Al")).collect()
        [Row(name='Alice')]
        """
        return _call("startswith", self, other)

    def endswith(self, other: Any) -> Column:
        """Whether the string ends with ``other``, as ``pyspark.sql.Column.endswith``.

        Examples
        --------
        >>> df = spark.createDataFrame([("Alice",), ("Bob",)], ["name"])
        >>> df.filter(df.name.endswith("ice")).collect()
        [Row(name='Alice')]
        """
        return _call("endswith", self, other)

    def substr(self, startPos: Any, length: Any) -> Column:
        """The substring from ``startPos`` of ``length`` characters, as
        ``pyspark.sql.Column.substr`` (positions start at 1).

        Examples
        --------
        >>> df = spark.createDataFrame([("Alice",), ("Bob",)], ["name"])
        >>> df.select(df.name.substr(1, 3).alias("col")).collect()
        [Row(col='Ali'), Row(col='Bob')]
        """
        if type(startPos) is not type(length):
            raise PySparkTypeError(
                errorClass="NOT_SAME_TYPE",
                messageParameters={
                    "arg_name1": "startPos",
                    "arg_name2": "length",
                    "arg_type1": type(startPos).__name__,
                    "arg_type2": type(length).__name__,
                },
            )
        if isinstance(startPos, int) and isinstance(length, int):
            return _call("substr", self, startPos, length)
        if isinstance(startPos, Column) and isinstance(length, Column):
            return _call("substr", self, startPos, length)
        raise not_expected_type("startPos", "Column or int", startPos)

    def bitwiseOR(self, other: Any) -> Column:
        """The bitwise OR with ``other``, as ``pyspark.sql.Column.bitwiseOR``.

        .. note:: The engine has no bitwise operators yet: this builds the
            node ``SELECT a | b`` builds, so it fails with the engine's own
            ``KRISHIV_UNSUPPORTED_FEATURE`` error, as the SQL does.

        Examples
        --------
        >>> df = spark.createDataFrame([(170, 75)], ["a", "b"])
        >>> df.select(df.a.bitwiseOR(df.b)).collect()
        [Row(value=235)]
        """
        return _call("|", self, other)

    def bitwiseAND(self, other: Any) -> Column:
        """The bitwise AND with ``other``, as ``pyspark.sql.Column.bitwiseAND``.

        .. note:: The engine has no bitwise operators yet: this builds the
            node ``SELECT a & b`` builds, so it fails with the engine's own
            ``KRISHIV_UNSUPPORTED_FEATURE`` error, as the SQL does.

        Examples
        --------
        >>> df = spark.createDataFrame([(170, 75)], ["a", "b"])
        >>> df.select(df.a.bitwiseAND(df.b)).collect()
        [Row(value=10)]
        """
        return _call("&", self, other)

    def bitwiseXOR(self, other: Any) -> Column:
        """The bitwise XOR with ``other``, as ``pyspark.sql.Column.bitwiseXOR``.

        .. note:: The engine has no bitwise operators yet: this builds the
            node ``SELECT a ^ b`` builds, so it fails with the engine's own
            ``KRISHIV_UNSUPPORTED_FEATURE`` error, as the SQL does.

        Examples
        --------
        >>> df = spark.createDataFrame([(170, 75)], ["a", "b"])
        >>> df.select(df.a.bitwiseXOR(df.b)).collect()
        [Row(value=225)]
        """
        return _call("^", self, other)

    def getItem(self, key: Any) -> Column:
        """The element at ``key`` of an array, or by key of a map, as
        ``pyspark.sql.Column.getItem``.

        .. note:: The engine has no nested types yet: this always fails with
            the engine's ``KRISHIV_UNSUPPORTED_FEATURE`` error.

        Examples
        --------
        >>> df = spark.createDataFrame([([1, 2],)], ["l"])
        >>> df.select(df.l.getItem(0)).collect()
        [Row(value=1)]
        """
        if isinstance(key, Column):
            import warnings

            warnings.warn(
                "A column as 'key' in getItem is deprecated as of Spark 3.0, and will not "
                "be supported in the future release. Use `column[key]` or `column.key` syntax "
                "instead.",
                FutureWarning,
                stacklevel=2,
            )
        raise unsupported_feature("Column.getItem")

    def getField(self, name: Any) -> Column:
        """The field ``name`` of a struct, as ``pyspark.sql.Column.getField``.

        .. note:: The engine has no nested types yet: this always fails with
            the engine's ``KRISHIV_UNSUPPORTED_FEATURE`` error.

        Examples
        --------
        >>> df = spark.createDataFrame([((1, "b"),)], ["r"])
        >>> df.select(df.r.getField("b")).collect()
        [Row(value='b')]
        """
        if isinstance(name, Column):
            import warnings

            warnings.warn(
                "A column as 'name' in getField is deprecated as of Spark 3.0, and will not "
                "be supported in the future release. Use `column[name]` or `column.name` syntax "
                "instead.",
                FutureWarning,
                stacklevel=2,
            )
        raise unsupported_feature("Column.getField")

    def withField(self, fieldName: str, col: Column) -> Column:
        """The struct with field ``fieldName`` added or replaced, as
        ``pyspark.sql.Column.withField``.

        .. note:: The engine has no nested types yet: this always fails with
            the engine's ``KRISHIV_UNSUPPORTED_FEATURE`` error.
        """
        if not isinstance(fieldName, str):
            raise not_expected_type("fieldName", "str", fieldName)
        if not isinstance(col, Column):
            raise not_expected_type("col", "Column", col)
        raise unsupported_feature("Column.withField")

    def dropFields(self, *fieldNames: str) -> Column:
        """The struct without the named fields, as
        ``pyspark.sql.Column.dropFields``.

        .. note:: The engine has no nested types yet: this always fails with
            the engine's ``KRISHIV_UNSUPPORTED_FEATURE`` error.
        """
        for name in fieldNames:
            if not isinstance(name, str):
                raise not_expected_type("fieldNames", "str", name)
        raise unsupported_feature("Column.dropFields")

    def isin(self, *cols: Any) -> Column:
        values = cols[0] if len(cols) == 1 and isinstance(cols[0], list | tuple | set) else cols
        return Column(self._c.isin([_operand(v) for v in values]))

    def between(self, lowerBound: Any, upperBound: Any) -> Column:
        return Column(self._c.between(_operand(lowerBound), _operand(upperBound)))

    def when(self, condition: Column, value: Any) -> Column:
        """Appends a ``WHEN condition THEN value`` branch, as
        ``pyspark.sql.Column.when`` (on a column that is not a ``when``
        chain, starts a new one, as in PySpark).

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
        >>> from krishiv.sql import functions as F
        >>> df.select(df.name, F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0)).show()
        +-----+------------------------------------------------------------+
        | name|CASE WHEN (age > 4) THEN 1 WHEN (age < 3) THEN -1 ELSE 0 END|
        +-----+------------------------------------------------------------+
        |Alice|                                                          -1|
        |  Bob|                                                           1|
        +-----+------------------------------------------------------------+
        """
        if not isinstance(condition, Column):
            raise not_expected_type("condition", "Column", condition)
        if isinstance(self, _WhenColumn):
            return _WhenColumn([*self._branches, (condition._c, _operand(value))], self._else)
        return _WhenColumn([(condition._c, _operand(value))])

    def otherwise(self, value: Any) -> Column:
        """Sets the ``ELSE`` value of a ``when`` chain, as
        ``pyspark.sql.Column.otherwise``. Only a column built by
        :func:`krishiv.sql.functions.when` (or chained :meth:`when`) has an
        ``ELSE`` to set.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
        >>> from krishiv.sql import functions as F
        >>> df.select(df.name, F.when(df.age > 3, 1).otherwise(0)).show()
        +-----+-------------------------------------+
        | name|CASE WHEN (age > 3) THEN 1 ELSE 0 END|
        +-----+-------------------------------------+
        |Alice|                                    0|
        |  Bob|                                    1|
        +-----+-------------------------------------+
        """
        if not isinstance(self, _WhenColumn):
            raise PySparkValueError("otherwise() can only be applied on a column previously generated by when()")
        return _WhenColumn(self._branches, _operand(value))

    def alias(self, *alias: str, **kwargs: Any) -> Column:
        if kwargs:
            raise PySparkValueError(f"unsupported alias argument(s): {sorted(kwargs)}")
        if len(alias) != 1:
            raise PySparkValueError("alias takes exactly one name")
        if not isinstance(alias[0], str):
            raise not_expected_type("alias", "str", alias[0])
        return Column(self._c.alias(alias[0]))

    name = alias

    def cast(self, dataType: DataType | str) -> Column:
        return Column(self._c.cast(_simple_string_of(dataType)))

    def try_cast(self, dataType: DataType | str) -> Column:
        """Casts to ``dataType``, with invalid input as NULL instead of an
        error, as ``pyspark.sql.Column.try_cast``.

        Examples
        --------
        >>> df = spark.createDataFrame([("123",), ("Bob",), (None,)], ["name"])
        >>> df.select(df.name.try_cast("int")).collect()
        [Row(value=123), Row(value=None), Row(value=None)]
        """
        return Column(self._c.try_cast(_simple_string_of(dataType)))

    astype = cast

    # Sort orders: Spark's defaults are nulls first ascending, nulls last descending.
    def asc(self) -> Column:
        return Column(self._c.sort_order(True, True))

    def asc_nulls_first(self) -> Column:
        return self.asc()

    def asc_nulls_last(self) -> Column:
        return Column(self._c.sort_order(True, False))

    def desc(self) -> Column:
        return Column(self._c.sort_order(False, False))

    def desc_nulls_first(self) -> Column:
        return Column(self._c.sort_order(False, True))

    def desc_nulls_last(self) -> Column:
        return self.desc()


class _WhenColumn(Column):
    """A ``CASE WHEN`` chain: ``branches`` of ``(condition, value)`` native
    columns and an optional ``ELSE`` native column. Chained :meth:`when`
    appends a branch (before the ``ELSE``, if set); :meth:`otherwise` sets
    the ``ELSE``. The native expression is rebuilt from the branches every
    time, so it is exactly the node the SQL ``CASE WHEN`` builds."""

    __slots__ = ("_branches", "_else")

    def __init__(
        self, branches: list[tuple[_native.Column, _native.Column]], else_: _native.Column | None = None
    ) -> None:
        parts: list[_native.Column] = [c for pair in branches for c in pair]
        if else_ is not None:
            parts.append(else_)
        super().__init__(_native.call_function("when", parts))
        self._branches = branches
        self._else = else_
