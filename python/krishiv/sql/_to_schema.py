# SPDX-License-Identifier: Apache-2.0
"""``DataFrame.to(schema)``: Spark's ``Project.matchSchema``, checked against PySpark 4.2.0.

Each field of the target schema is found by name (case rules of ``spark.sql.caseSensitive``) in the input, in
the target's order; input fields the target lacks are dropped. A target field the input lacks reads ``NULL`` if
it is nullable and is ``UNRESOLVED_COLUMN.WITH_SUGGESTION`` (``UNRESOLVED_FIELD.WITH_SUGGESTION`` inside a
struct) if not. A nullable input into a non-nullable field is ``NULLABLE_COLUMN_OR_FIELD``. A value is cast
only where Spark's store assignment allows it (numeric to numeric, anything atomic to string, date, timestamp
and timestamp_ntz among themselves, else ``INVALID_COLUMN_OR_FIELD_DATA_TYPE``), arrays and maps element-wise,
structs recursively by name. A struct inside an array or map whose fields would need reordering is not
supported (``NOT_IMPLEMENTED``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from krishiv.errors import AnalysisException, not_implemented
from krishiv.sql import functions as F
from krishiv.sql.column import Column, _literal
from krishiv.sql.types import (
    ArrayType,
    AtomicType,
    CharType,
    DataType,
    DateType,
    DecimalType,
    MapType,
    NumericType,
    StringType,
    StructType,
    TimestampNTZType,
    TimestampType,
    VarcharType,
)

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame

_DATETIME = (DateType, TimestampType, TimestampNTZType)


def _quote(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _path(parts: list[str]) -> str:
    return ".".join(_quote(p) for p in parts)


def _same_name(a: str, b: str, case_sensitive: bool) -> bool:
    return a == b if case_sensitive else a.lower() == b.lower()


def _has_char_or_varchar(t: DataType) -> bool:
    if isinstance(t, CharType | VarcharType):
        return True
    if isinstance(t, ArrayType):
        return _has_char_or_varchar(t.elementType)
    if isinstance(t, MapType):
        return _has_char_or_varchar(t.keyType) or _has_char_or_varchar(t.valueType)
    if isinstance(t, StructType):
        return any(_has_char_or_varchar(f.dataType) for f in t.fields)
    return False


def _bad_type(path: list[str], have: DataType, want: DataType) -> AnalysisException:
    return AnalysisException(
        f"[INVALID_COLUMN_OR_FIELD_DATA_TYPE] Column or field {_path(path)} is of type "
        f'"{have.simpleString().upper()}" while it\'s required to be "{want.simpleString().upper()}". SQLSTATE: 42000',
        errorClass="INVALID_COLUMN_OR_FIELD_DATA_TYPE",
        messageParameters={
            "name": _path(path),
            "type": f'"{have.simpleString().upper()}"',
            "expectedType": f'"{want.simpleString().upper()}"',
        },
        sqlState="42000",
    )


def _assignable(have: DataType, want: DataType) -> bool:
    """Whether a value of ``have`` may be stored as ``want`` (non-struct types; Spark's store assignment)."""
    if have == want:
        return True
    if isinstance(have, NumericType) and isinstance(want, NumericType):
        return True
    if isinstance(have, DecimalType) and isinstance(want, DecimalType):
        return True
    if type(want) is StringType and isinstance(have, AtomicType):
        return True
    if isinstance(have, _DATETIME) and isinstance(want, _DATETIME):
        return True
    if isinstance(have, ArrayType) and isinstance(want, ArrayType):
        return _assignable(have.elementType, want.elementType)
    if isinstance(have, MapType) and isinstance(want, MapType):
        return _assignable(have.keyType, want.keyType) and _assignable(have.valueType, want.valueType)
    return False


def _holds_struct(t: DataType) -> bool:
    if isinstance(t, StructType):
        return True
    if isinstance(t, ArrayType):
        return _holds_struct(t.elementType)
    if isinstance(t, MapType):
        return _holds_struct(t.keyType) or _holds_struct(t.valueType)
    return False


def _reconcile(
    expr: Column, have: DataType, have_nullable: bool, want: DataType, want_nullable: bool, path: list[str], case: bool
) -> Column:
    if have_nullable and not want_nullable:
        raise AnalysisException(
            f"[NULLABLE_COLUMN_OR_FIELD] Column or field {_path(path)} is nullable while it's required to be "
            "non-nullable. SQLSTATE: 42000",
            errorClass="NULLABLE_COLUMN_OR_FIELD",
            messageParameters={"name": _path(path)},
            sqlState="42000",
        )
    if isinstance(have, StructType) and isinstance(want, StructType):
        fields = _match_fields(expr, have, want, path, case)
        return F.when(expr.isNotNull(), F.named_struct(*[x for pair in fields for x in pair]))
    if _holds_struct(have) or _holds_struct(want):
        if have == want:
            return expr
        if isinstance(have, StructType) != isinstance(want, StructType) or not (
            isinstance(have, ArrayType | MapType) and type(have) is type(want)
        ):
            raise _bad_type(path, have, want)
        raise not_implemented("DataFrame.to with a struct inside an array or map whose type differs")
    if not _assignable(have, want):
        raise _bad_type(path, have, want)
    return expr if have == want else expr.cast(want)


def _match_fields(
    parent: Column | None, have: StructType, want: StructType, path: list[str], case: bool
) -> list[tuple[Column, Column]]:
    """``(name literal, value)`` pairs for the fields of ``want``, read from ``parent`` (``None``: the row)."""
    out: list[tuple[Column, Column]] = []
    for field in want.fields:
        found = next((f for f in have.fields if _same_name(f.name, field.name, case)), None)
        here = [*path, field.name]
        if found is None:
            if not field.nullable:
                suggestions = ", ".join(_quote(f.name) for f in have.fields)
                if parent is None:
                    raise AnalysisException(
                        f"[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column, variable, or function parameter with name "
                        f"{_quote(field.name)} cannot be resolved. Did you mean one of the following? "
                        f"[{suggestions}]. SQLSTATE: 42703",
                        errorClass="UNRESOLVED_COLUMN.WITH_SUGGESTION",
                        messageParameters={"objectName": _quote(field.name), "proposal": suggestions},
                        sqlState="42703",
                    )
                raise AnalysisException(
                    f"[UNRESOLVED_FIELD.WITH_SUGGESTION] A field with name {_quote(field.name)} cannot be resolved "
                    f"with the struct-type column {_path(path)}. Did you mean one of the following? "
                    f"[{suggestions}]. SQLSTATE: 42703",
                    errorClass="UNRESOLVED_FIELD.WITH_SUGGESTION",
                    messageParameters={
                        "fieldName": _quote(field.name),
                        "columnName": _path(path),
                        "proposal": suggestions,
                    },
                    sqlState="42703",
                )
            value = Column(_literal(None)).cast(field.dataType)
        else:
            source = F.col(_quote(found.name)) if parent is None else parent.getField(found.name)
            value = _reconcile(source, found.dataType, found.nullable, field.dataType, field.nullable, here, case)
        out.append((F.lit(field.name), value))
    return out


def to_schema(df: DataFrame, schema: StructType, case_sensitive: bool) -> DataFrame:
    """``df`` reconciled with ``schema``, as ``Dataset.to``."""
    if any(_has_char_or_varchar(f.dataType) for f in schema.fields):
        raise AnalysisException(
            "[UNSUPPORTED_CHAR_OR_VARCHAR_AS_STRING] The char/varchar type can't be used in the table schema.\n"
            "If you want Spark treat them as string type as same as Spark 3.0 and earlier, please set "
            '"spark.sql.legacy.charVarcharAsString" to "true". SQLSTATE: 0A000',
            errorClass="UNSUPPORTED_CHAR_OR_VARCHAR_AS_STRING",
            sqlState="0A000",
        )
    pairs = _match_fields(None, df.schema, schema, [], case_sensitive)
    return df.select(*[value.alias(schema.fields[i].name) for i, (_, value) in enumerate(pairs)])
