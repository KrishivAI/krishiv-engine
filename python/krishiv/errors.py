# SPDX-License-Identifier: Apache-2.0
"""Exceptions, shaped like ``pyspark.errors`` (Spark 4.2); no dependency on ``pyspark``.

Engine errors carry a Spark error class and SQLSTATE. The JVM exception type Spark
would have thrown is not carried, so :func:`_from_native` picks the class from the
error class and SQLSTATE (see ``_class_for``).
"""

from __future__ import annotations

import re
from typing import Any

# The message templates of the client-side errors this package raises (the wording of
# PySpark 4.2's error classes of the same names).
_TEMPLATES = {
    "NOT_IMPLEMENTED": "<feature> is not implemented.",
    "CANNOT_SET_TOGETHER": "<arg_list> should not be set together.",
    "NOT_EXPECTED_TYPE": "Argument `<arg_name>` should be <expected_type>, got <arg_type>.",
    "CANNOT_CONVERT_COLUMN_INTO_BOOL": (
        "Cannot convert column into bool: please use '&' for 'and', '|' for 'or', "
        "'~' for 'not' when building DataFrame boolean expressions."
    ),
    "UNSUPPORTED_OPERATION": "<operation> is not supported.",
    "VALUE_NOT_ALLOWED": "Value for `<arg_name>` has to be amongst the following values: <allowed_values>.",
    "CANNOT_BE_NONE": "Argument `<arg_name>` cannot be None.",
    "UNSUPPORTED_DATA_TYPE": "Unsupported DataType `<data_type>`.",
}


class PySparkException(Exception):
    """Base of all Krishiv errors (``pyspark.errors.PySparkException``)."""

    def __init__(
        self,
        message: str | None = None,
        errorClass: str | None = None,
        messageParameters: dict[str, str] | None = None,
        contexts: list[Any] | None = None,
        sqlState: str | None = None,
    ) -> None:
        if message is None:
            if errorClass is None:
                raise ValueError("either message or errorClass must be given")
            message = _render(errorClass, messageParameters or {})
        super().__init__(message)
        self._message = message
        self._error_class = errorClass
        self._message_parameters = messageParameters
        self._contexts = contexts or []
        self._sql_state = sqlState

    def getCondition(self) -> str | None:
        """The error class (condition), e.g. ``TABLE_OR_VIEW_NOT_FOUND``."""
        return self._error_class

    def getErrorClass(self) -> str | None:
        """Deprecated name of :meth:`getCondition`."""
        return self._error_class

    def getSqlState(self) -> str | None:
        return self._sql_state

    def getMessage(self) -> str:
        return self._message

    def getMessageParameters(self) -> dict[str, str] | None:
        return self._message_parameters

    def getQueryContext(self) -> list[Any]:
        return list(self._contexts)

    def __str__(self) -> str:
        return self._message


#: The base class under Krishiv's own name.
KrishivException = PySparkException


class AnalysisException(PySparkException):
    """The query failed analysis (unresolved names, type mismatches, ...)."""


class ParseException(AnalysisException):
    """The SQL text did not parse."""


class ArithmeticException(PySparkException):
    """Arithmetic overflow or division by zero."""


class ArrayIndexOutOfBoundsException(PySparkException):
    pass


class DateTimeException(PySparkException):
    pass


class IllegalArgumentException(PySparkException):
    pass


class NumberFormatException(IllegalArgumentException):
    pass


class UnsupportedOperationException(PySparkException):
    pass


class SparkRuntimeException(PySparkException):
    pass


class SparkUpgradeException(PySparkException):
    pass


class SparkNoSuchElementException(PySparkException):
    pass


class QueryExecutionException(PySparkException):
    pass


class UnknownException(PySparkException):
    pass


class PySparkTypeError(PySparkException, TypeError):
    pass


class PySparkValueError(PySparkException, ValueError):
    pass


class PySparkAttributeError(PySparkException, AttributeError):
    pass


class PySparkRuntimeError(PySparkException, RuntimeError):
    pass


class PySparkNotImplementedError(PySparkException, NotImplementedError):
    pass


class PySparkImportError(PySparkException, ImportError):
    pass


def _render(error_class: str, parameters: dict[str, str]) -> str:
    template = _TEMPLATES.get(error_class, error_class)
    text = re.sub(r"<(\w+)>", lambda m: str(parameters.get(m.group(1), m.group(0))), template)
    return f"[{error_class}] {text}"


# Classes (by prefix) with their own Spark exception type, before the SQLSTATE rules.
_BY_PREFIX: tuple[tuple[str, type[PySparkException]], ...] = (
    ("CAST_INVALID_INPUT", NumberFormatException),
    ("INCONSISTENT_BEHAVIOR_CROSS_VERSION", SparkUpgradeException),
    ("INVALID_ARRAY_INDEX", ArrayIndexOutOfBoundsException),
    ("INVALID_PARAMETER_VALUE", IllegalArgumentException),
    ("PARSE_", ParseException),
    ("UNSUPPORTED_DATATYPE", ParseException),
    ("INVALID_SQL_SYNTAX", ParseException),
)
# By SQLSTATE: the analyzer's class 42 (42601, syntax, is the parser's), arithmetic
# (22012 division by zero, 22003 overflow), datetime (22007, 22008), 0A000 unsupported.
_BY_SQL_STATE: dict[str, type[PySparkException]] = {
    "42601": ParseException,
    "22012": ArithmeticException,
    "22003": ArithmeticException,
    "22007": DateTimeException,
    "22008": DateTimeException,
    "22023": IllegalArgumentException,
    "0A000": UnsupportedOperationException,
}


def _class_for(error_class: str, sql_state: str | None) -> type[PySparkException]:
    for prefix, cls in _BY_PREFIX:
        if error_class.startswith(prefix):
            return cls
    if sql_state in _BY_SQL_STATE:
        return _BY_SQL_STATE[sql_state]
    if sql_state is not None and sql_state.startswith("42"):
        return AnalysisException
    return SparkRuntimeException


def _from_native(error_class: str, sql_state: str | None, message: str, parameters: dict[str, str]) -> PySparkException:
    """The exception for an engine error (called by ``krishiv._native``)."""
    return _class_for(error_class, sql_state)(
        message, errorClass=error_class, messageParameters=parameters, sqlState=sql_state
    )


def not_expected_type(arg_name: str, expected: str, value: object) -> PySparkTypeError:
    return PySparkTypeError(
        errorClass="NOT_EXPECTED_TYPE",
        messageParameters={"arg_name": arg_name, "expected_type": expected, "arg_type": type(value).__name__},
    )


def not_implemented(feature: str) -> PySparkNotImplementedError:
    return PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": feature})
