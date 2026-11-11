# SPDX-License-Identifier: Apache-2.0
# Ported from Apache Spark v4.2.0, python/pyspark/sql/udf.py (Apache License 2.0; see NOTICE): the same
# classes and checks; the JVM handle (`_judf`) is the engine's (`_native.AnonymousPythonUdf`).
"""``pyspark.sql.udf``: :class:`UserDefinedFunction` and :class:`UDFRegistration`.

A UDF is not bound to a session: its first call fixes its wrapper in the process (an unregistered UDF the
engine resolves in any session, never by name from SQL) and builds a Column shown ``name(args)``, as Spark's
``PythonUDF`` is (``<lambda>`` for a lambda). ``spark.udf.register`` makes a name SQL-callable in one session.

Divergence (design owner question 2): only the Arrow paths are implemented. A UDF Spark would run on its
pickled-row path (``SQL_BATCHED_UDF``, 100: ``useArrow=False``,
``spark.sql.execution.pythonUDF.arrow.enabled=false`` or no pandas) keeps ``evalType == 100`` and runs
Arrow-batched, whose result coercion differs from Spark's pickled path in some cases.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from krishiv.errors import PySparkNotImplementedError, PySparkTypeError
from krishiv.sql.column import Column, _named
from krishiv.sql.types import ArrayType, DataType, MapType, StringType, StructType, _parse_datatype_string

if TYPE_CHECKING:
    from krishiv.sql.session import SparkSession

__all__ = ["PythonEvalType", "UDFRegistration", "UserDefinedFunction"]


class PythonEvalType:
    """``pyspark.util.PythonEvalType`` (``pyspark/util.py:646-688``)."""

    NON_UDF = 0

    SQL_BATCHED_UDF = 100
    SQL_ARROW_BATCHED_UDF = 101

    SQL_SCALAR_PANDAS_UDF = 200
    SQL_GROUPED_MAP_PANDAS_UDF = 201
    SQL_GROUPED_AGG_PANDAS_UDF = 202
    SQL_WINDOW_AGG_PANDAS_UDF = 203
    SQL_SCALAR_PANDAS_ITER_UDF = 204
    SQL_MAP_PANDAS_ITER_UDF = 205
    SQL_COGROUPED_MAP_PANDAS_UDF = 206
    SQL_MAP_ARROW_ITER_UDF = 207
    SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE = 208
    SQL_GROUPED_MAP_ARROW_UDF = 209
    SQL_COGROUPED_MAP_ARROW_UDF = 210
    SQL_TRANSFORM_WITH_STATE_PANDAS_UDF = 211
    SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF = 212
    SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF = 213
    SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF = 214
    SQL_GROUPED_MAP_ARROW_ITER_UDF = 215
    SQL_GROUPED_MAP_PANDAS_ITER_UDF = 216
    SQL_GROUPED_AGG_PANDAS_ITER_UDF = 217

    SQL_SCALAR_ARROW_UDF = 250
    SQL_SCALAR_ARROW_ITER_UDF = 251
    SQL_GROUPED_AGG_ARROW_UDF = 252
    SQL_WINDOW_AGG_ARROW_UDF = 253
    SQL_GROUPED_AGG_ARROW_ITER_UDF = 254

    SQL_TABLE_UDF = 300
    SQL_ARROW_TABLE_UDF = 301
    SQL_ARROW_UDTF = 302


#: The eval types the engine runs (``SQL_BATCHED_UDF`` on the Arrow-batched path).
_RUNNABLE = {
    PythonEvalType.SQL_BATCHED_UDF: PythonEvalType.SQL_ARROW_BATCHED_UDF,
    PythonEvalType.SQL_ARROW_BATCHED_UDF: PythonEvalType.SQL_ARROW_BATCHED_UDF,
    PythonEvalType.SQL_SCALAR_PANDAS_UDF: PythonEvalType.SQL_SCALAR_PANDAS_UDF,
    PythonEvalType.SQL_SCALAR_ARROW_UDF: PythonEvalType.SQL_SCALAR_ARROW_UDF,
}


def _engine_ddl(dt: DataType) -> str:
    """``dt`` as a DDL string the engine parses (struct field names backquoted)."""
    if isinstance(dt, StructType):
        fields = ",".join("`" + f.name.replace("`", "``") + "`:" + _engine_ddl(f.dataType) for f in dt.fields)
        return f"struct<{fields}>"
    if isinstance(dt, ArrayType):
        return f"array<{_engine_ddl(dt.elementType)}>"
    if isinstance(dt, MapType):
        return f"map<{_engine_ddl(dt.keyType)},{_engine_ddl(dt.valueType)}>"
    return dt.simpleString()


def _engine_registration(
    func: Callable[..., Any], return_type: DataType, eval_type: int
) -> tuple[int, str, Callable[..., Any]]:
    """What the engine registers a UDF with: the eval type it runs, the return type's DDL and the wrapper.
    The wrapper sees the return type as the engine declares it (from the DDL; results are nullable)."""
    from krishiv._udf.worker import make_wrapper

    run = _RUNNABLE.get(eval_type)
    if run is None:
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": f"Python UDFs of eval type {eval_type}"},
        )
    ddl = _engine_ddl(return_type)
    return run, ddl, make_wrapper(func, run, _parse_datatype_string(ddl))


# --- pyspark/sql/udf.py:73-85 ---
def _create_udf(
    f: Callable[..., Any],
    returnType: DataType | str,
    evalType: int,
    name: str | None = None,
    deterministic: bool = True,
) -> Any:
    """Create a regular(non-Arrow-optimized) Python UDF."""
    # Set the name of the UserDefinedFunction object to be the name of function f
    udf_obj = UserDefinedFunction(f, returnType=returnType, name=name, evalType=evalType, deterministic=deterministic)
    return udf_obj._wrapped()


# --- pyspark/sql/udf.py:88-145 ---
def _create_py_udf(
    f: Callable[..., Any],
    returnType: DataType | str,
    useArrow: bool | None = None,
) -> Any:
    """Create a regular/Arrow-optimized Python UDF."""
    is_arrow_enabled = False
    if useArrow is None:
        from krishiv.sql import session as _session

        active = _session._active
        is_arrow_enabled = (
            False if active is None else active.conf.get("spark.sql.execution.pythonUDF.arrow.enabled") == "true"
        )
    else:
        is_arrow_enabled = useArrow

    if is_arrow_enabled:
        from krishiv._udf.pandas_utils import require_minimum_pandas_version, require_minimum_pyarrow_version

        try:
            require_minimum_pandas_version()
            require_minimum_pyarrow_version()
        except ImportError:
            is_arrow_enabled = False
            warnings.warn(
                "Arrow optimization failed to enable because PyArrow or Pandas is not installed. "
                "Falling back to a non-Arrow-optimized UDF.",
                RuntimeWarning,
                stacklevel=3,
            )

    eval_type: int | None = None
    if useArrow is None:
        # If the user doesn't explicitly set useArrow
        from krishiv._udf.typehints import infer_eval_type_for_udf

        try:
            # Try to infer the eval type from type hints
            eval_type = infer_eval_type_for_udf(f)
        except Exception:
            warnings.warn("Cannot infer the eval type from type hints. ", UserWarning, stacklevel=3)

    if eval_type is None:
        if is_arrow_enabled:
            # Arrow optimized Python UDF
            eval_type = PythonEvalType.SQL_ARROW_BATCHED_UDF
        else:
            # Fallback to Regular Python UDF
            eval_type = PythonEvalType.SQL_BATCHED_UDF
            # Krishiv: the pickled-row path is not implemented (design owner question 2).
            warnings.warn(
                "Krishiv runs Python UDFs on the Arrow path only: this UDF (evalType SQL_BATCHED_UDF) runs "
                "Arrow-batched, whose result coercion differs from Spark's pickled path in some cases.",
                UserWarning,
                stacklevel=3,
            )

    return _create_udf(f, returnType, eval_type)


class UserDefinedFunction:
    """
    User defined function in Python

    Notes
    -----
    The constructor of this class is not supposed to be directly called.
    Use :meth:`krishiv.sql.functions.udf` or :meth:`krishiv.sql.functions.pandas_udf`
    to create this instance.
    """

    # --- pyspark/sql/udf.py:161-208 ---
    def __init__(
        self,
        func: Callable[..., Any],
        returnType: DataType | str = StringType(),  # noqa: B008  # Spark's default, an immutable type
        name: str | None = None,
        evalType: int = PythonEvalType.SQL_BATCHED_UDF,
        deterministic: bool = True,
    ):
        if not callable(func):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "callable",
                    "arg_name": "func",
                    "arg_type": type(func).__name__,
                },
            )

        if not isinstance(returnType, DataType | str):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "DataType or str",
                    "arg_name": "returnType",
                    "arg_type": type(returnType).__name__,
                },
            )

        if not isinstance(evalType, int):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "int",
                    "arg_name": "evalType",
                    "arg_type": type(evalType).__name__,
                },
            )

        self.func = func
        self._returnType = returnType
        # Stores the engine's handle, once initialized (Spark's `_judf`).
        self._returnType_placeholder: DataType | None = None
        self._handle_placeholder: Any = None
        self._name: str = name or str(func.__name__ if hasattr(func, "__name__") else func.__class__.__name__)
        self.evalType = evalType
        self.deterministic = deterministic
        #: Krishiv: whether calls may overlap on the UDF pool (free-threaded Python); not in Spark.
        self._thread_safe = True

    # --- pyspark/sql/udf.py:210-248 (the scalar kinds the engine runs) ---
    @staticmethod
    def _check_return_type(returnType: DataType, evalType: int) -> None:
        from krishiv._udf.pandas_types import to_arrow_type

        if evalType == PythonEvalType.SQL_ARROW_BATCHED_UDF:
            try:
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={"feature": f"Invalid return type with Arrow-optimized Python UDF: {returnType}"},
                ) from None
        elif evalType in (PythonEvalType.SQL_SCALAR_PANDAS_UDF, PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF):
            try:
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={"feature": f"Invalid return type with scalar Pandas UDFs: {returnType}"},
                ) from None
        elif evalType in (PythonEvalType.SQL_SCALAR_ARROW_UDF, PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF):
            try:
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={"feature": f"Invalid return type with scalar Arrow UDFs: {returnType}"},
                ) from None

    # --- pyspark/sql/udf.py:391-402 ---
    @property
    def returnType(self) -> DataType:
        if self._returnType_placeholder is None:
            if isinstance(self._returnType, DataType):
                self._returnType_placeholder = self._returnType
            else:
                self._returnType_placeholder = _parse_datatype_string(self._returnType)

        UserDefinedFunction._check_return_type(self._returnType_placeholder, self.evalType)
        return self._returnType_placeholder

    # --- pyspark/sql/udf.py:404-426 (`_judf`, `_create_judf`) ---
    @property
    def _handle(self) -> Any:
        # As Spark's `_judf`: a concurrent first call may make two handles; either is correct.
        if self._handle_placeholder is None:
            self._handle_placeholder = self._create_handle(self.func)
        return self._handle_placeholder

    def _create_handle(self, func: Callable[..., Any]) -> Any:
        from krishiv import _native

        run, ddl, wrapper = _engine_registration(func, self.returnType, self.evalType)
        return _native.AnonymousPythonUdf(self._name, run, ddl, self.deterministic, self._thread_safe, wrapper)

    # --- pyspark/sql/udf.py:428-510 (no profilers) ---
    def __call__(self, *args: Any, **kwargs: Any) -> Column:
        if kwargs:
            raise PySparkNotImplementedError(
                errorClass="NOT_IMPLEMENTED",
                messageParameters={"feature": "keyword arguments of Python UDFs"},
            )
        return Column(self._handle.call([_named(arg, "col") for arg in args]))

    # --- pyspark/sql/udf.py:515-548 ---
    def _wrapped(self) -> Any:
        """
        Wrap this udf with a function and attach docstring from func
        """

        # It is possible for a callable instance without __name__ attribute or/and
        # __module__ attribute to be wrapped here. For example, functools.partial. In this case,
        # we should avoid wrapping the attributes from the wrapped function to the wrapper
        # function. So, we take out these attribute names from the default names to set and
        # then manually assign it after being wrapped.
        assignments = tuple(a for a in functools.WRAPPER_ASSIGNMENTS if a != "__name__" and a != "__module__")

        @functools.wraps(self.func, assigned=assignments)
        def wrapper(*args: Any, **kwargs: Any) -> Column:
            return self(*args, **kwargs)

        wrapper.__name__ = self._name
        wrapper.__module__ = (
            self.func.__module__ if hasattr(self.func, "__module__") else self.func.__class__.__module__
        )

        wrapper.func = self.func  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        wrapper.returnType = self.returnType  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        wrapper.evalType = self.evalType  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        wrapper.deterministic = self.deterministic  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        wrapper.asNondeterministic = functools.wraps(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
            self.asNondeterministic
        )(lambda: self.asNondeterministic()._wrapped())
        wrapper._unwrapped = self  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        return wrapper

    # --- pyspark/sql/udf.py:550-560 ---
    def asNondeterministic(self) -> UserDefinedFunction:
        """
        Updates UserDefinedFunction to nondeterministic.
        """
        # As Spark (SPARK-23233): drop the handle so the next call makes one with 'deterministic' updated.
        self._handle_placeholder = None
        self.deterministic = False
        return self


class UDFRegistration:
    """
    Wrapper for user-defined function registration. This instance can be accessed by
    :attr:`spark.udf`.
    """

    def __init__(self, sparkSession: SparkSession):
        self.sparkSession = sparkSession

    # --- pyspark/sql/udf.py:574-727 ---
    def register(
        self,
        name: str,
        f: Callable[..., Any],
        returnType: DataType | str | None = None,
    ) -> Any:
        """Register a Python function (including lambda function) or a user-defined function
        as a SQL function; the user-defined function under ``name``."""

        # This is to check whether the input function is from a user-defined function or
        # Python function.
        if hasattr(f, "asNondeterministic"):
            if returnType is not None:
                raise PySparkTypeError(
                    errorClass="CANNOT_SPECIFY_RETURN_TYPE_FOR_UDF",
                    messageParameters={"arg_name": "f", "return_type": str(returnType)},
                )
            udf_like: Any = f
            if udf_like.evalType not in [
                PythonEvalType.SQL_BATCHED_UDF,
                PythonEvalType.SQL_ARROW_BATCHED_UDF,
                PythonEvalType.SQL_SCALAR_PANDAS_UDF,
                PythonEvalType.SQL_SCALAR_ARROW_UDF,
                PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF,
                PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF,
                PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
                PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF,
                PythonEvalType.SQL_GROUPED_AGG_PANDAS_ITER_UDF,
                PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF,
            ]:
                raise PySparkTypeError(
                    errorClass="INVALID_UDF_EVAL_TYPE",
                    messageParameters={
                        "eval_type": "SQL_BATCHED_UDF, SQL_ARROW_BATCHED_UDF, "
                        "SQL_SCALAR_PANDAS_UDF, SQL_SCALAR_ARROW_UDF, "
                        "SQL_SCALAR_PANDAS_ITER_UDF, SQL_SCALAR_ARROW_ITER_UDF, "
                        "SQL_GROUPED_AGG_PANDAS_UDF, SQL_GROUPED_AGG_ARROW_UDF, "
                        "SQL_GROUPED_AGG_PANDAS_ITER_UDF or SQL_GROUPED_AGG_ARROW_ITER_UDF"
                    },
                )
            source_udf = _create_udf(
                udf_like.func,
                returnType=udf_like.returnType,
                name=name,
                evalType=udf_like.evalType,
                deterministic=udf_like.deterministic,
            )
            register_udf = source_udf._unwrapped
            return_udf = register_udf
        else:
            if returnType is None:
                returnType = StringType()
            return_udf = _create_udf(f, returnType=returnType, evalType=PythonEvalType.SQL_BATCHED_UDF, name=name)
            register_udf = return_udf._unwrapped
        run, ddl, wrapper = _engine_registration(register_udf.func, register_udf.returnType, register_udf.evalType)
        self.sparkSession._native.register_python_udf(
            name, run, ddl, register_udf.deterministic, register_udf._thread_safe, wrapper
        )
        return return_udf
