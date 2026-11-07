# SPDX-License-Identifier: Apache-2.0
# Ported from Apache Spark v4.2.0, python/pyspark/sql/pandas/functions.py (Apache License 2.0; see NOTICE):
# `PandasUDFType`, `ArrowUDFType` (lines 32-56), `vectorized_udf`, `_validate_vectorized_udf` and
# `_create_vectorized_udf` (lines 783-1031) unchanged but for their imports and the Connect branch.
"""``pyspark.sql.pandas.functions``: :func:`pandas_udf` and :func:`arrow_udf` (design P2-T059 unit 2c).

The engine runs the scalar kinds (``PandasUDFType.SCALAR``, ``ArrowUDFType.SCALAR``); a UDF of another kind is
made as in Spark and raises ``NOT_IMPLEMENTED`` when called.
"""

# ruff: noqa: SIM108, B028
# The ported functions keep Spark's spelling, for a readable diff against the tag.

import functools
import warnings
from inspect import getfullargspec, signature
from typing import Any, get_type_hints

from krishiv._udf.pandas_utils import require_minimum_pandas_version, require_minimum_pyarrow_version
from krishiv._udf.typehints import infer_eval_type
from krishiv.errors import PySparkTypeError, PySparkValueError
from krishiv.sql.types import DataType
from krishiv.sql.udf import PythonEvalType, _create_udf

__all__ = ["ArrowUDFType", "PandasUDFType", "arrow_udf", "pandas_udf"]


class PandasUDFType:
    """Pandas UDF Types. See :meth:`pyspark.sql.functions.pandas_udf`."""

    SCALAR = PythonEvalType.SQL_SCALAR_PANDAS_UDF

    SCALAR_ITER = PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF

    GROUPED_MAP = PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF

    GROUPED_AGG = PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF

    GROUPED_AGG_ITER = PythonEvalType.SQL_GROUPED_AGG_PANDAS_ITER_UDF


class ArrowUDFType:
    """Arrow UDF Types. See :meth:`pyspark.sql.functions.arrow_udf`."""

    SCALAR = PythonEvalType.SQL_SCALAR_ARROW_UDF

    SCALAR_ITER = PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF

    GROUPED_AGG = PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF

    GROUPED_AGG_ITER = PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF


def arrow_udf(f: Any = None, returnType: Any = None, functionType: Any = None) -> Any:
    """Creates an Arrow UDF (``pyspark.sql.functions.arrow_udf``, ``pandas/functions.py:58-396``): the
    function takes and returns ``pyarrow.Array`` values; ``returnType`` is required."""
    require_minimum_pyarrow_version()

    return vectorized_udf(f, returnType, functionType, "arrow")


def pandas_udf(f: Any = None, returnType: Any = None, functionType: Any = None) -> Any:
    """Creates a pandas UDF (``pyspark.sql.functions.pandas_udf``, ``pandas/functions.py:399-781``): the
    function takes and returns ``pandas.Series`` (a ``pandas.DataFrame`` for a struct); ``returnType`` is
    required. Without pandas this raises ``PACKAGE_NOT_INSTALLED``."""
    require_minimum_pandas_version()
    require_minimum_pyarrow_version()

    return vectorized_udf(f, returnType, functionType, "pandas")


def vectorized_udf(
    f=None,
    returnType=None,
    functionType=None,
    kind: str = "pandas",
):
    assert kind in ["pandas", "arrow"], "kind should be either 'pandas' or 'arrow'"

    # decorator @pandas_udf(returnType, functionType)
    is_decorator = f is None or isinstance(f, (str, DataType))

    if is_decorator:
        # If DataType has been passed as a positional argument
        # for decorator use it as a returnType
        return_type = f or returnType

        if functionType is not None:
            # @pandas_udf(dataType, functionType=functionType)
            # @pandas_udf(returnType=dataType, functionType=functionType)
            eval_type = functionType
        elif returnType is not None and isinstance(returnType, int):
            # @pandas_udf(dataType, functionType)
            eval_type = returnType
        else:
            # @pandas_udf(dataType) or @pandas_udf(returnType=dataType)
            eval_type = None
    else:
        return_type = returnType

        if functionType is not None:
            eval_type = functionType
        else:
            eval_type = None

    if return_type is None:
        raise PySparkTypeError(
            errorClass="CANNOT_BE_NONE",
            messageParameters={"arg_name": "returnType"},
        )

    if kind == "pandas" and eval_type not in [
        PythonEvalType.SQL_SCALAR_PANDAS_UDF,
        PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF,
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_ITER_UDF,
        PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_AGG_PANDAS_ITER_UDF,
        PythonEvalType.SQL_MAP_PANDAS_ITER_UDF,
        PythonEvalType.SQL_MAP_ARROW_ITER_UDF,
        PythonEvalType.SQL_COGROUPED_MAP_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PANDAS_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF,
        PythonEvalType.SQL_GROUPED_MAP_ARROW_UDF,
        PythonEvalType.SQL_GROUPED_MAP_ARROW_ITER_UDF,
        PythonEvalType.SQL_COGROUPED_MAP_ARROW_UDF,
        None,
    ]:  # None means it should infer the type from type hints.
        raise PySparkTypeError(
            errorClass="INVALID_PANDAS_UDF_TYPE",
            messageParameters={
                "arg_name": "functionType",
                "arg_type": str(eval_type),
            },
        )
    if kind == "arrow" and eval_type not in [
        PythonEvalType.SQL_SCALAR_ARROW_UDF,
        PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF,
        PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF,
        PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF,
        None,
    ]:  # None means it should infer the type from type hints.
        raise PySparkTypeError(
            errorClass="INVALID_PANDAS_UDF_TYPE",
            messageParameters={
                "arg_name": "functionType",
                "arg_type": str(eval_type),
            },
        )

    if is_decorator:
        return functools.partial(
            _create_vectorized_udf,
            returnType=return_type,
            evalType=eval_type,
            kind=kind,
        )
    else:
        return _create_vectorized_udf(
            f=f,
            returnType=return_type,
            evalType=eval_type,
            kind=kind,
        )


# validate the pandas udf and return the adjusted eval type
def _validate_vectorized_udf(f, evalType, kind: str = "pandas") -> int:
    assert kind in ["pandas", "arrow"], "kind should be either 'pandas' or 'arrow'"

    argspec = getfullargspec(f)

    # pandas UDF by type hints.
    if evalType in [
        PythonEvalType.SQL_SCALAR_PANDAS_UDF,
        PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF,
        PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_AGG_PANDAS_ITER_UDF,
    ]:
        warnings.warn(
            "In Python 3.6+ and Spark 3.0+, it is preferred to specify type hints for "
            "pandas UDF instead of specifying pandas UDF type which will be deprecated "
            "in the future releases. See SPARK-28264 for more details.",
            UserWarning,
        )
    elif evalType in [
        PythonEvalType.SQL_SCALAR_ARROW_UDF,
        PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF,
        PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF,
        PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF,
    ]:
        warnings.warn(
            "It is preferred to specify type hints for arrow UDF instead of specifying arrow UDF type.",
            UserWarning,
        )
    elif evalType in [
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_ITER_UDF,
        PythonEvalType.SQL_MAP_PANDAS_ITER_UDF,
        PythonEvalType.SQL_MAP_ARROW_ITER_UDF,
        PythonEvalType.SQL_COGROUPED_MAP_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PANDAS_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF,
        PythonEvalType.SQL_GROUPED_MAP_ARROW_UDF,
        PythonEvalType.SQL_GROUPED_MAP_ARROW_ITER_UDF,
        PythonEvalType.SQL_COGROUPED_MAP_ARROW_UDF,
        PythonEvalType.SQL_ARROW_BATCHED_UDF,
    ]:
        # In case of 'SQL_GROUPED_MAP_PANDAS_UDF', deprecation warning is being triggered
        # at `apply` instead.
        # In case of 'SQL_MAP_PANDAS_ITER_UDF', 'SQL_MAP_ARROW_ITER_UDF' and
        # 'SQL_COGROUPED_MAP_PANDAS_UDF', the evaluation type will always be set.
        # In case of 'SQL_ARROW_BATCHED_UDF', no deprecation warning is required since it is not
        # exposed to users.
        pass
    elif len(argspec.annotations) > 0:
        try:
            type_hints = get_type_hints(f)
        except NameError:
            type_hints = {}
        evalType = infer_eval_type(signature(f), type_hints, kind)
        assert evalType is not None

    if evalType is None:
        # Set default is scalar UDF.
        if kind == "pandas":
            evalType = PythonEvalType.SQL_SCALAR_PANDAS_UDF
        else:
            evalType = PythonEvalType.SQL_SCALAR_ARROW_UDF

    kind_str = "pandas_udfs" if kind == "pandas" else "arrow_udfs"

    if (
        (
            evalType == PythonEvalType.SQL_SCALAR_PANDAS_UDF
            or evalType == PythonEvalType.SQL_SCALAR_ARROW_UDF
            or evalType == PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF
            or evalType == PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF
        )
        and len(argspec.args) == 0
        and argspec.varargs is None
    ):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": f"0-arg {kind_str} are not supported. "
                f"Instead, create a 1-arg {kind_str} and ignore the arg in your function.",
            },
        )

    if evalType == PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF and len(argspec.args) not in (1, 2):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": f"{kind_str} with function type GROUPED_MAP or the function in "
                "groupby.applyInPandas must take either one argument (data) or "
                "two arguments (key, data).",
            },
        )

    if evalType == PythonEvalType.SQL_GROUPED_MAP_PANDAS_ITER_UDF and len(argspec.args) not in (
        1,
        2,
    ):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in groupby.applyInPandas with iterator API must take "
                "either one argument (batches: Iterator[pandas.DataFrame]) or two arguments "
                "(key, batches: Iterator[pandas.DataFrame]).",
            },
        )

    if evalType == PythonEvalType.SQL_GROUPED_MAP_ARROW_UDF and len(argspec.args) not in (1, 2):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in groupby.applyInArrow must take either one argument "
                "(data) or two arguments (key, data).",
            },
        )

    if evalType == PythonEvalType.SQL_COGROUPED_MAP_PANDAS_UDF and len(argspec.args) not in (2, 3):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in cogroup.applyInPandas must take either two arguments "
                "(left, right) or three arguments (key, left, right).",
            },
        )

    if evalType == PythonEvalType.SQL_COGROUPED_MAP_ARROW_UDF and len(argspec.args) not in (2, 3):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in cogroup.applyInArrow must take either two arguments "
                "(left, right) or three arguments (key, left, right).",
            },
        )

    return evalType


def _create_vectorized_udf(f, returnType, evalType, kind):
    evalType = _validate_vectorized_udf(f, evalType, kind)
    # Krishiv: no Spark Connect client (`pandas/functions.py:1025-1028`).
    return _create_udf(f, returnType, evalType)
