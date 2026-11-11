# SPDX-License-Identifier: Apache-2.0
"""The Python side of embedded Python UDFs (design P2-T059 unit 2c).

``worker`` builds the wrapper the native runtime calls for each eval type (``wrapper(*pyarrow.Array) ->
pyarrow.Array``). The conversions it runs are PySpark 4.2.0's, vendored unchanged but for their imports
(``conversion``, ``pandas_types``, ``pandas_utils``, ``typehints``, ``loose_version``; Apache License 2.0, see
``NOTICE``); ``types`` is what they import as ``pyspark.sql.types``.
"""
