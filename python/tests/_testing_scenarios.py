# SPDX-License-Identifier: Apache-2.0
"""Scenarios for the ``krishiv.testing`` vs ``pyspark.testing`` differential (data only, no imports of either).

Each scenario is ``(id, kind, args, kwargs)``. ``kind`` is ``rows`` (``args`` = actual rows, expected rows, each a
list of dicts) or ``schema`` (``args`` = actual fields, expected fields, each ``(name, type, nullable)`` with
``type`` a ``types`` class name without ``Type``, or ``("array", elem, contains_null)`` for arrays).
``tests/gen_pyspark_testing_fixture.py`` runs them through real ``pyspark.testing`` (4.2.0) into
``fixtures/pyspark-testing-4.2.0.json``; ``test_testing_differential.py`` runs the same through ``krishiv.testing``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

NAN = float("nan")
INF = float("inf")


def _r(**kw: object) -> dict[str, object]:
    return dict(kw)


SCENARIOS: list[tuple[str, str, tuple, dict]] = [
    ("rows-equal", "rows", ([_r(a=1, b="x")], [_r(a=1, b="x")]), {}),
    ("rows-both-empty", "rows", ([], []), {}),
    ("rows-one-differs", "rows", ([_r(a=1)], [_r(a=2)]), {}),
    ("rows-extra-row", "rows", ([_r(a=1), _r(a=2)], [_r(a=1)]), {}),
    ("rows-missing-row", "rows", ([_r(a=1)], [_r(a=1), _r(a=2)]), {}),
    ("rows-order-ignored", "rows", ([_r(a=1), _r(a=2)], [_r(a=2), _r(a=1)]), {}),
    ("rows-order-checked", "rows", ([_r(a=1), _r(a=2)], [_r(a=2), _r(a=1)]), {"checkRowOrder": True}),
    ("rows-null-vs-value", "rows", ([_r(a=None)], [_r(a=1)]), {}),
    ("rows-null-equal", "rows", ([_r(a=None)], [_r(a=None)]), {}),
    ("rows-float-within-rtol", "rows", ([_r(a=1.0)], [_r(a=1.000001)]), {}),
    ("rows-float-outside-rtol", "rows", ([_r(a=1.0)], [_r(a=1.001)]), {}),
    ("rows-float-custom-rtol", "rows", ([_r(a=1.0)], [_r(a=1.001)]), {"rtol": 1e-2}),
    ("rows-float-custom-atol", "rows", ([_r(a=0.0)], [_r(a=0.5)]), {"atol": 1.0}),
    ("rows-float-nan-equal", "rows", ([_r(a=NAN)], [_r(a=NAN)]), {}),
    ("rows-float-nan-vs-value", "rows", ([_r(a=NAN)], [_r(a=1.0)]), {}),
    ("rows-float-inf-equal", "rows", ([_r(a=INF)], [_r(a=INF)]), {}),
    ("rows-float-inf-vs-neginf", "rows", ([_r(a=INF)], [_r(a=-INF)]), {}),
    ("rows-array-float-tolerance", "rows", ([_r(a=[1.0, 2.0])], [_r(a=[1.0, 2.0000001])]), {}),
    ("rows-array-length", "rows", ([_r(a=[1, 2])], [_r(a=[1, 2, 3])]), {}),
    ("rows-string-case", "rows", ([_r(a="A")], [_r(a="a")]), {}),
    ("rows-decimal-equal", "rows", ([_r(a=Decimal("1.10"))], [_r(a=Decimal("1.10"))]), {}),
    ("rows-decimal-differs", "rows", ([_r(a=Decimal("1.10"))], [_r(a=Decimal("1.11"))]), {}),
    ("rows-date-differs", "rows", ([_r(a=dt.date(2020, 1, 1))], [_r(a=dt.date(2020, 1, 2))]), {}),
    ("rows-int-vs-float", "rows", ([_r(a=1)], [_r(a=1.0)]), {}),
    ("rows-bool-vs-int", "rows", ([_r(a=True)], [_r(a=1)]), {}),
    ("rows-two-columns-one-differs", "rows", ([_r(a=1, b="x")], [_r(a=1, b="y")]), {}),
    ("rows-column-names-differ", "rows", ([_r(a=1)], [_r(b=1)]), {}),
    (
        "rows-many-differ-max-errors",
        "rows",
        ([_r(a=i) for i in range(5)], [_r(a=i + 10) for i in range(5)]),
        {"maxErrors": 2},
    ),
    (
        "rows-show-only-diff",
        "rows",
        ([_r(a=1), _r(a=2)], [_r(a=1), _r(a=3)]),
        {"checkRowOrder": True, "showOnlyDiff": True},
    ),
    (
        "rows-include-diff-rows",
        "rows",
        ([_r(a=1), _r(a=2)], [_r(a=1), _r(a=3)]),
        {"checkRowOrder": True, "includeDiffRows": True},
    ),
    ("rows-nested-row-list", "rows", ([_r(a=[_r(x=1)])], [_r(a=[_r(x=1)])]), {}),
    ("rows-map-equal", "rows", ([_r(a={"k": 1})], [_r(a={"k": 1})]), {}),
    ("rows-map-differs", "rows", ([_r(a={"k": 1})], [_r(a={"k": 2})]), {}),
    ("schema-equal", "schema", ([("a", "Integer", True)], [("a", "Integer", True)]), {}),
    ("schema-type-differs", "schema", ([("a", "Integer", True)], [("a", "Long", True)]), {}),
    ("schema-name-differs", "schema", ([("a", "Integer", True)], [("b", "Integer", True)]), {}),
    ("schema-nullable-differs-default", "schema", ([("a", "Integer", True)], [("a", "Integer", False)]), {}),
    (
        "schema-nullable-differs-strict",
        "schema",
        ([("a", "Integer", True)], [("a", "Integer", False)]),
        {"ignoreNullable": False},
    ),
    ("schema-extra-field", "schema", ([("a", "Integer", True), ("b", "String", True)], [("a", "Integer", True)]), {}),
    (
        "schema-column-order",
        "schema",
        ([("a", "Integer", True), ("b", "String", True)], [("b", "String", True), ("a", "Integer", True)]),
        {},
    ),
    (
        "schema-column-order-ignored",
        "schema",
        ([("a", "Integer", True), ("b", "String", True)], [("b", "String", True), ("a", "Integer", True)]),
        {"ignoreColumnOrder": True},
    ),
    (
        "schema-column-name-ignored",
        "schema",
        ([("a", "Integer", True)], [("b", "Integer", True)]),
        {"ignoreColumnName": True},
    ),
    (
        "schema-array-contains-null",
        "schema",
        ([("a", ("array", "Integer", True), True)], [("a", ("array", "Integer", False), True)]),
        {},
    ),
    (
        "schema-array-contains-null-strict",
        "schema",
        ([("a", ("array", "Integer", True), True)], [("a", ("array", "Integer", False), True)]),
        {"ignoreNullable": False},
    ),
]
