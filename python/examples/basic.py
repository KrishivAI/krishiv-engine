# SPDX-License-Identifier: Apache-2.0
"""Typed example for ``mypy --strict``: a query and its notebook rendering."""

from krishiv.sql import DataFrame, SparkSession


def main() -> None:
    spark: SparkSession = SparkSession.builder.create()
    df: DataFrame = spark.sql("SELECT 1 AS a, 'x' AS b")
    print(repr(df))
    html: str | None = df._repr_html_()
    assert html is not None
    print(html)


if __name__ == "__main__":
    main()
