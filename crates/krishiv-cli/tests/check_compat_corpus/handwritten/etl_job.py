"""Nightly orders ETL.

Reads the batch from the loader and calls prepare(x) with defaults; see the
wiki page for select the rows from orders (and friends).
"""

import logging
import sys

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

log = logging.getLogger(__name__)


def main(path):
    spark = SparkSession.builder.appName("etl").getOrCreate()
    log.info("loading %s (%d files)", path, 3)
    df = spark.read.parquet(path)
    df = df.withColumn("day", F.to_date("ts")).withColumn("tag", F.lit("#etl"))
    df = df.select("day", F.sentences("text"), F.col("amount"))
    df = df.groupBy("day").agg(F.sum("amount"), F.countDistinct("user"))
    daily = spark.sql(
        """
        SELECT day, legacy_score(amount) AS score,
               kll_sketch_agg_bigint(amount) AS sketch  -- sentences(x) in a comment
        FROM daily_totals
        """
    )
    return df.join(daily, "day").filter("amount > 0")


if __name__ == "__main__":
    main(sys.argv[1])  # F.xpath_float('x') is not a call

# ruff: noqa
