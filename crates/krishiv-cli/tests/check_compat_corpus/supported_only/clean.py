"""Reads the batch from the loader and calls prepare(x) with defaults."""

import torch.nn.functional as F
import pyspark.sql.functions as sf


def run(x, items):
    msg = "%s(%d) select rows from the table (see docs)" % ("a", 1)
    print(f"computing {len(items)} items with with(x)")
    y = F.relu(x) + F.softmax(x, dim=-1)
    z = sf.when(sf.col("a") > 1, 1).otherwise(0)
    w = sf.col("a").cast("decimal(10, 2)").alias("b")
    return y, z, w, msg, sf.lit("#"), sf.sum_distinct("a"), sf.create_map("a", "b")
