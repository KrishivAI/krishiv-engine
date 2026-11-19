from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

spark = SparkSession.builder.getOrCreate()
events = spark.readStream.format("rate").load()
windowed = (
    events.withWatermark("timestamp", "10 minutes")
    .groupBy(sf.window("timestamp", "5 minutes"))
    .agg(sf.count("*").alias("n"), sf.approx_count_distinct("value"))
)
scored = windowed.selectExpr("n", "theta_sketch_agg(n) AS t", "abs(n)")
out = scored.where("hll_sketch_estimate(n) > 3")
out = out.withColumn("when", sf.when(sf.col("n") > 1, sf.lit("many")).otherwise("few"))
