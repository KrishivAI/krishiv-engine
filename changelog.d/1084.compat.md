A PySpark 3.5 client connected to `krishiv server` now gets Spark 3.5 answers: ANSI mode is off by default, so `1/0` and a
failed cast return NULL instead of raising. Each Spark Connect session follows the Spark release its client declares
(`--spark-profile auto`, the default); 4.0 to 4.2 clients now get their own release's behaviour too, and
`--spark-profile 4.3` (or `krishiv.connect.sparkProfile`) restores the previous single-profile behaviour.
