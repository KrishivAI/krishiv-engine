-- query 16: nope(1)
SELECT spark_partition_id(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(spark_partition_id)';
SELECT regexp_replace(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(regexp_replace)';
SELECT make_interval(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(make_interval)';
SELECT dayname(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(dayname)';
