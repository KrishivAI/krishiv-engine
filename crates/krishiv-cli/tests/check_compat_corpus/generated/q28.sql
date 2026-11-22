-- query 28: nope(1)
SELECT percentile(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(percentile)';
SELECT tuple_sketch_estimate_integer(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_estimate_integer)';
SELECT try_reflect(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
