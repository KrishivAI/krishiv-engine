-- query 19: nope(1)
SELECT kll_sketch_get_quantile_double(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_double)';
SELECT array_join(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(array_join)';
SELECT unix_timestamp(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(unix_timestamp)';
