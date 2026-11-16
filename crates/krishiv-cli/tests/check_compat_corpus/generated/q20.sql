-- query 20: nope(1)
SELECT tan(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(tan)';
SELECT vector_avg(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(vector_avg)';
SELECT time_from_millis(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(time_from_millis)';
SELECT hll_union(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(hll_union)';
SELECT legacy_fn_11(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
