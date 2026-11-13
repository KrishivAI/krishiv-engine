-- query 21: nope(1)
SELECT time(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(time)';
SELECT vector_inner_product(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(vector_inner_product)';
SELECT time_to_micros(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(time_to_micros)';
SELECT hll_sketch_estimate(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(hll_sketch_estimate)';
SELECT legacy_fn_11(a, b) AS c5 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
