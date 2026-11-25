-- query 15: nope(1)
SELECT degrees(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(degrees)';
SELECT ifnull(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(ifnull)';
SELECT approx_top_k_estimate(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(approx_top_k_estimate)';
SELECT schema_of_variant_agg(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(schema_of_variant_agg)';
