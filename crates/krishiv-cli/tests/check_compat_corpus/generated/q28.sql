-- query 28: nope(1)
SELECT percentile_approx(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(percentile_approx)';
SELECT tuple_sketch_estimate_double(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_estimate_double)';
SELECT theta_union_agg(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
