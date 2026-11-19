-- query 5: nope(1)
SELECT sequence(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(sequence)';
SELECT theta_union_agg(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
SELECT kll_sketch_to_string_bigint(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_bigint)';
SELECT approx_top_k_accumulate(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT legacy_fn_1(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
SELECT make_interval(a, b) AS c6 FROM t5 WHERE x IN (1, 2) AND y = 'x(make_interval)';
