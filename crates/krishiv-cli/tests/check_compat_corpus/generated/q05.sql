-- query 5: nope(1)
SELECT session_user(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(session_user)';
SELECT theta_sketch_estimate(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(theta_sketch_estimate)';
SELECT kll_sketch_merge_float(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(kll_sketch_merge_float)';
SELECT approx_top_k_accumulate(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT legacy_fn_1(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_1)';
SELECT make_time(a, b) AS c6 FROM t5 WHERE x IN (1, 2) AND y = 'x(make_time)';
