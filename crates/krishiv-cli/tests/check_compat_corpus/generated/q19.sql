-- query 19: nope(1)
SELECT theta_sketch_agg(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(theta_sketch_agg)';
SELECT session_window(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(session_window)';
SELECT theta_union_agg(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(theta_union_agg)';
SELECT legacy_fn_26(a, b) AS c4 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT space(a, b) AS c5 FROM t19 WHERE x IN (1, 2) AND y = 'x(space)';
