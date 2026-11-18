-- query 6: nope(1)
SELECT from_json(a, b) AS c1 FROM t6 WHERE x IN (1, 2) AND y = 'x(from_json)';
SELECT legacy_fn_3(a, b) AS c2 FROM t6 WHERE x IN (1, 2) AND y = 'x(legacy_fn_3)';
SELECT theta_sketch_estimate(a, b) AS c3 FROM t6 WHERE x IN (1, 2) AND y = 'x(theta_sketch_estimate)';
SELECT sha2(a, b) AS c4 FROM t6 WHERE x IN (1, 2) AND y = 'x(sha2)';
