-- query 5: nope(1)
SELECT legacy_fn_16(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT levenshtein(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(levenshtein)';
SELECT legacy_fn_35(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT min(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(min)';
SELECT theta_union(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(theta_union)';
