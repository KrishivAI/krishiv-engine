-- query 6: nope(1)
SELECT xpath_short(a, b) AS c1 FROM t6 WHERE x IN (1, 2) AND y = 'x(xpath_short)';
SELECT xpath_string(a, b) AS c2 FROM t6 WHERE x IN (1, 2) AND y = 'x(xpath_string)';
SELECT percentile_disc(a, b) AS c3 FROM t6 WHERE x IN (1, 2) AND y = 'x(percentile_disc)';
SELECT array_min(a, b) AS c4 FROM t6 WHERE x IN (1, 2) AND y = 'x(array_min)';
SELECT to_variant_object(a, b) AS c5 FROM t6 WHERE x IN (1, 2) AND y = 'x(to_variant_object)';
