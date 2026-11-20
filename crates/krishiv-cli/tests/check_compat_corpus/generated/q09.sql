-- query 9: nope(1)
SELECT array_compact(a, b) AS c1 FROM t9 WHERE x IN (1, 2) AND y = 'x(array_compact)';
SELECT schema_of_variant_agg(a, b) AS c2 FROM t9 WHERE x IN (1, 2) AND y = 'x(schema_of_variant_agg)';
SELECT ceil(a, b) AS c3 FROM t9 WHERE x IN (1, 2) AND y = 'x(ceil)';
SELECT tinyint(a, b) AS c4 FROM t9 WHERE x IN (1, 2) AND y = 'x(tinyint)';
SELECT array_except(a, b) AS c5 FROM t9 WHERE x IN (1, 2) AND y = 'x(array_except)';
