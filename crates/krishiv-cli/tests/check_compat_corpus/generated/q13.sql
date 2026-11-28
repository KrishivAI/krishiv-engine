-- query 13: nope(1)
SELECT unix_millis(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(unix_millis)';
SELECT legacy_fn_28(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';
SELECT legacy_fn_37(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_37)';
SELECT tuple_difference_integer(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_difference_integer)';
SELECT shiftrightunsigned(a, b) AS c5 FROM t13 WHERE x IN (1, 2) AND y = 'x(shiftrightunsigned)';
