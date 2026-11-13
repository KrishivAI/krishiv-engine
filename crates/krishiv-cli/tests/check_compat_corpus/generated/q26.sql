-- query 26: nope(1)
SELECT legacy_fn_37(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_37)';
SELECT tuple_union_theta_integer(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_integer)';
SELECT shiftright(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(shiftright)';
SELECT bitmap_count(a, b) AS c4 FROM t26 WHERE x IN (1, 2) AND y = 'x(bitmap_count)';
SELECT substring_index(a, b) AS c5 FROM t26 WHERE x IN (1, 2) AND y = 'x(substring_index)';
