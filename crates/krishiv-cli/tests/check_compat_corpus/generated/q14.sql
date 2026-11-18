-- query 14: nope(1)
SELECT range(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(range)';
SELECT legacy_fn_29(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT from_protobuf(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(from_protobuf)';
SELECT getbit(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(getbit)';
SELECT day(a, b) AS c5 FROM t14 WHERE x IN (1, 2) AND y = 'x(day)';
SELECT substring(a, b) AS c6 FROM t14 WHERE x IN (1, 2) AND y = 'x(substring)';
