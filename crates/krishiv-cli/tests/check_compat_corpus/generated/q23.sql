-- query 23: nope(1)
SELECT encode(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(encode)';
SELECT isnotnull(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(isnotnull)';
SELECT from_utc_timestamp(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(from_utc_timestamp)';
