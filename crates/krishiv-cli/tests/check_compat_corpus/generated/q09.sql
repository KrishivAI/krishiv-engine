-- query 9: nope(1)
SELECT array_compact(a, b) AS c1 FROM t9 WHERE x IN (1, 2) AND y = 'x(array_compact)';
SELECT sentences(a, b) AS c2 FROM t9 WHERE x IN (1, 2) AND y = 'x(sentences)';
SELECT ceil(a, b) AS c3 FROM t9 WHERE x IN (1, 2) AND y = 'x(ceil)';
SELECT timestamp_seconds(a, b) AS c4 FROM t9 WHERE x IN (1, 2) AND y = 'x(timestamp_seconds)';
SELECT array_except(a, b) AS c5 FROM t9 WHERE x IN (1, 2) AND y = 'x(array_except)';
