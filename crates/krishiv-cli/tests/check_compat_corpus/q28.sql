-- query 28: nope(1)
SELECT legacy_fn_34(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(legacy_fn_34)';
SELECT char_length(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(char_length)';
SELECT csc(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(csc)';
SELECT regexp(a, b) AS c4 FROM t28 WHERE x IN (1, 2) AND y = 'x(regexp)';
SELECT mean(a, b) AS c5 FROM t28 WHERE x IN (1, 2) AND y = 'x(mean)';
