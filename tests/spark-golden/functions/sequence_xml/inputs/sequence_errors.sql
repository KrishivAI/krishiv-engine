-- sequence: errors, maximum array length, conditional evaluation, columns
SET spark.sql.session.timeZone=UTC;
SELECT sequence(to_date('2018-01-01'), to_date('2018-03-01'), interval 0 month);
SELECT sequence(to_date('2018-01-01'), to_date('2018-03-01'), interval '0' month);
SELECT sequence(to_date('2018-01-01'), to_date('2018-01-05'), interval '0' day);
SELECT sequence(to_date('2018-01-01'), to_date('2018-03-01'), interval -1 month);
SELECT sequence(to_date('2018-03-01'), to_date('2018-01-01'), interval 1 month);
SELECT sequence(to_date('2018-01-01'), to_date('2018-01-05'), interval -1 day);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-01-02 00:00:00', interval -1 hour);
SELECT sequence(timestamp '2018-01-02 00:00:00', timestamp '2018-01-01 00:00:00', interval 1 hour);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-01-02 00:00:00', interval 0 hour);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-01-02 00:00:00', interval 0 month 0 day);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-01-02 00:00:00', interval 1 month -2 days);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-03-02 00:00:00', interval 1 month -2 days);
SELECT sequence(timestamp '2018-03-02 00:00:00', timestamp '2018-01-01 00:00:00', interval -1 month 2 days);
SELECT sequence(to_date('2018-01-01'), to_date('2018-03-02'), interval 1 month -2 days);
SELECT sequence(to_date('2018-01-01'), to_date('2018-03-02'), interval 1 month 0 days);
SELECT sequence(to_date('2018-03-02'), to_date('2018-01-01'), interval -1 month -2 days);
-- array length limit (Spark: MAX_ROUNDED_ARRAY_LENGTH = 2147483632)
SELECT size(sequence(to_date('2018-01-01'), to_date('2018-01-05'), interval 1 day));
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-12-31 00:00:00', interval 1 microsecond);
SELECT size(sequence(to_date('0001-01-01'), to_date('9999-12-31'), interval 1 day));
SELECT sequence(timestamp '0001-01-01 00:00:00', timestamp '9999-12-31 00:00:00', interval 1 microsecond);
SELECT sequence(timestamp '0001-01-01 00:00:00', timestamp '9999-12-31 00:00:00', interval 1 millisecond);
SELECT size(sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-01-01 00:00:05', interval 1 microsecond));
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-02-01 00:00:00', interval 1 microsecond);
SELECT sequence(timestamp_ntz '2018-01-01 00:00:00', timestamp_ntz '2018-12-31 00:00:00', interval 1 microsecond);
SELECT sequence(1, 2147483647);
SELECT sequence(1L, 2147483648L);
SELECT sequence(1, 2147483633);
SELECT sequence(-2147483648, 2147483647);
SELECT sequence(-2147483648, 2147483647, 2147483647);
SELECT sequence(to_date('2018-01-01'), to_date('2018-01-05'), interval 106751991 days);
SELECT sequence(to_date('2018-01-01'), to_date('2018-01-05'), interval 9999999999 days);
-- column inputs and conditional evaluation
CREATE OR REPLACE TEMPORARY VIEW sq AS SELECT * FROM VALUES
  (1, to_date('2018-01-01'), to_date('2018-03-01'), interval 1 month),
  (2, to_date('2018-01-01'), to_date('2018-03-01'), interval 0 month),
  (3, to_date('2018-03-01'), to_date('2018-01-01'), interval 1 month),
  (4, to_date('2018-01-01'), NULL, interval 1 month),
  (5, to_date('2018-01-31'), to_date('2018-05-31'), interval 1 month) AS t(id, a, b, s);
SELECT id, sequence(a, b, s) FROM sq WHERE id IN (1, 4, 5) ORDER BY id;
SELECT id, sequence(a, b, s) FROM sq ORDER BY id;
SELECT id, CASE WHEN id = 1 THEN sequence(a, b, s) ELSE array() END FROM sq ORDER BY id;
SELECT id, IF(id = 1, sequence(a, b, s), NULL) FROM sq ORDER BY id;
SELECT id, COALESCE(sequence(a, b, s), array(a)) FROM sq WHERE id = 1 OR id = 4 ORDER BY id;
SELECT id, CASE WHEN s > interval 0 month THEN sequence(a, b, s) END FROM sq WHERE id <> 3 ORDER BY id;
SELECT id, CASE WHEN a <= b THEN sequence(a, b, s) END FROM sq WHERE id <> 2 ORDER BY id;
SELECT id FROM sq WHERE id = 1 AND size(sequence(a, b, s)) = 3;
SELECT id FROM sq WHERE id <> 2 AND size(sequence(a, b, s)) = 3 ORDER BY id;
SELECT id, size(sequence(a, b, s)) FROM sq WHERE id IN (1, 5) ORDER BY id;
CREATE OR REPLACE TEMPORARY VIEW sqt AS SELECT * FROM VALUES
  (1, timestamp '2018-01-01 00:00:00', timestamp '2018-01-01 03:00:00', interval 1 hour),
  (2, timestamp '2018-01-01 00:00:00', timestamp '2018-01-01 03:00:00', interval 0 hour),
  (3, timestamp '2018-01-01 03:00:00', timestamp '2018-01-01 00:00:00', interval 1 hour) AS t(id, a, b, s);
SELECT id, sequence(a, b, s) FROM sqt WHERE id = 1;
SELECT id, sequence(a, b, s) FROM sqt;
SELECT id, CASE WHEN id = 1 THEN sequence(a, b, s) END FROM sqt ORDER BY id;
-- ANSI off: same errors
SET spark.sql.ansi.enabled=false;
SELECT sequence(to_date('2018-01-01'), to_date('2018-03-01'), interval 0 month);
SELECT sequence(to_date('2018-03-01'), to_date('2018-01-01'), interval 1 month);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-01-02 00:00:00', interval 0 hour);
SELECT sequence(timestamp '2018-01-02 00:00:00', timestamp '2018-01-01 00:00:00', interval 1 hour);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-12-31 00:00:00', interval 1 microsecond);
SELECT sequence(1, 2147483647);
SELECT id, sequence(a, b, s) FROM sq ORDER BY id;
SET spark.sql.ansi.enabled=true;
-- DataFrame-style type errors
SELECT sequence(to_date('2018-01-01'), to_date('2018-01-05'), interval 1 day, interval 1 day);
SELECT sequence(true, false);
SELECT sequence(1.5, 3.5);
SELECT sequence('a', 'b', interval 1 day);
SELECT sequence(to_date('2018-01-01'), to_date('2018-01-05'), interval 1 day, 1, 2);
SELECT sequence();
