-- sequence with a DST session time zone (America/Los_Angeles: 2018-03-11 spring forward 02:00->03:00, 2018-11-04 fall back 02:00->01:00)
SET spark.sql.session.timeZone=America/Los_Angeles;
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval 1 day);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval '1' day);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval 24 hours);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval '24' hour);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval '1 0:0:0' day to second);
SELECT sequence(timestamp '2018-03-10 12:00:00', timestamp '2018-03-12 12:00:00', interval 1 day);
SELECT sequence(timestamp '2018-03-10 12:00:00', timestamp '2018-03-12 12:00:00', interval 24 hours);
SELECT sequence(timestamp '2018-03-10 12:00:00', timestamp '2018-03-12 12:00:00', interval '1' day);
SELECT sequence(timestamp '2018-03-10 12:00:00', timestamp '2018-03-12 12:00:00', interval '24' hour);
SELECT sequence(timestamp '2018-03-11 00:00:00', timestamp '2018-03-11 05:00:00', interval 1 hour);
SELECT sequence(timestamp '2018-03-11 00:00:00', timestamp '2018-03-11 05:00:00', interval '1' hour);
SELECT sequence(timestamp '2018-03-10 02:30:00', timestamp '2018-03-13 02:30:00', interval 1 day);
SELECT sequence(timestamp '2018-03-10 02:30:00', timestamp '2018-03-13 02:30:00', interval '1' day);
SELECT sequence(timestamp '2018-03-10 02:30:00', timestamp '2018-03-13 02:30:00', interval 24 hours);
SELECT sequence(timestamp '2018-03-11 02:30:00', timestamp '2018-03-12 02:30:00', interval 1 day);
SELECT sequence(timestamp '2018-02-11 02:30:00', timestamp '2018-04-11 02:30:00', interval 1 month);
SELECT sequence(timestamp '2018-02-11 02:30:00', timestamp '2018-04-11 02:30:00', interval '1' month);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval 1 day 1 hour);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval 1 month 1 day);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00');
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', 86400);
SELECT sequence(timestamp '2018-03-13 00:00:00', timestamp '2018-03-10 00:00:00', interval -1 day);
SELECT sequence(timestamp '2018-03-13 00:00:00', timestamp '2018-03-10 00:00:00', interval '-1' day);
SELECT sequence(timestamp '2018-03-13 00:00:00', timestamp '2018-03-10 00:00:00', interval -24 hours);
SELECT sequence(timestamp '2018-03-12 02:30:00', timestamp '2018-03-10 02:30:00', interval -1 day);
-- fall back
SELECT sequence(timestamp '2018-11-03 00:00:00', timestamp '2018-11-06 00:00:00', interval 1 day);
SELECT sequence(timestamp '2018-11-03 00:00:00', timestamp '2018-11-06 00:00:00', interval '1' day);
SELECT sequence(timestamp '2018-11-03 00:00:00', timestamp '2018-11-06 00:00:00', interval 24 hours);
SELECT sequence(timestamp '2018-11-03 00:00:00', timestamp '2018-11-06 00:00:00', interval '24' hour);
SELECT sequence(timestamp '2018-11-03 12:00:00', timestamp '2018-11-05 12:00:00', interval 1 day);
SELECT sequence(timestamp '2018-11-03 12:00:00', timestamp '2018-11-05 12:00:00', interval 24 hours);
SELECT sequence(timestamp '2018-11-04 00:00:00', timestamp '2018-11-04 04:00:00', interval 1 hour);
SELECT sequence(timestamp '2018-11-04 01:30:00', timestamp '2018-11-05 01:30:00', interval 1 day);
SELECT sequence(timestamp '2018-10-04 01:30:00', timestamp '2018-12-04 01:30:00', interval 1 month);
-- default step with a DST zone, timestamps from instants
SELECT sequence(to_timestamp('2018-03-11 09:00:00Z'), to_timestamp('2018-03-11 12:00:00Z'), interval 1 hour);
SELECT sequence(to_timestamp('2018-03-10 08:00:00Z'), to_timestamp('2018-03-12 08:00:00Z'), interval 1 day);
-- DATE steps in DST zone
SELECT sequence(to_date('2018-03-10'), to_date('2018-03-13'), interval 1 day);
SELECT sequence(to_date('2018-03-10'), to_date('2018-03-13'), interval 24 hours);
SELECT sequence(to_date('2018-03-10'), to_date('2018-03-13'));
SELECT sequence(to_date('2018-03-10'), to_date('2018-03-13'), interval 1 month);
-- TIMESTAMP_NTZ is zone independent
SELECT sequence(timestamp_ntz '2018-03-10 00:00:00', timestamp_ntz '2018-03-13 00:00:00', interval 1 day);
SELECT sequence(timestamp_ntz '2018-03-10 00:00:00', timestamp_ntz '2018-03-13 00:00:00', interval 24 hours);
SELECT sequence(timestamp_ntz '2018-03-11 00:00:00', timestamp_ntz '2018-03-11 05:00:00', interval 1 hour);
SELECT sequence(timestamp_ntz '2018-11-04 00:00:00', timestamp_ntz '2018-11-04 04:00:00', interval 1 hour);
SELECT sequence(timestamp_ntz '2018-03-10 02:30:00', timestamp_ntz '2018-03-13 02:30:00', interval 1 day);
-- another zone with a half-hour offset and Southern Hemisphere DST
SET spark.sql.session.timeZone=Australia/Lord_Howe;
SELECT sequence(timestamp '2018-04-01 00:00:00', timestamp '2018-04-02 00:00:00', interval 6 hours);
SELECT sequence(timestamp '2018-10-06 00:00:00', timestamp '2018-10-09 00:00:00', interval 1 day);
SELECT sequence(timestamp '2018-10-06 00:00:00', timestamp '2018-10-09 00:00:00', interval 24 hours);
SET spark.sql.session.timeZone=Asia/Kolkata;
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-01-01 03:00:00', interval 1 hour);
SELECT sequence(timestamp '2018-01-01 00:00:00', timestamp '2018-03-01 00:00:00', interval 1 month);
-- ANSI off in a DST zone
SET spark.sql.session.timeZone=America/Los_Angeles;
SET spark.sql.ansi.enabled=false;
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval 1 day);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval 24 hours);
SELECT sequence(timestamp '2018-03-10 00:00:00', timestamp '2018-03-13 00:00:00', interval '1' day);
SET spark.sql.ansi.enabled=true;
SET spark.sql.session.timeZone=UTC;
