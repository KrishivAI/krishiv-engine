-- Entry, last-result rule, design note section 2.1. Each case: the script, then (if it is shown) a probe.
-- CASE empty_script
BEGIN END;
-- CASE begin_not_atomic
BEGIN NOT ATOMIC SELECT 1 AS a; END;
-- CASE top_level_label
lbl: BEGIN SELECT 1; END lbl;
-- CASE explain_script
EXPLAIN BEGIN SELECT 1; END;
-- CASE execute_immediate_script
EXECUTE IMMEDIATE 'BEGIN SELECT 1; END';
-- CASE scripting_disabled
SET spark.sql.scripting.enabled = false;
-- next
BEGIN SELECT 1; END;
-- CASE last_result_wins
BEGIN SELECT 1 AS a; SELECT 2 AS b; END;
-- CASE set_is_not_result
BEGIN DECLARE x INT DEFAULT 1; SELECT 1 AS a; SET x = 2; END;
-- CASE insert_is_not_result
BEGIN SELECT 1 AS a; INSERT INTO s VALUES (5); END;
-- next
SELECT * FROM s;
-- CASE show_is_not_result
BEGIN SELECT 1 AS a; SHOW TABLES; END;
-- CASE describe_is_not_result
BEGIN SELECT 1 AS a; DESCRIBE TABLE s; END;
-- CASE explain_alone
BEGIN EXPLAIN SELECT 1; END;
-- CASE no_result_statement
BEGIN DECLARE x INT DEFAULT 1; SET x = 2; END;
-- CASE loop_select_last_row
BEGIN DECLARE i INT DEFAULT 0; WHILE i < 3 DO SET i = i + 1; SELECT i AS v; END WHILE; END;
-- CASE values_is_result
BEGIN VALUES (1, 'a'), (2, 'b'); END;
-- CASE eager_insert_visible
BEGIN INSERT INTO s VALUES (7); END;
-- next
SELECT * FROM s;
-- CASE error_after_result
BEGIN SELECT 1 AS a; SELECT 1/0; END;
-- CASE error_after_insert_keeps_insert
BEGIN INSERT INTO s VALUES (9); SELECT 1/0; END;
-- next
SELECT * FROM s;
-- CASE ansi_off_divide
SET spark.sql.ansi.enabled = false;
-- next
BEGIN SELECT 1/0 AS q; END;
-- CASE ansi_on_divide_handler
BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 'caught' AS c; SELECT 1/0; END;
-- CASE ansi_off_divide_handler_not_run
SET spark.sql.ansi.enabled = false;
-- next
BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 'caught' AS c; SELECT 1/0 AS q; END;
-- CASE unresolved_variable_keeps_earlier_insert
BEGIN INSERT INTO s VALUES (1); SET nope = 1; END;
-- next
SELECT count(*) AS n FROM s;
-- CASE unresolved_column_keeps_earlier_insert
BEGIN INSERT INTO s VALUES (1); SELECT nope; END;
-- next
SELECT count(*) AS n FROM s;
