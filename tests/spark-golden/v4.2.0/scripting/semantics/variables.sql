-- Variables, design note section 2.2.
-- CASE declare_default_inferred
BEGIN DECLARE x DEFAULT 5L; SELECT x, typeof(x) AS t; END;
-- CASE declare_no_default_is_null
BEGIN DECLARE x INT; SELECT x; END;
-- CASE declare_multi
BEGIN DECLARE a, b INT DEFAULT 4; SELECT a, b; END;
-- CASE declare_variable_keyword_is_local
BEGIN DECLARE VARIABLE q INT DEFAULT 3; SELECT q; END;
-- next
SELECT q;
-- CASE declare_default_reads_earlier_local
BEGIN DECLARE a INT DEFAULT 2; DECLARE b INT DEFAULT a * 10; SELECT b; END;
-- CASE declare_default_reads_outer_same_name
BEGIN DECLARE x INT DEFAULT 1; BEGIN DECLARE x INT DEFAULT x + 1; SELECT x; END; END;
-- CASE declare_default_cast_string
BEGIN DECLARE a INT DEFAULT '12'; SELECT a; END;
-- CASE declare_default_cast_invalid
BEGIN DECLARE a INT DEFAULT 'abc'; SELECT a; END;
-- CASE set_cast_invalid
BEGIN DECLARE a INT; SET a = 'q'; SELECT a; END;
-- CASE declare_overflow
BEGIN DECLARE a TINYINT DEFAULT 300; SELECT a; END;
-- CASE declare_decimal_rounding
BEGIN DECLARE d DECIMAL(5,2) DEFAULT 1.005; SELECT d; END;
-- CASE declare_null_default
BEGIN DECLARE a INT DEFAULT NULL; SELECT a IS NULL AS n; END;
-- CASE scope_shadow_and_gone
BEGIN DECLARE x INT DEFAULT 1; BEGIN DECLARE x INT DEFAULT 2; SET x = 3; END; SELECT x; END;
-- CASE qualified_by_label
BEGIN lb: BEGIN DECLARE x INT DEFAULT 1; BEGIN DECLARE x INT DEFAULT 2; SELECT x, lb.x AS outer_x; END; END lb; END;
-- CASE set_qualified_by_label
BEGIN lb: BEGIN DECLARE x INT DEFAULT 1; BEGIN DECLARE x INT DEFAULT 2; SET lb.x = 5; SELECT x, lb.x AS outer_x; END; END lb; END;
-- CASE case_insensitive
BEGIN Lbl: BEGIN DECLARE Abc INT DEFAULT 4; SELECT aBC, LBL.abc AS q; END lbl; END;
-- CASE local_first_then_session
DECLARE sv INT DEFAULT 7;
-- next
BEGIN DECLARE sv INT DEFAULT 1; SET sv = 8; SELECT sv, session.sv AS s1, system.session.sv AS s2; END;
-- next
SELECT sv;
-- CASE session_variable_set_from_script
DECLARE v INT DEFAULT 1;
-- next
BEGIN DECLARE k INT DEFAULT 42; SET session.v = k; END;
-- next
SELECT v;
-- CASE session_variable_read_unqualified
DECLARE v2 INT DEFAULT 11;
-- next
BEGIN SET v2 = v2 + 1; SELECT v2; END;
-- next
SELECT v2;
-- CASE set_multiple
BEGIN DECLARE a INT; DECLARE b INT; SET a = 1, b = 2; SELECT a, b; END;
-- CASE set_tuple_subquery
BEGIN DECLARE a INT; DECLARE b STRING; SET (a, b) = (SELECT 1, 'z'); SELECT a, b; END;
-- CASE set_default
BEGIN DECLARE x INT DEFAULT 7; SET x = 9; SET x = DEFAULT; SELECT x; END;
-- CASE set_rhs_evaluated_before_assign
BEGIN DECLARE a INT DEFAULT 1; DECLARE b INT DEFAULT 2; SET a = b, b = a; SELECT a, b; END;
-- CASE set_config_is_assignment
BEGIN SET spark.sql.ansi.enabled = false; END;
-- CASE set_unknown
BEGIN SET nope = 1; END;
-- CASE set_duplicate_targets
BEGIN DECLARE a INT; SET a = 1, a = 2; END;
-- CASE set_too_many_rows
BEGIN DECLARE a INT; SET a = (SELECT id FROM range(3)); END;
-- CASE set_zero_rows
BEGIN DECLARE a INT DEFAULT 5; SET a = (SELECT id FROM range(3) WHERE id > 9); SELECT a; END;
-- CASE variable_already_exists
BEGIN DECLARE x INT; DECLARE x INT; END;
-- CASE declare_subquery_default
BEGIN DECLARE x INT DEFAULT (SELECT 1); END;
-- CASE declare_no_type_no_default
BEGIN DECLARE x; END;
-- CASE drop_temporary_variable
DECLARE tv INT DEFAULT 1;
-- next
BEGIN DROP TEMPORARY VARIABLE tv; END;
-- CASE unaliased_column_name_over_local
-- KRISHIV-EXPECT: column name `typeof(x)` (named from the expression as written; Spark names it with a random internal scope id)
BEGIN DECLARE x INT DEFAULT 12; SELECT typeof(x); END;
-- CASE unaliased_column_name_over_local_plain
BEGIN DECLARE x INT DEFAULT 12; SELECT x; END;
-- CASE local_in_insert_values
BEGIN DECLARE k INT DEFAULT 31; INSERT INTO s VALUES (k); END;
-- next
SELECT * FROM s;
-- CASE local_in_ctas
BEGIN DECLARE k INT DEFAULT 32; CREATE TABLE t2 AS SELECT k AS i; END;
-- next
SELECT * FROM t2;
-- CASE local_in_explain
BEGIN DECLARE k INT DEFAULT 33; EXPLAIN SELECT k; END;
-- CASE local_in_execute_immediate_using
BEGIN DECLARE k INT DEFAULT 5; EXECUTE IMMEDIATE 'SELECT ? + 1 AS r' USING k; END;
-- CASE local_in_execute_immediate_into
BEGIN DECLARE k INT; EXECUTE IMMEDIATE 'SELECT 41 + 1' INTO k; SELECT k; END;
-- CASE local_not_in_execute_immediate_text
BEGIN DECLARE k INT DEFAULT 5; EXECUTE IMMEDIATE 'SELECT k + 1'; END;
-- CASE temp_function_over_local
BEGIN DECLARE k INT DEFAULT 7; CREATE TEMPORARY FUNCTION f() RETURNS INT RETURN k; SELECT f() AS r; END;
-- next
SELECT f() AS r;
-- CASE temp_view_over_local
BEGIN DECLARE k INT DEFAULT 7; CREATE TEMPORARY VIEW v AS SELECT k AS c; END;
-- next
SELECT * FROM v;
-- CASE temp_view_persists_after_script
BEGIN CREATE TEMPORARY VIEW v2 AS SELECT 1 AS c; END;
-- next
SELECT * FROM v2;
-- CASE local_in_where_over_table
INSERT INTO s VALUES (1), (2), (3);
-- next
BEGIN DECLARE k INT DEFAULT 2; SELECT i FROM s WHERE i >= k ORDER BY i; END;
-- CASE local_in_cache_table
BEGIN DECLARE k INT DEFAULT 3; CACHE TABLE ct AS SELECT k AS c; END;
-- next
SELECT * FROM ct;
-- CASE set_session_from_local_ddl_after
DECLARE w STRING DEFAULT 'a';
-- next
BEGIN DECLARE k STRING DEFAULT 'z'; SET session.w = k; END;
-- next
SELECT w;
