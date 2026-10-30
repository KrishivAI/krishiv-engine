-- Structural errors and their phases: each case INSERTs before the construct; n = 1 means the script ran that far.
-- CASE labels_mismatch
BEGIN INSERT INTO s VALUES (1); a: BEGIN SELECT 1; END b; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE end_label_without_begin_label
BEGIN INSERT INTO s VALUES (1); BEGIN SELECT 1; END b; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE label_reuses_enclosing
BEGIN INSERT INTO s VALUES (1); a: BEGIN a: BEGIN SELECT 1; END a; END a; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE for_variable_reuses_label
BEGIN INSERT INTO s VALUES (1); a: BEGIN FOR a AS SELECT 1 AS x DO SELECT x; END FOR; END a; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE label_session_forbidden
BEGIN INSERT INTO s VALUES (1); session: BEGIN SELECT 1; END session; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE label_system_forbidden
BEGIN INSERT INTO s VALUES (1); system: BEGIN SELECT 1; END system; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE for_variable_session_forbidden
BEGIN INSERT INTO s VALUES (1); FOR session AS SELECT 1 AS x DO SELECT x; END FOR; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE qualified_label
BEGIN INSERT INTO s VALUES (1); a.b: BEGIN SELECT 1; END a.b; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE leave_unknown_label
BEGIN INSERT INTO s VALUES (1); LEAVE nope; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE iterate_unknown_label
BEGIN INSERT INTO s VALUES (1); ITERATE nope; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE iterate_compound_label
BEGIN INSERT INTO s VALUES (1); c: BEGIN ITERATE c; END c; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE declare_after_statement
BEGIN INSERT INTO s VALUES (1); BEGIN SELECT 1; DECLARE x INT; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE declare_after_handler
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 1; DECLARE x INT; SELECT 2; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE declare_in_if_body
BEGIN INSERT INTO s VALUES (1); IF true THEN DECLARE x INT; SELECT 1; END IF; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE declare_in_while_body
BEGIN INSERT INTO s VALUES (1); WHILE false DO DECLARE x INT; END WHILE; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE declare_qualified_local
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE a.x INT; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE declare_or_replace_local
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE OR REPLACE x INT; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_after_statement
BEGIN INSERT INTO s VALUES (1); BEGIN SELECT 1; DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 2; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_in_if_body
-- KRISHIV-EXPECT: error INVALID_HANDLER_DECLARATION.WRONG_PLACE_OF_DECLARATION (before execution; Spark 4.2 crashes with a Java NoSuchElementException)
BEGIN INSERT INTO s VALUES (1); IF true THEN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 2; SELECT 1; END IF; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_unknown_condition
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR no_such_cond SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_sqlexception_combined
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION, DIVIDE_BY_ZERO SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_notfound_combined
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR NOT FOUND, SQLSTATE '22012' SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_sqlexception_notfound
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION, NOT FOUND SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_duplicate_condition
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR DIVIDE_BY_ZERO, DIVIDE_BY_ZERO SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_duplicate_sqlstate
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLSTATE '22012', SQLSTATE '22012' SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE two_handlers_same_condition
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR DIVIDE_BY_ZERO SELECT 2; DECLARE EXIT HANDLER FOR DIVIDE_BY_ZERO SELECT 3; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE two_handlers_same_sqlstate
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLSTATE '22012' SELECT 2; DECLARE EXIT HANDLER FOR SQLSTATE '22012' SELECT 3; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE two_handlers_sqlexception
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 2; DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 3; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE condition_declared_twice
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE c CONDITION; DECLARE c CONDITION; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE condition_after_handler
-- KRISHIV-EXPECT: error INVALID_ERROR_CONDITION_DECLARATION.NOT_AT_START_OF_COMPOUND_STATEMENT (before execution; Spark 4.2 crashes with a Java NoSuchElementException)
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 2; DECLARE c CONDITION; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE condition_qualified_name
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE a.c CONDITION; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE sqlstate_class_00
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE c CONDITION FOR SQLSTATE '00001'; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE sqlstate_class_01
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLSTATE '01000' SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE sqlstate_class_xx
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE c CONDITION FOR SQLSTATE 'XX000'; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE sqlstate_too_short
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE c CONDITION FOR SQLSTATE '1234'; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE sqlstate_lowercase
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE c CONDITION FOR SQLSTATE 'abc12'; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE sqlstate_valid_user
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE c CONDITION FOR SQLSTATE '45000'; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE continue_handler
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE CONTINUE HANDLER FOR SQLEXCEPTION SELECT 2; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE cursor_declare
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE cur CURSOR FOR SELECT 1; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE cursor_open
BEGIN INSERT INTO s VALUES (1); BEGIN SELECT 1; OPEN cur; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE signal
BEGIN INSERT INTO s VALUES (1); BEGIN SIGNAL SQLSTATE '45000'; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE resignal
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION RESIGNAL; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE get_diagnostics
BEGIN INSERT INTO s VALUES (1); BEGIN GET DIAGNOSTICS x = ROW_COUNT; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE select_into
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE x INT; SELECT 1 INTO x; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE handler_body_if
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION IF true THEN SELECT 2; END IF; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE variable_already_exists
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE x INT; DECLARE x INT; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE variable_already_exists_case
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE x INT; DECLARE X INT; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE variable_already_exists_one_declare
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE a, A INT; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE duplicate_assignments
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE a INT DEFAULT 0; SET a = 1, a = 2; SELECT a; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE default_subquery
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE x INT DEFAULT (SELECT 1); SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE declare_no_type_no_default
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE x; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE drop_temporary_variable
BEGIN INSERT INTO s VALUES (1); BEGIN SELECT 1; DROP TEMPORARY VARIABLE v; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE set_config
BEGIN INSERT INTO s VALUES (1); BEGIN SET spark.sql.ansi.enabled = false; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE set_bare
BEGIN INSERT INTO s VALUES (1); BEGIN SET; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE set_key
BEGIN INSERT INTO s VALUES (1); BEGIN SET spark.sql.ansi.enabled; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE set_time_zone
BEGIN INSERT INTO s VALUES (1); BEGIN SET TIME ZONE 'UTC'; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE reset
BEGIN INSERT INTO s VALUES (1); BEGIN RESET; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE set_variable_keyword
BEGIN INSERT INTO s VALUES (1); BEGIN DECLARE x INT; SET VARIABLE x = 1; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE unresolved_variable_runtime
BEGIN INSERT INTO s VALUES (1); BEGIN SET nope = 1; SELECT 1; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE unresolved_column_runtime
BEGIN INSERT INTO s VALUES (1); BEGIN SELECT nope; END; END;
-- next
SELECT count(*) AS n FROM s;

-- CASE explain_script_nested
BEGIN INSERT INTO s VALUES (1); BEGIN EXPLAIN BEGIN SELECT 1; END; END; END;
-- next
SELECT count(*) AS n FROM s;
