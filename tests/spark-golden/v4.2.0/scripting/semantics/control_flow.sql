-- Control flow, design note section 2.3.
-- CASE if_elseif_else
BEGIN DECLARE x INT DEFAULT 2; IF x = 1 THEN SELECT 'one'; ELSEIF x = 2 THEN SELECT 'two'; ELSE SELECT 'other'; END IF; END;
-- CASE if_null_condition_is_false
BEGIN DECLARE b BOOLEAN; IF b THEN SELECT 'then'; ELSE SELECT 'else'; END IF; END;
-- CASE if_non_boolean
BEGIN IF 1 THEN SELECT 'then'; END IF; END;
-- CASE if_void_condition
BEGIN IF NULL THEN SELECT 'then'; END IF; END;
-- CASE if_condition_subquery_too_many_rows
BEGIN IF (SELECT id > 0 FROM range(2)) THEN SELECT 'then'; END IF; END;
-- CASE if_condition_string
BEGIN IF 'true' THEN SELECT 'then'; END IF; END;
-- CASE simple_case
BEGIN DECLARE x INT DEFAULT 2; CASE x WHEN 1 THEN SELECT 'one'; WHEN 2 THEN SELECT 'two'; ELSE SELECT 'other'; END CASE; END;
-- CASE simple_case_null_takes_else
BEGIN CASE NULL WHEN NULL THEN SELECT 'match'; ELSE SELECT 'else'; END CASE; END;
-- CASE simple_case_no_match_no_else
BEGIN DECLARE x INT DEFAULT 9; INSERT INTO s VALUES (1); CASE x WHEN 1 THEN SELECT 'one'; END CASE; END;
-- next
SELECT count(*) AS n FROM s;
-- CASE searched_case
BEGIN DECLARE x INT DEFAULT 5; CASE WHEN x < 3 THEN SELECT 'small'; WHEN x < 10 THEN SELECT 'medium'; END CASE; END;
-- CASE searched_case_no_match_no_else
BEGIN CASE WHEN false THEN SELECT 'never'; END CASE; END;
-- CASE while_loop
BEGIN DECLARE i INT DEFAULT 0; WHILE i < 4 DO SET i = i + 1; END WHILE; SELECT i; END;
-- CASE repeat_runs_body_first
BEGIN DECLARE i INT DEFAULT 10; REPEAT SET i = i + 1; UNTIL i > 0 END REPEAT; SELECT i; END;
-- CASE loop_leave
BEGIN DECLARE i INT DEFAULT 0; lp: LOOP SET i = i + 1; IF i >= 3 THEN LEAVE lp; END IF; END LOOP lp; SELECT i; END;
-- CASE leave_compound
BEGIN DECLARE i INT DEFAULT 0; c: BEGIN SET i = 1; LEAVE c; SET i = 2; END c; SELECT i; END;
-- CASE iterate_loop
BEGIN DECLARE i INT DEFAULT 0; DECLARE s INT DEFAULT 0; lp: WHILE i < 5 DO SET i = i + 1; IF i = 3 THEN ITERATE lp; END IF; SET s = s + i; END WHILE lp; SELECT s; END;
-- CASE iterate_outer_from_inner_unlabelled
BEGIN DECLARE i INT DEFAULT 0; DECLARE j INT DEFAULT 0; DECLARE n INT DEFAULT 0; outer_l: WHILE i < 3 DO SET i = i + 1; WHILE j < 10 DO SET j = j + 1; SET n = n + 1; IF j % 2 = 0 THEN ITERATE outer_l; END IF; END WHILE; END WHILE outer_l; SELECT i, j, n; END;
-- CASE leave_nested_loops
BEGIN DECLARE i INT DEFAULT 0; o: LOOP SET i = i + 1; LOOP SET i = i + 10; LEAVE o; END LOOP; END LOOP o; SELECT i; END;
-- CASE for_basic
BEGIN DECLARE s INT DEFAULT 0; FOR r AS SELECT id FROM range(4) DO SET s = s + r.id; END FOR; SELECT s; END;
-- CASE for_bare_column
BEGIN DECLARE s INT DEFAULT 0; FOR r AS SELECT id FROM range(4) DO SET s = s + id; END FOR; SELECT s; END;
-- CASE for_without_variable
BEGIN DECLARE s INT DEFAULT 0; FOR SELECT id FROM range(4) DO SET s = s + id; END FOR; SELECT s; END;
-- CASE for_zero_rows
BEGIN DECLARE s INT DEFAULT 0; FOR r AS SELECT id FROM range(4) WHERE id > 10 DO SET s = 1; END FOR; SELECT s; END;
-- CASE for_snapshot_semantics
INSERT INTO s VALUES (1), (2);
-- next
BEGIN FOR r AS SELECT i FROM s DO INSERT INTO s VALUES (r.i + 10); END FOR; END;
-- next
SELECT i FROM s ORDER BY i;
-- CASE for_assign_loop_variable
BEGIN DECLARE out INT DEFAULT 0; FOR r AS SELECT id FROM range(1) DO SET id = 10; SET out = r.id; END FOR; SELECT out; END;
-- CASE for_variable_gone_after
BEGIN FOR r AS SELECT 1 AS id DO SELECT 1; END FOR; SELECT id; END;
-- CASE for_duplicate_column_names
BEGIN FOR r AS SELECT 1 AS a, 2 AS a DO SELECT 1; END FOR; END;
-- CASE for_unnamed_expression_column
BEGIN FOR r AS SELECT 1 + 1 DO SELECT 1; END FOR; END;
-- CASE for_local_same_name_as_loop_variable
BEGIN DECLARE r INT DEFAULT 5; FOR r AS SELECT id FROM range(1) DO SELECT r.id AS q; END FOR; SELECT r; END;
-- CASE for_with_label_and_leave
BEGIN DECLARE s INT DEFAULT 0; lp: FOR r AS SELECT id FROM range(10) DO IF r.id = 3 THEN LEAVE lp; END IF; SET s = s + 1; END FOR lp; SELECT s; END;
-- CASE for_iterate
BEGIN DECLARE s INT DEFAULT 0; lp: FOR r AS SELECT id FROM range(5) DO IF r.id = 2 THEN ITERATE lp; END IF; SET s = s + r.id; END FOR lp; SELECT s; END;
-- CASE for_column_types
BEGIN FOR r AS SELECT 1 AS a, 'x' AS b, 1.5D AS c, CAST(NULL AS INT) AS d DO SELECT typeof(r.a) AS ta, typeof(r.b) AS tb, typeof(r.c) AS tc, r.d IS NULL AS dn; END FOR; END;
-- CASE for_query_error_handler
BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 'caught' AS c; FOR r AS SELECT nope FROM range(2) DO SELECT 1; END FOR; END;
-- CASE condition_in_while_false_null
BEGIN DECLARE b BOOLEAN; WHILE b DO SELECT 'never'; END WHILE; SELECT 'done'; END;
-- CASE condition_two_columns
BEGIN IF (SELECT 1, 2) THEN SELECT 1; END IF; END;
-- CASE nested_compound_labels_case_insensitive
BEGIN A: BEGIN SELECT 1 AS a; B: BEGIN SELECT 2 AS b; END b; END a; END;
