# T196 unit 0: SQL scripting phases on Spark 4.2.0

Run: `apache/spark:4.2.0` (`local[2]`, in-memory catalog, ANSI on, `spark.sql.session.timeZone=UTC`), 2026-10-30,
by `tests/scripts/gen_scripting_semantics.py` over `tests/spark-golden/v4.2.0/scripting/semantics/structural.sql`
(full outputs: `structural.json` beside it). Every case runs `BEGIN INSERT INTO s VALUES (1); <construct> END;` and then
`SELECT count(*) FROM s`: "before execution" means the table stayed empty, "at run time" means the INSERT happened
before the error. Scope ids in parameters are normalised to `<scope>`.

| Case | Result | Parameters | Phase |
|---|---|---|---|
| `labels_mismatch` | `LABELS_MISMATCH` (42K0L) | beginLabel=`a`, endLabel=`b` | before execution (no INSERT) |
| `end_label_without_begin_label` | `END_LABEL_WITHOUT_BEGIN_LABEL` (42K0L) | endLabel=`b` | before execution (no INSERT) |
| `label_reuses_enclosing` | `LABEL_OR_FOR_VARIABLE_ALREADY_EXISTS` (42K0L) | label=`a` | before execution (no INSERT) |
| `for_variable_reuses_label` | `LABEL_OR_FOR_VARIABLE_ALREADY_EXISTS` (42K0L) | label=`a` | before execution (no INSERT) |
| `label_session_forbidden` | `LABEL_OR_FOR_VARIABLE_NAME_FORBIDDEN` (42K0L) | label=`session` | before execution (no INSERT) |
| `label_system_forbidden` | `LABEL_OR_FOR_VARIABLE_NAME_FORBIDDEN` (42K0L) | label=`system` | before execution (no INSERT) |
| `for_variable_session_forbidden` | `LABEL_OR_FOR_VARIABLE_NAME_FORBIDDEN` (42K0L) | label=`session` | before execution (no INSERT) |
| `qualified_label` | `PARSE_SYNTAX_ERROR` (42601) | error='.', hint= | before execution (no INSERT) |
| `leave_unknown_label` | `INVALID_LABEL_USAGE.DOES_NOT_EXIST` (42K0L) | labelName=NOPE, statementType=LEAVE | before execution (no INSERT) |
| `iterate_unknown_label` | `INVALID_LABEL_USAGE.DOES_NOT_EXIST` (42K0L) | labelName=NOPE, statementType=ITERATE | before execution (no INSERT) |
| `iterate_compound_label` | `INVALID_LABEL_USAGE.ITERATE_IN_COMPOUND` (42K0L) | labelName=C | before execution (no INSERT) |
| `declare_after_statement` | `INVALID_VARIABLE_DECLARATION.ONLY_AT_BEGINNING` (42K0M) | varName=`x` | before execution (no INSERT) |
| `declare_after_handler` | `INVALID_VARIABLE_DECLARATION.ONLY_AT_BEGINNING` (42K0M) | varName=`x` | before execution (no INSERT) |
| `declare_in_if_body` | `INVALID_VARIABLE_DECLARATION.NOT_ALLOWED_IN_SCOPE` (42K0M) | varName=`x` | before execution (no INSERT) |
| `declare_in_while_body` | `INVALID_VARIABLE_DECLARATION.NOT_ALLOWED_IN_SCOPE` (42K0M) | varName=`x` | before execution (no INSERT) |
| `declare_qualified_local` | `INVALID_VARIABLE_DECLARATION.QUALIFIED_LOCAL_VARIABLE` (42K0M) | varName=`a`.`x` | at run time (INSERT done) |
| `declare_or_replace_local` | `INVALID_VARIABLE_DECLARATION.REPLACE_LOCAL_VARIABLE` (42K0M) | varName=`x` | at run time (INSERT done) |
| `handler_after_statement` | `INVALID_HANDLER_DECLARATION.WRONG_PLACE_OF_DECLARATION` (42K0Q) |  | before execution (no INSERT) |
| `handler_in_if_body` | `java.util.NoSuchElementException` |  | before execution (no INSERT) |
| `handler_unknown_condition` | `INVALID_HANDLER_DECLARATION.CONDITION_NOT_FOUND` (42K0Q) | condition=NO_SUCH_COND | before execution (no INSERT) |
| `handler_sqlexception_combined` | `INVALID_HANDLER_DECLARATION.INVALID_CONDITION_COMBINATION` (42K0Q) |  | before execution (no INSERT) |
| `handler_notfound_combined` | `INVALID_HANDLER_DECLARATION.INVALID_CONDITION_COMBINATION` (42K0Q) |  | before execution (no INSERT) |
| `handler_sqlexception_notfound` | no error (script ran) |  | at run time (INSERT done) |
| `handler_duplicate_condition` | `INVALID_HANDLER_DECLARATION.DUPLICATE_CONDITION_IN_HANDLER_DECLARATION` (42K0Q) | condition=DIVIDE_BY_ZERO | before execution (no INSERT) |
| `handler_duplicate_sqlstate` | `INVALID_HANDLER_DECLARATION.DUPLICATE_SQLSTATE_IN_HANDLER_DECLARATION` (42K0Q) | sqlState=22012 | before execution (no INSERT) |
| `two_handlers_same_condition` | `DUPLICATE_EXCEPTION_HANDLER.CONDITION` (42734) | condition=DIVIDE_BY_ZERO | before execution (no INSERT) |
| `two_handlers_same_sqlstate` | `DUPLICATE_EXCEPTION_HANDLER.SQLSTATE` (42734) | sqlState=22012 | before execution (no INSERT) |
| `two_handlers_sqlexception` | `DUPLICATE_EXCEPTION_HANDLER.CONDITION` (42734) | condition=SQLEXCEPTION | before execution (no INSERT) |
| `condition_declared_twice` | `DUPLICATE_CONDITION_IN_SCOPE` (42734) | condition=`C` | before execution (no INSERT) |
| `condition_after_handler` | `java.util.NoSuchElementException` |  | before execution (no INSERT) |
| `condition_qualified_name` | `UNSUPPORTED_DATATYPE` (0A000) | typeName="CONDITION" | before execution (no INSERT) |
| `sqlstate_class_00` | `INVALID_SQLSTATE` (428B3) | sqlState=00001 | before execution (no INSERT) |
| `sqlstate_class_01` | `INVALID_SQLSTATE` (428B3) | sqlState=01000 | before execution (no INSERT) |
| `sqlstate_class_xx` | `INVALID_SQLSTATE` (428B3) | sqlState=XX000 | before execution (no INSERT) |
| `sqlstate_too_short` | `INVALID_SQLSTATE` (428B3) | sqlState=1234 | before execution (no INSERT) |
| `sqlstate_lowercase` | no error (script ran) |  | at run time (INSERT done) |
| `sqlstate_valid_user` | no error (script ran) |  | at run time (INSERT done) |
| `continue_handler` | `UNSUPPORTED_FEATURE.CONTINUE_EXCEPTION_HANDLER` (0A000) |  | before execution (no INSERT) |
| `cursor_declare` | `UNSUPPORTED_FEATURE.SQL_CURSOR` (0A000) |  | before execution (no INSERT) |
| `cursor_open` | `UNSUPPORTED_FEATURE.SQL_CURSOR` (0A000) |  | before execution (no INSERT) |
| `signal` | `PARSE_SYNTAX_ERROR` (42601) | error='SQLSTATE', hint= | before execution (no INSERT) |
| `resignal` | `PARSE_SYNTAX_ERROR` (42601) | error=';', hint= | before execution (no INSERT) |
| `get_diagnostics` | `PARSE_SYNTAX_ERROR` (42601) | error='DIAGNOSTICS', hint= | before execution (no INSERT) |
| `select_into` | `PARSE_SYNTAX_ERROR` (42601) | error='x', hint=: extra input 'x' | before execution (no INSERT) |
| `handler_body_if` | `PARSE_SYNTAX_ERROR` (42601) | error='true', hint= | before execution (no INSERT) |
| `variable_already_exists` | `VARIABLE_ALREADY_EXISTS` (42723) | variableName=`<scope>`.`x` | at run time (INSERT done) |
| `variable_already_exists_case` | `VARIABLE_ALREADY_EXISTS` (42723) | variableName=`<scope>`.`x` | at run time (INSERT done) |
| `variable_already_exists_one_declare` | `DUPLICATE_VARIABLE_NAME_INSIDE_DECLARE` (42734) | variableName=`<scope>`.`a` | at run time (INSERT done) |
| `duplicate_assignments` | `DUPLICATE_ASSIGNMENTS` (42701) | nameList=`a` | at run time (INSERT done) |
| `default_subquery` | `INVALID_DEFAULT_VALUE.SUBQUERY_EXPRESSION` (42623) | colName=`local`.`<scope>`.`x`, defaultValue=(SELECT 1), statement=DECLARE VARIABLE | at run time (INSERT done) |
| `declare_no_type_no_default` | `INVALID_SQL_SYNTAX.VARIABLE_TYPE_OR_DEFAULT_REQUIRED` (42000) |  | before execution (no INSERT) |
| `drop_temporary_variable` | `UNSUPPORTED_FEATURE.SQL_SCRIPTING_DROP_TEMPORARY_VARIABLE` (0A000) |  | at run time (INSERT done) |
| `set_config` | `UNRESOLVED_VARIABLE` (42883) | variableName=`spark`.`sql`.`ansi`.`enabled` | at run time (INSERT done) |
| `set_bare` | `PARSE_SYNTAX_ERROR` (42601) | error=';', hint= | before execution (no INSERT) |
| `set_key` | `PARSE_SYNTAX_ERROR` (42601) | error=';', hint= | before execution (no INSERT) |
| `set_time_zone` | `PARSE_SYNTAX_ERROR` (42601) | error='ZONE', hint=: missing EQ | before execution (no INSERT) |
| `reset` | `PARSE_SYNTAX_ERROR` (42601) | error=';', hint= | before execution (no INSERT) |
| `set_variable_keyword` | `PARSE_SYNTAX_ERROR` (42601) | error='x', hint=: extra input 'x' | before execution (no INSERT) |
| `unresolved_variable_runtime` | `UNRESOLVED_VARIABLE` (42883) | variableName=`nope` | at run time (INSERT done) |
| `unresolved_column_runtime` | `UNRESOLVED_COLUMN.WITHOUT_SUGGESTION` (42703) | objectName=`nope` | at run time (INSERT done) |
| `explain_script_nested` | `PARSE_SYNTAX_ERROR` (42601) | error='BEGIN', hint= | before execution (no INSERT) |

## Differences from design note section 2

- **Phase.** Run-time (the earlier `INSERT` stays): `INVALID_VARIABLE_DECLARATION.QUALIFIED_LOCAL_VARIABLE`,
  `INVALID_VARIABLE_DECLARATION.REPLACE_LOCAL_VARIABLE`, `VARIABLE_ALREADY_EXISTS`, `DUPLICATE_VARIABLE_NAME_INSIDE_DECLARE`,
  `DUPLICATE_ASSIGNMENTS`, `INVALID_DEFAULT_VALUE.SUBQUERY_EXPRESSION`, `UNSUPPORTED_FEATURE.SQL_SCRIPTING_DROP_TEMPORARY_VARIABLE`,
  `UNRESOLVED_VARIABLE`, `UNRESOLVED_COLUMN.*`. Section 2.5 listed the first two with the pre-execution `SSE` checks ("unit 0 confirms each
  one's phase"): they are raised only when the `DECLARE` executes. Every other section 2.5 class is raised before anything runs.
- **`DECLARE a.c CONDITION`** is not a `PARSE_SYNTAX_ERROR`: Spark reads it as a variable declaration of type `CONDITION` and fails with
  `UNSUPPORTED_DATATYPE` (`typeName="CONDITION"`) before execution.
- **`FOR SQLEXCEPTION, NOT FOUND`** (both together) is accepted; `INVALID_CONDITION_COMBINATION` is raised only when one of them is combined with a named
  condition or a SQLSTATE.
- **SQLSTATE validation** is looser than the design: lowercase `'abc12'` is accepted (no `INVALID_SQLSTATE`); `'00001'`, `'01000'`,
  `'XX000'` and the 4-character `'1234'` are rejected.
- **`DECLARE a, A INT`** (duplicate name within one `DECLARE`) raises `DUPLICATE_VARIABLE_NAME_INSIDE_DECLARE`, not `VARIABLE_ALREADY_EXISTS` (run time).
- **Exit handler declared in a compound that is the last statement of an `IF` or `WHILE` body** ends the script silently (nothing after the `IF`/`WHILE` runs, the
  result is empty) instead of continuing after the compound; with a following statement in the body, or in a `FOR` body, it continues as section 2.4 says
  (`handlers.sql`: `exit_handler_in_if_nested_compound`, `exit_handler_in_loop_body_trace`, `exit_in_if_body_compound_then_more`,
  `exit_in_while_body_compound_two_statements_after`, `exit_in_for_body_compound_then_more`).
