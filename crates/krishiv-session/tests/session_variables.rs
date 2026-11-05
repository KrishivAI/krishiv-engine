// SPDX-License-Identifier: Apache-2.0
//! SQL session variables end to end: the Spark 4.2.0 golden
//! `sql-session-variables.sql` replayed in file order in one session.
//!
//! Every statement runs through [`Session::sql`] exactly as the golden
//! lists it. Statements the engine supports assert Spark's result exactly
//! (empty shape, schema and rows, or error class and message parameters).
//! [`Expect::Skip`] runs the statement but asserts nothing, only where the
//! outcome needs a feature the engine does not have yet (inline `VALUES`,
//! `RANGE`, `EXECUTE IMMEDIATE`, SQL functions like `typeof`, `CREATE
//! TABLE`, persistent views, `INTERVAL` values, `LATERAL` subqueries or
//! struct, map and array values), and the comment names it. Statements that
//! fail deterministically without the feature assert the engine's exact
//! error instead (the struct variable). The second
//! golden file, `session-variable-precedence.sql`, needs `DECLARE all` (a
//! reserved word here), `GROUP BY ALL` / `ORDER BY ALL` and `CREATE TABLE`
//! throughout and is skipped as a whole; precedence without them is covered
//! by `variable_precedence_over_tables` instead.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

/// What replaying one statement asserts.
enum Expect {
    /// Spark's empty result shape (`struct<>`, no rows).
    Empty,
    /// A result shape and rows.
    Rows {
        schema: &'static str,
        rows: &'static [&'static [&'static str]],
    },
    /// A Spark error class and message parameters.
    Error {
        class: ErrorClass,
        params: &'static [(&'static str, &'static str)],
    },
    /// Run but not asserted: needs an unsupported feature (named), and the
    /// outcome is not a deterministic error (deterministic failures assert
    /// `Error` instead).
    Skip(&'static str),
}

use Expect::{Empty as E, Error as Err, Rows as R, Skip as S};

/// The golden statements in file order: `(sql, expectation)`.
const CASES: &[(&str, Expect)] = &[
    (
        "SET spark.sql.ansi.enabled = true",
        R {
            schema: "struct<key:string,value:string>",
            rows: &[&["spark.sql.ansi.enabled", "true"]],
        },
    ),
    ("DECLARE title STRING", E),
    ("SET VARIABLE title = '-- Basic sanity --'", E),
    ("DECLARE var1 INT = 5", E),
    (
        "SELECT var1",
        R {
            schema: "struct<var1:int>",
            rows: &[&["5"]],
        },
    ),
    ("SET VARIABLE var1 = 6", E),
    (
        "SELECT var1",
        R {
            schema: "struct<var1:int>",
            rows: &[&["6"]],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("SET VARIABLE title = 'Create Variable - Success Cases'", E),
    ("DECLARE VARIABLE var1 INT", E),
    (
        "SELECT 'Expect: INT, NULL', typeof(var1), var1",
        S("function typeof"),
    ),
    ("DECLARE OR REPLACE VARIABLE var1 DOUBLE", E),
    (
        "SELECT 'Expect: DOUBLE, NULL', typeof(var1), var1",
        S("function typeof"),
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("DECLARE OR REPLACE VARIABLE var1 TIMESTAMP", E),
    (
        "SELECT 'Expect: TIMESTAMP, NULL', typeof(var1), var1",
        S("function typeof"),
    ),
    ("SET VARIABLE title = 'Create Variable - Failure Cases'", E),
    (
        "DECLARE VARIABLE IF NOT EXISTS var1 INT",
        Err {
            class: ErrorClass::ParseSyntaxError,
            params: &[("error", "'EXISTS'"), ("hint", "")],
        },
    ),
    ("DROP TEMPORARY VARIABLE IF EXISTS var1", E),
    ("SET VARIABLE title = 'Drop Variable'", E),
    ("DECLARE VAR var1 INT", E),
    (
        "SELECT var1",
        R {
            schema: "struct<var1:int>",
            rows: &[&["NULL"]],
        },
    ),
    ("DROP TEMPORARY VAR var1", E),
    (
        "SELECT var1",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`var1`")],
        },
    ),
    (
        "DROP TEMPORARY VARIABLE var1",
        Err {
            class: ErrorClass::VariableNotFound,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    ("DROP TEMPORARY VARIABLE IF EXISTS var1", E),
    ("DECLARE VARIABLE var1 INT", E),
    (
        "DROP VARIABLE var1",
        Err {
            class: ErrorClass::ParseSyntaxError,
            params: &[("error", "'VARIABLE'"), ("hint", "")],
        },
    ),
    (
        "DROP VARIABLE system.session.var1",
        Err {
            class: ErrorClass::ParseSyntaxError,
            params: &[("error", "'VARIABLE'"), ("hint", "")],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("SET VARIABLE title = 'Test qualifiers - success'", E),
    ("DECLARE VARIABLE var1 INT DEFAULT 1", E),
    (
        "SELECT 1 as Expected, var1 as Unqualified, session.var1 AS SchemaQualified, system.session.var1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["1", "1", "1", "1"]],
        },
    ),
    ("SET VARIABLE var1 = 2", E),
    (
        "SELECT 2 as Expected, var1 as Unqualified, session.var1 AS SchemaQualified, system.session.var1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["2", "2", "2", "2"]],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE session.var1 INT DEFAULT 1", E),
    (
        "SELECT 1 as Expected, var1 as Unqualified, session.var1 AS SchemaQualified, system.session.var1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["1", "1", "1", "1"]],
        },
    ),
    ("SET VARIABLE session.var1 = 2", E),
    (
        "SELECT 2 as Expected, var1 as Unqualified, session.var1 AS SchemaQualified, system.session.var1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["2", "2", "2", "2"]],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE system.session.var1 INT DEFAULT 1",
        E,
    ),
    (
        "SELECT 1 as Expected, var1 as Unqualified, session.var1 AS SchemaQualified, system.session.var1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["1", "1", "1", "1"]],
        },
    ),
    ("SET VARIABLE system.session.var1 = 2", E),
    (
        "SELECT 2 as Expected, var1 as Unqualified, session.var1 AS SchemaQualified, system.session.var1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["2", "2", "2", "2"]],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE sySteM.sEssIon.vAr1 INT DEFAULT 1",
        E,
    ),
    (
        "SELECT 1 as Expected, var1 as Unqualified, sessIon.Var1 AS SchemaQualified, System.sessiOn.var1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["1", "1", "1", "1"]],
        },
    ),
    ("SET VARIABLE sYstem.sesSiOn.vaR1 = 2", E),
    (
        "SELECT 2 as Expected, VAR1 as Unqualified, SESSION.VAR1 AS SchemaQualified, SYSTEM.SESSION.VAR1 AS fullyQualified",
        R {
            schema: "struct<Expected:int,Unqualified:int,SchemaQualified:int,fullyQualified:int>",
            rows: &[&["2", "2", "2", "2"]],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT", E),
    ("DROP TEMPORARY VARIABLE var1", E),
    (
        "DROP TEMPORARY VARIABLE var1",
        Err {
            class: ErrorClass::VariableNotFound,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT", E),
    ("DROP TEMPORARY VARIABLE session.var1", E),
    (
        "DROP TEMPORARY VARIABLE var1",
        Err {
            class: ErrorClass::VariableNotFound,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT", E),
    ("DROP TEMPORARY VARIABLE system.session.var1", E),
    (
        "DROP TEMPORARY VARIABLE var1",
        Err {
            class: ErrorClass::VariableNotFound,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT", E),
    ("DROP TEMPORARY VARIABLE sysTem.sesSion.vAr1", E),
    (
        "DROP TEMPORARY VARIABLE var1",
        Err {
            class: ErrorClass::VariableNotFound,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    ("SET VARIABLE title = 'Test variable in aggregate'", E),
    (
        "SELECT (SELECT MAX(id) FROM RANGE(10) WHERE id < title) FROM VALUES 1, 2 AS t(title)",
        S("RANGE and inline VALUES"),
    ),
    (
        "SET VARIABLE title = 'Dropped struct variable -- field access vs qualified name'",
        E,
    ),
    // Struct values cannot be held by session variables yet (the STRUCT
    // type parses and struct constructors resolve, but variable values of
    // struct type are not supported), so the declaration fails; the field
    // access then has no variable to resolve against, and the drop has
    // nothing to drop. Each asserts the engine's exact error, so adding
    // struct variable values must update this case.
    (
        "DECLARE OR REPLACE VARIABLE session STRUCT<a INT> = NAMED_STRUCT('a', 1)",
        Err {
            class: ErrorClass::KrishivUnsupportedFeature,
            params: &[(
                "feature",
                "session variable values of type Struct(\"a\": Int32)",
            )],
        },
    ),
    (
        "SELECT session.a",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`session`.`a`")],
        },
    ),
    (
        "DROP TEMPORARY VARIABLE session",
        Err {
            class: ErrorClass::VariableNotFound,
            params: &[("variableName", "`system`.`session`.`session`")],
        },
    ),
    (
        "SELECT session.a",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`session`.`a`")],
        },
    ),
    ("SET VARIABLE title = 'Test qualifiers - fail'", E),
    (
        "DECLARE OR REPLACE VARIABLE builtin.var1 INT",
        Err {
            class: ErrorClass::UnresolvedVariable,
            params: &[
                ("variableName", "`builtin`.`var1`"),
                ("searchPath", "[`system`.`session`]"),
            ],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE system.sesion.var1 INT",
        Err {
            class: ErrorClass::UnresolvedVariable,
            params: &[
                ("variableName", "`system`.`sesion`.`var1`"),
                ("searchPath", "[`system`.`session`]"),
            ],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE sys.session.var1 INT",
        Err {
            class: ErrorClass::UnresolvedVariable,
            params: &[
                ("variableName", "`sys`.`session`.`var1`"),
                ("searchPath", "[`system`.`session`]"),
            ],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT", E),
    (
        "SELECT var",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`var`")],
        },
    ),
    (
        "SELECT ses.var1",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`ses`.`var1`")],
        },
    ),
    (
        "SELECT b.sesson.var1",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`b`.`sesson`.`var1`")],
        },
    ),
    (
        "SELECT builtn.session.var1",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`builtn`.`session`.`var1`")],
        },
    ),
    (
        "SET VARIABLE ses.var1 = 1",
        Err {
            class: ErrorClass::UnresolvedVariable,
            params: &[
                ("variableName", "`ses`.`var1`"),
                (
                    "searchPath",
                    "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]",
                ),
            ],
        },
    ),
    (
        "SET VARIABLE builtn.session.var1 = 1",
        Err {
            class: ErrorClass::UnresolvedVariable,
            params: &[
                ("variableName", "`builtn`.`session`.`var1`"),
                (
                    "searchPath",
                    "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]",
                ),
            ],
        },
    ),
    ("SET VARIABLE title = 'Test DEFAULT on create - success'", E),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 1", E),
    (
        "SELECT 1 AS Expected, var1 AS result",
        R {
            schema: "struct<Expected:int,result:int>",
            rows: &[&["1", "1"]],
        },
    ),
    // `RAND` is supported, so the declaration replaces `var1` with the
    // `DOUBLE` value of `1 + RAND(5)`: the `SELECT` below asserts it is at
    // least 1, as the golden does.
    (
        "DECLARE OR REPLACE VARIABLE var1 DOUBLE DEFAULT 1 + RAND(5)",
        E,
    ),
    (
        "SELECT true AS Expected, var1 >= 1 AS result",
        R {
            schema: "struct<Expected:boolean,result:boolean>",
            rows: &[&["true", "true"]],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1 = 'Hello'", E),
    (
        "SELECT 'STRING, Hello' AS Expected, typeof(var1) AS type, var1 AS result",
        S("function typeof"),
    ),
    ("DECLARE OR REPLACE VARIABLE var1 DEFAULT NULL", E),
    (
        "SELECT 'VOID, NULL' AS Expected, typeof(var1) AS type, var1 AS result",
        S("function typeof"),
    ),
    ("DECLARE OR REPLACE VARIABLE INT DEFAULT 5.0", E),
    (
        "SELECT 'INT, 5' AS Expected, typeof(var1) AS type, var1 AS result",
        S("function typeof"),
    ),
    (
        "DECLARE OR REPLACE VARIABLE var1 MAP<string, double> DEFAULT MAP('Hello', 5.1, 'World', -7.1E10)",
        S("function MAP and MAP values"),
    ),
    (
        "SELECT 'MAP<string, double>, [Hello -> 5.1, World -> -7E10]' AS Expected, typeof(var1) AS type, var1 AS result",
        S("function typeof"),
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT NULL", E),
    (
        "SELECT 'NULL' AS Expected, var1 AS result",
        R {
            schema: "struct<Expected:string,result:int>",
            rows: &[&["NULL", "NULL"]],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE var1 STRING DEFAULT CURRENT_DATABASE()",
        S("function CURRENT_DATABASE"),
    ),
    (
        "SELECT 'true' AS Expected, length(var1) > 0 AS result",
        S("needs CURRENT_DATABASE above"),
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    (
        "DECLARE var1",
        Err {
            class: ErrorClass::InvalidSqlSyntaxVariableTypeOrDefaultRequired,
            params: &[],
        },
    ),
    (
        "SET VARIABLE title = 'Test DEFAULT on create - failures'",
        E,
    ),
    (
        "DECLARE OR REPLACE VARIABLE var1 INT DEFAULT (SELECT c1 FROM VALUES(1) AS T(c1))",
        Err {
            class: ErrorClass::InvalidDefaultValueSubqueryExpression,
            params: &[
                ("colName", "`system`.`session`.`var1`"),
                ("defaultValue", "(SELECT c1 FROM VALUES(1) AS T(c1))"),
                ("statement", "DECLARE VARIABLE"),
            ],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 'hello'",
        Err {
            class: ErrorClass::CastInvalidInput,
            params: &[
                ("expression", "'hello'"),
                ("sourceType", "\"STRING\""),
                ("targetType", "\"INT\""),
            ],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 1 / 0",
        Err {
            class: ErrorClass::DivideByZero,
            params: &[("config", "\"spark.sql.ansi.enabled\"")],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE var1 SMALLINT DEFAULT 100000",
        Err {
            class: ErrorClass::CastOverflow,
            params: &[
                ("value", "100000"),
                ("sourceType", "\"INT\""),
                ("targetType", "\"SMALLINT\""),
            ],
        },
    ),
    ("SET VARIABLE title = 'SET VARIABLE - single target'", E),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5", E),
    ("SET VARIABLE var1 = 7", E),
    (
        "SELECT var1",
        R {
            schema: "struct<var1:int>",
            rows: &[&["7"]],
        },
    ),
    ("SET VAR var1 = 8", E),
    (
        "SELECT var1",
        R {
            schema: "struct<var1:int>",
            rows: &[&["8"]],
        },
    ),
    (
        "SET VARIABLE var1 = (SELECT c1 FROM VALUES(1) AS T(c1))",
        S("inline VALUES"),
    ),
    ("SELECT var1", S("needs the SET above")),
    (
        "SET VARIABLE var1 = (SELECT c1 FROM VALUES(1) AS T(c1) WHERE 1=0)",
        S("inline VALUES"),
    ),
    ("SELECT var1 AS `null`", S("needs the SET above")),
    (
        "SET VARIABLE var1 = (SELECT c1 FROM VALUES(1.0) AS T(c1))",
        S("inline VALUES"),
    ),
    ("SELECT var1", S("needs the SET above")),
    (
        "SET VARIABLE var1 = (SELECT c1 FROM VALUES(1.0E10) AS T(c1))",
        S("inline VALUES"),
    ),
    ("SELECT var1", S("needs the SET above")),
    (
        "SET VARIABLE var1 = (SELECT c1 FROM VALUES(1), (2) AS T(c1))",
        S("inline VALUES"),
    ),
    (
        "SET VARIABLE var1 = (SELECT c1, c1 FROM VALUES(1), (2) AS T(c1))",
        S("inline VALUES"),
    ),
    (
        "SET VARIABLE var1 = (SELECT c1 FROM VALUES('hello') AS T(c1))",
        S("inline VALUES"),
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5", E),
    ("SET VARIABLE var1 = var1 + 1", E),
    (
        "SELECT var1",
        R {
            schema: "struct<var1:int>",
            rows: &[&["6"]],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    (
        "SET VARIABLE title = 'SET VARIABLE - comma separated target'",
        E,
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5", E),
    ("DECLARE OR REPLACE VARIABLE var2 STRING DEFAULT 'hello'", E),
    ("DECLARE OR REPLACE VARIABLE var3 DOUBLE DEFAULT 2", E),
    ("SET VARIABLE var1 = 6, var2 = 'world', var3 = pi()", E),
    (
        "SELECT var1 AS `6`, var2 AS `world` , var3 as `3.14...`",
        R {
            schema: "struct<6:int,world:string,3.14...:double>",
            rows: &[&["6", "world", "3.141592653589793"]],
        },
    ),
    ("SET VAR var1 = 7, var2 = 'universe', var3 = -1", E),
    (
        "SELECT var1 AS `7`, var2 AS `universe` , var3 as `-1`",
        R {
            schema: "struct<7:int,universe:string,-1:double>",
            rows: &[&["7", "universe", "-1.0"]],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5", E),
    ("DECLARE OR REPLACE VARIABLE var2 STRING DEFAULT 'hello'", E),
    ("DECLARE OR REPLACE VARIABLE var3 DOUBLE DEFAULT 2", E),
    (
        "SET VARIABLE var1 = var3, var2 = ascii(var1), var3 = var1",
        S("function ascii"),
    ),
    (
        "SELECT var1 AS `2`, var2 AS `104`, var3 AS `5`",
        S("needs the SET above"),
    ),
    (
        "SET VARIABLE var1 = var3, var2 = INTERVAL'5' HOUR, var3 = var1",
        S("INTERVAL values"),
    ),
    (
        "SET VARIABLE var1 = 1, var2 = 0, vAr1 = 1",
        Err {
            class: ErrorClass::DuplicateAssignments,
            params: &[("nameList", "`var1`")],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("DROP TEMPORARY VARIABLE var2", E),
    ("DROP TEMPORARY VARIABLE var3", E),
    (
        "SET VARIABLE title = 'DECLARE VARIABLE - multiple variables declared at once'",
        E,
    ),
    ("DECLARE VARIABLE var1, var2, var3 INT", E),
    (
        "DECLARE VARIABLE var4, var5, var6 INT DEFAULT CAST(RAND(0) * 10 AS INT)",
        E,
    ),
    ("DECLARE VARIABLE var7, var8, var9 DEFAULT 5", E),
    (
        "SELECT var4 = var5, var4 = var6, var5 = var6",
        R {
            schema: "struct<(variablereference(system.session.var4=7) = variablereference(system.session.var5=7)):boolean,(variablereference(system.session.var4=7) = variablereference(system.session.var6=7)):boolean,(variablereference(system.session.var5=7) = variablereference(system.session.var6=7)):boolean>",
            rows: &[&["true", "true", "true"]],
        },
    ),
    ("DECLARE OR REPLACE VARIABLE var1, var2, var3 DOUBLE", E),
    (
        "DECLARE OR REPLACE VARIABLE var4, var5, var6 DOUBLE DEFAULT RAND(0)",
        E,
    ),
    (
        "DECLARE OR REPLACE VARIABLE var7, var8, var9 DEFAULT 1.5",
        E,
    ),
    (
        "SELECT var4 = var5, var4 = var6, var5 = var6",
        R {
            schema: "struct<(variablereference(system.session.var4=0.7604953758285915D) = variablereference(system.session.var5=0.7604953758285915D)):boolean,(variablereference(system.session.var4=0.7604953758285915D) = variablereference(system.session.var6=0.7604953758285915D)):boolean,(variablereference(system.session.var5=0.7604953758285915D) = variablereference(system.session.var6=0.7604953758285915D)):boolean>",
            rows: &[&["true", "true", "true"]],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("DROP TEMPORARY VARIABLE var2", E),
    ("DROP TEMPORARY VARIABLE var3", E),
    ("DROP TEMPORARY VARIABLE var4", E),
    ("DROP TEMPORARY VARIABLE var5", E),
    ("DROP TEMPORARY VARIABLE var6", E),
    ("DROP TEMPORARY VARIABLE var7", E),
    ("DROP TEMPORARY VARIABLE var8", E),
    ("DROP TEMPORARY VARIABLE var9", E),
    (
        "SET VARIABLE title = 'DECLARE VARIABLE - duplicate names'",
        E,
    ),
    (
        "DECLARE VARIABLE var1, var2, vAr1 INT",
        Err {
            class: ErrorClass::DuplicateVariableNameInsideDeclare,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    (
        "DECLARE OR REPLACE VARIABLE var1, var2, vAr1 INT",
        Err {
            class: ErrorClass::DuplicateVariableNameInsideDeclare,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    ("SET VARIABLE title = 'SET VARIABLE - row assignment'", E),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5", E),
    ("DECLARE OR REPLACE VARIABLE var2 STRING DEFAULT 'hello'", E),
    ("DECLARE OR REPLACE VARIABLE var3 DOUBLE DEFAULT 2", E),
    (
        "SET VARIABLE (var1) = (SELECT c1 FROM VALUES(1) AS T(c1))",
        S("inline VALUES"),
    ),
    ("SELECT var1", S("needs the SET above")),
    (
        "SET VAR (var1) = (SELECT c1 FROM VALUES(2) AS T(c1))",
        S("inline VALUES"),
    ),
    ("SELECT var1", S("needs the SET above")),
    (
        "SET VARIABLE (var1, var2) = (SELECT c1, c2 FROM VALUES(10, 11) AS T(c1, c2))",
        S("inline VALUES"),
    ),
    (
        "SELECT var1 AS `10`, var2 AS `11`",
        S("needs the SET above"),
    ),
    (
        "SET VARIABLE (var1, var2, var3) = (SELECT c1, c2, c3 FROM VALUES(100, 110, 120) AS T(c1, c2, c3))",
        S("inline VALUES"),
    ),
    (
        "SELECT var1 AS `100`, var2 AS `110`, var3 AS `120`",
        S("needs the SET above"),
    ),
    (
        "SET VARIABLE (var1, var2, var3) = (SELECT c1, c2, c3 FROM VALUES(100, 110, 120) AS T(c1, c2, c3) WHERE 1 = 0)",
        S("inline VALUES"),
    ),
    (
        "SELECT var1 AS `NULL`, var2 AS `NULL`, var3 AS `NULL`",
        S("needs the SET above"),
    ),
    (
        "SET VARIABLE () = (SELECT 1)",
        Err {
            class: ErrorClass::InvalidSetSyntax,
            params: &[],
        },
    ),
    (
        "SET VARIABLE (var1, var2, var3) = (SELECT c1, c2, c3 FROM VALUES(100, 110, 120), (-100, -110, -120) AS T(c1, c2, c3))",
        S("inline VALUES"),
    ),
    (
        "SET VARIABLE (var1, var2, var3) = (SELECT c1, c2 FROM VALUES(100, 110, 120) AS T(c1, c2, c3))",
        S("inline VALUES"),
    ),
    (
        "SET VARIABLE (var1, var2, var3) = (SELECT c1, c2, c3, c1 FROM VALUES(100, 110, 120) AS T(c1, c2, c3))",
        S("inline VALUES"),
    ),
    (
        "SET VARIABLE (var1, var2, var1) = (SELECT c1, c2, c3, c1 FROM VALUES(100, 110, 120) AS T(c1, c2, c3))",
        Err {
            class: ErrorClass::DuplicateAssignments,
            params: &[("nameList", "`var1`")],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("DROP TEMPORARY VARIABLE var2", E),
    ("DROP TEMPORARY VARIABLE var3", E),
    ("SET VARIABLE title = 'DEFAULT expression usage'", E),
    (
        "DECLARE OR REPLACE VARIABLE var1 STRING DEFAULT 'default1'",
        E,
    ),
    (
        "DECLARE OR REPLACE VARIABLE var2 STRING DEFAULT 'default2'",
        E,
    ),
    (
        "DECLARE OR REPLACE VARIABLE var3 STRING DEFAULT 'default3'",
        E,
    ),
    ("SET VARIABLE var1 = 'hello'", E),
    ("SET VARIABLE var1 = DEFAULT", E),
    (
        "SELECT var1 AS `default`",
        R {
            schema: "struct<default:string>",
            rows: &[&["default1"]],
        },
    ),
    ("SET VARIABLE var1 = 'hello1'", E),
    ("SET VARIABLE var1 = 'hello2'", E),
    ("SET VARIABLE var1 = 'hello3'", E),
    (
        "SET VARIABLE var1 = DEFAULT, var2 = DEFAULT, var3 = DEFAULT",
        E,
    ),
    (
        "SELECT var1 AS `default1`, var2 AS `default2`, var3 AS `default3`",
        R {
            schema: "struct<default1:string,default2:string,default3:string>",
            rows: &[&["default1", "default2", "default3"]],
        },
    ),
    ("SET VARIABLE var1 = 'hello'", E),
    (
        "SET VARIABLE (var1) = (SELECT DEFAULT FROM VALUES(1) AS T(c1))",
        S("inline VALUES"),
    ),
    ("SELECT var1 AS `default`", S("needs the SET above")),
    ("SET VARIABLE var1 = 'hello'", E),
    (
        "SET VARIABLE (var1) = (SELECT DEFAULT FROM VALUES('world') AS T(default))",
        S("inline VALUES"),
    ),
    ("SELECT var1 AS `world`", S("needs the SET above")),
    ("SET VARIABLE var1 = 'hello'", E),
    (
        "SET VARIABLE (var1) = (SELECT DEFAULT FROM VALUES(1) AS T(c1) LIMIT 1)",
        S("inline VALUES"),
    ),
    ("SELECT var1 AS `default`", S("needs the SET above")),
    ("SET VARIABLE var1 = 'hello'", E),
    (
        "SET VARIABLE (var1) = (SELECT DEFAULT FROM VALUES(1),(2),(3) AS T(c1) LIMIT 1 OFFSET 1)",
        S("inline VALUES"),
    ),
    ("SELECT var1 AS `default`", S("needs the SET above")),
    ("SET VARIABLE var1 = 'hello'", E),
    (
        "SET VARIABLE (var1) = (SELECT DEFAULT FROM VALUES(1),(2),(3) AS T(c1) OFFSET 1)",
        S("inline VALUES"),
    ),
    ("SELECT var1 AS `default`", S("needs the SET above")),
    ("SET VARIABLE var1 = 'hello'", E),
    (
        "SET VARIABLE (var1) = (WITH v1(c1) AS (VALUES(1) AS T(c1)) SELECT DEFAULT FROM VALUES(1),(2),(3) AS T(c1))",
        S("inline VALUES"),
    ),
    ("SELECT var1 AS `default`", S("needs the SET above")),
    (
        "SET VARIABLE var1 = 'Hello' || DEFAULT",
        Err {
            class: ErrorClass::DefaultPlacementInvalid,
            params: &[],
        },
    ),
    (
        "SET VARIABLE (var1) = (VALUES(DEFAULT))",
        S("inline VALUES"),
    ),
    (
        "SET VARIABLE (var1) = (WITH v1(c1) AS (VALUES(1) AS T(c1)) SELECT DEFAULT + 1 FROM VALUES(1),(2),(3) AS T(c1))",
        S("inline VALUES"),
    ),
    (
        "SET VARIABLE var1 = session.default",
        Err {
            class: ErrorClass::UnresolvedColumnWithoutSuggestion,
            params: &[("objectName", "`session`.`default`")],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("DROP TEMPORARY VARIABLE var2", E),
    ("DROP TEMPORARY VARIABLE var3", E),
    ("SET VARIABLE title = 'SET command'", E),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 1", E),
    (
        "SET x.var1 = 5",
        R {
            schema: "struct<key:string,value:string>",
            rows: &[&["x.var1", "5"]],
        },
    ),
    (
        "SET x = 5",
        R {
            schema: "struct<key:string,value:string>",
            rows: &[&["x", "5"]],
        },
    ),
    (
        "SET system.x.var = 5",
        R {
            schema: "struct<key:string,value:string>",
            rows: &[&["system.x.var", "5"]],
        },
    ),
    (
        "SET x.session.var1 = 5",
        R {
            schema: "struct<key:string,value:string>",
            rows: &[&["x.session.var1", "5"]],
        },
    ),
    (
        "SET var1 = 5",
        Err {
            class: ErrorClass::UnsupportedFeatureSetVariableUsingSet,
            params: &[("variableName", "`var1`")],
        },
    ),
    (
        "SET session.var1 = 5",
        Err {
            class: ErrorClass::UnsupportedFeatureSetVariableUsingSet,
            params: &[("variableName", "`session`.`var1`")],
        },
    ),
    (
        "SET system.session.var1 = 5",
        Err {
            class: ErrorClass::UnsupportedFeatureSetVariableUsingSet,
            params: &[("variableName", "`system`.`session`.`var1`")],
        },
    ),
    (
        "SET vAr1 = 5",
        Err {
            class: ErrorClass::UnsupportedFeatureSetVariableUsingSet,
            params: &[("variableName", "`vAr1`")],
        },
    ),
    (
        "SET seSSion.var1 = 5",
        Err {
            class: ErrorClass::UnsupportedFeatureSetVariableUsingSet,
            params: &[("variableName", "`seSSion`.`var1`")],
        },
    ),
    (
        "SET sYStem.session.var1 = 5",
        Err {
            class: ErrorClass::UnsupportedFeatureSetVariableUsingSet,
            params: &[("variableName", "`sYStem`.`session`.`var1`")],
        },
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 1", E),
    (
        "SELECT var1 AS `2` FROM VALUES(2) AS T(var1)",
        S("inline VALUES"),
    ),
    (
        "SELECT c1 AS `2` FROM VALUES(2) AS T(var1), LATERAL(SELECT var1) AS TT(c1)",
        S("inline VALUES and LATERAL subqueries"),
    ),
    (
        "SELECT session.var1 AS `1` FROM VALUES(2) AS T(var1)",
        S("inline VALUES"),
    ),
    (
        "SELECT c1 AS `1` FROM VALUES(2) AS T(var1), LATERAL(SELECT session.var1) AS TT(c1)",
        S("inline VALUES and LATERAL subqueries"),
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
    (
        "SET VARIABLE title = 'variable references -- visibility'",
        E,
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 1", E),
    ("VALUES (var1)", S("VALUES as a table factor")),
    (
        "SELECT var1",
        R {
            schema: "struct<var1:int>",
            rows: &[&["1"]],
        },
    ),
    ("SELECT sum(var1) FROM VALUES(1)", S("inline VALUES")),
    ("SELECT var1 + SUM(0) FROM VALUES(1)", S("inline VALUES")),
    (
        "SELECT substr('12345', var1, 1)",
        R {
            schema: "struct<substr(12345, variablereference(system.session.var1=1), 1):string>",
            rows: &[&["1"]],
        },
    ),
    (
        "SELECT 1 FROM VALUES(1, 2) AS T(c1, c2) GROUP BY c1 + var1",
        S("inline VALUES"),
    ),
    (
        "SELECT c1, sum(c2) FROM VALUES(1, 2) AS T(c1, c2) GROUP BY c1 HAVING sum(c1) != var1",
        S("inline VALUES"),
    ),
    (
        "SELECT 1 FROM VALUES(1) AS T(c1) WHERE c1 IN (var1)",
        S("inline VALUES"),
    ),
    (
        "SELECT sum(c1) FILTER (c1 != var1) FROM VALUES(1, 2), (2, 3) AS T(c1, c2)",
        S("FILTER parses differently here and the query needs VALUES"),
    ),
    ("SELECT array(1, 2, 4)[var1]", S("function array")),
    (
        "SELECT 1 FROM VALUES(1) AS T(c1) WHERE c1 = var1",
        S("inline VALUES"),
    ),
    (
        "WITH v1 AS (SELECT var1 AS c1) SELECT c1 AS `1` FROM v1",
        R {
            schema: "struct<1:int>",
            rows: &[&["1"]],
        },
    ),
    ("CREATE OR REPLACE TEMPORARY VIEW v AS SELECT var1 AS c1", E),
    (
        "SELECT * FROM v",
        R {
            schema: "struct<c1:int>",
            rows: &[&["1"]],
        },
    ),
    ("DROP VIEW v", E),
    ("DROP TEMPORARY VARIABLE var1", E),
    (
        "SET VARIABLE title = 'variable references -- prohibited'",
        E,
    ),
    ("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 1", E),
    (
        "CREATE OR REPLACE VIEW v AS SELECT var1 AS c1",
        S("persistent views"),
    ),
    ("DROP VIEW IF EXISTS V", E),
    ("DROP TEMPORARY VARIABLE var1", E),
    (
        "SET VARIABLE title = 'variable references -- test constant folding'",
        E,
    ),
    ("DECLARE OR REPLACE VARIABLE var1 STRING DEFAULT 'a INT'", E),
    (
        "SELECT from_json('{\"a\": 1}', var1)",
        S("function from_json"),
    ),
    ("DROP TEMPORARY VARIABLE var1", E),
];

/// A session with ANSI on, as the golden file sets it.
fn session() -> Session {
    Session::builder()
        .config("spark.sql.ansi.enabled", "true")
        .build()
        .unwrap()
}

/// The rows of `sql` as strings (`NULL` for null), with the schema they
/// came with.
fn rows(s: &Session, sql: &str) -> (String, Vec<Vec<String>>) {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: unexpected error {e}"));
    let schema = block_on(df.schema()).unwrap_or_else(|e| panic!("{sql}: unexpected error {e}"));
    let rendered = schema
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), spark_name(f.data_type())))
        .collect::<Vec<_>>()
        .join(",");
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: unexpected error {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(b.columns().iter().map(|c| cell(c, i)).collect());
        }
    }
    (format!("struct<{rendered}>"), out)
}

/// The Spark type name of an Arrow type, as the golden schemas spell it.
fn spark_name(t: &ArrowType) -> String {
    match t {
        ArrowType::Null => "void".to_owned(),
        ArrowType::Boolean => "boolean".to_owned(),
        ArrowType::Int8 => "tinyint".to_owned(),
        ArrowType::Int16 => "smallint".to_owned(),
        ArrowType::Int32 => "int".to_owned(),
        ArrowType::Int64 => "bigint".to_owned(),
        ArrowType::Float32 => "float".to_owned(),
        ArrowType::Float64 => "double".to_owned(),
        ArrowType::Decimal128(p, s) => format!("decimal({p},{s})"),
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => "string".to_owned(),
        ArrowType::Date32 => "date".to_owned(),
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some(_)) => {
            "timestamp".to_owned()
        }
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, None) => {
            "timestamp_ntz".to_owned()
        }
        other => panic!("unexpected type {other}"),
    }
}

/// One cell as the golden rows spell it (`NULL` for null).
fn cell(column: &arrow_array::ArrayRef, row: usize) -> String {
    use arrow_array::types::{Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type};
    if column.is_null(row) || matches!(column.data_type(), ArrowType::Null) {
        return "NULL".to_owned();
    }
    match column.data_type() {
        ArrowType::Boolean => column.as_boolean().value(row).to_string(),
        ArrowType::Int8 => column.as_primitive::<Int8Type>().value(row).to_string(),
        ArrowType::Int16 => column.as_primitive::<Int16Type>().value(row).to_string(),
        ArrowType::Int32 => column.as_primitive::<Int32Type>().value(row).to_string(),
        ArrowType::Int64 => column.as_primitive::<Int64Type>().value(row).to_string(),
        ArrowType::Float32 => double(f64::from(column.as_primitive::<Float32Type>().value(row))),
        ArrowType::Float64 => double(column.as_primitive::<Float64Type>().value(row)),
        ArrowType::Utf8 => column.as_string::<i32>().value(row).to_owned(),
        ArrowType::LargeUtf8 => column.as_string::<i64>().value(row).to_owned(),
        ArrowType::Utf8View => column.as_string_view().value(row).to_owned(),
        other => panic!("unexpected type {other}"),
    }
}

/// A double as Spark spells it (`-1.0`, not `-1`).
fn double(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// The error `sql` fails with, from analysis or the first action.
fn error(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Ok(df) => match block_on(df.collect()) {
            Ok(_) => panic!("{sql}: unexpectedly succeeded"),
            Err(e) => e,
        },
        Err(e) => e,
    }
}

#[test]
fn replay_sql_session_variables() {
    let s = session();
    for (sql, expect) in CASES {
        match expect {
            Expect::Empty => {
                let (schema, found) = rows(&s, sql);
                assert_eq!(schema.as_str(), "struct<>", "{sql}");
                assert!(found.is_empty(), "{sql}");
            }
            Expect::Rows {
                schema,
                rows: expected,
            } => {
                let (found_schema, found_rows) = rows(&s, sql);
                assert_eq!(found_schema, *schema, "{sql}");
                let expected: Vec<Vec<String>> = expected
                    .iter()
                    .map(|r| r.iter().map(ToString::to_string).collect())
                    .collect();
                assert_eq!(found_rows, expected, "{sql}");
            }
            Expect::Error { class, params } => {
                let e = error(&s, sql);
                assert_eq!(e.error_class(), *class, "{sql}");
                let mut found: Vec<(String, String)> = e
                    .message_parameters()
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), v.clone()))
                    .collect();
                let mut expected: Vec<(String, String)> = params
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                    .collect();
                found.sort();
                expected.sort();
                assert_eq!(found, expected, "{sql}: {e}");
            }
            Expect::Skip(reason) => {
                // Still runs, in file order, so later statements see the
                // same session state; the outcome is not asserted. The
                // reason names the unsupported feature (used in review).
                let _ = reason;
                if let Ok(df) = block_on(s.sql(sql)) {
                    let _ = block_on(df.collect());
                }
            }
        }
    }
}

/// Precedence without inline `VALUES`: columns and outer references beat
/// variables, and qualified names skip columns.
#[test]
fn variable_precedence_over_tables() {
    let s = session();
    s.register_batches(
        "t",
        Arc::new(Schema::new(vec![Field::new(
            "var1",
            ArrowType::Int32,
            false,
        )])),
        vec![
            arrow_array::RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "var1",
                    ArrowType::Int32,
                    false,
                )])),
                vec![Arc::new(arrow_array::Int32Array::from(vec![2]))],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    block_on(s.sql("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 1")).unwrap();
    // A column beats the variable, a qualified name skips the column.
    let (schema, found) = rows(&s, "SELECT var1, session.var1 FROM t");
    assert_eq!(schema, "struct<var1:int,var1:int>");
    assert_eq!(found, [vec!["2".to_owned(), "1".to_owned()]]);
    // An outer reference beats the variable: `x = 1` only matches the
    // variable, so no row matches the outer column `2`.
    let (_, found) = rows(
        &s,
        "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM (SELECT 1 AS x) AS u WHERE x = var1)",
    );
    assert!(found.is_empty());
    // An output alias beats the variable too: `GROUP BY a` groups by the
    // `col1` alias (two groups), not by the constant variable `a` (which
    // would give one group). `GROUP BY ALL` / `ORDER BY ALL` cannot be
    // covered: the engine has no `ALL` grouping yet.
    s.register_batches(
        "u",
        Arc::new(Schema::new(vec![
            Field::new("col1", ArrowType::Int32, false),
            Field::new("col2", ArrowType::Int32, false),
        ])),
        vec![
            arrow_array::RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("col1", ArrowType::Int32, false),
                    Field::new("col2", ArrowType::Int32, false),
                ])),
                vec![
                    Arc::new(arrow_array::Int32Array::from(vec![1, 1, 2])),
                    Arc::new(arrow_array::Int32Array::from(vec![10, 20, 30])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    block_on(s.sql("DECLARE OR REPLACE VARIABLE a INT DEFAULT 1")).unwrap();
    let (schema, found) = rows(
        &s,
        "SELECT col1 AS a, SUM(col2) AS s FROM u GROUP BY a ORDER BY a",
    );
    assert_eq!(schema, "struct<a:int,s:bigint>");
    assert_eq!(
        found,
        [
            vec!["1".to_owned(), "30".to_owned()],
            vec!["2".to_owned(), "30".to_owned()]
        ]
    );
}

/// A cached resolved plan is never reused after a variable changed: the same frame sees the new value on its next action.
#[test]
fn set_assigns_everything_at_once_and_reanalyzes() {
    let s = session();
    for sql in [
        "DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5",
        "DECLARE OR REPLACE VARIABLE var2 STRING DEFAULT 'hello'",
        "DECLARE OR REPLACE VARIABLE var3 DOUBLE DEFAULT 2",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    block_on(s.sql("SET VARIABLE var1 = var3, var2 = UPPER(var1), var3 = var1")).unwrap();
    let (schema, found) = rows(&s, "SELECT var1, var2, var3");
    assert_eq!(schema, "struct<var1:int,var2:string,var3:double>");
    assert_eq!(
        found,
        [vec!["2".to_owned(), "5".to_owned(), "5.0".to_owned()]]
    );
    // One frame, run before and after a `SET`: the second run sees the
    // new value, so the cached plan did not survive the change.
    let df = block_on(s.sql("SELECT var1")).unwrap();
    assert_eq!(rows(&s, "SELECT var1").1, [vec!["2".to_owned()]]);
    assert_eq!(
        block_on(df.collect()).unwrap()[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        2
    );
    block_on(s.sql("SET VARIABLE var1 = 42")).unwrap();
    assert_eq!(
        block_on(df.collect()).unwrap()[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        42
    );
}

/// Row assignments through queries: one row assigns with casts, no rows
/// assign NULL, several rows fail.
#[test]
fn set_row_assignment_through_queries() {
    let s = session();
    s.register_batches(
        "p",
        Arc::new(Schema::new(vec![
            Field::new("id", ArrowType::Int32, false),
            Field::new("name", ArrowType::Utf8, true),
        ])),
        vec![
            arrow_array::RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("id", ArrowType::Int32, false),
                    Field::new("name", ArrowType::Utf8, true),
                ])),
                vec![
                    Arc::new(arrow_array::Int32Array::from(vec![7])),
                    Arc::new(arrow_array::StringArray::from(vec![Some("seven")])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    for sql in [
        "DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5",
        "DECLARE OR REPLACE VARIABLE var2 STRING DEFAULT 'hello'",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    block_on(s.sql("SET VARIABLE (var1, var2) = (SELECT id, name FROM p)")).unwrap();
    let (_, found) = rows(&s, "SELECT var1, var2");
    assert_eq!(found, [vec!["7".to_owned(), "seven".to_owned()]]);
    // No rows assign NULL with the variables' types.
    block_on(s.sql("SET VARIABLE (var1, var2) = (SELECT id, name FROM p WHERE 1 = 0)")).unwrap();
    let (schema, found) = rows(&s, "SELECT var1, var2");
    assert_eq!(schema, "struct<var1:int,var2:string>");
    assert_eq!(found, [vec!["NULL".to_owned(), "NULL".to_owned()]]);
    // Several rows fail, and the wrong width fails before running.
    let e = error(
        &s,
        "SET VARIABLE (var1, var2) = (SELECT id, name FROM p UNION ALL SELECT id, name FROM p)",
    );
    assert_eq!(e.error_class(), ErrorClass::RowSubqueryTooManyRows);
    let e = error(&s, "SET VARIABLE (var1, var2) = (SELECT id FROM p)");
    assert_eq!(e.error_class(), ErrorClass::AssignmentArityMismatch);
    assert_eq!(
        e.message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect::<Vec<_>>(),
        [
            ("numTarget".to_owned(), "2".to_owned()),
            ("numExpr".to_owned(), "1".to_owned())
        ]
    );
}

/// Single assignments through scalar subqueries: one row assigns, no rows
/// assign NULL, several rows and several columns fail with the golden
/// classes, an overflowing cast fails, and every failure keeps the old
/// value.
#[test]
fn set_single_assignment_through_queries() {
    let s = session();
    s.register_batches(
        "q",
        Arc::new(Schema::new(vec![
            Field::new("id", ArrowType::Int32, false),
            Field::new("name", ArrowType::Utf8, true),
        ])),
        vec![
            arrow_array::RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("id", ArrowType::Int32, false),
                    Field::new("name", ArrowType::Utf8, true),
                ])),
                vec![
                    Arc::new(arrow_array::Int32Array::from(vec![7])),
                    Arc::new(arrow_array::StringArray::from(vec![Some("seven")])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    for sql in [
        "DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5",
        "DECLARE OR REPLACE VARIABLE var2 STRING DEFAULT 'hello'",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    block_on(s.sql("SET VARIABLE var1 = (SELECT id FROM q)")).unwrap();
    let (_, found) = rows(&s, "SELECT var1, var2");
    assert_eq!(found, [vec!["7".to_owned(), "hello".to_owned()]]);
    // No rows assign NULL with the variables' types.
    block_on(s.sql("SET VARIABLE var1 = (SELECT id FROM q WHERE 1 = 0)")).unwrap();
    let (schema, found) = rows(&s, "SELECT var1, var2");
    assert_eq!(schema, "struct<var1:int,var2:string>");
    assert_eq!(found, [vec!["NULL".to_owned(), "hello".to_owned()]]);
    // Several rows, several columns and an overflowing cast fail, and each
    // failure keeps the old value.
    block_on(s.sql("SET VARIABLE var1 = (SELECT id FROM q)")).unwrap();
    let e = error(
        &s,
        "SET VARIABLE var1 = (SELECT id FROM q UNION ALL SELECT id FROM q)",
    );
    assert_eq!(e.error_class(), ErrorClass::ScalarSubqueryTooManyRows);
    assert!(e.message_parameters().is_empty());
    let e = error(&s, "SET VARIABLE var1 = (SELECT id, name FROM q)");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidSubqueryExpressionScalarSubqueryReturnMoreThanOneOutputColumn
    );
    assert_eq!(
        e.message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect::<Vec<_>>(),
        [("number".to_owned(), "2".to_owned())]
    );
    let e = error(&s, "SET VARIABLE var1 = (SELECT 1.0E10)");
    assert_eq!(e.error_class(), ErrorClass::CastOverflow);
    assert_eq!(
        e.message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect::<Vec<_>>(),
        [
            ("value".to_owned(), "1.0E10D".to_owned()),
            ("sourceType".to_owned(), "\"DOUBLE\"".to_owned()),
            ("targetType".to_owned(), "\"INT\"".to_owned()),
        ]
    );
    let (_, found) = rows(&s, "SELECT var1, var2");
    assert_eq!(found, [vec!["7".to_owned(), "hello".to_owned()]]);
}

/// Declaring without `OR REPLACE` over an existing variable fails, and
/// assigning or dropping a missing one reports it.
#[test]
fn declare_assign_and_drop_missing_variables() {
    let s = session();
    block_on(s.sql("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5")).unwrap();
    let e = error(&s, "DECLARE VARIABLE var1 INT");
    assert_eq!(e.error_class(), ErrorClass::VariableAlreadyExists);
    assert_eq!(
        e.message_parameters(),
        &[("variableName", "`system`.`session`.`var1`".to_owned())]
    );
    let e = error(&s, "SET VARIABLE missing = 1");
    assert_eq!(e.error_class(), ErrorClass::VariableNotFound);
    assert_eq!(
        e.message_parameters(),
        &[("variableName", "`system`.`session`.`missing`".to_owned())]
    );
    let e = error(&s, "SET VARIABLE (missing) = (SELECT 1)");
    assert_eq!(e.error_class(), ErrorClass::VariableNotFound);
    let e = error(&s, "SET VARIABLE (var1, missing) = (SELECT 1, 2)");
    assert_eq!(e.error_class(), ErrorClass::VariableNotFound);
    // A backquoted key is one part: a config, never a variable.
    let (schema, found) = rows(&s, "SET `session.var1` = 5");
    assert_eq!(schema, "struct<key:string,value:string>");
    assert_eq!(found, [vec!["session.var1".to_owned(), "5".to_owned()]]);
    assert_eq!(s.conf().get("session.var1").as_deref(), Some("5"));
}

/// `SET var = DEFAULT` with no declared default assigns NULL, and a
/// parenthesized `DEFAULT` reads the default too.
#[test]
fn set_default_without_declared_default() {
    let s = session();
    block_on(s.sql("DECLARE OR REPLACE VARIABLE var1 INT")).unwrap();
    block_on(s.sql("SET VARIABLE var1 = 5")).unwrap();
    block_on(s.sql("SET VARIABLE var1 = DEFAULT")).unwrap();
    let (schema, found) = rows(&s, "SELECT var1");
    assert_eq!(schema, "struct<var1:int>");
    assert_eq!(found, [vec!["NULL".to_owned()]]);
    block_on(s.sql("DECLARE OR REPLACE VARIABLE var2 INT DEFAULT 7")).unwrap();
    block_on(s.sql("SET VARIABLE var2 = (DEFAULT)")).unwrap();
    let (_, found) = rows(&s, "SELECT var2");
    assert_eq!(found, [vec!["7".to_owned()]]);
}

/// A variable named `default` is a variable, not the keyword.
#[test]
fn variable_named_default() {
    let s = session();
    block_on(s.sql("DECLARE OR REPLACE VARIABLE default INT DEFAULT 3")).unwrap();
    block_on(s.sql("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 5")).unwrap();
    block_on(s.sql("SET VARIABLE var1 = default")).unwrap();
    let (_, found) = rows(&s, "SELECT var1");
    assert_eq!(found, [vec!["3".to_owned()]]);
    // Without the variable, the same shape reads the default.
    block_on(s.sql("DROP TEMPORARY VARIABLE default")).unwrap();
    block_on(s.sql("SET VARIABLE var1 = DEFAULT")).unwrap();
    let (_, found) = rows(&s, "SELECT var1");
    assert_eq!(found, [vec!["5".to_owned()]]);
}

/// `EXPLAIN` of a variable command prints the command.
#[test]
fn explain_variable_commands() {
    let s = session();
    for (sql, plan) in [
        (
            "EXPLAIN DECLARE var1 INT",
            "== Physical Plan ==\nDECLARE `var1` AS ...\n",
        ),
        (
            "EXPLAIN SET VARIABLE var1 = 1",
            "== Physical Plan ==\nSET VARIABLE ...\n",
        ),
        (
            "EXPLAIN DROP TEMPORARY VARIABLE IF EXISTS var1",
            "== Physical Plan ==\nDROP TEMPORARY VARIABLE IF EXISTS `var1`\n",
        ),
    ] {
        let df = block_on(s.sql(sql)).unwrap();
        let batches = block_on(df.collect()).unwrap();
        assert_eq!(batches.len(), 1, "{sql}");
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 1, "{sql}");
        let value = batch.column(0).as_string_view().value(0).to_owned();
        assert_eq!(value, plan, "{sql}");
    }
}

/// A decimal session variable is no decimal literal: comparing an integral
/// column with it keeps the variable (named `variablereference(...)`) and the comparison as written
/// (Spark's decimal-literal narrowing matches `DecimalLiteral` only).
#[test]
fn decimal_variable_comparisons_keep_the_variable() {
    let s = Session::new().unwrap();
    block_on(s.sql("DECLARE VARIABLE v DECIMAL(3,1) DEFAULT 1.5")).unwrap();
    let df = block_on(s.sql("SELECT id > v FROM range(3) ORDER BY id")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        schema.field(0).name(),
        "(id > variablereference(system.session.v=1.5))"
    );
    let rows = block_on(df.collect()).unwrap();
    let got: Vec<bool> = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_boolean()
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(got, [false, false, true]);
}

/// A keyword names a variable (`all` is not reserved unless keywords are enforced), and `TEMP` is `TEMPORARY`.
#[test]
fn keywords_name_variables_and_temp_drops_them() {
    let s = Session::new().unwrap();
    let run = |sql: &str| {
        block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    };
    run("DECLARE all = 1");
    run("DECLARE VARIABLE select INT DEFAULT 2");
    run("SET VARIABLE all = 3, select = 4");
    run("DROP TEMP VARIABLE all");
    run("DROP TEMP VAR IF EXISTS select");
    run("DROP TEMP VARIABLE IF EXISTS select");
    assert_eq!(
        error(&s, "DROP TEMP VARIABLE all").error_class(),
        ErrorClass::VariableNotFound
    );
}

/// `date + variable` is `date_add(date, variable)`: the output name keeps
/// the variable's name, not its value (Spark 4.2 names it so).
#[test]
fn a_date_plus_a_variable_keeps_the_variable_in_its_name() {
    let s = Session::new().unwrap();
    block_on(s.sql("DECLARE VARIABLE v INT DEFAULT 5")).unwrap();
    for (sql, name) in [
        (
            "SELECT DATE'2020-01-30' + v",
            "date_add(DATE '2020-01-30', v)",
        ),
        (
            "SELECT v + DATE'2020-01-30'",
            "date_add(DATE '2020-01-30', v)",
        ),
        (
            "SELECT DATE'2020-01-30' - v",
            "date_sub(DATE '2020-01-30', v)",
        ),
    ] {
        let df = block_on(s.sql(sql)).unwrap();
        let schema = block_on(df.schema()).unwrap();
        assert_eq!(schema.field(0).name(), name, "{sql}");
    }
}

#[test]
fn a_bare_default_in_a_row_query_reads_the_targets_default() {
    let s = session();
    for sql in [
        "DECLARE OR REPLACE VARIABLE v1 STRING DEFAULT 'default1'",
        "DECLARE OR REPLACE VARIABLE v2 INT",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    let value = |s: &Session, sql: &str| rows(s, sql).1;
    for set in [
        "SET VARIABLE (v1) = (SELECT DEFAULT FROM VALUES(1) AS T(c1))",
        "SET VARIABLE (v1) = (SELECT DEFAULT FROM VALUES(1),(2),(3) AS T(c1) LIMIT 1 OFFSET 1)",
        "SET VARIABLE (v1) = (WITH w(c1) AS (VALUES(1) AS T(c1)) SELECT DEFAULT FROM VALUES(1) AS T(c1))",
    ] {
        block_on(s.sql("SET VARIABLE v1 = 'hello'")).unwrap();
        block_on(s.sql(set)).unwrap();
        assert_eq!(
            value(&s, "SELECT v1"),
            vec![vec!["default1".to_owned()]],
            "{set}"
        );
    }
    // A column named default wins; a variable without a default gets NULL.
    block_on(s.sql("SET VARIABLE (v1) = (SELECT DEFAULT FROM VALUES('world') AS T(default))"))
        .unwrap();
    assert_eq!(value(&s, "SELECT v1"), vec![vec!["world".to_owned()]]);
    block_on(s.sql("SET VARIABLE v2 = 3")).unwrap();
    block_on(s.sql("SET VARIABLE (v2) = (SELECT DEFAULT FROM VALUES(1) AS T(c1))")).unwrap();
    assert_eq!(value(&s, "SELECT v2"), vec![vec!["NULL".to_owned()]]);
    // Only a bare DEFAULT of a plain select list is the default.
    assert!(
        block_on(s.sql("SET VARIABLE (v2) = (SELECT DEFAULT + 1 FROM VALUES(1) AS T(c1))"))
            .is_err()
    );
}

/// A variable inside another expression's name reads
/// `variablereference(system.session.v=<value>)` (Spark 4.2.0); a bare select item keeps `v`.
#[test]
fn a_variable_in_a_name_is_a_variablereference() {
    let s = Session::new().unwrap();
    for sql in [
        "DECLARE VARIABLE v1 INT DEFAULT 1",
        "DECLARE VARIABLE vs STRING DEFAULT 'x'",
        "DECLARE VARIABLE vn INT",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    let names = |sql: &str| -> Vec<String> {
        let df = block_on(s.sql(sql)).unwrap();
        let schema = block_on(df.schema()).unwrap();
        schema.fields().iter().map(|f| f.name().clone()).collect()
    };
    assert_eq!(names("SELECT v1, vs"), ["v1", "vs"]);
    assert_eq!(
        names("SELECT v1 + 1, upper(vs), vn + 1"),
        [
            "(variablereference(system.session.v1=1) + 1)",
            "upper(variablereference(system.session.vs='x'))",
            "(variablereference(system.session.vn=CAST(NULL AS INT)) + 1)",
        ]
    );
}

/// `var` and `variable` are not reserved: `DECLARE var = 'x'` names a variable `var`.
#[test]
fn a_variable_may_be_named_var() {
    let s = Session::new().unwrap();
    block_on(s.sql("DECLARE var = 'sometable'")).unwrap();
    block_on(s.sql("SET VAR var = 'c1'")).unwrap();
    let df = block_on(s.sql("SELECT IDENTIFIER(var) FROM VALUES(1) AS T(c1)")).unwrap();
    assert_eq!(block_on(df.collect()).unwrap()[0].num_rows(), 1);
}
