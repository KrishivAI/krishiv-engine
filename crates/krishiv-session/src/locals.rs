// SPDX-License-Identifier: Apache-2.0
//! The local variables of a running SQL script.
//!
//! A script's statements run on a *handle* of the session that carries the script's local frames
//! ([`Session::with_locals`]). Every command builds its [`QueryEnv`] from its handle, so each one
//! (a query, `INSERT ... VALUES (k)`, `CREATE TABLE ... AS SELECT k`, `SET session.v = k`,
//! `EXPLAIN SELECT k`, `CACHE TABLE ... AS SELECT k`) sees the locals without any command taking
//! them as a parameter. The frames are immutable values of one handle: a change to a local makes a
//! new handle, so a query that already started never sees one, and no other query of the session
//! (the handle is a clone; the shared state holds no frames) ever sees them.

use std::sync::Arc;

use krishiv_plan::analyzer::LocalFrame;

use crate::session::Session;

impl Session {
    /// A handle of this session whose queries also resolve `locals` (outermost frame first).
    pub(crate) fn with_locals(&self, locals: Vec<LocalFrame>) -> Session {
        Session {
            state: Arc::clone(&self.state),
            locals: locals.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use futures::executor::block_on;
    use krishiv_plan::Literal;
    use krishiv_plan::analyzer::SessionVariable;
    use krishiv_types::DataType;

    use super::*;

    fn frame(label: Option<&str>, vars: &[(&str, i32)]) -> LocalFrame {
        LocalFrame {
            label: label.map(str::to_owned),
            variables: vars
                .iter()
                .map(|(k, v)| {
                    (
                        (*k).to_owned(),
                        SessionVariable {
                            data_type: DataType::Integer,
                            value: Literal::Integer(*v),
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        }
    }

    fn first_int(session: &Session, sql: &str) -> Result<i64, String> {
        let batches = block_on(async { session.sql(sql).await?.collect().await })
            .map_err(|e| e.error_class().name().to_owned())?;
        let column = batches[0].column(0);
        Ok(match column.data_type() {
            arrow_schema::DataType::Int32 => i64::from(
                column
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .value(0),
            ),
            arrow_schema::DataType::Int64 => column
                .as_primitive::<arrow_array::types::Int64Type>()
                .value(0),
            other => return Err(format!("unexpected {other}")),
        })
    }

    #[test]
    fn only_the_handle_with_frames_sees_them() {
        let session = Session::new().unwrap();
        let handle = session.with_locals(vec![frame(Some("lbl"), &[("k", 5)])]);
        assert_eq!(first_int(&handle, "SELECT k + 1"), Ok(6));
        assert_eq!(first_int(&handle, "SELECT lbl.k"), Ok(5));
        assert_eq!(
            first_int(&session, "SELECT k + 1"),
            Err("UNRESOLVED_COLUMN.WITHOUT_SUGGESTION".to_owned())
        );
        // A handle made from a handle replaces the frames; the first is unchanged.
        let next = handle.with_locals(vec![frame(Some("lbl"), &[("k", 9)])]);
        assert_eq!(first_int(&next, "SELECT k"), Ok(9));
        assert_eq!(first_int(&handle, "SELECT k"), Ok(5));
    }

    #[test]
    fn every_command_builds_its_env_from_its_handle() {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::new().unwrap();
        session
            .conf()
            .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
            .unwrap();
        let handle = session.with_locals(vec![frame(None, &[("k", 7)])]);
        let run = |sql: &str| {
            block_on(async { handle.sql(sql).await?.collect().await })
                .unwrap_or_else(|e| panic!("{sql}: {e}"));
        };
        run("CREATE TABLE t (a INT) USING parquet");
        run("INSERT INTO t VALUES (k), (k + 1)");
        run("CREATE TABLE u USING parquet AS SELECT k AS a");
        run("DECLARE sv INT");
        run("SET VARIABLE sv = k");
        run("EXPLAIN SELECT k");
        run("CREATE TEMPORARY VIEW v AS SELECT 1 AS a");
        run("INSERT INTO t SELECT a + k FROM u");
        assert_eq!(first_int(&session, "SELECT sum(a) FROM t"), Ok(7 + 8 + 14));
        assert_eq!(first_int(&session, "SELECT a FROM u"), Ok(7));
        assert_eq!(first_int(&session, "SELECT sv"), Ok(7));
    }
}
