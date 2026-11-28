// SPDX-License-Identifier: Apache-2.0
//! `CREATE FUNCTION ... USING PYTHON` over a persisted Python UDF (P3-T072d).

use krishiv_common::error::KrishivError;

use super::{RoutineAlreadyExists, quote, unsupported};
use crate::dataframe::DataFrame;
use crate::session::Session;

impl Session {
    /// `CREATE [OR REPLACE] FUNCTION name AS 'python' USING PYTHON 'registered'` (P3-T072d): keeps
    /// the Python UDF the session registered as `registered` in the server's function catalog,
    /// under that name.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` without a function catalog, for a name other than the
    /// registered one or a UDF the session did not register; `ROUTINE_ALREADY_EXISTS` for a
    /// stored function without `OR REPLACE` or `IF NOT EXISTS`.
    pub(crate) async fn exec_create_persisted_function(
        &self,
        f: &krishiv_sql::ast::CreatePersistedFunction,
    ) -> Result<DataFrame, KrishivError> {
        let Some(hook) = self.persisted_functions() else {
            return Err(unsupported(
                "persisting Python UDFs in a session without a function catalog",
            ));
        };
        let [part] = &f.name.parts[..] else {
            return Err(unsupported("qualified names of persisted Python UDFs"));
        };
        if !part.value.eq_ignore_ascii_case(&f.registered) {
            return Err(unsupported(
                "a persisted Python UDF under another name than the one it was registered with",
            ));
        }
        if !f.or_replace && hook.exists(self, &part.value).await? {
            if f.if_not_exists {
                return self.empty_frame().await;
            }
            return Err(RoutineAlreadyExists {
                new_routine_type: "function".to_owned(),
                routine_name: quote(&part.value),
                existing_routine_type: "function".to_owned(),
            }
            .into());
        }
        hook.persist(self, &f.registered).await?;
        self.empty_frame().await
    }
}
