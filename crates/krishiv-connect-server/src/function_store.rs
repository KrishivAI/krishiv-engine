// SPDX-License-Identifier: Apache-2.0
//! [`SqliteFunctionStore`]: the persisted Python UDFs (P3-T072c) in the embedded SQLite catalog
//! (`krishiv_catalogs::SqliteCatalog`, table `python_functions`), one row per (tenant, name).

use std::sync::Arc;

use krishiv_catalogs::sqlite::{FunctionChange, SqliteCatalog, StoredPythonFunction};
use krishiv_common::error::KrishivError;
use krishiv_connect::translate::PythonUdf;
use sha2::{Digest, Sha256};

use crate::udf::{FunctionStore, StoreChange};

/// A [`FunctionStore`] over a SQLite catalog file.
#[derive(Debug, Clone)]
pub struct SqliteFunctionStore {
    catalog: Arc<SqliteCatalog>,
}

impl SqliteFunctionStore {
    /// The store over `catalog`.
    #[must_use]
    pub fn new(catalog: Arc<SqliteCatalog>) -> Self {
        Self { catalog }
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn store_change(c: FunctionChange) -> StoreChange {
    match c {
        FunctionChange::Done => StoreChange::Done,
        FunctionChange::Absent => StoreChange::Absent,
        FunctionChange::OwnedByOther => StoreChange::OwnedByOther,
    }
}

impl FunctionStore for SqliteFunctionStore {
    fn list(&self, tenant: Option<&str>) -> Result<Vec<PythonUdf>, KrishivError> {
        let mut udfs = Vec::new();
        for f in self.catalog.python_functions(tenant)? {
            // A payload that no longer matches its digest (a damaged or edited file) is not run.
            if hex(&Sha256::digest(&f.payload)) != f.digest {
                tracing::warn!(target: "krishiv::udf", function = %f.name, "a persisted Python UDF fails its digest");
                continue;
            }
            // A row that no longer converts (a newer build wrote it) is skipped, not fatal.
            match PythonUdf::from_stored(
                f.name.clone(),
                f.deterministic,
                f.eval_type,
                &f.return_type,
                f.payload,
                f.python_ver,
            ) {
                Ok(udf) => udfs.push(udf),
                Err(e) => tracing::warn!(
                    target: "krishiv::udf",
                    function = %f.name,
                    error = %e,
                    "a persisted Python UDF cannot be read"
                ),
            }
        }
        Ok(udfs)
    }

    fn put(
        &self,
        tenant: Option<&str>,
        owner: &str,
        udf: &PythonUdf,
        override_owner: bool,
    ) -> Result<StoreChange, KrishivError> {
        let digest = hex(&Sha256::digest(&udf.command));
        self.catalog
            .put_python_function(
                tenant,
                &StoredPythonFunction::new(
                    udf.name.clone(),
                    udf.eval_type,
                    udf.deterministic,
                    udf.return_type.json(),
                    udf.command.clone(),
                    digest,
                    udf.python_ver.clone(),
                    owner,
                ),
                override_owner,
            )
            .map(store_change)
    }

    fn owner(&self, tenant: Option<&str>, name: &str) -> Result<Option<String>, KrishivError> {
        Ok(self.catalog.python_function(tenant, name)?.map(|f| f.owner))
    }

    fn remove(
        &self,
        tenant: Option<&str>,
        name: &str,
        caller: &str,
        override_owner: bool,
    ) -> Result<StoreChange, KrishivError> {
        self.catalog
            .drop_python_function(tenant, name, caller, override_owner)
            .map(store_change)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_types::DataType;

    fn udf(name: &str) -> PythonUdf {
        PythonUdf::from_stored(name.into(), true, 101, "bigint", vec![9, 9], "3.12".into())
            .expect("udf")
    }

    #[test]
    fn a_struct_return_type_keeps_its_field_names_and_nullability() {
        let store = SqliteFunctionStore::new(Arc::new(
            SqliteCatalog::open_in_memory("f", false).expect("catalog"),
        ));
        let json = r#"{"type":"struct","fields":[{"name":"a b","type":"long","nullable":false,"metadata":{}}]}"#;
        let udf = PythonUdf::from_stored("s".into(), true, 101, json, vec![1], "3.12".into())
            .expect("udf");
        store.put(None, "alice", &udf, false).expect("put");
        let listed = store.list(None).expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].return_type, udf.return_type);
    }

    #[test]
    fn a_function_round_trips_per_tenant() {
        let store = SqliteFunctionStore::new(Arc::new(
            SqliteCatalog::open_in_memory("f", false).expect("catalog"),
        ));
        store
            .put(Some("acme"), "alice", &udf("Plus1"), false)
            .expect("put");
        let listed = store.list(Some("acme")).expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "plus1");
        assert_eq!(listed[0].return_type, DataType::Long);
        assert_eq!(listed[0].command, vec![9, 9]);
        assert!(store.list(Some("globex")).expect("list").is_empty());
        assert_eq!(
            store
                .remove(Some("acme"), "PLUS1", "alice", false)
                .expect("remove"),
            StoreChange::Done
        );
        assert_eq!(
            store
                .remove(Some("acme"), "plus1", "alice", false)
                .expect("remove"),
            StoreChange::Absent
        );
    }

    #[test]
    fn a_row_this_build_cannot_read_is_skipped() {
        let catalog = Arc::new(SqliteCatalog::open_in_memory("f", false).expect("catalog"));
        catalog
            .put_python_function(
                None,
                &StoredPythonFunction::new(
                    "bad",
                    999,
                    true,
                    "bigint",
                    vec![1],
                    hex(&Sha256::digest([1])),
                    "3.12",
                    "x",
                ),
                false,
            )
            .expect("put");
        // A row whose payload no longer matches its digest is skipped before conversion.
        catalog
            .put_python_function(
                None,
                &StoredPythonFunction::new(
                    "damaged",
                    101,
                    true,
                    "bigint",
                    vec![1],
                    "d",
                    "3.12",
                    "x",
                ),
                false,
            )
            .expect("put");
        let store = SqliteFunctionStore::new(catalog);
        store.put(None, "x", &udf("good"), false).expect("put");
        let names: Vec<_> = store
            .list(None)
            .expect("list")
            .into_iter()
            .map(|u| u.name)
            .collect();
        assert_eq!(names, ["good"]);
    }
}
