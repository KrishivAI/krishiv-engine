// SPDX-License-Identifier: Apache-2.0
//! The namespace a command names: `CREATE`/`USE`, the commands of Spark's v1 session catalog and
//! `COMMENT ON SCHEMA`, which each refuse other shapes of a name in their own way.

use krishiv_common::error::{
    KrishivError, MissingDatabaseForV1SessionCatalog, NestedDatabaseUnsupportedByV1SessionCatalog,
    RequiresSinglePartNamespace, SchemaNotFound,
};
use krishiv_sql::ast::DdlName;

use super::unsupported;
use crate::catalog::CATALOG;
use crate::session::{QueryEnv, Session};

/// The last part of a namespace name as written: what Spark's `USE`, `DESCRIBE` and `DROP` name
/// when the namespace is missing.
pub(super) fn written_namespace(name: &DdlName, env: &QueryEnv) -> Result<String, KrishivError> {
    Ok(crate::access::ddl_parts(name, env)?
        .pop()
        .unwrap_or_default())
}

/// `` `a`.`b` ``.
fn quoted<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    parts
        .into_iter()
        .map(|p| format!("`{}`", p.replace('`', "``")))
        .collect::<Vec<_>>()
        .join(".")
}

impl Session {
    /// The namespace named by `name` (one part, or the session catalog and one part).
    pub(crate) fn namespace_name(
        &self,
        name: &DdlName,
        env: &QueryEnv,
    ) -> Result<String, KrishivError> {
        let parts = crate::access::ddl_parts(name, env)?;
        let fold = |s: &str| {
            if env.analysis.conf.case_sensitive {
                s.to_owned()
            } else {
                s.to_lowercase()
            }
        };
        match parts.as_slice() {
            [ns] => self.namespace_arg(&fold(ns)),
            [catalog, ns] if catalog.eq_ignore_ascii_case(crate::catalog::CATALOG) => Ok(fold(ns)),
            [catalog, _] => Err(unsupported(&format!(
                "namespaces in the catalog `{catalog}`"
            ))),
            _ => Err(RequiresSinglePartNamespace {
                session_catalog: crate::catalog::CATALOG.to_owned(),
                identifier: format!("`{}`", parts.join("`.`")),
            }
            .into()),
        }
    }

    /// The database a command of Spark's v1 session catalog names (`DESCRIBE`, `DROP` and
    /// `ALTER NAMESPACE`, `SHOW ... IN`, `ANALYZE TABLES`; Spark 4.3.0 run): a leading
    /// `spark_catalog` is the catalog, and exactly one part names a database. A declared
    /// catalog's and `system`'s names are read as [`Self::namespace_name`] reads them.
    pub(crate) fn database_name(
        &self,
        name: &DdlName,
        env: &QueryEnv,
    ) -> Result<String, KrishivError> {
        let parts = crate::access::ddl_parts(name, env)?;
        let nested = match parts.as_slice() {
            [catalog] if catalog.eq_ignore_ascii_case(CATALOG) => {
                return Err(MissingDatabaseForV1SessionCatalog {}.into());
            }
            [catalog, rest @ ..] if catalog.eq_ignore_ascii_case(CATALOG) => rest,
            [first, _, ..] if !self.is_catalog_name(first) => {
                self.refuse_declared_current()?;
                parts.as_slice()
            }
            _ => return self.namespace_name(name, env),
        };
        if let [_] = nested {
            return self.namespace_name(name, env);
        }
        Err(NestedDatabaseUnsupportedByV1SessionCatalog {
            namespace: quoted(nested.iter().map(String::as_str)),
        }
        .into())
    }

    /// The namespace of `COMMENT ON SCHEMA`: anything but one database of the session catalog is
    /// a missing schema, named with every part (Spark 4.3.0 run).
    pub(super) fn commented_namespace(
        &self,
        name: &DdlName,
        env: &QueryEnv,
    ) -> Result<String, KrishivError> {
        let parts = crate::access::ddl_parts(name, env)?;
        let rest = match parts.as_slice() {
            [catalog, rest @ ..] if catalog.eq_ignore_ascii_case(CATALOG) => rest,
            [first, _, ..] if !self.is_catalog_name(first) => {
                self.refuse_declared_current()?;
                parts.as_slice()
            }
            _ => return self.namespace_name(name, env),
        };
        if let [_] = rest {
            return self.namespace_name(name, env);
        }
        Err(SchemaNotFound {
            schema_name: quoted(std::iter::once(CATALOG).chain(rest.iter().map(String::as_str))),
        }
        .into())
    }
}
