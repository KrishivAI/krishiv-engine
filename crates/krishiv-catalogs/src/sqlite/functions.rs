// SPDX-License-Identifier: Apache-2.0
//! Python UDFs kept in the catalog file (P3-T072c): one row per (tenant, name), so a tenant's later sessions, also
//! after a restart, find the function a client registered with `persist`. The payload is client code the engine never
//! unpickles; the row carries its digest, eval type, the client's Python version and the principal that stored it.

use krishiv_common::error::KrishivError;
use rusqlite::{OptionalExtension, params};

use super::SqliteCatalog;
use super::store::{Failure, bump_epoch};

/// The longest payload one stored function may have (bytes): a bound on what a client can make the file hold.
pub(super) const MAX_STORED_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;

/// A Python UDF as the catalog keeps it.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StoredPythonFunction {
    /// The function's name, as registered (matched without regard to case when read back).
    pub name: String,
    /// Spark's eval type, as sent.
    pub eval_type: i32,
    /// `deterministic`.
    pub deterministic: bool,
    /// The return type as a Spark catalog string.
    pub return_type: String,
    /// The cloudpickled payload.
    pub payload: Vec<u8>,
    /// Lower-case hex SHA-256 of the payload.
    pub digest: String,
    /// The client's Python `major.minor`.
    pub python_ver: String,
    /// The principal that stored it.
    pub owner: String,
    /// The catalog epoch of the change that made it what it is.
    pub version: i64,
}

impl std::fmt::Debug for StoredPythonFunction {
    // No payload bytes: client code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredPythonFunction")
            .field("name", &self.name)
            .field("eval_type", &self.eval_type)
            .field("digest", &self.digest)
            .field("payload_len", &self.payload.len())
            .field("owner", &self.owner)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl StoredPythonFunction {
    /// A function to store; `version` is set by the catalog.
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "the stored row's columns")]
    pub fn new(
        name: impl Into<String>,
        eval_type: i32,
        deterministic: bool,
        return_type: impl Into<String>,
        payload: Vec<u8>,
        digest: impl Into<String>,
        python_ver: impl Into<String>,
        owner: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            eval_type,
            deterministic,
            return_type: return_type.into(),
            payload,
            digest: digest.into(),
            python_ver: python_ver.into(),
            owner: owner.into(),
            version: 0,
        }
    }
}

fn key(tenant: Option<&str>) -> &str {
    tenant.unwrap_or("")
}

impl SqliteCatalog {
    /// Stores `function` for `tenant` (`None`: the server's one tenant), replacing a function of the same name.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`, also for a payload over `MAX_STORED_PAYLOAD_BYTES`.
    pub fn put_python_function(
        &self,
        tenant: Option<&str>,
        function: &StoredPythonFunction,
    ) -> Result<(), KrishivError> {
        if function.payload.len() > MAX_STORED_PAYLOAD_BYTES {
            return Err(self.fault(
                &function.name,
                Failure::Sql(format!(
                    "the payload of {} bytes is over the {MAX_STORED_PAYLOAD_BYTES} byte limit",
                    function.payload.len()
                )),
            ));
        }
        let name = self.fold(&function.name);
        self.change(&function.name, |tx| {
            let version = bump_epoch(tx)?;
            tx.execute(
                "INSERT OR REPLACE INTO python_functions \
                 (tenant, name, eval_type, deterministic, return_type, payload, digest, python_ver, owner, version) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    key(tenant),
                    name,
                    function.eval_type,
                    function.deterministic,
                    function.return_type,
                    function.payload,
                    function.digest,
                    function.python_ver,
                    function.owner,
                    version,
                ],
            )?;
            Ok(((), true))
        })
    }

    /// Removes `tenant`'s function `name`; whether there was one.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`.
    pub fn drop_python_function(
        &self,
        tenant: Option<&str>,
        name: &str,
    ) -> Result<bool, KrishivError> {
        let name = self.fold(name);
        self.change(&name, |tx| {
            let n = tx.execute(
                "DELETE FROM python_functions WHERE tenant = ?1 AND name = ?2",
                params![key(tenant), name],
            )?;
            if n > 0 {
                bump_epoch(tx)?;
            }
            Ok((n > 0, n > 0))
        })
    }

    /// `tenant`'s function `name`, if stored.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`.
    pub fn python_function(
        &self,
        tenant: Option<&str>,
        name: &str,
    ) -> Result<Option<StoredPythonFunction>, KrishivError> {
        let name = self.fold(name);
        self.read(&name, |conn| {
            Ok(conn
                .query_row(
                    "SELECT name, eval_type, deterministic, return_type, payload, digest, python_ver, owner, version \
                     FROM python_functions WHERE tenant = ?1 AND name = ?2",
                    params![key(tenant), name],
                    row_of,
                )
                .optional()?)
        })
    }

    /// Every function stored for `tenant`, by name.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`.
    pub fn python_functions(
        &self,
        tenant: Option<&str>,
    ) -> Result<Vec<StoredPythonFunction>, KrishivError> {
        self.read("python_functions", |conn| {
            let mut stmt = conn.prepare(
                "SELECT name, eval_type, deterministic, return_type, payload, digest, python_ver, owner, version \
                 FROM python_functions WHERE tenant = ?1 ORDER BY name",
            )?;
            let rows = stmt.query_map(params![key(tenant)], row_of)?;
            Ok(rows.collect::<Result<Vec<_>, _>>()?)
        })
    }
}

fn row_of(r: &rusqlite::Row<'_>) -> rusqlite::Result<StoredPythonFunction> {
    Ok(StoredPythonFunction {
        name: r.get(0)?,
        eval_type: r.get(1)?,
        deterministic: r.get(2)?,
        return_type: r.get(3)?,
        payload: r.get(4)?,
        digest: r.get(5)?,
        python_ver: r.get(6)?,
        owner: r.get(7)?,
        version: r.get(8)?,
    })
}
