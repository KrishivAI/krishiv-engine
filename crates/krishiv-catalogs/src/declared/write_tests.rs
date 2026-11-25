// SPDX-License-Identifier: Apache-2.0
//! Writes into declared Iceberg REST tables (PR-36c part 2) over real loopback HTTP: a fake REST
//! catalog that applies each `CommitTableRequest` as a server does (requirements checked, updates
//! applied by the connector's test helper, the metadata file written), a provider opened through
//! the declared catalog with `option.write = true`, an append committed through it and read back
//! by SQL; a lost 200 found by the `snapshots=all` probe; a 409 after a foreign commit rebased;
//! read-only by default.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::executor::block_on;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_connectors_iceberg::testing::{
    LocalIo, WrittenTable, append, apply_commit, write_table,
};
use krishiv_exec::planner::TableProvider;
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::{
    CallerCatalog, CatalogCaller, CatalogFactory, ScanInputs, ScopedStores,
};
use krishiv_store::egress::PrivateAddresses;

use super::IcebergRestFactory;
use super::read_tests::{pairs, rows, session};
use super::tests::{FORM, decl, deps, ident, path};
use crate::iceberg_rest::test_server::{Reply, Seen, Server, serve};

const TABLE: &str = "/v1/namespaces/db/tables/t";

/// What the fake catalog does with the next commit.
#[derive(Debug, Clone, Copy)]
pub(super) enum Fault {
    /// Applied and answered.
    Answer,
    /// Applied; the connection closes without an answer.
    LostAfterApply,
}

/// The catalog's one table.
pub(super) struct Lake {
    pub(super) metadata: serde_json::Value,
    location: String,
    root: String,
    version: u32,
    script: VecDeque<Fault>,
}

impl Lake {
    fn load_body(&self) -> String {
        serde_json::json!({
            "metadata-location": self.location,
            "metadata": self.metadata,
            "config": {},
        })
        .to_string()
    }

    fn commit(&mut self, body: &[u8]) -> Reply {
        let fault = self.script.pop_front().unwrap_or(Fault::Answer);
        self.version += 1;
        let next = format!(
            "{}/metadata/v{}-rest.metadata.json",
            self.root, self.version
        );
        match apply_commit(&self.metadata, &self.location, body, &next) {
            Ok(m) => {
                self.metadata = m;
                self.location = next;
                match fault {
                    Fault::Answer => Reply::Json(200, self.load_body()),
                    Fault::LostAfterApply => Reply::Raw(Vec::new()),
                }
            }
            Err(refusal) => Reply::Json(refusal.status, refusal.body()),
        }
    }
}

/// A REST catalog over `table` with `script` for its commits.
pub(super) fn catalog(table: &WrittenTable, script: &[Fault]) -> (Server, Arc<Mutex<Lake>>) {
    let lake = Arc::new(Mutex::new(Lake {
        metadata: table.metadata.clone(),
        location: table.metadata_location.clone(),
        root: table.location.clone(),
        version: 100,
        script: script.iter().copied().collect(),
    }));
    let state = Arc::clone(&lake);
    let server = serve(move |r: &Seen| match (r.method.as_str(), path(r)) {
        (_, "/v1/config") => Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()),
        ("GET", TABLE) => Reply::Json(200, state.lock().unwrap().load_body()),
        ("POST", TABLE) => state.lock().unwrap().commit(r.body.as_bytes()),
        _ => Reply::Json(
            404,
            r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#.into(),
        ),
    });
    (server, lake)
}

/// The declared catalog `lake` on `server` (`option.write` when given), its loads never cached.
fn built(server: &Server, write: Option<&str>) -> Result<Arc<dyn CallerCatalog>, KrishivError> {
    let mut pairs = vec![
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", server.base.as_str()),
        ("krishiv.catalog.lake.storage", "allowed-prefixes"),
    ];
    if let Some(w) = write {
        pairs.push(("krishiv.catalog.lake.option.write", w));
    }
    let d = decl(&pairs, DeclMode::Embedded);
    let deps = deps(DeclMode::Embedded, PrivateAddresses::Allow);
    IcebergRestFactory.validate(&d, &deps)?;
    IcebergRestFactory::build_with_ttl(&d, &deps, Duration::ZERO)
}

/// `db.t` of `catalog` opened as a provider, at its current version.
fn open(catalog: &Arc<dyn CallerCatalog>, io: &LocalIo) -> Arc<dyn TableProvider> {
    let loaded = block_on(catalog.load_declared(
        &CatalogCaller::server("lake"),
        &ident(&["db"], "t"),
        None,
        FORM,
    ))
    .unwrap()
    .unwrap();
    let config = Arc::new(Config::new());
    let stores = ScopedStores::new(Arc::clone(&config), Arc::clone(io.io().scheduler()), "/");
    let inputs = ScanInputs::new(io.io(), io.paths(), config, 1, stores);
    block_on(loaded.table.unwrap().open(inputs)).unwrap()
}

fn commits(server: &Server) -> usize {
    server
        .seen()
        .iter()
        .filter(|s| s.method == "POST" && path(s) == TABLE)
        .count()
}

fn probes(server: &Server) -> Vec<String> {
    server
        .seen()
        .iter()
        .filter(|s| s.method == "GET" && path(s) == TABLE && s.target.contains('?'))
        .map(|s| s.target.clone())
        .collect()
}

fn read_back(server: &Server) -> Vec<(i64, String)> {
    rows(&session(server, None), "SELECT id, name FROM lake.db.t")
}

/// An append into a writable declared table: one `CommitTableRequest` (identifier, the
/// `assert-ref-snapshot-id` of the snapshot the table was opened at), committed, and the rows
/// read back through SQL.
#[test]
fn an_append_commits_through_the_rest_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let (server, lake) = catalog(&table, &[]);
    let io = LocalIo::new(dir.path()).unwrap();
    let cat = built(&server, Some("true")).unwrap();
    let provider = open(&cat, &io);
    assert!(provider.capabilities().batch_write);
    append(&*provider, &[(4, "d")]).unwrap();
    assert_eq!(commits(&server), 1);
    let post = server
        .seen()
        .into_iter()
        .find(|s| s.method == "POST")
        .unwrap();
    let body: serde_json::Value = serde_json::from_str(&post.body).unwrap();
    assert_eq!(
        body["identifier"],
        serde_json::json!({"namespace": ["db"], "name": "t"})
    );
    let head = table.snapshots[1];
    assert!(
        body["requirements"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["type"] == "assert-ref-snapshot-id" && r["snapshot-id"] == head),
        "{body}"
    );
    assert_eq!(
        lake.lock().unwrap().metadata["snapshots"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        read_back(&server),
        pairs(&[(1, "a"), (2, "b"), (3, "c"), (4, "d")])
    );
}

/// The 200 is lost after the catalog applied the commit: the loop probes with `snapshots=all`,
/// finds its snapshot and stops; nothing is sent twice.
#[test]
fn a_lost_200_is_found_by_the_snapshots_all_probe() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let (server, _) = catalog(&table, &[Fault::LostAfterApply]);
    let io = LocalIo::new(dir.path()).unwrap();
    let cat = built(&server, Some("true")).unwrap();
    append(&*open(&cat, &io), &[(4, "d")]).unwrap();
    assert_eq!(commits(&server), 1);
    assert_eq!(probes(&server), [format!("{TABLE}?snapshots=all")]);
    assert_eq!(
        read_back(&server),
        pairs(&[(1, "a"), (2, "b"), (3, "c"), (4, "d")])
    );
}

/// Two writers opened at the same snapshot: the second's first request fails its
/// `assert-ref-snapshot-id` (409), the loop refreshes with `snapshots=all` and rebases onto the
/// first writer's snapshot; both appends are in the table once.
#[test]
fn a_409_after_a_foreign_commit_is_rebased() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let (server, _) = catalog(&table, &[]);
    let io = LocalIo::new(dir.path()).unwrap();
    let cat = built(&server, Some("true")).unwrap();
    let first = open(&cat, &io);
    let second = open(&cat, &io);
    append(&*first, &[(4, "d")]).unwrap();
    append(&*second, &[(5, "e")]).unwrap();
    assert_eq!(commits(&server), 3);
    assert_eq!(probes(&server), [format!("{TABLE}?snapshots=all")]);
    assert_eq!(
        read_back(&server),
        pairs(&[(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")])
    );
}

/// A landed commit drops the table's cached `loadTable` answer: with a 30 s TTL a writer opened
/// after a commit starts from the metadata that commit left (no 409, no probe).
#[test]
fn a_landed_commit_drops_the_cached_load() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let (server, _) = catalog(&table, &[]);
    let io = LocalIo::new(dir.path()).unwrap();
    let d = decl(
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            ("krishiv.catalog.lake.uri", server.base.as_str()),
            ("krishiv.catalog.lake.storage", "allowed-prefixes"),
            ("krishiv.catalog.lake.option.write", "true"),
        ],
        DeclMode::Embedded,
    );
    let deps = deps(DeclMode::Embedded, PrivateAddresses::Allow);
    IcebergRestFactory.validate(&d, &deps).unwrap();
    let cat = IcebergRestFactory::build_with_ttl(&d, &deps, Duration::from_secs(30)).unwrap();
    let first = open(&cat, &io);
    append(&*first, &[(4, "d")]).unwrap();
    let second = open(&cat, &io);
    append(&*second, &[(5, "e")]).unwrap();
    assert_eq!(commits(&server), 2);
    assert!(probes(&server).is_empty());
}

/// Without `option.write` (or with `false`) the tables stay read-only: no write capability, no
/// writer; a value other than true/false is refused at startup.
#[test]
fn declared_tables_are_read_only_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let table = write_table(dir.path()).unwrap();
    let (server, _) = catalog(&table, &[]);
    let io = LocalIo::new(dir.path()).unwrap();
    for write in [None, Some("false")] {
        let cat = built(&server, write).unwrap();
        let provider = open(&cat, &io);
        assert!(!provider.capabilities().batch_write);
        assert!(
            provider
                .row_level(krishiv_exec::planner::RowLevelOperation::Delete)
                .is_none()
        );
        let e = append(&*provider, &[(4, "d")]).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation
        );
    }
    assert_eq!(commits(&server), 0);
    let e = built(&server, Some("maybe")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert!(e.to_string().contains("option.write"), "{e}");
}
