// SPDX-License-Identifier: Apache-2.0
//! An Iceberg table on `abfss://` (lakehouse benchmark gate 2, `design/P3-lakehouse-benchmark-iceberg.md`
//! item 5): written to Azurite with the account key through iceberg-rust's in-memory catalog (metadata,
//! manifest list and manifest through Krishiv's I/O, the Parquet file put here), then loaded, planned and
//! scanned through SQL by a second scheduler whose only credential is an Azure bearer token (the Hadoop
//! ABFS keys `fs.azure.account.auth.type.<host> = OAuth`, `fs.azure.account.oauth2.access.token.<host>`).
//!
//! Needs Azurite in OAuth mode (`--oauth basic`, https): `KRISHIV_TEST_AZURE_OAUTH_ENDPOINT`, container
//! `krishiv-test`, the token `KRISHIV_TEST_AZURE_BEARER` and `SSL_CERT_FILE` trusting the emulator's
//! certificate; `tests/scripts/object_store_emulators.sh` makes all of them. Without the endpoint the test
//! checks only that the reader's paths place an `abfss://` location with no I/O.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use iceberg::spec::Struct;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;
use krishiv_exec::files::{ObjectPaths, PathLocator as _};
use krishiv_io::{IoConfig, IoScheduler, StoreId, StoreRegistry};

use super::provider::IcebergTableProvider;
use super::provider_tests::{
    batch, data_file, id_field, int64s, ixn_schema, provider_session, strings,
};
use super::storage::{KrishivStorageFactory, StorageRegistration};
use super::table::IcebergTable;
use crate::plan::VersionSpec;

/// Azurite's published development account key (public, fixed by Microsoft; not a secret).
const AZURITE_KEY: &str =
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==";
const HOST: &str = "devstoreaccount1.dfs.core.windows.net";

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn config(pairs: &[(&str, &str)]) -> Arc<Config> {
    let mut full = vec![("krishiv.io.requestTimeout", "10s")];
    full.extend(pairs.iter().copied());
    Arc::new(Config::from_pairs(full).unwrap().0)
}

/// A scheduler on `rt`'s I/O side whose cloud paths resolve with `config`, and a registration over it.
fn registration(
    rt: &tokio::runtime::Runtime,
    config: &Arc<Config>,
) -> (Arc<IoScheduler>, Arc<ObjectPaths>, StorageRegistration) {
    let sched = IoScheduler::with_runtime(IoConfig::from_config(config), rt.handle().clone());
    let paths = Arc::new(ObjectPaths::new(
        StoreId::new("file://"),
        "/",
        Arc::clone(&sched),
        Arc::new(StoreRegistry::new(Arc::clone(config))),
    ));
    let reg = StorageRegistration::new(sched.context(), Arc::clone(&paths) as _);
    (sched, paths, reg)
}

fn parquet(b: &arrow_array::RecordBatch) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut w = parquet::arrow::ArrowWriter::try_new(&mut buf, b.schema(), None).unwrap();
    w.write(b).unwrap();
    w.close().unwrap();
    buf
}

#[test]
fn iceberg_scan_over_abfss_with_a_bearer_token() {
    let (auth_key, token_key, account_key) = (
        format!("fs.azure.account.auth.type.{HOST}"),
        format!("fs.azure.account.oauth2.access.token.{HOST}"),
        format!("fs.azure.account.key.{HOST}"),
    );
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let Some(endpoint) = env_nonempty("KRISHIV_TEST_AZURE_OAUTH_ENDPOINT") else {
        // Offline: a bearer-token reader places an abfss:// location without I/O.
        let reader = config(&[
            ("fs.azure.endpoint", "https://127.0.0.1:1/devstoreaccount1"),
            (auth_key.as_str(), "OAuth"),
            (token_key.as_str(), "header.payload.signature"),
        ]);
        let (_sched, paths, _reg) = registration(&rt, &reader);
        let at = paths
            .locate(&format!(
                "abfss://krishiv-test@{HOST}/w/db/t/metadata/v1.metadata.json"
            ))
            .unwrap();
        assert_eq!(at.path.as_ref(), "w/db/t/metadata/v1.metadata.json");
        return;
    };
    // With the endpoint set, missing tokens are a broken setup, not an offline run.
    let token = env_nonempty("KRISHIV_TEST_AZURE_BEARER").unwrap_or_else(|| {
        panic!("KRISHIV_TEST_AZURE_OAUTH_ENDPOINT is set without KRISHIV_TEST_AZURE_BEARER")
    });
    let other_audience = env_nonempty("KRISHIV_TEST_AZURE_BEARER_OTHER_AUDIENCE").unwrap_or_else(|| {
        panic!("KRISHIV_TEST_AZURE_OAUTH_ENDPOINT is set without KRISHIV_TEST_AZURE_BEARER_OTHER_AUDIENCE")
    });
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let warehouse = format!(
        "abfss://krishiv-test@{HOST}/krishiv-test/iceberg-{}-{nanos}",
        std::process::id()
    );

    // Writer: the account key, the in-memory catalog's files through Krishiv's I/O.
    let writer = config(&[
        ("fs.azure.endpoint", endpoint.as_str()),
        (account_key.as_str(), AZURITE_KEY),
    ]);
    let (wsched, wpaths, wreg) = registration(&rt, &writer);
    let location = rt.block_on(async {
        let catalog = iceberg::memory::MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(KrishivStorageFactory {
                name: wreg.name().to_owned(),
            }))
            .with_runtime(wreg.runtime().unwrap())
            .load(
                "mem",
                [(
                    iceberg::memory::MEMORY_CATALOG_WAREHOUSE.to_owned(),
                    warehouse.clone(),
                )]
                .into(),
            )
            .await
            .unwrap();
        let db = NamespaceIdent::new("db".to_owned());
        catalog.create_namespace(&db, HashMap::new()).await.unwrap();
        let table = catalog
            .create_table(
                &db,
                TableCreation::builder()
                    .name("t".to_owned())
                    .schema(ixn_schema())
                    .build(),
            )
            .await
            .unwrap();
        let rows = batch(
            vec![
                id_field("id", arrow_schema::DataType::Int64, 1),
                id_field("x", arrow_schema::DataType::Int32, 2),
                id_field("name", arrow_schema::DataType::Utf8, 3),
            ],
            vec![
                int64s(&[1, 2, 3]),
                Arc::new(arrow_array::Int32Array::from(vec![10, 20, 30])),
                strings(&["a", "b", "c"]),
            ],
        );
        let bytes = parquet(&rows);
        let file = format!("{}/data/f0.parquet", table.metadata().location());
        let at = wpaths.locate(&file).unwrap();
        wsched
            .context()
            .put_object(&at.store, &at.path, Bytes::from(bytes.clone()))
            .await
            .unwrap();
        let tx = iceberg::transaction::Transaction::new(&table);
        let tx = iceberg::transaction::ApplyTransactionAction::apply(
            tx.fast_append().add_data_files(vec![data_file(
                &file,
                bytes.len() as u64,
                3,
                Struct::empty(),
            )]),
            tx,
        )
        .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        table.metadata_location().unwrap().to_owned()
    });
    assert!(location.starts_with("abfss://"), "{location}");

    // Reader: only the bearer token. Metadata, manifests and the data file all come through it.
    let bearer = |token: &str| {
        config(&[
            ("fs.azure.endpoint", endpoint.as_str()),
            (auth_key.as_str(), "OAuth"),
            (token_key.as_str(), token),
        ])
    };
    // Azurite checks the audience: a token for another one cannot load the table, so the
    // scan below depends on the bearer token, not on an open emulator.
    let (_osched, _opaths, oreg) = registration(&rt, &bearer(&other_audience));
    let refused = rt
        .block_on(IcebergTable::load(&oreg, &location, "db", "t"))
        .err()
        .unwrap();
    assert_eq!(
        refused.error_class(),
        ErrorClass::KrishivObjectStoreRequestFailed,
        "{refused}"
    );
    let (_rsched, _rpaths, rreg) = registration(&rt, &bearer(&token));
    let table = rt
        .block_on(IcebergTable::load(&rreg, &location, "db", "t"))
        .unwrap();
    let provider = rt
        .block_on(IcebergTableProvider::open(
            &table,
            &VersionSpec::Current,
            &Config::new(),
            4,
        ))
        .unwrap();
    let session = provider_session(provider);
    assert_eq!(
        session.rows("SELECT id, x, name FROM t WHERE x > 10 ORDER BY id"),
        vec!["2|20|b", "3|30|c"]
    );
    // The writer's registration lives to here so the catalog's `FileIO` name resolves for the whole write.
    drop(wreg);
}
