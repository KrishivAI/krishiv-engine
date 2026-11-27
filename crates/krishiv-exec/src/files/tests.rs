// SPDX-License-Identifier: Apache-2.0
//! Tests of `files.rs` (split out to keep it under 2,000 lines).
use super::*;

#[test]
fn an_overlapped_data_column_keeps_its_place_as_the_partition_column() {
    use arrow_schema::{DataType as A, Field, Schema};
    let files: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("a", A::Utf8View, true),
        Field::new("K", A::Utf8View, true),
        Field::new("v", A::Utf8View, true),
    ]));
    let parts: Vec<FieldRef> = vec![
        Arc::new(Field::new("k", A::Int32, true)),
        Arc::new(Field::new("j", A::Int32, true)),
    ];
    let names = |s: &SchemaRef| {
        s.fields()
            .iter()
            .map(|f| format!("{}:{}", f.name(), f.data_type()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names(&with_partition_columns(&files, &parts, false)),
        ["a:Utf8View", "k:Int32", "v:Utf8View", "j:Int32"]
    );
    assert_eq!(
        names(&with_partition_columns(&files, &parts, true)),
        [
            "a:Utf8View",
            "K:Utf8View",
            "v:Utf8View",
            "k:Int32",
            "j:Int32"
        ]
    );
}

fn locate(path: &str) -> Result<LocatedPath> {
    LocalPaths::new(StoreId::new("file"), "/work/dir").locate(path)
}

#[test]
fn local_paths_are_qualified_like_hadoop() {
    for (input, path, display) in [
        ("/a/b.parquet", "a/b.parquet", "file:/a/b.parquet"),
        ("nope", "work/dir/nope", "file:/work/dir/nope"),
        ("../x/./y/", "work/x/y", "file:/work/x/y"),
        ("file:/a/b", "a/b", "file:/a/b"),
        ("file:///a/b", "a/b", "file:/a/b"),
        ("file://localhost/a", "a", "file:/a"),
        ("/", "", "file:/"),
    ] {
        let at = locate(input).unwrap();
        assert_eq!(at.path.as_ref(), path, "{input}");
        assert_eq!(at.display, display, "{input}");
        assert_eq!(at.store, StoreId::new("file"));
        assert_eq!(locate(&at.display).unwrap(), at, "{input}");
    }
    for input in ["s3a://bucket/k", "ftp://x/y", "file://other/a"] {
        let e = locate(input).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{input}"
        );
    }
    // A colon later in a name is no scheme.
    assert_eq!(locate("/a:b").unwrap().display, "file:/a:b");
    assert_eq!(locate("1a:b").unwrap().display, "file:/work/dir/1a:b");
}

fn object_paths() -> ObjectPaths {
    use krishiv_common::config::Config;
    use krishiv_io::{IoConfig, IoScheduler, StoreRegistry};

    let config = Config::new();
    ObjectPaths::new(
        StoreId::new("file"),
        "/work/dir",
        IoScheduler::new(IoConfig::from_config(&config)),
        Arc::new(StoreRegistry::new(Arc::new(config))),
    )
}

/// Qualifying never builds a store: cloud paths print their canonical
/// form whatever the build, local paths print as `LocalPaths` does, and
/// local locating is the same through the options form.
#[test]
fn object_paths_qualify_without_building_stores() {
    let paths = object_paths();
    for (input, display) in [
        ("s3a://bucket/k", "s3a://bucket/k"),
        ("gs://bucket/k", "gs://bucket/k"),
        ("/a/b.parquet", "file:/a/b.parquet"),
        ("rel", "file:/work/dir/rel"),
        ("file:/a/b", "file:/a/b"),
    ] {
        assert_eq!(paths.qualified_display(input).unwrap(), display, "{input}");
    }
    assert_eq!(
        paths
            .qualified_display("ftp://x/y")
            .unwrap_err()
            .error_class()
            .name(),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    let options = BTreeMap::new();
    assert_eq!(
        paths.locate_with_options("/a/b", &options).unwrap(),
        paths.locate("/a/b").unwrap()
    );
}

/// Without the `cloud` feature a cloud path locates to the unsupported
/// error rather than a store, with or without per-read options.
#[cfg(not(feature = "cloud"))]
#[test]
fn cloud_paths_need_the_cloud_feature_to_locate() {
    let paths = object_paths();
    let options = BTreeMap::new();
    for e in [
        paths.locate("s3://bucket/k").unwrap_err(),
        paths
            .locate_with_options("s3://bucket/k", &options)
            .unwrap_err(),
    ] {
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    }
}

#[cfg(feature = "cloud")]
#[test]
fn cloud_urls_locate_in_their_bucket_stores() {
    use krishiv_common::config::Config;
    use krishiv_io::{IoConfig, IoScheduler, StoreRegistry};

    let (config, _) = Config::from_pairs([
        ("fs.s3a.endpoint", "http://127.0.0.1:9000"),
        ("fs.azure.account.name", "acct"),
        (
            "fs.azure.account.key.acct",
            "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
        ),
    ])
    .unwrap();
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let paths = ObjectPaths::new(
        StoreId::new("file"),
        "/work/dir",
        sched,
        Arc::new(StoreRegistry::new(Arc::new(config))),
    );
    for (input, store, path, display) in [
        ("s3a://bucket/k", "s3://bucket", "k", "s3a://bucket/k"),
        ("s3://bucket/", "s3://bucket", "", "s3://bucket"),
        ("gs://bucket/k", "gs://bucket", "k", "gs://bucket/k"),
        (
            "abfss://cont@acct.dfs.core.windows.net/k",
            "abfss://cont@acct.dfs.core.windows.net",
            "k",
            "abfss://cont@acct.dfs.core.windows.net/k",
        ),
        ("az://cont/k", "abfss://cont", "k", "az://cont/k"),
        ("/a/b.parquet", "file", "a/b.parquet", "file:/a/b.parquet"),
    ] {
        let at = paths.locate(input).unwrap();
        // A store built from `fs.*` settings is registered as
        // `<store>#cfg:<hash>`; its display is the location.
        assert_eq!(
            at.store.to_string(),
            StoreId::new(store).to_string(),
            "{input}"
        );
        if store == "file" {
            assert_eq!(at.store, StoreId::new(store), "{input}");
        } else {
            assert!(
                at.store.key().contains("#cfg:"),
                "{input}: {}",
                at.store.key()
            );
        }
        assert_eq!(at.path.as_ref(), path, "{input}");
        assert_eq!(at.display, display, "{input}");
        assert_eq!(paths.locate(&at.display).unwrap(), at, "{input}");
    }
    for input in ["ftp://x/y", "s3a://AKIA:secret@bucket/k"] {
        let e = paths.locate(input).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{input}"
        );
    }
}

#[test]
fn hidden_names_follow_spark() {
    for name in [".x", "_SUCCESS", "_temporary", "a._COPYING_"] {
        assert!(hidden(name), "{name}");
    }
    for name in [
        "a.parquet",
        "_k=1",
        "k=_1",
        "_metadata",
        "_common_metadata",
        "_metadata.parquet",
        "_common_metadata2",
    ] {
        assert!(!hidden(name), "{name}");
    }
    assert!(!is_data_path("_metadata") && !is_data_path("_common_metadata2"));
    assert!(is_data_path("a.parquet") && is_data_path("_k=1"));
    assert_eq!(display_name("parquet"), "Parquet");
    assert_eq!(display_name("avro"), "Avro");
    assert_eq!(display_name("json"), "JSON");
    assert_eq!(display_name("mock"), "mock");
}

#[cfg(feature = "cloud")]
#[test]
fn cloud_locating_honours_per_read_options() {
    use std::collections::BTreeMap;

    use krishiv_common::config::Config;
    use krishiv_io::{IoConfig, IoScheduler, StoreRegistry};

    fn options(pairs: &[(&str, &str)]) -> BTreeMap<String, OptionValue> {
        pairs
            .iter()
            .map(|&(k, v)| (k.to_owned(), OptionValue::from_user(k, v)))
            .collect()
    }

    // An invalid session flag fails a plain locate ...
    let (config, _) = Config::from_pairs([("fs.s3a.path.style.access", "banana")]).unwrap();
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let paths = ObjectPaths::new(
        StoreId::new("file"),
        "/work/dir",
        sched,
        Arc::new(StoreRegistry::new(Arc::new(config))),
    );
    let error = paths.locate("s3://bucket/k").unwrap_err();
    assert_eq!(error.error_class().name(), "KRISHIV_INVALID_CONFIG_VALUE");
    // ... but a valid per-read option overrides it, and an invalid
    // per-read option beats a valid session value the other way.
    let at = paths
        .locate_with_options(
            "s3://bucket/k",
            &options(&[("fs.s3a.path.style.access", "true")]),
        )
        .unwrap();
    assert_eq!(at.display, "s3://bucket/k");
    let (good, _) = Config::from_pairs([("fs.s3a.path.style.access", "true")]).unwrap();
    let sched = IoScheduler::new(IoConfig::from_config(&good));
    let paths = ObjectPaths::new(
        StoreId::new("file"),
        "/work/dir",
        sched,
        Arc::new(StoreRegistry::new(Arc::new(good))),
    );
    let error = paths
        .locate_with_options(
            "s3://bucket/k",
            &options(&[("fs.s3a.path.style.access", "banana")]),
        )
        .unwrap_err();
    assert_eq!(error.error_class().name(), "KRISHIV_INVALID_CONFIG_VALUE");
    assert!(!error.to_string().contains("banana"));
    // Qualifying never builds a store: it succeeds despite the invalid
    // session flag, and unknown schemes still fail as before.
    assert_eq!(
        paths.qualified_display("s3://bucket/k").unwrap(),
        "s3://bucket/k"
    );
    for input in ["ftp://x/y", "s3a://AKIA:secret@bucket/k"] {
        assert_eq!(
            paths
                .locate_with_options(input, &options(&[]))
                .unwrap_err()
                .error_class()
                .name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{input}"
        );
        assert_eq!(
            paths
                .qualified_display(input)
                .unwrap_err()
                .error_class()
                .name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{input}"
        );
    }
}
