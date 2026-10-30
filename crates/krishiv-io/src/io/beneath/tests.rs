// SPDX-License-Identifier: Apache-2.0
//! `BeneathStore` against a real directory tree: nothing is created, written
//! or deleted outside the anchor, with `openat2` and with the component walk.

use std::os::unix::fs::symlink;
use std::path::{Path as FsPath, PathBuf};

use bytes::Bytes;
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{
    GetOptions, MultipartUpload, ObjectStore, PutMultipartOptions, PutOptions, PutPayload,
};

use super::{AnchorError, BeneathStore, is_staging};

struct Fixture {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    outside: PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap().join("root");
    let outside = tmp.path().canonicalize().unwrap().join("outside");
    std::fs::create_dir_all(root.join("a")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("keep"), b"outside").unwrap();
    Fixture {
        _tmp: tmp,
        root,
        outside,
    }
}

/// The anchor for a write to `<root>/a/t`, with either resolution.
fn store(f: &Fixture, components: bool) -> BeneathStore {
    let target = f.root.join("a/t");
    if components {
        BeneathStore::anchor_components(&target).unwrap()
    } else {
        BeneathStore::anchor(&target).unwrap()
    }
}

/// The object path of `rel` below `<root>/a`.
fn obj(f: &Fixture, rel: &str) -> Path {
    obj_in(&f.root.join("a"), rel)
}

fn obj_in(dir: &FsPath, rel: &str) -> Path {
    let mut parts: Vec<String> = dir
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(n) => Some(n.to_str().unwrap().to_owned()),
            _ => None,
        })
        .collect();
    parts.extend(rel.split('/').filter(|s| !s.is_empty()).map(str::to_owned));
    Path::from_iter(parts.iter().map(String::as_str))
}

fn tree(dir: &FsPath) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let e = e.unwrap();
            let p = e.path();
            if std::fs::symlink_metadata(&p).unwrap().is_dir() {
                stack.push(p);
            } else {
                out.push((p.clone(), std::fs::read(&p).unwrap_or_default()));
            }
        }
    }
    out.sort();
    out
}

async fn put(s: &BeneathStore, at: &Path, body: &'static [u8]) -> object_store::Result<()> {
    s.put_opts(at, PutPayload::from_static(body), PutOptions::default())
        .await
        .map(|_| ())
}

fn denied<T: std::fmt::Debug>(r: object_store::Result<T>) {
    match r {
        Err(object_store::Error::PermissionDenied { .. }) => {}
        other => panic!("expected PermissionDenied, got {other:?}"),
    }
}

#[tokio::test]
async fn a_put_creates_parents_and_leaves_no_staging_file() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        put(&s, &obj(&f, "t/x/part-0"), b"hello").await.unwrap();
        assert_eq!(
            std::fs::read(f.root.join("a/t/x/part-0")).unwrap(),
            b"hello"
        );
        assert_eq!(tree(&f.root.join("a")).len(), 1, "no staging file is left");
        // Overwrites the same name.
        put(&s, &obj(&f, "t/x/part-0"), b"again").await.unwrap();
        assert_eq!(
            std::fs::read(f.root.join("a/t/x/part-0")).unwrap(),
            b"again"
        );
    }
}

#[tokio::test]
async fn put_modes_other_than_overwrite_are_refused() {
    let f = fixture();
    let s = store(&f, false);
    let opts = PutOptions {
        mode: object_store::PutMode::Create,
        ..PutOptions::default()
    };
    let e = s
        .put_opts(&obj(&f, "t/x"), PutPayload::from_static(b"x"), opts)
        .await
        .unwrap_err();
    assert!(
        matches!(e, object_store::Error::NotImplemented { .. }),
        "{e}"
    );
    assert!(matches!(
        s.copy_opts(
            &obj(&f, "t/x"),
            &obj(&f, "t/y"),
            object_store::CopyOptions::default()
        )
        .await,
        Err(object_store::Error::NotImplemented { .. })
    ));
    assert!(matches!(
        s.get_opts(&obj(&f, "t/x"), GetOptions::default()).await,
        Err(object_store::Error::NotImplemented { .. })
    ));
}

#[tokio::test]
async fn multipart_parts_arrive_in_any_order_and_abort_cleans_up() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        let at = obj(&f, "t/big");
        let mut up = s
            .put_multipart_opts(&at, PutMultipartOptions::default())
            .await
            .unwrap();
        let p1 = up.put_part(PutPayload::from(Bytes::from_static(b"aaaa")));
        let p2 = up.put_part(PutPayload::from(Bytes::from_static(b"bbbb")));
        // Completed out of order: each part knows its offset.
        p2.await.unwrap();
        p1.await.unwrap();
        up.complete().await.unwrap();
        assert_eq!(std::fs::read(f.root.join("a/t/big")).unwrap(), b"aaaabbbb");
        assert_eq!(tree(&f.root.join("a/t")).len(), 1);

        let mut up = s
            .put_multipart_opts(&obj(&f, "t/gone"), PutMultipartOptions::default())
            .await
            .unwrap();
        up.put_part(PutPayload::from(Bytes::from_static(b"zz")))
            .await
            .unwrap();
        up.abort().await.unwrap();
        assert!(!f.root.join("a/t/gone").exists());
        assert_eq!(
            tree(&f.root.join("a/t")).len(),
            1,
            "the staging file is gone"
        );
        // A finished upload refuses more.
        assert!(up.put_part(PutPayload::from_static(b"q")).await.is_err());
        up.abort().await.unwrap();
    }
}

#[tokio::test]
async fn head_sees_files_not_directories_and_links_as_themselves() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        put(&s, &obj(&f, "t/x"), b"12345").await.unwrap();
        symlink(f.outside.join("keep"), f.root.join("a/t/lnk")).unwrap();
        let head = |rel: &str| {
            let (s, at) = (s.clone(), obj(&f, rel));
            async move {
                s.get_opts(
                    &at,
                    GetOptions {
                        head: true,
                        ..GetOptions::default()
                    },
                )
                .await
            }
        };
        let meta = head("t/x").await.unwrap().meta;
        assert_eq!(meta.size, 5);
        assert!(meta.e_tag.is_some());
        assert!(matches!(
            head("t").await,
            Err(object_store::Error::NotFound { .. })
        ));
        assert!(matches!(
            head("t/missing").await,
            Err(object_store::Error::NotFound { .. })
        ));
        assert!(matches!(
            head("nope/x").await,
            Err(object_store::Error::NotFound { .. })
        ));
        // The link's own size (the length of its target path), not the file's.
        let lnk = head("t/lnk").await.unwrap().meta;
        assert_eq!(lnk.size, f.outside.join("keep").as_os_str().len() as u64);
    }
}

#[tokio::test]
async fn deleting_a_link_removes_the_link_only() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        std::fs::create_dir_all(f.root.join("a/t")).unwrap();
        symlink(&f.outside, f.root.join("a/t/dir")).unwrap();
        symlink(f.outside.join("keep"), f.root.join("a/t/file")).unwrap();
        let paths = vec![
            Ok(obj(&f, "t/dir")),
            Ok(obj(&f, "t/file")),
            Ok(obj(&f, "t/nope")),
        ];
        let results: Vec<_> = s
            .delete_stream(futures::stream::iter(paths).boxed())
            .collect()
            .await;
        assert!(results[0].is_ok() && results[1].is_ok());
        assert!(matches!(
            results[2],
            Err(object_store::Error::NotFound { .. })
        ));
        assert!(std::fs::symlink_metadata(f.root.join("a/t/dir")).is_err());
        assert_eq!(std::fs::read(f.outside.join("keep")).unwrap(), b"outside");
    }
}

use futures::StreamExt;

#[tokio::test]
async fn listing_does_not_follow_links_and_skips_staging_files() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        put(&s, &obj(&f, "t/x/deep/part"), b"1").await.unwrap();
        put(&s, &obj(&f, "t/top"), b"22").await.unwrap();
        symlink(&f.outside, f.root.join("a/t/link")).unwrap();
        symlink(&f.outside, f.root.join("a/t/x/link2")).unwrap();
        std::fs::write(f.root.join("a/t/top#7"), b"staging").unwrap();

        let mut listed: Vec<String> = s
            .list(Some(&obj(&f, "t")))
            .map_ok(|m| m.location.to_string())
            .try_collect()
            .await
            .unwrap();
        listed.sort();
        let rel: Vec<_> = listed
            .iter()
            .map(|p| p.rsplit_once("root/a/").unwrap().1.to_owned())
            .collect();
        // The links are objects of their own; nothing under them is listed.
        assert_eq!(rel, ["t/link", "t/top", "t/x/deep/part", "t/x/link2"]);

        let level = s.list_with_delimiter(Some(&obj(&f, "t"))).await.unwrap();
        let prefixes: Vec<_> = level
            .common_prefixes
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            prefixes.len(),
            1,
            "a link is never a common prefix: {prefixes:?}"
        );
        assert!(prefixes[0].ends_with("t/x"));
        let objects: Vec<_> = level
            .objects
            .iter()
            .map(|m| m.location.to_string())
            .collect();
        assert!(objects.iter().any(|o| o.ends_with("t/link")));
        assert!(objects.iter().any(|o| o.ends_with("t/top")));
        assert_eq!(objects.len(), 2);

        // A missing prefix lists nothing.
        let none: Vec<_> = s
            .list(Some(&obj(&f, "nothing")))
            .try_collect()
            .await
            .unwrap();
        assert!(none.is_empty());
        // The whole anchor.
        let all: Vec<_> = s.list(None).try_collect().await.unwrap();
        assert_eq!(all.len(), 4);
    }
}

#[tokio::test]
async fn a_link_as_an_intermediate_directory_is_denied() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        std::fs::create_dir_all(f.root.join("a/t")).unwrap();
        symlink(&f.outside, f.root.join("a/t/sub")).unwrap();
        let before = tree(&f.outside);
        denied(put(&s, &obj(&f, "t/sub/x"), b"x").await);
        denied(
            s.put_multipart_opts(&obj(&f, "t/sub/y"), PutMultipartOptions::default())
                .await
                .map(|_| ()),
        );
        denied(
            s.get_opts(
                &obj(&f, "t/sub/x"),
                GetOptions {
                    head: true,
                    ..GetOptions::default()
                },
            )
            .await,
        );
        let deleted: Vec<_> = s
            .delete_stream(futures::stream::iter([Ok(obj(&f, "t/sub/keep"))]).boxed())
            .collect()
            .await;
        denied(deleted.into_iter().next().unwrap());
        denied(
            s.list(Some(&obj(&f, "t/sub")))
                .try_collect::<Vec<_>>()
                .await,
        );
        assert_eq!(tree(&f.outside), before);
    }
}

#[tokio::test]
async fn a_link_as_the_final_name_is_replaced_never_written_through() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        std::fs::create_dir_all(f.root.join("a/t")).unwrap();
        symlink(f.outside.join("keep"), f.root.join("a/t/part")).unwrap();
        put(&s, &obj(&f, "t/part"), b"new").await.unwrap();
        assert_eq!(std::fs::read(f.outside.join("keep")).unwrap(), b"outside");
        assert_eq!(std::fs::read(f.root.join("a/t/part")).unwrap(), b"new");
        assert!(
            !std::fs::symlink_metadata(f.root.join("a/t/part"))
                .unwrap()
                .is_symlink()
        );
    }
}

#[tokio::test]
async fn staging_names_that_exist_are_skipped_with_o_excl() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        std::fs::create_dir_all(f.root.join("a/t")).unwrap();
        // `part#1` is a link to an outside file, `part#2` a plain file.
        symlink(f.outside.join("keep"), f.root.join("a/t/part#1")).unwrap();
        std::fs::write(f.root.join("a/t/part#2"), b"mine").unwrap();
        put(&s, &obj(&f, "t/part"), b"data").await.unwrap();
        assert_eq!(std::fs::read(f.outside.join("keep")).unwrap(), b"outside");
        assert_eq!(std::fs::read(f.root.join("a/t/part#2")).unwrap(), b"mine");
        assert_eq!(std::fs::read(f.root.join("a/t/part")).unwrap(), b"data");
    }
}

#[tokio::test]
async fn a_hard_link_at_the_final_name_keeps_the_outside_file() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        std::fs::create_dir_all(f.root.join("a/t")).unwrap();
        std::fs::hard_link(f.outside.join("keep"), f.root.join("a/t/part")).unwrap();
        put(&s, &obj(&f, "t/part"), b"new").await.unwrap();
        assert_eq!(std::fs::read(f.outside.join("keep")).unwrap(), b"outside");
        assert_eq!(std::fs::read(f.root.join("a/t/part")).unwrap(), b"new");
    }
}

#[tokio::test]
async fn paths_outside_the_anchor_are_denied() {
    let f = fixture();
    let s = store(&f, false);
    let outside = obj_in(&f.outside, "x");
    denied(put(&s, &outside, b"x").await);
    let sibling = obj_in(&f.root, "b/x");
    denied(put(&s, &sibling, b"x").await);
    denied(s.list(Some(&sibling)).try_collect::<Vec<_>>().await);
    // The anchor itself is not an object.
    denied(put(&s, &obj_in(&f.root, "a"), b"x").await);
    assert_eq!(tree(&f.outside).len(), 1);
}

#[test]
fn anchor_is_the_deepest_existing_directory_above_the_target() {
    for components in [false, true] {
        let f = fixture();
        let open = |p: &FsPath| {
            if components {
                BeneathStore::anchor_components(p)
            } else {
                BeneathStore::anchor(p)
            }
        };
        let base = |s: &BeneathStore| s.inner.base.last().cloned();
        // Missing target and parents: the anchor is `a`.
        let s = open(&f.root.join("a/x/y/z")).unwrap();
        assert_eq!(base(&s).as_deref(), Some("a"));
        // An existing directory target: still its parent.
        std::fs::create_dir_all(f.root.join("a/x")).unwrap();
        let s = open(&f.root.join("a/x")).unwrap();
        assert_eq!(base(&s).as_deref(), Some("a"));
        // An existing file target: its parent too.
        std::fs::write(f.root.join("a/file"), b"f").unwrap();
        let s = open(&f.root.join("a/file")).unwrap();
        assert_eq!(base(&s).as_deref(), Some("a"));
        // A file where a directory is needed stops the walk there.
        let e = open(&f.root.join("a/file/sub/t"));
        assert!(matches!(e, Err(AnchorError::Swapped(_))), "{e:?}");
        // A link on an ancestor is a swap.
        symlink(&f.outside, f.root.join("lnk")).unwrap();
        let e = open(&f.root.join("lnk/x/t"));
        assert!(matches!(e, Err(AnchorError::Swapped(_))), "{e:?}");
        // The root's child: the anchor is `/`.
        let s = open(FsPath::new("/zz-no-such-dir")).unwrap();
        assert!(s.inner.base.is_empty());
        // Relative or odd paths are refused.
        assert!(matches!(open(FsPath::new("a/b")), Err(AnchorError::Io(_))));
        assert!(matches!(
            open(FsPath::new("/a/../b")),
            Err(AnchorError::Io(_))
        ));
        // The display names the anchor.
        assert!(
            store(&f, components)
                .to_string()
                .starts_with("BeneathStore(")
        );
    }
}

#[tokio::test]
async fn a_renamed_anchor_keeps_receiving_files_inside_the_root() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        put(&s, &obj(&f, "t/first"), b"1").await.unwrap();
        // The anchor directory is renamed away and a link to an outside
        // directory put in its place.
        std::fs::rename(f.root.join("a"), f.root.join("moved")).unwrap();
        symlink(&f.outside, f.root.join("a")).unwrap();
        let before = tree(&f.outside);
        put(&s, &obj(&f, "t/second"), b"2").await.unwrap();
        assert_eq!(tree(&f.outside), before, "nothing landed outside the root");
        assert_eq!(std::fs::read(f.root.join("moved/t/second")).unwrap(), b"2");
    }
}

#[tokio::test]
async fn a_directory_swapped_for_a_link_during_the_write_is_denied() {
    for components in [false, true] {
        let f = fixture();
        let s = store(&f, components);
        put(&s, &obj(&f, "t/first"), b"1").await.unwrap();
        // `t`, created by the write, becomes a link out of the root.
        std::fs::rename(f.root.join("a/t"), f.root.join("a/t-old")).unwrap();
        symlink(&f.outside, f.root.join("a/t")).unwrap();
        let before = tree(&f.outside);
        denied(put(&s, &obj(&f, "t/second"), b"2").await);
        denied(
            s.put_multipart_opts(&obj(&f, "t/third"), PutMultipartOptions::default())
                .await
                .map(|_| ()),
        );
        assert_eq!(tree(&f.outside), before);
        assert!(
            f.root.join("a/t-old/first").exists(),
            "earlier files stay inside the root"
        );
    }
}

#[test]
fn staging_names_are_recognized() {
    assert!(is_staging("part-0#1") && is_staging("a#23"));
    // As `LocalFileSystem` reads a name: the text after the first `#`.
    assert!(!is_staging("a#1#2") && !is_staging("a#b#23"));
    assert!(
        !is_staging("part-0")
            && !is_staging("part#")
            && !is_staging("part#x")
            && !is_staging("a#1b")
    );
}

#[test]
fn a_directory_name_object_paths_would_encode_is_refused_up_front() {
    let f = fixture();
    for name in ["jos\u{e9}", "a~b", "a#b", "a%b"] {
        std::fs::create_dir_all(f.root.join(name)).unwrap();
        let e = BeneathStore::anchor(&f.root.join(name).join("out"));
        assert!(
            matches!(e, Err(AnchorError::UnsupportedName(ref n)) if n == name),
            "{name}: {e:?}"
        );
    }
    // The target's own name is not checked: it is encoded like any part name.
    BeneathStore::anchor(&f.root.join("a/jos\u{e9}")).unwrap();
}

// Counts this process's descriptors through procfs, so Linux only; the
// allowance covers tests running beside it under plain `cargo test`, far
// below the 300 a listing that kept every directory open would hold.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_listing_holds_one_descriptor_however_wide_the_tree_is() {
    let f = fixture();
    let s = store(&f, false);
    for i in 0..300 {
        std::fs::create_dir_all(f.root.join(format!("a/t/d{i}/e"))).unwrap();
        std::fs::write(f.root.join(format!("a/t/d{i}/e/f")), b"x").unwrap();
    }
    // A low descriptor limit would fail a listing that keeps every pending
    // directory open; the walk re-opens each directory when its turn comes.
    let before = open_fds();
    let all: Vec<_> = s.list(Some(&obj(&f, "t"))).try_collect().await.unwrap();
    assert_eq!(all.len(), 300);
    assert!(open_fds() <= before + 24, "no descriptor is left open");
}

#[cfg(target_os = "linux")]
fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").map_or(0, Iterator::count)
}
