// SPDX-License-Identifier: Apache-2.0
//! The resource group and configuration RPCs of `krishiv.admin.v1` (design P3-T066 sections 4,
//! 5.4 and 5.5): who may call them, the etag rules of a replace, and the batch rules of
//! `SetConfigDefaults`.
#![cfg(all(feature = "admin-api", feature = "session"))]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use futures::future::BoxFuture;
use krishiv_server::admin_v1::{AdminConfig, SessionAdmin, router};
use krishiv_server::guard::{Access, Caller, Guard, GuardRequest, Refusal};
use krishiv_session::EngineRuntime;
use krishiv_session::auth::{Authenticated, Method, Role};
use krishiv_session::{Principal, PrincipalKind};
use serde_json::Value;
use tower::ServiceExt;

/// `Authorization: Bearer <name>:<role>` is the caller `name` with that role.
struct Roles;

impl Guard for Roles {
    fn check<'a>(&'a self, r: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>> {
        Box::pin(async move { Self::who(&r).map(|_| ()) })
    }

    fn authenticate<'a>(&'a self, r: GuardRequest<'a>) -> BoxFuture<'a, Result<Caller, Refusal>> {
        Box::pin(async move {
            let (name, role) = Self::who(&r)?;
            let mut caller = Caller::anonymous();
            name.clone_into(&mut caller.principal);
            caller.role = Some(role.as_str().to_owned());
            caller.method = "token".to_owned();
            caller.identity = Some(Arc::new(Authenticated::new(
                Principal::new(name, PrincipalKind::User),
                Some(role),
                Method::Token,
            )));
            Ok(caller)
        })
    }
}

impl Roles {
    fn who(r: &GuardRequest<'_>) -> Result<(String, Role), Refusal> {
        let _ = Access::Authenticated;
        let token = r.token.ok_or_else(|| panic!("no token"))?;
        let (name, role) = token.split_once(':').unwrap();
        let role = match role {
            "viewer" => Role::Viewer,
            "operator" => Role::Operator,
            _ => Role::Admin,
        };
        Ok((name.to_owned(), role))
    }
}

const GROUPS: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 10}],
  "defaultGroup": "root"
}"#;

const GROUPS_V2: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 8, "maxQueued": 10}],
  "defaultGroup": "root"
}"#;

struct Fixture {
    app: Router,
    path: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

fn fixture(with_groups: bool) -> Fixture {
    fixture_with(with_groups, &[], &[])
}

/// A fixture whose server has the `krishiv.auth.*` startup settings `auth` and the engine
/// settings `conf` (keys the registry may not know).
fn fixture_with(with_groups: bool, auth: &[(&str, &str)], conf: &[(&str, &str)]) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resource-groups.json");
    std::fs::write(&path, GROUPS).unwrap();
    let mut builder = EngineRuntime::builder();
    if with_groups {
        builder = builder.config(
            "krishiv.workload.resourceGroups.file",
            path.to_str().unwrap(),
        );
    }
    for (k, v) in conf {
        builder = builder.config(*k, *v);
    }
    let rt = builder.build().unwrap();
    let app = router(
        Arc::new(
            SessionAdmin::new(rt).with_startup_auth(
                auth.iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                    .collect(),
            ),
        ),
        Arc::new(Roles),
        AdminConfig {
            loopback: true,
            allowed_hosts: Vec::new(),
            idempotency_ttl: Duration::from_secs(60),
        },
    );
    Fixture {
        app,
        path,
        _dir: dir,
    }
}

impl Fixture {
    async fn call(
        &self,
        who: &str,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(method)
            .uri(format!("/api/admin/v1{path}"))
            .header(header::HOST, "localhost:7700")
            .header(header::AUTHORIZATION, format!("Bearer {who}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.unwrap_or_default().to_owned()))
            .unwrap();
        let response = self.app.clone().oneshot(req).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|e| panic!("{status}: not JSON ({e}): {bytes:?}"))
        };
        (status, body)
    }

    async fn replace(&self, who: &str, document: &str, etag: &str) -> (StatusCode, Value) {
        let body = serde_json::json!({ "document": document, "etag": etag }).to_string();
        self.call(who, "PUT", "/resourceGroups/config", Some(&body))
            .await
    }
}

#[tokio::test]
async fn groups_are_listed_to_everyone_but_the_document_needs_monitor() {
    let f = fixture(true);
    let (status, body) = f.call("o:operator", "GET", "/resourceGroups", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["groups"][0]["groupId"], "root");
    assert_eq!(body["groups"][0]["running"], 0);
    // A viewer holds no query: it sees no group.
    let (status, body) = f.call("v:viewer", "GET", "/resourceGroups", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["groups"], serde_json::json!([]), "{body}");
    for (method, path) in [
        ("GET", "/resourceGroups/config"),
        ("POST", "/resourceGroups/config:validate"),
    ] {
        let (status, _) = f.call("v:viewer", method, path, Some("{}")).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{path}");
    }
    let (status, body) = f
        .call("o:operator", "GET", "/resourceGroups/config", None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body["etag"].as_str().is_some_and(|e| e.len() == 64),
        "{body}"
    );
    assert_eq!(body["fileEtag"], body["etag"]);
}

#[tokio::test]
async fn validate_reports_the_loader_error_and_changes_nothing() {
    let f = fixture(true);
    let (_, before) = f
        .call("o:operator", "GET", "/resourceGroups/config", None)
        .await;
    let good = serde_json::json!({ "document": GROUPS_V2 }).to_string();
    let (status, body) = f
        .call(
            "o:operator",
            "POST",
            "/resourceGroups/config:validate",
            Some(&good),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["valid"], true);
    let bad = serde_json::json!({ "document": r#"{"rootGroups": [], "cpuQuota": 1}"# }).to_string();
    let (_, body) = f
        .call(
            "o:operator",
            "POST",
            "/resourceGroups/config:validate",
            Some(&bad),
        )
        .await;
    assert_eq!(body["valid"], false, "{body}");
    assert_eq!(
        body["problems"][0]["errorClass"],
        "KRISHIV_RESOURCE_GROUP_CONFIG_INVALID"
    );
    let (_, after) = f
        .call("o:operator", "GET", "/resourceGroups/config", None)
        .await;
    assert_eq!(before["etag"], after["etag"]);
    assert_eq!(
        std::fs::read_to_string(&f.path).unwrap(),
        GROUPS,
        "validation leaves the file as it was"
    );
}

#[tokio::test]
async fn replace_needs_manage_and_a_fresh_etag_and_writes_the_file() {
    let f = fixture(true);
    let (_, current) = f
        .call("o:operator", "GET", "/resourceGroups/config", None)
        .await;
    let etag = current["etag"].as_str().unwrap().to_owned();
    let (status, _) = f.replace("o:operator", GROUPS_V2, &etag).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "an operator may not replace");

    let (status, body) = f.replace("a:admin", GROUPS_V2, "not-the-etag").await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");
    assert_eq!(
        body["error"]["errorClass"],
        "KRISHIV_ADMIN_PRECONDITION_FAILED"
    );

    let (status, body) = f.replace("a:admin", GROUPS_V2, &etag).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_ne!(body["etag"].as_str().unwrap(), etag);
    let on_disk: Value = serde_json::from_str(&std::fs::read_to_string(&f.path).unwrap()).unwrap();
    let expected: Value = serde_json::from_str(GROUPS_V2).unwrap();
    assert_eq!(on_disk, expected, "the file holds the new document");
    let (_, after) = f.call("o:operator", "GET", "/resourceGroups", None).await;
    assert_eq!(after["groups"][0]["hardConcurrencyLimit"], 8);

    // The old etag is stale now.
    let (status, _) = f.replace("a:admin", GROUPS, &etag).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED);
}

#[tokio::test]
async fn reload_picks_up_a_file_edited_by_hand_and_refuses_an_invalid_one() {
    let f = fixture(true);
    let (status, _) = f
        .call(
            "o:operator",
            "POST",
            "/resourceGroups/config:reload",
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    std::fs::write(&f.path, GROUPS_V2).unwrap();
    let (status, body) = f
        .call(
            "a:admin",
            "POST",
            "/resourceGroups/config:reload",
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, after) = f.call("o:operator", "GET", "/resourceGroups", None).await;
    assert_eq!(after["groups"][0]["hardConcurrencyLimit"], 8);
    std::fs::write(&f.path, "{ not json").unwrap();
    let (status, body) = f
        .call(
            "a:admin",
            "POST",
            "/resourceGroups/config:reload",
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (_, still) = f.call("o:operator", "GET", "/resourceGroups", None).await;
    assert_eq!(
        still["groups"][0]["hardConcurrencyLimit"], 8,
        "the old tree stays"
    );
}

#[tokio::test]
async fn without_a_group_file_the_reads_are_empty_and_the_changes_are_read_only() {
    let f = fixture(false);
    let (status, body) = f
        .call("o:operator", "GET", "/resourceGroups/config", None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["document"], "", "{body}");
    let (status, body) = f.replace("a:admin", GROUPS, "").await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");
    assert_eq!(
        body["error"]["errorClass"],
        "KRISHIV_RESOURCE_GROUP_CONFIG_READ_ONLY"
    );
    let (status, _) = f
        .call(
            "a:admin",
            "POST",
            "/resourceGroups/config:reload",
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED);
}

#[tokio::test]
async fn get_config_lists_the_auth_settings_withheld_from_an_operator_and_shown_to_an_admin() {
    let f = fixture_with(
        false,
        &[
            ("krishiv.auth.tokens.ops.secret", "file:/run/secrets/ops"),
            ("krishiv.auth.tokens.ops.role", "admin"),
        ],
        &[],
    );
    let (status, _) = f.call("v:viewer", "GET", "/config", None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let url = "/config?prefix=krishiv.auth.&pageSize=500";
    let (status, body) = f.call("o:operator", "GET", url, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entries = body["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2, "{body}");
    for e in entries {
        assert_eq!(e["withheld"], true, "{e}");
        assert!(e["value"].is_null(), "{e}");
        assert_eq!(e["scope"], "cluster");
        assert_eq!(e["source"], "startup");
    }
    let (_, body) = f.call("a:admin", "GET", url, None).await;
    let entries = body["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2, "{body}");
    let secret = entries
        .iter()
        .find(|e| e["key"] == "krishiv.auth.tokens.ops.secret")
        .unwrap();
    assert_eq!(secret["value"], "file:/run/secrets/ops");
    assert_eq!(secret["withheld"], false);
}

#[tokio::test]
async fn get_config_never_shows_an_inline_spark_token_but_shows_a_reference() {
    let f = fixture_with(
        false,
        &[
            ("spark.connect.authenticate.token", "s3cr3t-inline"),
            ("krishiv.auth.tokens.ops.secret", "env:OPS_TOKEN"),
        ],
        &[],
    );
    for prefix in ["spark.", "krishiv.auth."] {
        for who in ["o:operator", "a:admin"] {
            let (status, body) = f
                .call(
                    who,
                    "GET",
                    &format!("/config?prefix={prefix}&pageSize=500"),
                    None,
                )
                .await;
            assert_eq!(status, StatusCode::OK, "{who} {prefix}: {body}");
            assert!(!body.to_string().contains("s3cr3t-inline"), "{who}: {body}");
            if prefix == "spark." {
                let row = body["entries"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|e| e["key"] == "spark.connect.authenticate.token")
                    .unwrap_or_else(|| panic!("{who}: no token row in {body}"));
                assert_eq!(row["withheld"], true, "{who}: {row}");
                assert!(row["value"].is_null(), "{who}: {row}");
            }
        }
    }
    let (_, body) = f
        .call(
            "a:admin",
            "GET",
            "/config?prefix=spark.connect&pageSize=500",
            None,
        )
        .await;
    assert_eq!(body["entries"][0]["withheld"], true, "{body}");
    let (_, body) = f
        .call(
            "a:admin",
            "GET",
            "/config?prefix=krishiv.auth.&pageSize=500",
            None,
        )
        .await;
    assert_eq!(body["entries"][0]["value"], "env:OPS_TOKEN", "{body}");
}

#[tokio::test]
async fn get_config_withholds_the_path_allowlists_from_an_operator_over_the_rpc() {
    let f = fixture_with(false, &[], &[("krishiv.io.local.allowedRoots", "/tmp")]);
    let url = "/config?prefix=krishiv.io.local.allowedRoots";
    let (_, body) = f.call("o:operator", "GET", url, None).await;
    assert_eq!(body["entries"][0]["withheld"], true, "{body}");
    assert!(body["entries"][0]["value"].is_null());
    let (_, body) = f.call("a:admin", "GET", url, None).await;
    assert_eq!(body["entries"][0]["value"], "/tmp", "{body}");
}

#[tokio::test]
async fn get_config_never_shows_a_credential_named_startup_key() {
    let f = fixture_with(
        false,
        &[],
        &[
            ("spark.hadoop.fs.s3a.secret.key", "s3cr3t"),
            ("spark.some.plain", "visible"),
        ],
    );
    let url = "/config?prefix=spark.&pageSize=500";
    for who in ["o:operator", "a:admin"] {
        let (_, body) = f.call(who, "GET", url, None).await;
        let text = body.to_string();
        assert!(!text.contains("s3cr3t"), "{who}: {text}");
        let row = body["entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["key"] == "spark.hadoop.fs.s3a.secret.key")
            .unwrap();
        assert_eq!(row["withheld"], true);
    }
    let plain = |body: &Value| {
        body["entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["key"] == "spark.some.plain")
            .cloned()
            .unwrap()
    };
    let (_, body) = f.call("o:operator", "GET", url, None).await;
    assert_eq!(
        plain(&body)["withheld"],
        true,
        "an operator sees no unvetted value"
    );
    let (_, body) = f.call("a:admin", "GET", url, None).await;
    assert_eq!(plain(&body)["value"], "visible");
}

#[tokio::test]
async fn validate_reports_every_problem_with_its_path() {
    let f = fixture(true);
    let doc = r#"{"rootGroups": [
        {"name": "a", "hardConcurrencyLimit": -1, "maxQueued": 1},
        {"name": "b", "maxQueued": 1}],
      "defaultGroup": "a"}"#;
    let body = serde_json::json!({ "document": doc }).to_string();
    let (_, body) = f
        .call(
            "o:operator",
            "POST",
            "/resourceGroups/config:validate",
            Some(&body),
        )
        .await;
    assert_eq!(body["valid"], false, "{body}");
    let paths: Vec<&str> = body["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["path"].as_str().unwrap())
        .collect();
    assert_eq!(
        paths,
        [
            "rootGroups[0].hardConcurrencyLimit",
            "rootGroups[1].hardConcurrencyLimit"
        ],
        "{body}"
    );
}

#[tokio::test]
async fn validate_reports_every_problem_of_one_root_with_its_paths() {
    let f = fixture(true);
    let doc = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 1,
        "subGroups": [
          {"name": "a", "hardConcurrencyLimit": 0, "maxQueued": 1},
          {"name": "b", "hardConcurrencyLimit": 1, "maxQueued": -1}]}],
      "defaultGroup": "root.zzz"}"#;
    let body = serde_json::json!({ "document": doc }).to_string();
    let (_, body) = f
        .call(
            "o:operator",
            "POST",
            "/resourceGroups/config:validate",
            Some(&body),
        )
        .await;
    let paths: Vec<&str> = body["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["path"].as_str().unwrap())
        .collect();
    assert_eq!(
        paths,
        [
            "rootGroups[0].subGroups[0].hardConcurrencyLimit",
            "rootGroups[0].subGroups[1].maxQueued",
            "defaultGroup"
        ],
        "{body}"
    );
}

#[tokio::test]
async fn a_replayed_request_id_answers_the_stored_response_and_applies_once() {
    let f = fixture(true);
    let (_, current) = f
        .call("a:admin", "GET", "/resourceGroups/config", None)
        .await;
    let etag = current["etag"].as_str().unwrap();
    let body = serde_json::json!({
        "document": GROUPS_V2, "etag": etag, "requestId": "req-1"
    })
    .to_string();
    let (first, a) = f
        .call("a:admin", "PUT", "/resourceGroups/config", Some(&body))
        .await;
    assert_eq!(first, StatusCode::OK, "{a}");
    // Applied again it would answer 412 (the etag moved); the replay answers the stored one.
    let (second, b) = f
        .call("a:admin", "PUT", "/resourceGroups/config", Some(&body))
        .await;
    assert_eq!(second, StatusCode::OK, "{b}");
    assert_eq!(a["etag"], b["etag"]);

    let set = serde_json::json!({
        "set": {"krishiv.sql.ansi.enabled": "true"}, "requestId": "req-2"
    })
    .to_string();
    let (s1, x) = f
        .call("a:admin", "PATCH", "/config/defaults", Some(&set))
        .await;
    let (s2, y) = f
        .call("a:admin", "PATCH", "/config/defaults", Some(&set))
        .await;
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK), "{x} {y}");
    assert_eq!(x, y);

    // Unset the key under another id, then replay `req-2`: an implementation that ran it again
    // would set the key back to `true`.
    let unset = serde_json::json!({
        "unset": ["krishiv.sql.ansi.enabled"], "requestId": "req-3"
    })
    .to_string();
    let (status, body) = f
        .call("a:admin", "PATCH", "/config/defaults", Some(&unset))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (s3, z) = f
        .call("a:admin", "PATCH", "/config/defaults", Some(&set))
        .await;
    assert_eq!(s3, StatusCode::OK, "{z}");
    assert_eq!(x, z, "the replay answers the first response");
    let (_, entry) = f
        .call(
            "a:admin",
            "GET",
            "/config?prefix=krishiv.sql.ansi.enabled",
            None,
        )
        .await;
    assert_ne!(
        entry["entries"][0]["source"], "admin",
        "the replay did not set the key again: {entry}"
    );
}

#[tokio::test]
async fn a_request_dropped_mid_flight_still_stores_its_result_for_the_retry() {
    let f = fixture(true);
    let (_, current) = f
        .call("a:admin", "GET", "/resourceGroups/config", None)
        .await;
    let etag = current["etag"].as_str().unwrap();
    let body = serde_json::json!({
        "document": GROUPS_V2, "etag": etag, "requestId": "req-drop"
    })
    .to_string();
    // Poll the request once (it hands its mutation to the blocking pool and waits), then drop it:
    // a request cut off mid-flight.
    {
        let mut call =
            std::pin::pin!(f.call("a:admin", "PUT", "/resourceGroups/config", Some(&body)));
        assert!(
            std::future::poll_fn(|cx| std::task::Poll::Ready(call.as_mut().poll(cx).is_pending()))
                .await,
            "the request was dropped unfinished"
        );
    }
    // The detached mutation finishes on its own: the file holds the new document before any
    // retry could have written it.
    let expected: Value = serde_json::from_str(GROUPS_V2).unwrap();
    let mut written = false;
    for _ in 0..250 {
        let on_disk = std::fs::read_to_string(&f.path).unwrap_or_default();
        if serde_json::from_str::<Value>(&on_disk).is_ok_and(|d| d == expected) {
            written = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(written, "the dropped request's mutation still completed");
    // The retry with the same id gets the stored response; run again it would be a 412 (the etag
    // moved).
    let (status, answer) = f
        .call("a:admin", "PUT", "/resourceGroups/config", Some(&body))
        .await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let (_, now) = f
        .call("a:admin", "GET", "/resourceGroups/config", None)
        .await;
    assert_eq!(
        answer["etag"], now["etag"],
        "the stored result is the applied one"
    );
}

#[tokio::test]
async fn a_file_edited_after_it_was_read_is_a_412_on_replace() {
    let f = fixture(true);
    let (_, current) = f
        .call("a:admin", "GET", "/resourceGroups/config", None)
        .await;
    std::fs::write(&f.path, GROUPS_V2).unwrap();
    let (status, body) = f
        .replace("a:admin", GROUPS_V2, current["etag"].as_str().unwrap())
        .await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");
}

#[tokio::test]
async fn set_config_defaults_is_all_or_none_and_shows_as_admin_source() {
    let f = fixture(false);
    let key = "krishiv.sql.ansi.enabled";
    let set = |k: &str, v: &str| serde_json::json!({ "set": { k: v } }).to_string();
    let (status, _) = f
        .call(
            "o:operator",
            "PATCH",
            "/config/defaults",
            Some(&set(key, "true")),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "an operator may not set");

    // One bad key refuses the batch: the good key is not applied either.
    let (_, original) = f
        .call("a:admin", "GET", &format!("/config?prefix={key}"), None)
        .await;
    let batch =
        serde_json::json!({ "set": { key: "true" }, "unset": ["krishiv.auth.defaultRole"] })
            .to_string();
    let (status, body) = f
        .call("a:admin", "PATCH", "/config/defaults", Some(&batch))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["errorClass"], "CANNOT_MODIFY_STATIC_CONFIG");
    let (_, body) = f
        .call("a:admin", "GET", &format!("/config?prefix={key}"), None)
        .await;
    assert_eq!(
        body["entries"][0], original["entries"][0],
        "the valid change before the bad key was not applied"
    );

    let (status, body) = f
        .call(
            "a:admin",
            "PATCH",
            "/config/defaults",
            Some(&set(key, "true")),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = f
        .call("a:admin", "GET", &format!("/config?prefix={key}"), None)
        .await;
    assert_eq!(body["entries"][0]["source"], "admin");
    assert_eq!(body["entries"][0]["value"], "true");

    let unset = serde_json::json!({ "unset": [key] }).to_string();
    let (status, body) = f
        .call("a:admin", "PATCH", "/config/defaults", Some(&unset))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = f
        .call("a:admin", "GET", &format!("/config?prefix={key}"), None)
        .await;
    assert_ne!(body["entries"][0]["source"], "admin");
}

/// Collects the `krishiv::admin_action` records of the process as `field=value` strings.
#[derive(Clone, Default)]
struct Records(Arc<std::sync::Mutex<Vec<String>>>);

struct Fields(String);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write as _;
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }
}

impl tracing::Subscriber for Records {
    fn enabled(&self, m: &tracing::Metadata<'_>) -> bool {
        m.target() == "krishiv::admin_action"
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, e: &tracing::Event<'_>) {
        let mut f = Fields(String::new());
        e.record(&mut f);
        self.0.lock().unwrap().push(f.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn records() -> &'static Records {
    static RECORDS: std::sync::OnceLock<Records> = std::sync::OnceLock::new();
    RECORDS.get_or_init(|| {
        let r = Records::default();
        let _ = tracing::subscriber::set_global_default(r.clone());
        r
    })
}

/// Design 5.6 and 8: the group and defaults changes write one record each (allowed, denied,
/// failed) with the old and new etag or value; a credential-named key's value is in no field and
/// a Spark alias is recorded under its canonical key.
#[tokio::test]
async fn config_changes_write_action_records_with_old_and_new_values() {
    let records = records();
    let f = fixture(true);
    let (_, current) = f
        .call("rec:admin", "GET", "/resourceGroups/config", None)
        .await;
    let old = current["etag"].as_str().unwrap().to_owned();
    // allowed replace, a stale replace (failed), an operator's replace (denied)
    let (status, ok) = f.replace("rec:admin", GROUPS_V2, &old).await;
    assert_eq!(status, StatusCode::OK, "{ok}");
    let new = ok["etag"].as_str().unwrap().to_owned();
    f.replace("rec:admin", GROUPS, &old).await;
    f.replace("rec:operator", GROUPS, &new).await;
    // reload after an edit on disk
    std::fs::write(&f.path, GROUPS).unwrap();
    let (status, reloaded) = f
        .call(
            "rec:admin",
            "POST",
            "/resourceGroups/config:reload",
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{reloaded}");
    // defaults: the canonical key first, then the same key through a Spark alias (the record must
    // show the canonical old value), then a credential-named unregistered key (fails)
    let canonical = serde_json::json!({
        "set": {"krishiv.sql.avro.compression.codec": "snappy"}
    })
    .to_string();
    let (status, body) = f
        .call("rec:admin", "PATCH", "/config/defaults", Some(&canonical))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let alias = serde_json::json!({
        "set": {"spark.sql.avro.compression.codec": "deflate"}
    })
    .to_string();
    let (status, body) = f
        .call("rec:admin", "PATCH", "/config/defaults", Some(&alias))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let secret = serde_json::json!({ "set": {"my.secret.key": "topsecret-value"} }).to_string();
    let (status, _) = f
        .call("rec:admin", "PATCH", "/config/defaults", Some(&secret))
        .await;
    assert_ne!(status, StatusCode::OK);
    f.call("rec:operator", "PATCH", "/config/defaults", Some(&alias))
        .await;

    let everything: Vec<String> = records.0.lock().unwrap().clone();
    assert!(
        everything.iter().all(|l| !l.contains("topsecret-value")),
        "no record of the process carries the credential: {everything:#?}"
    );
    let log: Vec<String> = everything
        .iter()
        .filter(|l| l.contains("principal=\"rec\""))
        .cloned()
        .collect();
    // Each mutation writes exactly one record of its outcome.
    let find = |rpc: &str, result: &str| -> &String {
        let found: Vec<&String> = log
            .iter()
            .filter(|l| {
                l.contains(&format!("rpc=\"{rpc}\"")) && l.contains(&format!("result=\"{result}\""))
            })
            .collect();
        assert_eq!(found.len(), 1, "{rpc} {result} once in {log:#?}");
        found[0]
    };
    let replaced = find("ReplaceResourceGroupConfig", "ok");
    assert!(replaced.contains(&format!("{old} -> {new}")), "{replaced}");
    let stale = find("ReplaceResourceGroupConfig", "failed");
    assert!(stale.contains(&old) && stale.contains(&new), "{stale}");
    find("ReplaceResourceGroupConfig", "denied");
    let reload = find("ReloadResourceGroupConfig", "ok");
    assert!(reload.contains(&format!("{new} -> ")), "{reload}");
    let sets: Vec<&String> = log
        .iter()
        .filter(|l| l.contains("rpc=\"SetConfigDefaults\"") && l.contains("result=\"ok\""))
        .collect();
    assert_eq!(sets.len(), 2, "{log:#?}");
    assert!(
        sets[0].contains("set krishiv.sql.avro.compression.codec: ")
            && sets[0].contains("-> snappy"),
        "{}",
        sets[0]
    );
    let aliased = sets[1];
    assert!(
        aliased.contains("set krishiv.sql.avro.compression.codec: snappy -> deflate")
            && !aliased.contains("spark.sql.avro"),
        "{aliased}"
    );
    let failed = find("SetConfigDefaults", "failed");
    assert!(failed.contains("(withheld)"), "{failed}");
    find("SetConfigDefaults", "denied");
    assert!(
        log.iter()
            .all(|l| !l.contains("topsecret-value") && !l.contains("rec:admin")),
        "{log:#?}"
    );
}

#[tokio::test]
async fn a_symbolic_link_group_file_refuses_a_replace_as_a_precondition() {
    let f = fixture(true);
    let (_, current) = f
        .call("a:admin", "GET", "/resourceGroups/config", None)
        .await;
    let real = f.path.with_file_name("real.json");
    std::fs::rename(&f.path, &real).unwrap();
    std::os::unix::fs::symlink(&real, &f.path).unwrap();
    let (status, body) = f
        .replace("a:admin", GROUPS_V2, current["etag"].as_str().unwrap())
        .await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");
    assert!(
        body.to_string()
            .contains("KRISHIV_RESOURCE_GROUP_CONFIG_READ_ONLY"),
        "{body}"
    );
}

/// All seven new RPCs over gRPC on a real socket answer as over REST (design §8, transports agree).
#[tokio::test]
async fn the_group_and_config_rpcs_answer_over_grpc() {
    use krishiv_server::admin_v1::{
        GetConfigRequest, GetConfigResponse, GetResourceGroupConfigRequest,
        GetResourceGroupConfigResponse, ListResourceGroupsRequest, ListResourceGroupsResponse,
        ReloadResourceGroupConfigRequest, ReloadResourceGroupConfigResponse,
        ReplaceResourceGroupConfigRequest, ReplaceResourceGroupConfigResponse,
        SetConfigDefaultsRequest, SetConfigDefaultsResponse, ValidateResourceGroupConfigRequest,
        ValidateResourceGroupConfigResponse,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resource-groups.json");
    std::fs::write(&path, GROUPS).unwrap();
    let rt = EngineRuntime::builder()
        .config(
            "krishiv.workload.resourceGroups.file",
            path.to_str().unwrap(),
        )
        .build()
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(
        Arc::new(SessionAdmin::new(rt)),
        Arc::new(Roles),
        AdminConfig::for_listener(addr.ip(), Vec::new()),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut grpc = tonic::client::Grpc::new(channel);
    grpc.ready().await.unwrap();

    async fn call<Q, R>(
        grpc: &mut tonic::client::Grpc<tonic::transport::Channel>,
        who: &str,
        rpc: &str,
        request: Q,
    ) -> Result<R, tonic::Status>
    where
        Q: prost::Message + 'static,
        R: prost::Message + Default + 'static,
    {
        grpc.ready().await.unwrap();
        let mut req = tonic::Request::new(request);
        req.metadata_mut()
            .insert("authorization", format!("Bearer {who}").parse().unwrap());
        let path = format!("/krishiv.admin.v1.AdminService/{rpc}")
            .parse()
            .unwrap();
        grpc.unary(req, path, tonic_prost::ProstCodec::default())
            .await
            .map(tonic::Response::into_inner)
    }

    let listed: ListResourceGroupsResponse = call(
        &mut grpc,
        "a:admin",
        "ListResourceGroups",
        ListResourceGroupsRequest::default(),
    )
    .await
    .unwrap();
    assert_eq!(listed.groups.len(), 1);
    let got: GetResourceGroupConfigResponse = call(
        &mut grpc,
        "o:operator",
        "GetResourceGroupConfig",
        GetResourceGroupConfigRequest {},
    )
    .await
    .unwrap();
    let valid: ValidateResourceGroupConfigResponse = call(
        &mut grpc,
        "o:operator",
        "ValidateResourceGroupConfig",
        ValidateResourceGroupConfigRequest {
            document: "{}".to_owned(),
        },
    )
    .await
    .unwrap();
    assert!(!valid.valid && !valid.problems.is_empty());
    let denied: Result<ReplaceResourceGroupConfigResponse, _> = call(
        &mut grpc,
        "o:operator",
        "ReplaceResourceGroupConfig",
        ReplaceResourceGroupConfigRequest {
            document: GROUPS_V2.to_owned(),
            etag: got.etag.clone(),
            request_id: String::new(),
        },
    )
    .await;
    assert_eq!(denied.unwrap_err().code(), tonic::Code::PermissionDenied);
    let replaced: ReplaceResourceGroupConfigResponse = call(
        &mut grpc,
        "a:admin",
        "ReplaceResourceGroupConfig",
        ReplaceResourceGroupConfigRequest {
            document: GROUPS_V2.to_owned(),
            etag: got.etag.clone(),
            request_id: String::new(),
        },
    )
    .await
    .unwrap();
    assert_ne!(replaced.etag, got.etag);
    // An edit on disk is what a reload picks up: write the original document back first.
    std::fs::write(&path, GROUPS).unwrap();
    let reloaded: ReloadResourceGroupConfigResponse = call(
        &mut grpc,
        "a:admin",
        "ReloadResourceGroupConfig",
        ReloadResourceGroupConfigRequest::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        reloaded.etag, got.etag,
        "the reload read the edited file, not the installed document"
    );
    assert_ne!(reloaded.etag, replaced.etag);
    let current: GetResourceGroupConfigResponse = call(
        &mut grpc,
        "o:operator",
        "GetResourceGroupConfig",
        GetResourceGroupConfigRequest {},
    )
    .await
    .unwrap();
    assert_eq!(current.document, got.document);
    let config: GetConfigResponse = call(
        &mut grpc,
        "o:operator",
        "GetConfig",
        GetConfigRequest {
            prefix: "krishiv.sql.ansi".to_owned(),
            ..GetConfigRequest::default()
        },
    )
    .await
    .unwrap();
    assert!(
        config
            .entries
            .iter()
            .any(|e| e.key == "krishiv.sql.ansi.enabled"),
        "{config:?}"
    );
    assert!(
        config
            .entries
            .iter()
            .all(|e| e.key.starts_with("krishiv.sql.ansi")),
        "the prefix filters the keys: {config:?}"
    );
    let unfiltered: GetConfigResponse = call(
        &mut grpc,
        "o:operator",
        "GetConfig",
        GetConfigRequest {
            page_size: 500,
            ..GetConfigRequest::default()
        },
    )
    .await
    .unwrap();
    assert!(unfiltered.entries.len() > config.entries.len());
    let set: SetConfigDefaultsResponse = call(
        &mut grpc,
        "a:admin",
        "SetConfigDefaults",
        SetConfigDefaultsRequest {
            set: [("krishiv.sql.ansi.enabled".to_owned(), "true".to_owned())].into(),
            ..SetConfigDefaultsRequest::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(set.set_count, 1);
    server.abort();
}
