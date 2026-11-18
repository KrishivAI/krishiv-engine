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
    let rt = builder.build().unwrap();
    let app = router(
        Arc::new(SessionAdmin::new(rt)),
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
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
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
    assert!(
        body["groups"].as_array().is_none_or(Vec::is_empty),
        "{body}"
    );
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
    let on_disk = std::fs::read_to_string(&f.path).unwrap();
    assert!(
        on_disk.contains('8'),
        "the file holds the new document: {on_disk}"
    );
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
    assert!(
        body["document"].as_str().is_none_or(str::is_empty),
        "{body}"
    );
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
async fn get_config_lists_keys_with_source_and_withholds_the_auth_values() {
    let f = fixture(false);
    let (status, _) = f.call("v:viewer", "GET", "/config", None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = f
        .call(
            "o:operator",
            "GET",
            "/config?prefix=krishiv.auth.&pageSize=500",
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entries = body["entries"].as_array().unwrap();
    // The authentication settings never reach the runtime's configuration: nothing is listed
    // with a value, whoever asks.
    assert!(
        entries
            .iter()
            .all(|e| e["withheld"] == true && e["value"].is_null()),
        "{body}"
    );
    let (_, body) = f
        .call(
            "o:operator",
            "GET",
            "/config?prefix=krishiv.workload.resourceGroups.file",
            None,
        )
        .await;
    assert_eq!(
        body["entries"][0]["key"],
        "krishiv.workload.resourceGroups.file"
    );
    assert_eq!(body["entries"][0]["source"], "default");
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
    let batch = serde_json::json!({ "set": { key: "true", "krishiv.auth.defaultRole": "admin" } })
        .to_string();
    let (status, body) = f
        .call("a:admin", "PATCH", "/config/defaults", Some(&batch))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["errorClass"], "CANNOT_MODIFY_STATIC_CONFIG");
    let (_, body) = f
        .call("a:admin", "GET", &format!("/config?prefix={key}"), None)
        .await;
    assert_ne!(body["entries"][0]["source"], "admin", "{body}");

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
