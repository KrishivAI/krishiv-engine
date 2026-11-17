// SPDX-License-Identifier: Apache-2.0
//! [`AdminApi`] over the engine runtime: the version and the `system.capabilities` rows.

use futures::future::{BoxFuture, FutureExt};

use super::{
    ADMIN_V1_REVISION, AdminApi, AdminResult, Capability, GetCapabilitiesResponse,
    GetVersionResponse, PREVIEW,
};

/// The admin API of a `krishiv server` process. Creating it records the `admin.v1` surface row
/// in `system.capabilities` (`("surface", "admin.v1", <revision>, "preview" | "")`), so a client
/// that needs a field of a newer revision can check the revision first.
#[derive(Debug, Clone, Copy)]
pub struct SessionAdmin;

impl SessionAdmin {
    /// The admin API; registers the `admin.v1` capability row.
    #[must_use]
    pub fn new() -> Self {
        krishiv_session::register_surface(
            "admin.v1",
            &ADMIN_V1_REVISION.to_string(),
            if PREVIEW { "preview" } else { "" },
        );
        Self
    }
}

impl Default for SessionAdmin {
    fn default() -> Self {
        Self::new()
    }
}

impl AdminApi for SessionAdmin {
    fn get_version(&self) -> BoxFuture<'_, AdminResult<GetVersionResponse>> {
        async {
            Ok(GetVersionResponse {
                version: crate::VERSION.to_owned(),
                admin_v1_revision: ADMIN_V1_REVISION,
                preview: PREVIEW,
            })
        }
        .boxed()
    }

    fn get_capabilities(&self) -> BoxFuture<'_, AdminResult<GetCapabilitiesResponse>> {
        async {
            Ok(GetCapabilitiesResponse {
                capabilities: krishiv_session::capabilities()
                    .into_iter()
                    .map(|[kind, name, value, detail]| Capability {
                        kind,
                        name,
                        value,
                        detail,
                    })
                    .collect(),
            })
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_capability_rows_carry_the_surface_revision() {
        let admin = SessionAdmin::new();
        let version = admin
            .get_version()
            .await
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert_eq!(version.version, crate::VERSION);
        assert_eq!(version.admin_v1_revision, ADMIN_V1_REVISION);
        let rows = admin
            .get_capabilities()
            .await
            .unwrap_or_else(|e| panic!("{}", e.message))
            .capabilities;
        let row = rows
            .iter()
            .find(|c| c.kind == "surface" && c.name == "admin.v1")
            .unwrap_or_else(|| panic!("no admin.v1 surface row"));
        assert_eq!(row.value, ADMIN_V1_REVISION.to_string());
        assert_eq!(row.detail, "preview");
    }
}
