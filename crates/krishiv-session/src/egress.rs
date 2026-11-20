// SPDX-License-Identifier: Apache-2.0
//! The server's egress policy from its startup configuration:
//! `krishiv.io.egress.privateAddresses`, `allowedCidrs`, `proxy`, the credential-provider
//! endpoints of the server's environment, and the admin-set store endpoints that object-store
//! clients may reach at private addresses by name.
//!
//! The rule itself is `krishiv_io::egress` (krishiv-store). Embedded sessions default to `allow`;
//! `krishiv server` sets `deny` unless the admin chose otherwise.

use std::sync::Arc;

use krishiv_common::config::{
    Config, ConfigValue, IO_EGRESS_ALLOWED_CIDRS, IO_EGRESS_PRIVATE_ADDRESSES, IO_EGRESS_PROXY,
};
use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue};
use krishiv_io::egress::{Cidr, EgressPolicy, EgressScope, PrivateAddresses, ProviderEndpoints};

/// The `fs.*` endpoint keys an admin sets at startup (`krishiv-store`'s `S3_ENDPOINT`,
/// `GCS_ENDPOINT`, `GCS_STORAGE_ROOT_URL`, `AZURE_ENDPOINT`, `HF_ENDPOINT`): their hosts are the
/// object-store clients' by-name exemptions.
const STORE_ENDPOINT_KEYS: [&str; 5] = [
    "fs.s3a.endpoint",
    "fs.gs.endpoint",
    "fs.gs.storage.root.url",
    "fs.azure.endpoint",
    "fs.hf.endpoint",
];

fn invalid(key: &str, value: &str, reason: &str) -> KrishivError {
    KrishivInvalidConfigValue {
        value: format!("'{value}'"),
        key: format!("\"{key}\""),
        reason: reason.to_owned(),
    }
    .into()
}

/// The egress policy of `config`, reading the provider endpoints through `env`.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` for a network in `allowedCidrs` that does not parse.
pub fn egress_policy(
    config: &Config,
    env: impl Fn(&str) -> Option<String>,
) -> Result<Arc<EgressPolicy>, KrishivError> {
    let mode = match config
        .get(&IO_EGRESS_PRIVATE_ADDRESSES)
        .and_then(ConfigValue::as_str)
    {
        Some(v) if v.eq_ignore_ascii_case("deny") => PrivateAddresses::Deny,
        _ => PrivateAddresses::Allow,
    };
    let mut cidrs = Vec::new();
    if let Some(raw) = config
        .get(&IO_EGRESS_ALLOWED_CIDRS)
        .and_then(ConfigValue::as_str)
    {
        for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            cidrs.push(
                Cidr::parse(part)
                    .map_err(|reason| invalid(IO_EGRESS_ALLOWED_CIDRS.key, part, &reason))?,
            );
        }
    }
    Ok(EgressPolicy::new(
        mode,
        cidrs,
        ProviderEndpoints::from_env(env),
    ))
}

/// The admin's egress proxy (`krishiv.io.egress.proxy`), if set.
#[must_use]
pub fn egress_proxy(config: &Config) -> Option<String> {
    config
        .get(&IO_EGRESS_PROXY)
        .and_then(ConfigValue::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

/// The scope of the object-store clients: the hosts of the admin-set store endpoints of
/// `config`, each literal endpoint checked now (§5.2: no later point exists for a request the
/// store builds itself).
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` naming the key (never the address) for an endpoint whose
/// literal address the rule refuses even with the exemption (link-local, special-purpose).
pub fn store_scope(
    policy: &Arc<EgressPolicy>,
    config: &Config,
) -> Result<EgressScope, KrishivError> {
    let endpoints: Vec<(&str, String)> = STORE_ENDPOINT_KEYS
        .iter()
        .filter_map(|key| {
            config
                .get_unregistered(key)
                .map(|v| (*key, v.trim().to_owned()))
        })
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let scope = policy.scope(
        endpoints
            .iter()
            .filter_map(|(_, v)| krishiv_io::egress::host_of(v)),
    );
    for (key, endpoint) in &endpoints {
        if scope.check_endpoint(endpoint).is_err() {
            return Err(invalid(
                key,
                "<endpoint>",
                "the egress policy refuses this endpoint's address (private address); \
                 list its network in krishiv.io.egress.allowedCidrs if it is meant",
            ));
        }
    }
    Ok(scope)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(pairs: &[(&str, &str)]) -> Config {
        Config::from_pairs(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned())),
        )
        .unwrap()
        .0
    }

    #[test]
    fn embedded_default_allows_and_server_deny_refuses() {
        let none = |_: &str| None;
        let embedded = egress_policy(&config(&[]), none).unwrap();
        assert!(!embedded.enforced());
        let server = egress_policy(
            &config(&[("krishiv.io.egress.privateAddresses", "deny")]),
            none,
        )
        .unwrap();
        assert!(server.enforced());
    }

    #[test]
    fn allowed_cidrs_parse_or_fail_startup() {
        let none = |_: &str| None;
        let bad = config(&[("krishiv.io.egress.allowedCidrs", "10.0.0.0/8, nope")]);
        let err = egress_policy(&bad, none).unwrap_err().to_string();
        assert!(err.contains("krishiv.io.egress.allowedCidrs"), "{err}");
        let good = config(&[
            ("krishiv.io.egress.privateAddresses", "deny"),
            ("krishiv.io.egress.allowedCidrs", "10.0.0.0/8"),
        ]);
        let scope = egress_policy(&good, none).unwrap().scope::<_, &str>([]);
        assert!(scope.permits(None, "10.1.2.3".parse().unwrap()));
    }

    #[test]
    fn egress_proxy_only_when_set() {
        assert_eq!(egress_proxy(&config(&[])), None);
        assert_eq!(
            egress_proxy(&config(&[("krishiv.io.egress.proxy", "  ")])),
            None,
            "a blank proxy is no proxy"
        );
        assert_eq!(
            egress_proxy(&config(&[(
                "krishiv.io.egress.proxy",
                " http://proxy.internal:3128 "
            )])),
            Some("http://proxy.internal:3128".to_owned())
        );
    }

    #[test]
    fn literal_endpoint_checked_when_set() {
        let none = |_: &str| None;
        let deny = config(&[
            ("krishiv.io.egress.privateAddresses", "deny"),
            ("fs.s3a.endpoint", "http://169.254.169.254:9000"),
        ]);
        let policy = egress_policy(&deny, none).unwrap();
        let err = store_scope(&policy, &deny).unwrap_err().to_string();
        assert!(err.contains("fs.s3a.endpoint"), "{err}");
        assert!(!err.contains("169.254"), "{err}");
        // Review B1: the metadata address written in decimal, hex or short form is refused too.
        for endpoint in [
            "http://2852039166:80",
            "http://0xa9fea9fe",
            "http://169.254.43518",
            "http://169.254.169.254.",
        ] {
            let c = config(&[
                ("krishiv.io.egress.privateAddresses", "deny"),
                ("fs.s3a.endpoint", endpoint),
            ]);
            let policy = egress_policy(&c, none).unwrap();
            assert!(store_scope(&policy, &c).is_err(), "{endpoint}");
        }
        // An on-premises MinIO at a private literal is exempt by name.
        let minio = config(&[
            ("krishiv.io.egress.privateAddresses", "deny"),
            ("fs.s3a.endpoint", "http://10.0.0.5:9000"),
        ]);
        let policy = egress_policy(&minio, none).unwrap();
        let scope = store_scope(&policy, &minio).unwrap();
        assert!(scope.permits(None, "10.0.0.5".parse().unwrap()));
        assert!(!scope.permits(None, "10.0.0.6".parse().unwrap()));
    }
}
