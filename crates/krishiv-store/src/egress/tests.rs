// SPDX-License-Identifier: Apache-2.0
//! The address rule (design P3-T067 §5.1, §8 egress tests).

use std::collections::HashMap;
use std::net::IpAddr;

use super::*;

fn ip(text: &str) -> IpAddr {
    text.parse().unwrap()
}

fn deny(cidrs: &[&str], env: &[(&str, &str)]) -> Arc<EgressPolicy> {
    let env: HashMap<String, String> = env
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    EgressPolicy::new(
        PrivateAddresses::Deny,
        cidrs.iter().map(|c| Cidr::parse(c).unwrap()).collect(),
        ProviderEndpoints::from_env(|k| env.get(k).cloned()),
    )
}

#[test]
fn special_purpose_ranges_refused() {
    let scope = deny(&[], &[]).scope::<_, &str>([]);
    for refused in [
        "0.0.0.0",
        "0.1.2.3",
        "10.1.2.3",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "172.31.255.255",
        "192.0.0.8",
        "192.0.0.170",
        "192.0.2.1",
        "192.88.99.2",
        "192.168.1.1",
        "198.18.0.1",
        "198.19.255.255",
        "198.51.100.7",
        "203.0.113.9",
        "224.0.0.1",
        "240.0.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:169.254.169.254",
        "::10.0.0.1",
        "64:ff9b:1::1",
        "100::1",
        "100:0:0:1::1",
        "2001::1",
        "2001:2::1",
        "2001:10::1",
        "2001:db8::1",
        "2002:a00:1::1",
        "3fff::1",
        "5f00::1",
        "fc00::1",
        "fd12:3456::1",
        "fd00:ec2::254",
        "fe80::1",
        "fec0::1",
        "ff02::1",
    ] {
        assert!(!scope.permits(None, ip(refused)), "{refused} passed");
    }
    for allowed in [
        "8.8.8.8",
        "1.1.1.1",
        "172.32.0.1",
        "192.0.0.9",
        "192.0.0.10",
        "192.31.196.1",
        "192.52.193.1",
        "192.175.48.1",
        "::ffff:8.8.8.8",
        "2001:1::1",
        "2001:3::1",
        "2001:4:112::1",
        "2001:20::1",
        "2001:30::1",
        "2606:4700::1111",
        "2620:4f:8000::1",
    ] {
        assert!(scope.permits(None, ip(allowed)), "{allowed} refused");
    }
}

#[test]
fn nat64_embedded_ipv4_checked() {
    let scope = deny(&[], &[]).scope::<_, &str>([]);
    assert!(scope.permits(None, ip("64:ff9b::808:808")));
    assert!(!scope.permits(None, ip("64:ff9b::a9fe:a9fe")));
    assert!(!scope.permits(None, ip("64:ff9b::a00:1")));
}

#[test]
fn declared_catalog_host_exempt_by_name_only() {
    let scope = deny(&[], &[]).scope(["Catalog.Internal", "10.0.0.5"]);
    assert!(scope.permits(Some("catalog.internal"), ip("10.1.2.3")));
    assert!(scope.permits(Some("catalog.internal"), ip("127.0.0.1")));
    assert!(scope.permits(Some("catalog.internal"), ip("fd12::1")));
    // Another name resolving to the same private address is refused.
    assert!(!scope.permits(Some("other.internal"), ip("10.1.2.3")));
    // A literal is exempt only when it is the declared literal.
    assert!(scope.permits(None, ip("10.0.0.5")));
    assert!(!scope.permits(None, ip("10.0.0.6")));
    // Not for CGN or other special-purpose answers.
    assert!(!scope.permits(Some("catalog.internal"), ip("100.64.0.1")));
}

#[test]
fn link_local_never_exempt_by_name() {
    let scope = deny(&[], &[]).scope(["catalog.internal", "169.254.169.254"]);
    for a in [
        "169.254.169.254",
        "169.254.170.2",
        "fe80::1",
        "fd00:ec2::254",
    ] {
        assert!(!scope.permits(Some("catalog.internal"), ip(a)), "{a}");
        assert!(!scope.permits(None, ip(a)), "{a}");
    }
}

#[test]
fn exemption_scoped_to_owning_client() {
    let policy = deny(&[], &[]);
    let a = policy.scope(["a.internal"]);
    let b = policy.scope(["b.internal"]);
    assert!(a.permits(Some("a.internal"), ip("10.0.0.1")));
    assert!(!b.permits(Some("a.internal"), ip("10.0.0.1")));
    assert_ne!(a.fingerprint(), b.fingerprint());
    assert_eq!(a.fingerprint(), policy.scope(["A.internal"]).fingerprint());
}

#[test]
fn allowed_cidrs_admit() {
    let scope = deny(&["10.20.0.0/16", "fd00:ec2::254/128"], &[]).scope::<_, &str>([]);
    assert!(scope.permits(None, ip("10.20.3.4")));
    assert!(scope.permits(Some("any.name"), ip("::ffff:10.20.3.4")));
    assert!(!scope.permits(None, ip("10.21.0.1")));
    // An admin may open a link-local address only this way.
    assert!(scope.permits(None, ip("fd00:ec2::254")));
    assert!(Cidr::parse("10.0.0.0/33").is_err());
    assert!(Cidr::parse("bogus").is_err());
}

#[test]
fn provider_host_exemption_pins_address() {
    let scope = deny(
        &[],
        &[
            ("IDENTITY_ENDPOINT", "http://localhost:42356/msi/token"),
            ("GCE_METADATA_HOST", "gce-emulator:8080"),
        ],
    )
    .scope::<_, &str>([]);
    assert!(scope.permits(Some("metadata.google.internal"), ip("169.254.169.254")));
    for wrong in ["8.8.8.8", "10.0.0.1", "127.0.0.1", "169.254.170.2"] {
        assert!(
            !scope.permits(Some("metadata.google.internal"), ip(wrong)),
            "{wrong}"
        );
    }
    assert!(scope.permits(Some("localhost"), ip("127.0.0.1")));
    assert!(!scope.permits(Some("localhost"), ip("8.8.8.8")));
    assert!(!scope.permits(Some("localhost"), ip("10.0.0.1")));
    assert!(scope.permits(Some("gce-emulator"), ip("127.0.0.1")));
    assert!(scope.permits(Some("gce-emulator"), ip("169.254.169.254")));
    assert!(!scope.permits(Some("gce-emulator"), ip("192.168.0.1")));
}

#[test]
fn provider_literals_pass_only_for_store_built_requests() {
    let scope = deny(
        &[],
        &[(
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "http://127.0.0.9/creds",
        )],
    )
    .scope::<_, &str>([]);
    for literal in [
        "169.254.169.254",
        "169.254.170.2",
        "[fd00:ec2::254]",
        "127.0.0.9",
    ] {
        assert!(scope.permits_request_literal(literal).is_ok(), "{literal}");
    }
    assert!(scope.permits_request_literal("10.0.0.1").is_err());
    assert!(scope.permits_request_literal("example.com").is_ok());
    // A configured endpoint or a redirect hop gets no provider exemption.
    assert_eq!(
        scope.check_endpoint("http://169.254.169.254/latest"),
        Err(EgressRefusal::PrivateAddress)
    );
}

#[test]
fn literal_endpoint_checked_when_set() {
    let scope = deny(&[], &[]).scope(["10.0.0.5"]);
    assert!(scope.check_endpoint("http://10.0.0.5:9000").is_ok());
    assert!(scope.check_endpoint("http://10.0.0.6:9000").is_err());
    assert!(scope.check_endpoint("http://[::1]:9000/x").is_err());
    assert!(
        scope
            .check_endpoint("https://user@[fe80::1%25eth0]/")
            .is_err()
    );
    assert!(scope.check_endpoint("https://s3.amazonaws.com").is_ok());
    assert!(scope.check_endpoint("http://8.8.8.8").is_ok());
    // The refusal never names the address.
    let text = scope
        .check_endpoint("http://10.0.0.6")
        .unwrap_err()
        .to_string();
    assert!(!text.contains("10.0.0.6"), "{text}");
}

#[test]
fn filter_drops_refused_and_fails_when_none_left() {
    let scope = deny(&[], &[]).scope::<_, &str>([]);
    assert_eq!(
        scope.filter("x.example", &[ip("10.0.0.1"), ip("8.8.8.8")]),
        Ok(vec![ip("8.8.8.8")])
    );
    assert_eq!(
        scope.filter("x.example", &[ip("10.0.0.1"), ip("169.254.169.254")]),
        Err(EgressRefusal::NoAllowedAddress)
    );
}

#[test]
fn allow_mode_permits_everything() {
    let policy = EgressPolicy::new(
        PrivateAddresses::Allow,
        Vec::new(),
        ProviderEndpoints::default(),
    );
    let scope = policy.scope::<_, &str>([]);
    assert!(!scope.enforced());
    assert!(scope.permits(None, ip("169.254.169.254")));
    assert!(scope.check_endpoint("http://10.0.0.1").is_ok());
}

#[test]
fn host_of_forms() {
    assert_eq!(
        host_of("http://H.example:80/p").as_deref(),
        Some("h.example")
    );
    assert_eq!(host_of("h.example:80").as_deref(), Some("h.example"));
    assert_eq!(host_of("http://[::1]:80/").as_deref(), Some("::1"));
    assert_eq!(host_of("fd00:ec2::254").as_deref(), Some("fd00:ec2::254"));
    assert_eq!(host_of("http://u:p@h/").as_deref(), Some("h"));
    assert_eq!(host_of("http:///x"), None);
}

#[test]
fn redirect_hops_get_no_provider_exemption() {
    let scope = deny(&[], &[]).scope(["minio.internal"]);
    assert!(scope.check_redirect("169.254.169.254").is_err());
    assert!(scope.check_redirect("fd00:ec2::254").is_err());
    assert!(scope.check_redirect("metadata.google.internal").is_err());
    assert!(scope.check_redirect("10.0.0.1").is_err());
    assert!(scope.check_redirect("8.8.8.8").is_ok());
    assert!(scope.check_redirect("cdn.example.com").is_ok());
}

/// Review B1: every IPv4 form the WHATWG URL parser (and so reqwest) reads as an address is a
/// literal here, so no form slips past the literal checks as a "name".
#[test]
fn noncanonical_ipv4_literals_recognised() {
    let scope = deny(&[], &[]).scope::<_, &str>([]);
    for (form, canonical) in [
        ("2130706433", "127.0.0.1"),
        ("2852039166", "169.254.169.254"),
        ("0x7f.0.0.1", "127.0.0.1"),
        ("0177.0.0.1", "127.0.0.1"),
        ("127.1", "127.0.0.1"),
        ("10.1", "10.0.0.1"),
        ("127.0.0.1.", "127.0.0.1"),
        ("0xa9.0376.43518", "169.254.169.254"),
        ("0XA9FEA9FE", "169.254.169.254"),
    ] {
        assert_eq!(literal_ip(form), Some(ip(canonical)), "{form}");
        let url = format!("http://{form}:80/latest");
        assert_eq!(host_of(&url).as_deref(), Some(canonical), "{url}");
        assert_eq!(
            scope.check_endpoint(&url),
            Err(EgressRefusal::PrivateAddress),
            "{url}"
        );
        assert!(
            scope.permits_request_literal(form).is_err() || canonical == "169.254.169.254",
            "{form}"
        );
        assert!(scope.check_redirect(form).is_err(), "{form}");
    }
    // An exemption declared in one form covers the others: they are one address.
    let minio = deny(&[], &[]).scope(["167772165"]);
    assert!(minio.check_endpoint("http://10.0.0.5:9000").is_ok());
    // Names stay names.
    assert_eq!(literal_ip("s3.amazonaws.com"), None);
    assert_eq!(literal_ip("1x.example"), None);
}

/// Re-check of review 58a1: every refused host form gives a refusal from the startup check
/// (`check_endpoint`), the request-time check (`permits_request_literal`) and the redirect check,
/// in an IPv4 or IPv6 spelling, bracketed or not.
#[test]
fn every_refused_form_refused_at_startup_and_request_time() {
    let scope = deny(&[], &[("IDENTITY_ENDPOINT", "http://localhost:1/x")]).scope::<_, &str>([]);
    // (host as written, whether it is a provider literal object_store's own requests may use)
    for (host, provider) in [
        ("2130706433", false),
        ("2852039166", true),
        ("0x7f.0.0.1", false),
        ("0xa9fea9fe", true),
        ("0177.0.0.1", false),
        ("127.1", false),
        ("10.1", false),
        ("127.0.0.1.", false),
        ("::ffff:127.1", false),
        ("::ffff:7f00:1", false),
        // The mapped spellings of the metadata address are no provider literal: object_store
        // never writes them, so they are refused even for its own requests.
        ("[::ffff:169.254.169.254]", false),
        ("0:0:0:0:0:ffff:a9fe:a9fe", false),
    ] {
        let bracketed = if host.contains(':') && !host.starts_with('[') {
            format!("[{host}]")
        } else {
            host.to_owned()
        };
        for endpoint in [
            format!("http://{bracketed}:80/latest"),
            format!("https://{bracketed}/"),
            bracketed.clone(),
        ] {
            assert_eq!(
                scope.check_endpoint(&endpoint),
                Err(EgressRefusal::PrivateAddress),
                "startup: {endpoint}"
            );
        }
        for request_host in [host, bracketed.as_str()] {
            // A data request (no provider exemption) and a redirect hop are refused for every
            // form; a store-built request is refused unless it names a provider literal.
            assert!(
                scope.check_redirect(request_host).is_err(),
                "redirect: {request_host}"
            );
            assert_eq!(
                scope.permits_request_literal(request_host).is_ok(),
                provider,
                "request: {request_host}"
            );
        }
    }
    // Names are names: resolved and checked by the resolver.
    for name in ["s3.amazonaws.com", "storage.googleapis.com", "1x.example"] {
        assert!(
            scope.check_endpoint(&format!("https://{name}/")).is_ok(),
            "{name}"
        );
        assert!(scope.permits_request_literal(name).is_ok(), "{name}");
    }
    // The allow policy refuses nothing, malformed or not.
    let allow = EgressPolicy::new(
        PrivateAddresses::Allow,
        Vec::new(),
        ProviderEndpoints::default(),
    )
    .scope::<_, &str>([]);
    assert!(allow.check_endpoint("http://[::ffff:127.1]/").is_ok());
}
