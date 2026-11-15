// SPDX-License-Identifier: Apache-2.0
//! The egress address rule (design `P3-T067-declared-catalogs.md` §5.1; threat model T-C2, T-C7).
//!
//! In server mode every outbound connection the server makes for a catalog or for data passes an
//! **allow rule**: a destination address passes only when it is globally routable unicast, when it
//! lies in an admin-set `krishiv.io.egress.allowedCidrs` network, or when an exemption below admits
//! it. Everything else (loopback, link-local, RFC 1918, carrier-grade NAT, documentation,
//! benchmarking, reserved, multicast, unique-local, 6to4, local-use NAT64 ...) fails without being
//! listed.
//!
//! The special-purpose tables are the IANA IPv4 and IPv6 Special-Purpose Address Registries,
//! both last updated 2025-10-09. An entry marked "Globally Reachable: True" is global here; every
//! other entry is not. Three IPv6 forms that carry an IPv4 address are checked as the embedded
//! IPv4: IPv4-mapped `::ffff:0:0/96`, IPv4-compatible `::/96` and the well-known NAT64 prefix
//! `64:ff9b::/96`.
//!
//! Exemptions, each scoped to the client that owns it ([`EgressScope`]):
//! - **By name.** A host an admin declared (a catalog `uri` or token endpoint, an admin-set store
//!   endpoint) may resolve to an RFC 1918, unique-local or loopback address. Only the exact host
//!   name declared, only for the client that owns it. Link-local is never exempt by name.
//! - **Credential providers** (§5.1 N1). The ambient provider host names (`metadata.google.internal`,
//!   the hosts of `GCE_METADATA_HOST` / `GCE_METADATA_ROOT` and `IDENTITY_ENDPOINT`) are pinned:
//!   each may resolve only to the address it stands for, and any other answer is refused. The
//!   provider literal endpoints (`169.254.169.254`, `169.254.170.2`, ...) come only from the
//!   server's environment; [`EgressScope::permits_request_literal`] admits them for a request the
//!   store itself built, never for a redirect hop or a configured endpoint.

use std::collections::BTreeSet;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// What an address is, for the egress rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressKind {
    /// Globally routable unicast: allowed.
    Global,
    /// RFC 1918 or IPv6 unique-local: allowed for a host exempt by name.
    Private,
    /// Loopback: allowed for a host exempt by name.
    Loopback,
    /// Link-local, including the cloud metadata addresses: never exempt by name.
    LinkLocal,
    /// Any other special-purpose, multicast or reserved address: never exempt by name.
    Special,
}

/// The cloud metadata addresses inside otherwise private ranges (`fd00:ec2::254` is unique-local),
/// classed as link-local so no by-name exemption reaches them.
const METADATA_V6: [Ipv6Addr; 2] = [
    Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254),
    Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0023),
];

/// Classifies `ip` (see the module docs for the tables).
#[must_use]
pub fn classify(ip: IpAddr) -> AddressKind {
    match ip {
        IpAddr::V4(v4) => classify_v4(v4),
        IpAddr::V6(v6) => classify_v6(v6),
    }
}

fn in_v4(ip: Ipv4Addr, net: [u8; 4], prefix: u32) -> bool {
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - prefix)
    };
    (u32::from(ip) & mask) == (u32::from(Ipv4Addr::from(net)) & mask)
}

fn in_v6(ip: Ipv6Addr, net: Ipv6Addr, prefix: u32) -> bool {
    let mask = if prefix == 0 {
        0
    } else {
        u128::MAX << (128 - prefix)
    };
    (u128::from(ip) & mask) == (u128::from(net) & mask)
}

fn classify_v4(ip: Ipv4Addr) -> AddressKind {
    if in_v4(ip, [127, 0, 0, 0], 8) {
        return AddressKind::Loopback;
    }
    if in_v4(ip, [169, 254, 0, 0], 16) {
        return AddressKind::LinkLocal;
    }
    if in_v4(ip, [10, 0, 0, 0], 8)
        || in_v4(ip, [172, 16, 0, 0], 12)
        || in_v4(ip, [192, 168, 0, 0], 16)
    {
        return AddressKind::Private;
    }
    // "Globally Reachable: True" entries inside the blocks refused below.
    const GLOBAL: [([u8; 4], u32); 5] = [
        ([192, 0, 0, 9], 32),
        ([192, 0, 0, 10], 32),
        ([192, 31, 196, 0], 24),
        ([192, 52, 193, 0], 24),
        ([192, 175, 48, 0], 24),
    ];
    if GLOBAL.iter().any(|(net, p)| in_v4(ip, *net, *p)) {
        return AddressKind::Global;
    }
    const SPECIAL: [([u8; 4], u32); 11] = [
        ([0, 0, 0, 0], 8),
        ([100, 64, 0, 0], 10),
        ([192, 0, 0, 0], 24),
        ([192, 0, 2, 0], 24),
        ([192, 88, 99, 0], 24),
        ([198, 18, 0, 0], 15),
        ([198, 51, 100, 0], 24),
        ([203, 0, 113, 0], 24),
        // Multicast is not in the registry but is not unicast.
        ([224, 0, 0, 0], 4),
        // Reserved, with the limited broadcast address inside it.
        ([240, 0, 0, 0], 4),
        ([255, 255, 255, 255], 32),
    ];
    if SPECIAL.iter().any(|(net, p)| in_v4(ip, *net, *p)) {
        return AddressKind::Special;
    }
    AddressKind::Global
}

/// The IPv4 address an IPv6 address carries, for the three forms checked as IPv4.
fn embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let bits = u128::from(ip);
    let low = Ipv4Addr::from(u32::try_from(bits & 0xffff_ffff).ok()?);
    let mapped = in_v6(ip, Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0, 0), 96);
    let nat64 = in_v6(ip, Ipv6Addr::new(0x64, 0xff9b, 0, 0, 0, 0, 0, 0), 96);
    // IPv4-compatible (deprecated): `::a.b.c.d`, but not `::` or `::1`.
    let compatible = in_v6(ip, Ipv6Addr::UNSPECIFIED, 96) && bits > 1;
    (mapped || nat64 || compatible).then_some(low)
}

fn classify_v6(ip: Ipv6Addr) -> AddressKind {
    if let Some(v4) = embedded_v4(ip) {
        return classify_v4(v4);
    }
    if ip == Ipv6Addr::LOCALHOST {
        return AddressKind::Loopback;
    }
    if METADATA_V6.contains(&ip) || in_v6(ip, Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0), 10) {
        return AddressKind::LinkLocal;
    }
    if in_v6(ip, Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0), 7) {
        return AddressKind::Private;
    }
    // Only global unicast `2000::/3` can pass; everything outside it is special-purpose,
    // multicast or unassigned.
    if !in_v6(ip, Ipv6Addr::new(0x2000, 0, 0, 0, 0, 0, 0, 0), 3) {
        return AddressKind::Special;
    }
    // "Globally Reachable: True" entries inside 2001::/23.
    let global: [(Ipv6Addr, u32); 7] = [
        (Ipv6Addr::new(0x2001, 1, 0, 0, 0, 0, 0, 1), 128),
        (Ipv6Addr::new(0x2001, 1, 0, 0, 0, 0, 0, 2), 128),
        (Ipv6Addr::new(0x2001, 1, 0, 0, 0, 0, 0, 3), 128),
        (Ipv6Addr::new(0x2001, 3, 0, 0, 0, 0, 0, 0), 32),
        (Ipv6Addr::new(0x2001, 4, 0x112, 0, 0, 0, 0, 0), 48),
        (Ipv6Addr::new(0x2001, 0x20, 0, 0, 0, 0, 0, 0), 28),
        (Ipv6Addr::new(0x2001, 0x30, 0, 0, 0, 0, 0, 0), 28),
    ];
    if global.iter().any(|(net, p)| in_v6(ip, *net, *p)) {
        return AddressKind::Global;
    }
    let special: [(Ipv6Addr, u32); 4] = [
        // IETF protocol assignments (TEREDO, benchmarking, deprecated ORCHID, ...).
        (Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 23),
        (Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32),
        // 6to4: the relay decides where it goes.
        (Ipv6Addr::new(0x2002, 0, 0, 0, 0, 0, 0, 0), 16),
        (Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0), 20),
    ];
    if special.iter().any(|(net, p)| in_v6(ip, *net, *p)) {
        return AddressKind::Special;
    }
    AddressKind::Global
}

/// A network in CIDR form (`10.1.0.0/16`, `fd12::/48`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cidr {
    net: IpAddr,
    prefix: u32,
}

impl Cidr {
    /// Parses `a.b.c.d/n` or `x::y/n`. A bare address is a single-address network.
    ///
    /// # Errors
    /// The text is not an address with an in-range prefix length.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let (addr, prefix) = match text.split_once('/') {
            Some((a, p)) => (a, Some(p)),
            None => (text, None),
        };
        let net: IpAddr = addr
            .parse()
            .map_err(|_| format!("`{text}` is not a CIDR network"))?;
        let max = if net.is_ipv4() { 32 } else { 128 };
        let prefix = match prefix {
            Some(p) => p
                .parse::<u32>()
                .ok()
                .filter(|p| *p <= max)
                .ok_or_else(|| format!("`{text}` has no valid prefix length"))?,
            None => max,
        };
        Ok(Self { net, prefix })
    }

    /// Whether `ip` lies in this network (an IPv4 network also holds the IPv4-mapped forms).
    #[must_use]
    pub fn contains(&self, ip: IpAddr) -> bool {
        let ip = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
            IpAddr::V4(_) => ip,
        };
        match (self.net, ip) {
            (IpAddr::V4(net), IpAddr::V4(ip)) => in_v4(ip, net.octets(), self.prefix),
            (IpAddr::V6(net), IpAddr::V6(ip)) => in_v6(ip, net, self.prefix),
            _ => false,
        }
    }
}

/// Where a pinned provider host name may resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Pin {
    /// Exactly these addresses.
    Addresses(Vec<IpAddr>),
    /// Any loopback address.
    Loopback,
    /// These addresses or any loopback address.
    AddressesOrLoopback(Vec<IpAddr>),
}

impl Pin {
    fn admits(&self, ip: IpAddr) -> bool {
        let loopback = classify(ip) == AddressKind::Loopback;
        match self {
            Self::Addresses(list) => list.contains(&ip),
            Self::Loopback => loopback,
            Self::AddressesOrLoopback(list) => loopback || list.contains(&ip),
        }
    }
}

const GCE_METADATA_IP: IpAddr = IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254));

/// The server's credential-provider endpoints, computed once at startup from its environment
/// (§5.1 N1): host names pinned to the address each stands for, and the literal addresses the
/// providers call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderEndpoints {
    hosts: Vec<(String, Pin)>,
    literals: Vec<IpAddr>,
}

impl ProviderEndpoints {
    /// The endpoints object_store 0.14.2's ambient providers call, read through `env` (the
    /// process environment in production; a map in tests).
    ///
    /// Literal endpoints: AWS instance metadata `169.254.169.254` / `fd00:ec2::254`, ECS
    /// `169.254.170.2`, EKS pod identity `169.254.170.23` / `fd00:ec2::23`, Azure IMDS, and the
    /// literal hosts of `AWS_METADATA_ENDPOINT`, `AWS_CONTAINER_CREDENTIALS_FULL_URI` and
    /// `GCE_METADATA_IP`. Host names: `metadata.google.internal` pinned to `169.254.169.254`;
    /// the `GCE_METADATA_HOST` / `GCE_METADATA_ROOT` host pinned to that address or loopback (an
    /// emulator); the `IDENTITY_ENDPOINT` host pinned to loopback (Azure App Service).
    pub fn from_env(env: impl Fn(&str) -> Option<String>) -> Self {
        let mut out = Self {
            hosts: vec![(
                "metadata.google.internal".to_owned(),
                Pin::Addresses(vec![GCE_METADATA_IP]),
            )],
            literals: vec![
                GCE_METADATA_IP,
                IpAddr::V4(Ipv4Addr::new(169, 254, 170, 2)),
                IpAddr::V4(Ipv4Addr::new(169, 254, 170, 23)),
                IpAddr::V6(METADATA_V6[0]),
                IpAddr::V6(METADATA_V6[1]),
            ],
        };
        for var in [
            "AWS_METADATA_ENDPOINT",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "GCE_METADATA_IP",
        ] {
            if let Some(ip) = env(var)
                .as_deref()
                .and_then(host_of)
                .as_deref()
                .and_then(literal_ip)
            {
                out.literals.push(ip);
            }
        }
        for (var, pin) in [
            (
                "GCE_METADATA_HOST",
                Pin::AddressesOrLoopback(vec![GCE_METADATA_IP]),
            ),
            (
                "GCE_METADATA_ROOT",
                Pin::AddressesOrLoopback(vec![GCE_METADATA_IP]),
            ),
            ("IDENTITY_ENDPOINT", Pin::Loopback),
        ] {
            let Some(host) = env(var).as_deref().and_then(host_of) else {
                continue;
            };
            match literal_ip(&host) {
                Some(ip) => out.literals.push(ip),
                None => out.hosts.push((host, pin)),
            }
        }
        out
    }

    fn pin(&self, host: &str) -> Option<&Pin> {
        self.hosts
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(host))
            .map(|(_, pin)| pin)
    }
}

/// The host of `endpoint`: a URL (`http://host:port/path`) or a bare `host[:port]`. Lowercased,
/// IPv6 brackets removed, and an address literal in its canonical form whatever form it was
/// written in (`2130706433`, `0x7f.0.0.1`, `127.1` and `127.0.0.1.` are all `127.0.0.1`), as the
/// URL parser reqwest dials through reads it.
#[must_use]
pub fn host_of(endpoint: &str) -> Option<String> {
    let rest = endpoint
        .split_once("://")
        .map_or(endpoint, |(_, rest)| rest);
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    let host = if let Some(v6) = authority.strip_prefix('[') {
        v6.split_once(']').map(|(h, _)| h)?
    } else if authority.matches(':').count() > 1 {
        // A bare IPv6 address without brackets.
        authority
    } else {
        authority.split(':').next()?
    };
    if host.is_empty() {
        return None;
    }
    Some(literal_ip(host).map_or_else(|| host.to_ascii_lowercase(), |ip| ip.to_string()))
}

/// `host` as an IP address when it is a literal (brackets and an IPv6 zone removed). An IPv4
/// host is read by the WHATWG URL host parser (the `url` crate reqwest builds its requests
/// with), so every form a request would dial as an address is one here: decimal, octal and hex
/// parts, fewer than four parts, a trailing dot (review B1). A host that is no address is a name.
#[must_use]
pub fn literal_ip(host: &str) -> Option<IpAddr> {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let host = host.split('%').next().unwrap_or(host);
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(ip);
    }
    let parsed = if host.contains(':') {
        url::Host::parse(&format!("[{host}]"))
    } else {
        url::Host::parse(host)
    };
    match parsed {
        Ok(url::Host::Ipv4(v4)) => Some(IpAddr::V4(v4)),
        Ok(url::Host::Ipv6(v6)) => Some(IpAddr::V6(v6)),
        _ => None,
    }
}

/// Whether `host` is neither an address nor a host name the URL parser accepts: an IPv6-like
/// host that is not an address (`::ffff:127.1`), or a number-shaped IPv4 host out of range. No
/// request can be dialled to it, so the checks refuse it rather than leave it to the resolver.
fn malformed(host: &str) -> bool {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let bare = bare.split('%').next().unwrap_or(bare);
    if bare.is_empty() || literal_ip(bare).is_some() {
        return false;
    }
    bare.contains(':') || url::Host::parse(bare).is_err()
}

/// Whether the egress rule applies (`krishiv.io.egress.privateAddresses`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrivateAddresses {
    /// Every address is allowed (embedded mode's default).
    #[default]
    Allow,
    /// The allow rule of the module docs (server mode's default).
    Deny,
}

/// The startup-fixed egress policy: the mode, `allowedCidrs` and the provider endpoints.
#[derive(Debug)]
pub struct EgressPolicy {
    mode: PrivateAddresses,
    allowed_cidrs: Vec<Cidr>,
    providers: ProviderEndpoints,
    generation: u64,
}

static GENERATION: AtomicU64 = AtomicU64::new(1);

impl EgressPolicy {
    /// A policy; each one gets a process-unique generation, part of the client fingerprint.
    #[must_use]
    pub fn new(
        mode: PrivateAddresses,
        allowed_cidrs: Vec<Cidr>,
        providers: ProviderEndpoints,
    ) -> Arc<Self> {
        Arc::new(Self {
            mode,
            allowed_cidrs,
            providers,
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Whether the rule applies at all.
    #[must_use]
    pub fn enforced(&self) -> bool {
        self.mode == PrivateAddresses::Deny
    }

    /// A scope for one client, exempting `exempt_hosts` by name (each a host as declared: a
    /// name or an address literal; lowercased here).
    #[must_use]
    pub fn scope<I, S>(self: &Arc<Self>, exempt_hosts: I) -> EgressScope
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        EgressScope {
            policy: Arc::clone(self),
            exempt: exempt_hosts
                .into_iter()
                .map(|h| {
                    let h = h.as_ref().trim().to_ascii_lowercase();
                    literal_ip(&h).map_or(h, |ip| ip.to_string())
                })
                .filter(|h| !h.is_empty())
                .collect(),
        }
    }
}

/// Why a destination was refused. Never carries the address (§5.2: no message names it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EgressRefusal {
    /// The address is not globally routable and no exemption admits it.
    PrivateAddress,
    /// A host name resolved to no address the rule allows.
    NoAllowedAddress,
}

impl fmt::Display for EgressRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PrivateAddress => "private address",
            Self::NoAllowedAddress => "no allowed address",
        })
    }
}

impl std::error::Error for EgressRefusal {}

/// The egress rule as one client applies it: the policy plus the hosts that client may reach
/// at private addresses by name (§5.1, scoped to the owning client).
#[derive(Debug, Clone)]
pub struct EgressScope {
    policy: Arc<EgressPolicy>,
    exempt: BTreeSet<String>,
}

impl EgressScope {
    /// Whether the rule applies at all.
    #[must_use]
    pub fn enforced(&self) -> bool {
        self.policy.enforced()
    }

    /// Whether `ip`, reached through the name `host` (`None` for an address literal), passes.
    #[must_use]
    pub fn permits(&self, host: Option<&str>, ip: IpAddr) -> bool {
        if !self.policy.enforced() || self.policy.allowed_cidrs.iter().any(|c| c.contains(ip)) {
            return true;
        }
        // A provider host name is pinned: no other rule widens or narrows it.
        if let Some(pin) = host.and_then(|h| self.policy.providers.pin(h)) {
            return pin.admits(ip);
        }
        match classify(ip) {
            AddressKind::Global => true,
            AddressKind::Private | AddressKind::Loopback => {
                let name = host.map_or_else(|| ip.to_string(), str::to_ascii_lowercase);
                self.exempt.contains(&name)
            }
            AddressKind::LinkLocal | AddressKind::Special => false,
        }
    }

    /// The addresses of `resolved` (answers for the name `host`) the rule allows, or a refusal
    /// when none is left.
    ///
    /// # Errors
    /// [`EgressRefusal::NoAllowedAddress`] when every address is refused.
    pub fn filter(&self, host: &str, resolved: &[IpAddr]) -> Result<Vec<IpAddr>, EgressRefusal> {
        let kept: Vec<IpAddr> = resolved
            .iter()
            .copied()
            .filter(|ip| self.permits(Some(host), *ip))
            .collect();
        if kept.is_empty() {
            Err(EgressRefusal::NoAllowedAddress)
        } else {
            Ok(kept)
        }
    }

    /// Checks the host of a configured endpoint or a redirect target when it is an address
    /// literal (a name is checked when it resolves). The provider literals are **not** exempt
    /// here: they come only from the environment, never from a configured endpoint.
    ///
    /// # Errors
    /// [`EgressRefusal::PrivateAddress`] for a refused literal.
    pub fn check_endpoint(&self, endpoint: &str) -> Result<(), EgressRefusal> {
        if self.policy.enforced() && host_of(endpoint).as_deref().is_some_and(malformed) {
            return Err(EgressRefusal::PrivateAddress);
        }
        match host_of(endpoint).as_deref().and_then(literal_ip) {
            Some(ip) if !self.permits(None, ip) => Err(EgressRefusal::PrivateAddress),
            _ => Ok(()),
        }
    }

    /// Checks the host of a request the store built itself when it is an address literal. Unlike
    /// [`Self::check_endpoint`] the provider literals pass: object_store's ambient credential
    /// providers call them from the server's environment (§5.1 N1).
    ///
    /// # Errors
    /// [`EgressRefusal::PrivateAddress`] for a refused literal.
    pub fn permits_request_literal(&self, host: &str) -> Result<(), EgressRefusal> {
        if self.policy.enforced() && malformed(host) {
            return Err(EgressRefusal::PrivateAddress);
        }
        match literal_ip(host) {
            Some(ip)
                if !self.permits(None, ip) && !self.policy.providers.literals.contains(&ip) =>
            {
                Err(EgressRefusal::PrivateAddress)
            }
            _ => Ok(()),
        }
    }

    /// Checks one redirect hop's host (§5.2): an address literal must pass the rule without the
    /// provider literals, and a credential-provider host name is never a redirect target (its
    /// pin would otherwise admit the metadata address). Other names are checked when they
    /// resolve.
    ///
    /// # Errors
    /// [`EgressRefusal::PrivateAddress`] for a refused hop.
    pub fn check_redirect(&self, host: &str) -> Result<(), EgressRefusal> {
        if !self.policy.enforced() {
            return Ok(());
        }
        if malformed(host) {
            return Err(EgressRefusal::PrivateAddress);
        }
        match literal_ip(host) {
            Some(ip) if self.permits(None, ip) => Ok(()),
            Some(_) => Err(EgressRefusal::PrivateAddress),
            None if self.policy.providers.pin(host).is_some() => Err(EgressRefusal::PrivateAddress),
            None => Ok(()),
        }
    }

    /// The fingerprint component of a client with this scope (§5.2): the policy generation and
    /// the exempt host set, so clients of different scopes are never shared.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let hosts: Vec<&str> = self.exempt.iter().map(String::as_str).collect();
        format!(
            "policy:{}:{}:{}",
            self.policy.generation,
            u8::from(self.policy.enforced()),
            hosts.join(",")
        )
    }
}

#[cfg(test)]
mod tests;
