// SPDX-License-Identifier: Apache-2.0
//! Host hygiene of a server process (design P3-T001 §3.6, threat model
//! T-A1, T-H1, T-H3): which addresses need authentication, the root check
//! and core dumps off. Unix only; elsewhere the checks pass and core dumps
//! are left as they are.

use std::net::IpAddr;

/// Whether binding `address` without authentication is refused: any address
/// that is not loopback (ADR 0046 §6). `0.0.0.0` and `::` (every interface)
/// are not loopback.
#[must_use]
pub fn needs_auth(address: IpAddr) -> bool {
    !address.is_loopback()
}

/// Whether the process runs with effective uid 0.
#[must_use]
pub fn is_root() -> bool {
    #[cfg(unix)]
    {
        rustix::process::geteuid().is_root()
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// Sets the core-dump size limit to zero, soft and hard: a dump would hold
/// query data and credentials.
///
/// # Errors
/// The `setrlimit` failure.
pub fn disable_core_dumps() -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use rustix::process::{Resource, Rlimit, setrlimit};
        setrlimit(
            Resource::Core,
            Rlimit {
                current: Some(0),
                maximum: Some(0),
            },
        )
        .map_err(std::io::Error::from)
    }
    #[cfg(not(unix))]
    {
        Ok(())
    }
}

/// The core-dump size limit (soft), `None` for unlimited; `Some(0)` on a
/// platform without one.
#[must_use]
pub fn core_dump_limit() -> Option<u64> {
    #[cfg(unix)]
    {
        rustix::process::getrlimit(rustix::process::Resource::Core).current
    }
    #[cfg(not(unix))]
    {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_binds_without_auth() {
        for a in ["127.0.0.1", "127.1.2.3", "::1"] {
            assert!(!needs_auth(a.parse().unwrap()), "{a}");
        }
        for a in ["0.0.0.0", "::", "10.0.0.5", "192.168.1.1", "fe80::1"] {
            assert!(needs_auth(a.parse().unwrap()), "{a}");
        }
    }

    // nextest runs each test in a process of its own, so this changes no
    // other test's limit.
    #[test]
    fn core_dumps_off() {
        disable_core_dumps().unwrap();
        assert_eq!(core_dump_limit(), Some(0));
    }

    #[test]
    fn root_check_matches_the_uid() {
        #[cfg(unix)]
        assert_eq!(is_root(), rustix::process::geteuid().as_raw() == 0);
    }
}
