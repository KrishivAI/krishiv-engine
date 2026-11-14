// SPDX-License-Identifier: Apache-2.0
//! What the running kernel can enforce (note §4.4).
//!
//! The server probes once at startup and refuses server-mode UDFs when a required layer is missing, unless an
//! administrator turned the sandbox off. The probe never restricts the calling process: Landlock rulesets are
//! created with a hard requirement and dropped without `restrict_self`.

use crate::{Network, SandboxError, SandboxSpec};

/// The sandbox layers the running kernel supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Capabilities {
    /// The Landlock ABI level whose features are all available, among the levels the sandbox uses: 0 (no
    /// Landlock), 1 (files), 4 (TCP bind and connect) or 6 (abstract Unix socket and signal scoping).
    pub landlock_abi: u8,
    /// seccomp-bpf filters can be installed.
    pub seccomp: bool,
}

impl Capabilities {
    /// Capabilities with the given values, for tests and for the probe hook of a server that must behave as
    /// if a layer were missing (`sandbox_unavailable_refuses`).
    pub fn new(landlock_abi: u8, seccomp: bool) -> Self {
        Capabilities {
            landlock_abi,
            seccomp,
        }
    }

    /// Checks that these capabilities can enforce `spec`.
    ///
    /// # Errors
    /// [`SandboxError::Unavailable`] when Landlock or seccomp is missing, or when `spec` opens TCP ports and
    /// Landlock is older than ABI 4 (which first restricts TCP connect).
    pub fn require(&self, spec: &SandboxSpec) -> Result<(), SandboxError> {
        if self.landlock_abi == 0 {
            return Err(SandboxError::unavailable(
                "Landlock is not available (Linux 5.13 or later with Landlock in the boot LSM list is needed)",
            ));
        }
        if !self.seccomp {
            return Err(SandboxError::unavailable(
                "seccomp filters are not available (a kernel built with CONFIG_SECCOMP_FILTER is needed)",
            ));
        }
        if matches!(spec.network, Network::TcpConnect(_)) && self.landlock_abi < 4 {
            return Err(SandboxError::unavailable(format!(
                "a TCP port allowlist needs Landlock ABI 4 (Linux 6.7), and this kernel has ABI {}",
                self.landlock_abi
            )));
        }
        Ok(())
    }
}

/// Probes the running kernel.
pub fn probe() -> Capabilities {
    Capabilities {
        landlock_abi: imp::landlock_abi(),
        seccomp: imp::seccomp(),
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use landlock::{
        ABI, Access, AccessFs, AccessNet, CompatLevel, Compatible, Ruleset, RulesetAttr, Scope,
    };

    pub(super) fn landlock_abi() -> u8 {
        let hard = || Ruleset::default().set_compatibility(CompatLevel::HardRequirement);
        let fs = hard()
            .handle_access(AccessFs::from_all(ABI::V1))
            .and_then(Ruleset::create)
            .is_ok();
        if !fs {
            return 0;
        }
        let net = hard()
            .handle_access(AccessNet::from_all(ABI::V4))
            .and_then(Ruleset::create)
            .is_ok();
        if !net {
            return 1;
        }
        let scope = hard()
            .scope(Scope::from_all(ABI::V6))
            .and_then(Ruleset::create)
            .is_ok();
        if scope { 6 } else { 4 }
    }

    /// The kernel has seccomp when `/proc/self/status` reports the field (`CONFIG_SECCOMP`); filter mode is
    /// part of every kernel that has Landlock.
    pub(super) fn seccomp() -> bool {
        std::fs::read_to_string("/proc/self/status")
            .map(|s| s.lines().any(|l| l.starts_with("Seccomp:")))
            .unwrap_or(false)
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub(super) fn landlock_abi() -> u8 {
        0
    }

    pub(super) fn seccomp() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::ErrorClass;

    #[test]
    fn missing_layers_refuse_with_the_class() {
        let spec = SandboxSpec::new("/p", "/d");
        let tcp = SandboxSpec {
            network: Network::TcpConnect(vec![443]),
            ..SandboxSpec::new("/p", "/d")
        };
        for (caps, spec, needle) in [
            (Capabilities::new(0, true), &spec, "Landlock"),
            (Capabilities::new(6, false), &spec, "seccomp"),
            (Capabilities::new(3, true), &tcp, "ABI 4"),
        ] {
            match caps.require(spec) {
                Err(e) => {
                    assert_eq!(e.error_class(), ErrorClass::KrishivUdfSandboxUnavailable);
                    let text = e.to_string();
                    assert!(text.contains(needle), "{text}");
                    assert!(text.contains("krishiv.udf.sandbox"), "{text}");
                }
                Ok(()) => panic!("{caps:?} accepted {spec:?}"),
            }
        }
        assert!(Capabilities::new(1, true).require(&spec).is_ok());
        assert!(Capabilities::new(4, true).require(&tcp).is_ok());
    }

    /// The probe agrees with the kernel's own report. CI runs on kernels with Landlock ABI 6 or later (note
    /// §5), so there the sandbox tests cannot pass with nothing enforced.
    #[cfg(target_os = "linux")]
    #[test]
    fn probe_matches_the_kernel() {
        let caps = probe();
        // securityfs is not mounted in every container; compare only where it can be read.
        if let Ok(lsm) = std::fs::read_to_string("/sys/kernel/security/lsm") {
            assert_eq!(
                caps.landlock_abi > 0,
                lsm.split(',').any(|m| m.trim() == "landlock"),
                "{lsm}"
            );
        }
        assert!(caps.seccomp);
        if std::env::var_os("CI").is_some() {
            assert!(
                caps.landlock_abi >= 6,
                "CI runner has Landlock ABI {}",
                caps.landlock_abi
            );
        }
    }
}
