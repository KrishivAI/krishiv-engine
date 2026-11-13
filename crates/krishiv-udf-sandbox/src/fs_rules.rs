// SPDX-License-Identifier: Apache-2.0
//! The worker's Landlock ruleset (note §4.2 item 4).
//!
//! - execute: the interpreter and the ELF loader it names (the kernel opens the loader with the execute
//!   right). A program started through the loader stays under the same Landlock and seccomp layers, so it
//!   gains nothing over the interpreter itself;
//! - read: the spec's read paths (files and directories);
//! - read and write, but not execute: the private directory;
//! - read and write: `/dev/null`;
//! - TCP: bind denied, connect only to the spec's ports (ABI 4 or later);
//! - scopes: abstract Unix sockets and signals stay inside the worker (ABI 6 or later).
//!
//! Everything else is refused, so `/etc/passwd`, home directories and the server's spill and data roots are
//! unreadable.

use std::path::{Path, PathBuf};

use landlock::{
    ABI, Access, AccessFs, AccessNet, BitFlags, CompatLevel, Compatible, NetPort, PathBeneath,
    PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr, RulesetStatus, Scope,
};

use crate::{Network, SandboxError, SandboxSpec};

/// The newest ABI whose rights the ruleset asks for; older kernels get the subset they know (best effort,
/// checked against the probe before entry).
const TARGET: ABI = ABI::V6;

fn failed(what: &str, e: impl std::fmt::Display) -> SandboxError {
    SandboxError::unavailable(format!("Landlock {what} failed: {e}"))
}

fn read_only() -> BitFlags<AccessFs> {
    AccessFs::ReadFile | AccessFs::ReadDir
}

/// Builds the ruleset and restricts the calling thread (and every program it executes) to it.
pub(crate) fn restrict(spec: &SandboxSpec) -> Result<(), SandboxError> {
    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(TARGET))
        .map_err(|e| failed("ruleset", e))?
        .handle_access(AccessNet::from_all(TARGET))
        .map_err(|e| failed("ruleset", e))?
        .scope(Scope::from_all(TARGET))
        .map_err(|e| failed("ruleset", e))?
        .create()
        .map_err(|e| failed("ruleset", e))?;

    // Executables: required, so a missing interpreter or loader fails here and not as a vague exec error.
    let mut executables = vec![spec.interpreter.clone()];
    executables.extend(elf_loader(&spec.interpreter));
    for path in &executables {
        let fd =
            PathFd::new(path).map_err(|e| failed(&format!("opening {}", path.display()), e))?;
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, AccessFs::Execute | AccessFs::ReadFile))
            .map_err(|e| failed("execute rule", e))?;
    }
    for path in &spec.read_paths {
        // `sys.path` names entries that do not exist; they are skipped, not refused.
        if let Ok(fd) = PathFd::new(path) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, read_only()))
                .map_err(|e| failed("read rule", e))?;
        }
    }
    let dev_null = PathFd::new("/dev/null").map_err(|e| failed("opening /dev/null", e))?;
    ruleset = ruleset
        .add_rule(PathBeneath::new(
            dev_null,
            AccessFs::ReadFile | AccessFs::WriteFile | AccessFs::Truncate,
        ))
        .map_err(|e| failed("/dev/null rule", e))?;
    let private = PathFd::new(&spec.private_dir).map_err(|e| {
        failed(
            &format!(
                "opening the private directory {}",
                spec.private_dir.display()
            ),
            e,
        )
    })?;
    ruleset = ruleset
        .add_rule(PathBeneath::new(
            private,
            AccessFs::from_all(TARGET) & !AccessFs::Execute,
        ))
        .map_err(|e| failed("private directory rule", e))?;

    if let Network::TcpConnect(ports) = &spec.network {
        // A port allowlist must be enforced, never silently dropped on an old kernel.
        ruleset = ruleset.set_compatibility(CompatLevel::HardRequirement);
        for &port in ports {
            ruleset = ruleset
                .add_rule(NetPort::new(port, AccessNet::ConnectTcp))
                .map_err(|e| failed("TCP connect rule", e))?;
        }
    }

    let status = ruleset
        .restrict_self()
        .map_err(|e| failed("restrict_self", e))?;
    if status.ruleset == RulesetStatus::NotEnforced {
        return Err(SandboxError::unavailable(
            "Landlock accepted the ruleset but enforces none of it",
        ));
    }
    Ok(())
}

/// The program interpreter (`PT_INTERP`) of an ELF64 executable, if it names one. A static or non-ELF
/// interpreter has none.
pub(crate) fn elf_loader(executable: &Path) -> Option<PathBuf> {
    use std::io::Read;
    // The ELF header, the program headers and the `PT_INTERP` string sit at the start of the file (offset
    // 0x318 on today's toolchains); the first 64 KiB holds them without reading a whole interpreter.
    let mut head = Vec::new();
    std::fs::File::open(executable)
        .ok()?
        .take(HEAD_BYTES)
        .read_to_end(&mut head)
        .ok()?;
    pt_interp(&head)
}

const HEAD_BYTES: u64 = 64 * 1024;

fn pt_interp(elf: &[u8]) -> Option<PathBuf> {
    // e_ident: magic, ELFCLASS64 (2), ELFDATA2LSB (1).
    if elf.get(..6)? != b"\x7fELF\x02\x01" {
        return None;
    }
    let bytes = |at: usize, len: usize| elf.get(at..at.checked_add(len)?);
    let u16_at = |at: usize| Some(u16::from_le_bytes(bytes(at, 2)?.try_into().ok()?));
    let u32_at = |at: usize| Some(u32::from_le_bytes(bytes(at, 4)?.try_into().ok()?));
    let u64_at = |at: usize| {
        let v = u64::from_le_bytes(bytes(at, 8)?.try_into().ok()?);
        usize::try_from(v).ok()
    };
    let phoff = u64_at(0x20)?;
    let phentsize = usize::from(u16_at(0x36)?);
    let phnum = usize::from(u16_at(0x38)?);
    const PT_INTERP: u32 = 3;
    (0..phnum).find_map(|i| {
        let ph = phoff.checked_add(i.checked_mul(phentsize)?)?;
        if u32_at(ph)? != PT_INTERP {
            return None;
        }
        let offset = u64_at(ph.checked_add(8)?)?;
        let size = u64_at(ph.checked_add(32)?)?;
        let raw = bytes(offset, size)?;
        let name = raw.split(|&b| b == 0).next()?;
        let name = std::str::from_utf8(name).ok()?;
        (!name.is_empty()).then(|| PathBuf::from(name))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loader_of_a_dynamic_executable_is_found() {
        // The test binary itself is a dynamic ELF on every Linux CI target.
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let loader = elf_loader(&exe);
        assert!(loader.as_deref().is_some_and(Path::exists), "{loader:?}");
    }

    #[test]
    fn non_elf_and_truncated_files_have_no_loader() {
        assert_eq!(pt_interp(b"#!/bin/sh\n"), None);
        assert_eq!(pt_interp(b"\x7fELF\x02\x01\x01"), None);
        assert_eq!(pt_interp(b""), None);
        // A header whose program table points past the end.
        let mut elf = vec![0u8; 64];
        elf[..6].copy_from_slice(b"\x7fELF\x02\x01");
        elf[0x20..0x28].copy_from_slice(&u64::MAX.to_le_bytes());
        elf[0x36..0x38].copy_from_slice(&56u16.to_le_bytes());
        elf[0x38..0x3a].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(pt_interp(&elf), None);
    }
}
