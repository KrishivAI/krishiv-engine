// SPDX-License-Identifier: Apache-2.0
//! Start-up check of the CPU features this build was compiled for.
//!
//! The shipped builds target a CPU level above the architecture baseline
//! (`x86-64-v3` on x86-64, `neoverse-n1` on aarch64; see `README.md`,
//! "Supported CPUs"). On an older CPU the first instruction of that level
//! faults with SIGILL and gives the user nothing to act on. [`check`] runs
//! once, as early as possible, compares every feature the compiler was allowed
//! to assume with what the CPU reports, and returns an error that names the
//! missing features and the compat build to use instead.
//!
//! What is checked is decided at compile time with `cfg!(target_feature)`, so
//! a baseline (compat) build checks nothing and costs nothing. The detection
//! reads the CPU directly (`cpuid` on x86-64, `/proc/cpuinfo` on Linux arm64):
//! `is_x86_feature_detected!` and its arm64 twin answer `true` at compile time
//! for any feature the build enables, so they cannot detect an old CPU in the
//! very build that needs the check. The x86-64 check does not read `XCR0`
//! (that needs `unsafe`); it relies on the `OSXSAVE` bit for the OS side of AVX.
//!
//! Everything on the way to a verdict, including the failure path, is written
//! to stay plain scalar code: [`is_supported`] is three masked compares on
//! `cpuid` results, the failure path loops over a `static` table with a call in
//! every iteration, and no error type is moved by value. The compiler may use
//! the build's CPU level for any code (in a v3 build even a 16-byte copy or a
//! simple counting loop becomes VEX/AVX2/BMI code), so ordinary-looking code
//! here would fault on exactly the CPUs this exists for. Callers keep the code
//! before the check minimal (a `main` that only calls [`check_or_exit`] and
//! then an `#[inline(never)]` function) for the same reason. CI runs the v3
//! binary under QEMU with a pre-AVX2 CPU model to catch a regression.
//!
//! The Python extension does not use this module: raising a Python exception from
//! a v3-compiled extension on an old CPU faults too, so the wheels check the CPU
//! in pure Python (`python/krishiv/_cpu.py`) before the extension is imported.
//!
//! Limitation: the check cannot run before the program starts. Code that runs
//! first (the Rust runtime start-up and the dynamic loader) was not necessarily built
//! for the baseline, and the compiler may use a higher level anywhere in the
//! binary, including in code that runs before `main`. In practice nothing
//! before `main` is hot or vectorised, so the check is reliable on the CPUs it
//! targets, but it is best-effort rather than a guarantee.

use std::io::Write;

/// Name of the compat Python distribution (it imports as `krishiv`).
pub const COMPAT_WHEEL: &str = "krishiv-compat";
/// Name of the compat command line binary.
pub const COMPAT_BINARY: &str = "krishiv-compat";

/// The CPU level this build targets, named in the message.
#[cfg(target_arch = "x86_64")]
const LEVEL: &str = "x86-64-v3";
#[cfg(not(target_arch = "x86_64"))]
const LEVEL: &str = "neoverse-n1";

/// One required x86-64 feature: its name, where its bit is, and whether this build assumes it.
#[cfg(any(target_arch = "x86_64", test))]
struct X86Feature {
    name: &'static str,
    /// 0: `cpuid(1).ecx`, 1: `cpuid(7, 0).ebx`, 2: `cpuid(0x8000_0001).ecx`.
    reg: u8,
    /// The feature's bit in that register.
    bit: u32,
    compiled: bool,
}

#[cfg(any(target_arch = "x86_64", test))]
macro_rules! x86 {
    ($name:literal, $reg:literal, $bit:literal, $cfg:literal) => {
        X86Feature {
            name: $name,
            reg: $reg,
            bit: 1 << $bit,
            compiled: cfg!(target_feature = $cfg),
        }
    };
}

/// The x86-64-v3 features (`osxsave` stands for the OS having enabled the AVX state).
#[cfg(any(target_arch = "x86_64", test))]
static X86_V3: [X86Feature; 16] = [
    x86!("sse3", 0, 0, "sse3"),
    x86!("ssse3", 0, 9, "ssse3"),
    x86!("fma", 0, 12, "fma"),
    x86!("cmpxchg16b", 0, 13, "cmpxchg16b"),
    x86!("sse4.1", 0, 19, "sse4.1"),
    x86!("sse4.2", 0, 20, "sse4.2"),
    x86!("movbe", 0, 22, "movbe"),
    x86!("popcnt", 0, 23, "popcnt"),
    x86!("xsave", 0, 26, "xsave"),
    x86!("osxsave", 0, 27, "avx"),
    x86!("avx", 0, 28, "avx"),
    x86!("f16c", 0, 29, "f16c"),
    x86!("bmi1", 1, 3, "bmi1"),
    x86!("avx2", 1, 5, "avx2"),
    x86!("bmi2", 1, 8, "bmi2"),
    x86!("lzcnt", 2, 5, "lzcnt"),
];

/// The `cpuid` bits this build requires in register `reg` (see [`X86Feature::reg`]).
#[cfg(any(target_arch = "x86_64", test))]
const fn x86_required(reg: u8) -> u32 {
    let mut required = 0;
    let mut i = 0;
    while i < X86_V3.len() {
        if X86_V3[i].compiled && X86_V3[i].reg == reg {
            required |= X86_V3[i].bit;
        }
        i += 1;
    }
    required
}

/// `[leaf1.ecx, leaf7.ebx, ext.ecx]` of the running CPU.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn x86_regs() -> (u32, u32, u32) {
    use std::arch::x86_64::{__cpuid, __cpuid_count};
    let max = __cpuid(0).eax;
    let max_ext = __cpuid(0x8000_0000).eax;
    let leaf1 = if max >= 1 { __cpuid(1).ecx } else { 0 };
    let leaf7 = if max >= 7 { __cpuid_count(7, 0).ebx } else { 0 };
    let ext = if max_ext >= 0x8000_0001 {
        __cpuid(0x8000_0001).ecx
    } else {
        0
    };
    (leaf1, leaf7, ext)
}

/// Whether the running CPU has every feature this build was compiled to assume.
///
/// The passing path is three masked compares on `cpuid` results (see the module docs).
#[cfg(target_arch = "x86_64")]
pub fn is_supported() -> bool {
    // A static, read at run time: a compat build's all-zero masks are not a constant condition.
    static REQUIRED: [u32; 3] = [x86_required(0), x86_required(1), x86_required(2)];
    let (leaf1, leaf7, ext) = x86_regs();
    leaf1 & REQUIRED[0] == REQUIRED[0]
        && leaf7 & REQUIRED[1] == REQUIRED[1]
        && ext & REQUIRED[2] == REQUIRED[2]
}

/// Calls `sink` with the name of every x86-64-v3 feature a CPU with these `cpuid` results lacks,
/// counting every feature when `honor_build` is false, else only those this build assumes.
///
/// Failure path: each name goes straight to the sink (no collection is built or moved), and
/// every iteration makes a call, which keeps the loop scalar (see the module docs).
#[cfg(any(target_arch = "x86_64", test))]
#[cold]
#[inline(never)]
fn x86_missing(regs: (u32, u32, u32), honor_build: bool, sink: &mut dyn FnMut(&str)) {
    let mut first = true;
    for f in &X86_V3 {
        let word = match f.reg {
            0 => regs.0,
            1 => regs.1,
            _ => regs.2,
        };
        if (f.compiled || !honor_build) && word & f.bit == 0 {
            if !first {
                sink(", ");
            }
            first = false;
            sink(f.name);
        }
    }
}

/// The neoverse-n1 features: name, kernel `Features` flag, whether this build assumes it.
#[cfg(any(all(target_arch = "aarch64", target_os = "linux"), test))]
static ARM_N1: [(&str, &str, bool); 8] = [
    ("aes", "aes", cfg!(target_feature = "aes")),
    ("crc", "crc32", cfg!(target_feature = "crc")),
    ("dotprod", "asimddp", cfg!(target_feature = "dotprod")),
    ("fp16", "fphp", cfg!(target_feature = "fp16")),
    ("lse", "atomics", cfg!(target_feature = "lse")),
    ("rcpc", "lrcpc", cfg!(target_feature = "rcpc")),
    ("rdm", "asimdrdm", cfg!(target_feature = "rdm")),
    ("sha2", "sha2", cfg!(target_feature = "sha2")),
];

/// Calls `sink` with the name of every neoverse-n1 feature the `Features` line of
/// `/proc/cpuinfo` lacks (every feature when `honor_build` is false, else only those this build assumes).
#[cfg(any(all(target_arch = "aarch64", target_os = "linux"), test))]
fn arm_missing(features_line: &str, honor_build: bool, sink: &mut dyn FnMut(&str)) {
    let mut first = true;
    for (name, flag, compiled) in &ARM_N1 {
        if (*compiled || !honor_build) && !features_line.split_whitespace().any(|f| f == *flag) {
            if !first {
                sink(", ");
            }
            first = false;
            sink(name);
        }
    }
}

/// The `Features` flags of the first CPU in `/proc/cpuinfo`; `None` when it cannot be read.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn arm_features_line() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("Features"))?;
    Some(line.split_once(':').map_or("", |(_, v)| v).to_owned())
}

/// Whether the running CPU has every feature this build was compiled to assume.
///
/// Linux only (reads `/proc/cpuinfo`); when that cannot be read, `true`: the CPU cannot be told.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub fn is_supported() -> bool {
    arm_features_line().is_none_or(|l| {
        let mut any = false;
        arm_missing(&l, true, &mut |_| any = true);
        !any
    })
}

/// Nothing is checked on this target.
#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_os = "linux")
)))]
pub fn is_supported() -> bool {
    true
}

/// Calls `sink` with the pieces of the error text, in order: the CPU level, the
/// features the running CPU lacks and the compat build to use instead. Call it
/// only when [`is_supported`] is false. The pieces go out one by one so that the
/// failure path builds and moves no value (see the module docs).
#[cold]
#[inline(never)]
pub fn unsupported_message(sink: &mut dyn FnMut(&str)) {
    sink("this build of Krishiv requires a ");
    sink(LEVEL);
    sink(" CPU, but this CPU lacks: ");
    #[cfg(target_arch = "x86_64")]
    {
        let (leaf1, leaf7, ext) = x86_regs();
        x86_missing((leaf1, leaf7, ext), true, sink);
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    if let Some(line) = arm_features_line() {
        arm_missing(&line, true, sink);
    }
    // Compat builds are published for Linux x86-64 only.
    if cfg!(target_arch = "x86_64") {
        sink(". Use the compat build instead: `pip install ");
        sink(COMPAT_WHEEL);
        sink("` for Python, or the `");
        sink(COMPAT_BINARY);
        sink("` binary.");
    } else {
        sink(". No compat build is published for this architecture.");
    }
}

/// For a binary's `main`: when the CPU is too old, prints
/// [`unsupported_message`] to stderr and exits with status 1.
pub fn check_or_exit() {
    if !is_supported() {
        fail();
    }
}

#[cold]
#[inline(never)]
fn fail() -> ! {
    write_failure(&mut std::io::stderr());
    std::process::exit(1)
}

/// `error: <message>` and a newline, piece by piece (see the module docs).
fn write_failure(err: &mut impl Write) {
    // Best effort: a closed stderr leaves nothing to report to.
    let _ = err.write_all(b"error: ");
    unsupported_message(&mut |piece| {
        let _ = err.write_all(piece.as_bytes());
    });
    let _ = err.write_all(b"\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Nehalem-class CPU: SSE4.2 and POPCNT, no AVX family, no BMI, no LZCNT.
    const NEHALEM: (u32, u32, u32) = (1 | 1 << 9 | 1 << 13 | 1 << 19 | 1 << 20 | 1 << 23, 0, 0);
    /// A Haswell-class CPU: everything in x86-64-v3.
    const HASWELL: (u32, u32, u32) = (
        NEHALEM.0 | 1 << 12 | 1 << 22 | 1 << 26 | 1 << 27 | 1 << 28 | 1 << 29,
        1 << 3 | 1 << 5 | 1 << 8,
        1 << 5,
    );

    fn collect(f: impl FnOnce(&mut dyn FnMut(&str))) -> String {
        let mut out = String::new();
        f(&mut |p| out.push_str(p));
        out
    }

    #[test]
    fn haswell_has_every_x86_v3_feature() {
        assert_eq!(collect(|s| x86_missing(HASWELL, false, s)), "");
        assert_eq!(collect(|s| x86_missing(HASWELL, true, s)), "");
    }

    #[test]
    fn nehalem_lacks_the_avx_family_in_table_order() {
        assert_eq!(
            collect(|s| x86_missing(NEHALEM, false, s)),
            "fma, movbe, xsave, osxsave, avx, f16c, bmi1, avx2, bmi2, lzcnt"
        );
    }

    #[test]
    fn only_features_the_build_assumes_are_required() {
        let honored = collect(|s| x86_missing(NEHALEM, true, s));
        assert_eq!(honored.contains("avx2"), cfg!(target_feature = "avx2"));
        assert_eq!(honored.contains("fma"), cfg!(target_feature = "fma"));
        assert!(!honored.contains("sse4.2"));
        if !cfg!(target_feature = "avx2") {
            assert!(honored.is_empty());
        }
    }

    #[test]
    fn required_bits_follow_the_table() {
        // Every x86-64-v3 bit is set on the Haswell model, so the masks pass it.
        assert_eq!(HASWELL.0 & x86_required(0), x86_required(0));
        assert_eq!(HASWELL.1 & x86_required(1), x86_required(1));
        assert_eq!(HASWELL.2 & x86_required(2), x86_required(2));
        if cfg!(target_feature = "avx2") {
            assert_ne!(NEHALEM.1 & x86_required(1), x86_required(1));
        } else {
            assert_eq!(x86_required(1), 0);
        }
    }

    #[test]
    fn arm_features_are_read_from_the_kernel_flags() {
        let all = " fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp ssbs";
        assert_eq!(collect(|s| arm_missing(all, false, s)), "");
        // Cortex-A72 class: no atomics, dotprod, fp16, rdm, lrcpc.
        let a72 = " fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid";
        assert_eq!(
            collect(|s| arm_missing(a72, false, s)),
            "dotprod, fp16, lse, rcpc, rdm"
        );
    }

    #[test]
    fn message_names_the_level_and_the_compat_build() {
        let msg = collect(unsupported_message);
        assert!(
            msg.starts_with("this build of Krishiv requires a "),
            "{msg}"
        );
        assert!(msg.contains(LEVEL), "{msg}");
        if cfg!(target_arch = "x86_64") {
            assert!(
                msg.contains(COMPAT_WHEEL) && msg.contains(COMPAT_BINARY),
                "{msg}"
            );
        } else {
            assert!(msg.contains("No compat build"), "{msg}");
        }
    }

    #[test]
    fn failure_text_is_one_error_line() {
        let mut out = Vec::new();
        write_failure(&mut out);
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("error: this build of Krishiv requires a "),
            "{text}"
        );
        assert!(text.ends_with(".\n"), "{text}");
        assert_eq!(text.lines().count(), 1, "{text}");
    }

    #[test]
    fn check_or_exit_returns_on_a_supported_cpu() {
        // The test machine runs its own build, so this must not exit the test process.
        check_or_exit();
    }

    /// The machine running the tests built them, so it must pass its own check.
    #[test]
    fn running_cpu_satisfies_its_own_build() {
        assert!(is_supported());
        #[cfg(target_arch = "x86_64")]
        assert_eq!(collect(|s| x86_missing(x86_regs(), true, s)), "");
    }
}
