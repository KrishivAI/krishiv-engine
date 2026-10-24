// SPDX-License-Identifier: Apache-2.0
//! Hand-written `tar` (ustar) and `zip` (stored) writers for debug bundles.
//!
//! No new dependency: both formats are small enough to emit directly. The
//! writers are deterministic (sorted entries, fixed timestamps, fixed modes)
//! so two runs over the same input are byte-identical. Readers for both
//! formats live here too, for the tests.

use std::collections::BTreeMap;

/// A string that already went through redaction, the only content an
/// [`BundleArchive`] accepts, so an unredacted string cannot be written by
/// construction. The constructor replaces table paths with `<path-N>`
/// placeholders and applies the existing pattern redaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Redacted(String);

impl Redacted {
    /// `raw` with table paths replaced and pattern redaction applied.
    #[must_use]
    pub(crate) fn new(raw: &str, subs: &[PathSub]) -> Self {
        Self(apply_redact(&apply_paths(raw, subs)))
    }

    /// As [`Self::new`], also masking SQL string literals (`'...'`) and
    /// standalone numbers, for query text, plans and profiles that embed
    /// them. Placeholders and the redaction marker survive.
    #[must_use]
    pub(crate) fn new_query(raw: &str, subs: &[PathSub]) -> Self {
        let replaced = apply_paths(raw, subs);
        let redacted = apply_redact(&replaced);
        Self(mask_literals(&redacted))
    }

    /// Content already known to hold no secrets (fixed markers, numbers the
    /// engine produced): still run through the pattern redaction so every
    /// archived string went through it.
    #[must_use]
    pub(crate) fn vetted(raw: &str, subs: &[PathSub]) -> Self {
        Self::new(raw, subs)
    }

    /// The redacted text.
    #[must_use]
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    /// The bytes.
    #[must_use]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

/// One table-path substitution: every occurrence of `original` (and its file
/// name) becomes `placeholder` (`<path-1>`, ...).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PathSub {
    /// The placeholder, e.g. `<path-1>`.
    pub(crate) placeholder: String,
    /// Strings to replace, longest first (the full path, then components).
    patterns: Vec<String>,
}

impl PathSub {
    /// Substitutions for `paths` in order (`<path-1>`, ...).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn for_tables(paths: &[String]) -> Vec<Self> {
        paths
            .iter()
            .enumerate()
            .map(|(i, p)| Self::one(i + 1, std::slice::from_ref(p)))
            .collect()
    }

    /// Substitutions with one placeholder per table, where each table may
    /// be named by several spellings (a relative `--table` path and the
    /// absolute path the plan prints).
    #[must_use]
    pub(crate) fn for_table_spellings(tables: &[Vec<String>]) -> Vec<Self> {
        tables
            .iter()
            .enumerate()
            .map(|(i, spellings)| Self::one(i + 1, spellings))
            .collect()
    }

    fn one(number: usize, spellings: &[String]) -> Self {
        let placeholder = format!("<path-{number}>");
        let mut patterns = Vec::new();
        for p in spellings {
            patterns.push(p.clone());
            let file = std::path::Path::new(p)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !file.is_empty() && file != p.as_str() {
                patterns.push(file.to_owned());
            }
            // Each path component, so a recognizable directory name
            // never survives even if the plan shortens the path.
            for seg in p.split(['/', '\\']) {
                if seg.len() >= 2 && seg != file {
                    patterns.push(seg.to_owned());
                }
            }
        }
        patterns.sort_by_key(|s| std::cmp::Reverse(s.len()));
        patterns.dedup();
        Self {
            placeholder,
            patterns,
        }
    }
}

fn apply_paths(text: &str, subs: &[PathSub]) -> String {
    let mut out = text.to_owned();
    for sub in subs {
        for pat in &sub.patterns {
            if pat.is_empty() {
                continue;
            }
            // Skip single-component placeholders' own digits: patterns never
            // match `<path-N>`.
            if out.contains(pat.as_str()) {
                out = out.replace(pat.as_str(), &sub.placeholder);
            }
        }
    }
    out
}

fn apply_redact(text: &str) -> String {
    krishiv_common::redact::redact(text).into_owned()
}

/// Masks single-quoted string contents with `?` and standalone digit runs
/// with `0`, skipping `<...>` placeholders. Deterministic and idempotent
/// over its own output.
fn mask_literals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'<' {
            if let Some(end) = text[i..].find('>') {
                out.push_str(&text[i..i + end + 1]);
                i += end + 1;
                continue;
            }
            out.push('<');
            i += 1;
            continue;
        }
        if c == b'\'' {
            // Find the closing quote first, so a string holding a
            // placeholder or the redaction marker is kept verbatim.
            let mut j = i + 1;
            let mut closed = false;
            while j < bytes.len() {
                let d = bytes[j];
                if d == b'\\' {
                    j += 2;
                    continue;
                }
                if d == b'\'' {
                    if bytes.get(j + 1) == Some(&b'\'') {
                        j += 2;
                        continue;
                    }
                    closed = true;
                    break;
                }
                j += 1;
            }
            let end = if closed { j + 1 } else { bytes.len() };
            let literal = &text[i..end];
            if literal.contains("<path-") || literal.contains(krishiv_common::secret::REDACTED) {
                out.push_str(literal);
            } else {
                out.push_str("'?'");
            }
            i = end;
            continue;
        }
        if c.is_ascii_digit() {
            let prev_ok = i == 0 || {
                let p = bytes[i - 1];
                !(p.is_ascii_alphanumeric() || p == b'_')
            };
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let next_ok = j >= bytes.len() || {
                let n = bytes[j];
                !(n.is_ascii_alphanumeric() || n == b'_')
            };
            if prev_ok && next_ok {
                out.push('0');
            } else {
                out.push_str(&text[i..j]);
            }
            i = j;
            continue;
        }
        // ASCII fast path; non-ASCII copied by char.
        if c < 0x80 {
            out.push(c as char);
            i += 1;
        } else {
            let ch = text[i..].chars().next().unwrap_or('\u{FFFD}');
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Sorted archive contents: name to redacted content.
#[derive(Default)]
pub(crate) struct BundleArchive {
    files: BTreeMap<String, Redacted>,
}

impl BundleArchive {
    /// An empty archive.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds `name` with `content`. Names must be relative, UTF-8 and use
    /// `/`; duplicates overwrite (bundle names are unique).
    ///
    /// # Errors
    /// When `name` is empty, absolute, escapes (`..`) or too long.
    pub(crate) fn add(&mut self, name: &str, content: Redacted) -> Result<(), String> {
        check_name(name)?;
        self.files.insert(name.to_owned(), content);
        Ok(())
    }

    /// The archive as uncompressed ustar (512-byte headers, two zero
    /// blocks at the end).
    ///
    /// # Errors
    /// When a name or size does not fit the ustar fields.
    pub(crate) fn to_tar(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for (name, content) in &self.files {
            write_tar_entry(&mut out, name, content.as_bytes())?;
        }
        out.extend([0_u8; 1024]);
        Ok(out)
    }

    /// The archive as stored (uncompressed) zip with CRC-32.
    ///
    /// # Errors
    /// When a name or size does not fit the zip fields.
    pub(crate) fn to_zip(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, content) in &self.files {
            let bytes = content.as_bytes();
            let crc = crc32(bytes);
            let name_bytes = name.as_bytes();
            if name_bytes.len() > u16::MAX as usize {
                return Err(format!("zip name too long: {name}"));
            }
            let len64 = u64::try_from(bytes.len())
                .map_err(|_| format!("file too large for zip: {name}"))?;
            if len64 > u64::from(u32::MAX) {
                return Err(format!("file too large for zip: {name}"));
            }
            let offset = u32::try_from(out.len())
                .map_err(|_| format!("archive too large for zip: {name}"))?;
            write_u32_le(&mut out, 0x0403_4b50);
            write_u16_le(&mut out, 20);
            write_u16_le(&mut out, 0);
            write_u16_le(&mut out, 0);
            write_u16_le(&mut out, 0);
            write_u16_le(&mut out, 0x0021);
            write_u32_le(&mut out, crc);
            let size32 = u32::try_from(bytes.len())
                .map_err(|_| format!("file too large for zip: {name}"))?;
            write_u32_le(&mut out, size32);
            write_u32_le(&mut out, size32);
            let name16 = u16::try_from(name_bytes.len())
                .map_err(|_| format!("zip name too long: {name}"))?;
            write_u16_le(&mut out, name16);
            write_u16_le(&mut out, 0);
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(bytes);
            write_u32_le(&mut central, 0x0201_4b50);
            write_u16_le(&mut central, 20);
            write_u16_le(&mut central, 20);
            write_u16_le(&mut central, 0);
            write_u16_le(&mut central, 0);
            write_u16_le(&mut central, 0);
            write_u16_le(&mut central, 0x0021);
            write_u32_le(&mut central, crc);
            let size32 = u32::try_from(bytes.len())
                .map_err(|_| format!("file too large for zip: {name}"))?;
            write_u32_le(&mut central, size32);
            write_u32_le(&mut central, size32);
            let name16 = u16::try_from(name_bytes.len())
                .map_err(|_| format!("zip name too long: {name}"))?;
            write_u16_le(&mut central, name16);
            write_u16_le(&mut central, 0);
            write_u16_le(&mut central, 0);
            write_u16_le(&mut central, 0);
            write_u16_le(&mut central, 0);
            write_u32_le(&mut central, 0);
            write_u32_le(&mut central, offset);
            central.extend_from_slice(name_bytes);
        }
        let central_offset =
            u32::try_from(out.len()).map_err(|_| "archive too large for zip".to_owned())?;
        let central_size =
            u32::try_from(central.len()).map_err(|_| "archive too large for zip".to_owned())?;
        let count =
            u16::try_from(self.files.len()).map_err(|_| "too many files for zip".to_owned())?;
        out.extend_from_slice(&central);
        write_u32_le(&mut out, 0x0605_4b50);
        write_u16_le(&mut out, 0);
        write_u16_le(&mut out, 0);
        write_u16_le(&mut out, count);
        write_u16_le(&mut out, count);
        write_u32_le(&mut out, central_size);
        write_u32_le(&mut out, central_offset);
        write_u16_le(&mut out, 0);
        Ok(out)
    }

    /// Reads a [`Self::to_tar`] archive back (tests).
    ///
    /// # Errors
    /// When the bytes are not a ustar archive this writer produced.
    #[cfg(test)]
    pub(crate) fn read_tar(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let mut out = BTreeMap::new();
        let mut at = 0;
        let mut zeros = 0;
        while at + 512 <= bytes.len() {
            let header = &bytes[at..at + 512];
            at += 512;
            if header.iter().all(|&b| b == 0) {
                zeros += 1;
                if zeros == 2 {
                    break;
                }
                continue;
            }
            zeros = 0;
            let name = tar_name(header)?;
            let size = tar_octal(header, 124, 12, "size")?;
            let stored = tar_octal(header, 148, 8, "checksum")?;
            let mut sum = 0_u64;
            for (i, &b) in header.iter().enumerate() {
                sum += u64::from(if (148..156).contains(&i) { b' ' } else { b });
            }
            if sum != stored {
                return Err(format!("tar checksum mismatch for {name}"));
            }
            if header[156] != b'0' && header[156] != 0 {
                return Err(format!("unsupported tar entry {name}"));
            }
            let size_usize =
                usize::try_from(size).map_err(|_| format!("tar size too large: {name}"))?;
            if at + size_usize > bytes.len() {
                return Err(format!("tar entry truncated: {name}"));
            }
            out.insert(name, bytes[at..at + size_usize].to_vec());
            at += size_usize.next_multiple_of(512);
        }
        Ok(out)
    }

    /// Reads a [`Self::to_zip`] archive back (tests).
    ///
    /// # Errors
    /// When the bytes are not a stored zip this writer produced.
    #[cfg(test)]
    pub(crate) fn read_zip(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let mut out = BTreeMap::new();
        let mut at = 0;
        while at + 30 <= bytes.len() {
            let sig = u32_le(bytes, at);
            if sig == 0x0605_4b50 || sig == 0x0201_4b50 {
                break;
            }
            if sig != 0x0403_4b50 {
                return Err(format!("bad zip header at {at}"));
            }
            let method = u16_le(bytes, at + 8);
            if method != 0 {
                return Err("zip entry is not stored".to_owned());
            }
            let crc = u32_le(bytes, at + 14);
            let comp = u32_le(bytes, at + 18) as usize;
            let uncomp = u32_le(bytes, at + 22) as usize;
            if comp != uncomp {
                return Err("zip sizes differ".to_owned());
            }
            let name_len = u16_le(bytes, at + 26) as usize;
            let extra_len = u16_le(bytes, at + 28) as usize;
            let head = at + 30;
            if head + name_len + extra_len + comp > bytes.len() {
                return Err("zip entry truncated".to_owned());
            }
            let name = std::str::from_utf8(&bytes[head..head + name_len])
                .map_err(|_| "zip name is not UTF-8".to_owned())?
                .to_owned();
            let data_start = head + name_len + extra_len;
            let data = bytes[data_start..data_start + comp].to_vec();
            if crc32(&data) != crc {
                return Err(format!("zip CRC mismatch for {name}"));
            }
            out.insert(name, data);
            at = data_start + comp;
        }
        Ok(out)
    }
}

fn check_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("empty file name".to_owned());
    }
    if name.starts_with('/') || name.contains("..") || name.contains('\\') {
        return Err(format!("bad file name: {name}"));
    }
    if name.len() > 100 {
        return Err(format!("file name too long for ustar: {name}"));
    }
    Ok(())
}

fn write_tar_entry(out: &mut Vec<u8>, name: &str, data: &[u8]) -> Result<(), String> {
    if name.len() > 100 {
        return Err(format!("file name too long for ustar: {name}"));
    }
    let size64 = u64::try_from(data.len()).map_err(|_| format!("file too large: {name}"))?;
    if size64 >= 1 << 33 {
        return Err(format!("file too large for ustar: {name}"));
    }
    let mut header = [0_u8; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    octal_into(&mut header[100..108], 0o644, "mode")?;
    octal_into(&mut header[108..116], 0, "uid")?;
    octal_into(&mut header[116..124], 0, "gid")?;
    octal_into(&mut header[124..136], size64, "size")?;
    octal_into(&mut header[136..148], 0, "mtime")?;
    header[156] = b'0';
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    // Checksum over the header with the checksum field as spaces.
    let mut sum: u64 = 0;
    for (i, &b) in header.iter().enumerate() {
        sum += u64::from(if (148..156).contains(&i) { b' ' } else { b });
    }
    let chk = format!("{sum:06o}\0 ");
    header[148..156].copy_from_slice(chk.as_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
    let pad = data.len().next_multiple_of(512) - data.len();
    out.extend(std::iter::repeat_n(0, pad));
    Ok(())
}

fn octal_into(field: &mut [u8], value: u64, what: &str) -> Result<(), String> {
    let text = format!("{:0width$o}\0", value, width = field.len() - 1);
    if text.len() != field.len() {
        return Err(format!("{what} does not fit ustar"));
    }
    field.copy_from_slice(text.as_bytes());
    Ok(())
}

#[cfg(test)]
fn tar_name(header: &[u8]) -> Result<String, String> {
    let end = header[..100].iter().position(|&b| b == 0).unwrap_or(100);
    std::str::from_utf8(&header[..end])
        .map_err(|_| "tar name is not UTF-8".to_owned())
        .map(str::to_owned)
}

#[cfg(test)]
fn tar_octal(header: &[u8], at: usize, len: usize, what: &str) -> Result<u64, String> {
    let raw = &header[at..at + len];
    let end = raw.iter().position(|&b| b == 0 || b == b' ').unwrap_or(len);
    let text = std::str::from_utf8(&raw[..end]).map_err(|_| format!("bad tar {what}"))?;
    let text = text.trim();
    if text.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(text, 8).map_err(|_| format!("bad tar {what}"))
}

fn write_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
fn u16_le(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

#[cfg(test)]
fn u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// CRC-32 (IEEE) of `data`.
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0_u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "table index is below 256 by construction"
            )]
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 == 1 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in data {
        let index = ((crc ^ u32::from(b)) & 0xFF) as usize;
        crc = table[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subs() -> Vec<PathSub> {
        PathSub::for_tables(&["/data/secret-dir/file.parquet".to_owned()])
    }

    #[test]
    fn redacted_hides_paths_and_patterns() {
        let r = Redacted::new(
            "read /data/secret-dir/file.parquet password=hunter2 https://u:pw@h/x",
            &subs(),
        );
        assert!(!r.as_str().contains("secret-dir"));
        assert!(!r.as_str().contains("hunter2"));
        assert!(!r.as_str().contains("pw@h"));
        assert!(r.as_str().contains("<path-1>"));
        assert!(r.as_str().contains("*********(redacted)"));
    }

    #[test]
    fn query_redaction_masks_literals_but_keeps_placeholders() {
        let r = Redacted::new_query(
            "select * from t where name = 'alice' and id = 987654 and password=hunter2 and p='/data/secret-dir/file.parquet'",
            &subs(),
        );
        assert!(!r.as_str().contains("alice"));
        assert!(!r.as_str().contains("987654"));
        assert!(!r.as_str().contains("hunter2"));
        assert!(r.as_str().contains("<path-1>"));
        assert!(r.as_str().contains("*********(redacted)"));
    }

    #[test]
    fn tar_round_trips_sorted() {
        let mut a = BundleArchive::new();
        a.add("b.txt", Redacted::new("second", &[])).expect("add");
        a.add("a.txt", Redacted::new("first", &[])).expect("add");
        let bytes = a.to_tar().expect("tar");
        assert_eq!(bytes.len() % 512, 0);
        let back = BundleArchive::read_tar(&bytes).expect("read");
        assert_eq!(back.len(), 2);
        assert_eq!(back["a.txt"], b"first");
        assert_eq!(back["b.txt"], b"second");
        // Sorted: `a.txt` header comes first.
        let first = String::from_utf8_lossy(&bytes[..100]);
        assert!(first.starts_with("a.txt"), "{first}");
    }

    #[test]
    fn zip_round_trips_and_checks_crc() {
        let mut a = BundleArchive::new();
        a.add("q/001.sql", Redacted::new("select 0", &[]))
            .expect("add");
        a.add("empty.txt", Redacted::new("", &[])).expect("add");
        let bytes = a.to_zip().expect("zip");
        let back = BundleArchive::read_zip(&bytes).expect("read");
        assert_eq!(back["q/001.sql"], b"select 0");
        assert!(back["empty.txt"].is_empty());
        // Corrupting file data trips the CRC check.
        let mut bad = bytes;
        let at = bad
            .windows(b"select 0".len())
            .position(|w| w == b"select 0")
            .expect("data present");
        bad[at] ^= 0xFF;
        assert!(BundleArchive::read_zip(&bad).is_err());
    }

    #[test]
    fn rejects_bad_names() {
        let mut a = BundleArchive::new();
        assert!(a.add("/abs", Redacted::new("x", &[])).is_err());
        assert!(a.add("../esc", Redacted::new("x", &[])).is_err());
        assert!(a.add("", Redacted::new("x", &[])).is_err());
        assert!(a.add("a\\b", Redacted::new("x", &[])).is_err());
        assert!(a.add(&"n".repeat(101), Redacted::new("x", &[])).is_err());
    }

    #[test]
    fn crc32_matches_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
