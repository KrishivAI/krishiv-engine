// SPDX-License-Identifier: Apache-2.0
//! The names in an X.509 certificate a client presents (design §3.4; ADR 0046 §6): the subject
//! common names and the DNS and URI subject alternative names, read by a minimal DER walk.
//!
//! The chain was verified by rustls at the handshake; this only reads names out of the leaf.
//! The walk is strict and bounded: definite lengths only, no high tag numbers, no trailing
//! bytes where a structure ends, every length inside its parent. Anything else is
//! [`DerError`], which refuses the certificate. There is no recursion. `fuzz_targets/cert_names.rs`
//! fuzzes [`CertNames::from_der`].

/// What went wrong reading a certificate. Never carries certificate bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DerError {
    /// The input ended inside an element.
    Truncated,
    /// An element is not the one the certificate structure has at that place.
    Unexpected,
    /// A length form DER does not allow (indefinite, over four bytes, or not minimal).
    BadLength,
    /// Bytes left after the last element of a structure.
    Trailing,
    /// A name is not text of the type its tag says (or holds a control character).
    BadText,
    /// The certificate has two subject alternative name extensions.
    DuplicateExtension,
}

impl std::fmt::Display for DerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Truncated => "truncated element",
            Self::Unexpected => "unexpected element",
            Self::BadLength => "length not in DER form",
            Self::Trailing => "bytes after the last element",
            Self::BadText => "a name is not valid text",
            Self::DuplicateExtension => "two subjectAltName extensions",
        })
    }
}

/// The names of a certificate's subject.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CertNames {
    /// Every `commonName` attribute of the subject, in order.
    pub common_names: Vec<String>,
    /// Every `dNSName` of the `subjectAltName` extension, in order.
    pub dns: Vec<String>,
    /// Every `uniformResourceIdentifier` of the `subjectAltName` extension, in order.
    pub uris: Vec<String>,
}

// Universal tags.
const BOOLEAN: u8 = 0x01;
const INTEGER: u8 = 0x02;
const OCTET_STRING: u8 = 0x04;
const OID: u8 = 0x06;
const UTF8_STRING: u8 = 0x0c;
const PRINTABLE_STRING: u8 = 0x13;
const IA5_STRING: u8 = 0x16;
const SEQUENCE: u8 = 0x30;
const SET: u8 = 0x31;
// Context tags of `TBSCertificate` and `GeneralName`.
const VERSION: u8 = 0xa0;
const ISSUER_UID: u8 = 0x81;
const SUBJECT_UID: u8 = 0x82;
const EXTENSIONS: u8 = 0xa3;
const DNS_NAME: u8 = 0x82;
const URI: u8 = 0x86;

/// `id-at-commonName`, 2.5.4.3.
const OID_COMMON_NAME: &[u8] = &[0x55, 0x04, 0x03];
/// `id-ce-subjectAltName`, 2.5.29.17.
const OID_SUBJECT_ALT_NAME: &[u8] = &[0x55, 0x1d, 0x11];

/// A cursor over the elements of one DER structure.
struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    /// The tag of the next element, if any.
    fn peek(&self) -> Option<u8> {
        self.rest.first().copied()
    }

    /// The next element: its tag and contents.
    fn any(&mut self) -> Result<(u8, &'a [u8]), DerError> {
        let (&tag, after) = self.rest.split_first().ok_or(DerError::Truncated)?;
        // High tag numbers (low five bits all set) never occur in the fields read here.
        if tag & 0x1f == 0x1f {
            return Err(DerError::Unexpected);
        }
        let (&first, mut after) = after.split_first().ok_or(DerError::Truncated)?;
        let len = if first < 0x80 {
            usize::from(first)
        } else {
            let count = usize::from(first & 0x7f);
            // 0x80 is BER's indefinite form; more than four length bytes is no real certificate.
            if count == 0 || count > 4 {
                return Err(DerError::BadLength);
            }
            let (bytes, tail) = after.split_at_checked(count).ok_or(DerError::Truncated)?;
            after = tail;
            if bytes.first() == Some(&0) {
                return Err(DerError::BadLength);
            }
            let len = bytes
                .iter()
                .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
            if len < 0x80 {
                return Err(DerError::BadLength);
            }
            len
        };
        let (contents, tail) = after.split_at_checked(len).ok_or(DerError::Truncated)?;
        self.rest = tail;
        Ok((tag, contents))
    }

    /// The next element, which must have tag `tag`.
    fn expect(&mut self, tag: u8) -> Result<&'a [u8], DerError> {
        match self.any()? {
            (t, contents) if t == tag => Ok(contents),
            _ => Err(DerError::Unexpected),
        }
    }

    /// The next element if it has tag `tag`.
    fn optional(&mut self, tag: u8) -> Result<Option<&'a [u8]>, DerError> {
        if self.peek() == Some(tag) {
            self.expect(tag).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Nothing may follow.
    fn finish(&self) -> Result<(), DerError> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(DerError::Trailing)
        }
    }
}

impl CertNames {
    /// The names of the DER certificate `cert`.
    ///
    /// # Errors
    /// [`DerError`] when `cert` is not a DER certificate, or a name read is not valid text.
    pub fn from_der(cert: &[u8]) -> Result<Self, DerError> {
        let mut outer = Reader::new(cert);
        let certificate = outer.expect(SEQUENCE)?;
        outer.finish()?;
        let mut certificate = Reader::new(certificate);
        let tbs = certificate.expect(SEQUENCE)?;
        // signatureAlgorithm and signatureValue follow; rustls checked them.
        let mut tbs = Reader::new(tbs);
        tbs.optional(VERSION)?;
        tbs.expect(INTEGER)?; // serialNumber
        tbs.expect(SEQUENCE)?; // signature
        tbs.expect(SEQUENCE)?; // issuer
        tbs.expect(SEQUENCE)?; // validity
        let subject = tbs.expect(SEQUENCE)?;
        tbs.expect(SEQUENCE)?; // subjectPublicKeyInfo
        tbs.optional(ISSUER_UID)?;
        tbs.optional(SUBJECT_UID)?;
        let extensions = tbs.optional(EXTENSIONS)?;
        tbs.finish()?;

        let mut names = Self {
            common_names: common_names(subject)?,
            ..Self::default()
        };
        if let Some(extensions) = extensions {
            let mut explicit = Reader::new(extensions);
            let list = explicit.expect(SEQUENCE)?;
            explicit.finish()?;
            let mut list = Reader::new(list);
            let mut seen = false;
            while !list.is_empty() {
                let mut ext = Reader::new(list.expect(SEQUENCE)?);
                let id = ext.expect(OID)?;
                ext.optional(BOOLEAN)?; // critical
                let value = ext.expect(OCTET_STRING)?;
                ext.finish()?;
                if id == OID_SUBJECT_ALT_NAME {
                    if seen {
                        return Err(DerError::DuplicateExtension);
                    }
                    seen = true;
                    alt_names(value, &mut names)?;
                }
            }
        }
        Ok(names)
    }
}

/// The `commonName` values of the `Name` whose contents are `subject`
/// (`RDNSequence`: a sequence of sets of attribute type and value).
fn common_names(subject: &[u8]) -> Result<Vec<String>, DerError> {
    let mut out = Vec::new();
    let mut rdns = Reader::new(subject);
    while !rdns.is_empty() {
        let mut set = Reader::new(rdns.expect(SET)?);
        while !set.is_empty() {
            let mut attr = Reader::new(set.expect(SEQUENCE)?);
            let id = attr.expect(OID)?;
            let (tag, value) = attr.any()?;
            attr.finish()?;
            if id == OID_COMMON_NAME {
                out.push(text(tag, value)?);
            }
        }
    }
    Ok(out)
}

/// The DNS and URI names of the `GeneralNames` whose DER is `value`; other kinds are skipped.
fn alt_names(value: &[u8], names: &mut CertNames) -> Result<(), DerError> {
    let mut outer = Reader::new(value);
    let list = outer.expect(SEQUENCE)?;
    outer.finish()?;
    let mut list = Reader::new(list);
    while !list.is_empty() {
        match list.any()? {
            (DNS_NAME, v) => names.dns.push(text(IA5_STRING, v)?),
            (URI, v) => names.uris.push(text(IA5_STRING, v)?),
            _ => {}
        }
    }
    Ok(())
}

/// The string of a `DirectoryString` or `IA5String` element: UTF-8 for `UTF8String`, ASCII
/// for the others, never a control character. `TeletexString`, `BMPString` and
/// `UniversalString` are refused.
fn text(tag: u8, bytes: &[u8]) -> Result<String, DerError> {
    let s = match tag {
        UTF8_STRING => std::str::from_utf8(bytes).map_err(|_| DerError::BadText)?,
        PRINTABLE_STRING | IA5_STRING if bytes.is_ascii() => {
            std::str::from_utf8(bytes).map_err(|_| DerError::BadText)?
        }
        _ => return Err(DerError::BadText),
    };
    if s.chars().any(char::is_control) {
        return Err(DerError::BadText);
    }
    Ok(s.to_owned())
}
