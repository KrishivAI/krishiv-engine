// SPDX-License-Identifier: Apache-2.0
//! AWS Signature Version 4 for one JSON POST (design P3-lakehouse §2.3): Krishiv signs its own Glue requests with
//! `sha2` and a small HMAC instead of taking the AWS SDK (decision 8's reasoning for the REST client holds: a
//! credential is never handed to code that follows a server's redirect or override). Signed headers are exactly
//! `content-type`, `host`, `x-amz-date`, `x-amz-target` and, with temporary credentials, `x-amz-security-token`.

use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

const BLOCK: usize = 64;

fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// HMAC-SHA256 (RFC 2104).
pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut block = [0_u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&sha256(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|b| b ^ 0x36));
    inner.update(data);
    let inner: [u8; 32] = inner.finalize().into();
    let mut outer = Sha256::new();
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner);
    outer.finalize().into()
}

/// `YYYYMMDDTHHMMSSZ` of `unix` seconds (UTC).
pub(crate) fn amz_date(unix: u64) -> String {
    let days = unix / 86_400;
    let secs = unix % 86_400;
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = i64::try_from(days).unwrap_or(0) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        secs / 3600,
        secs % 3600 / 60,
        secs % 60
    )
}

/// The current time as an `x-amz-date` value.
pub(crate) fn now_amz_date() -> String {
    amz_date(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    )
}

/// What one signature covers.
pub(crate) struct Signing<'a> {
    pub(crate) method: &'a str,
    /// The `host` header value (the authority, with a non-default port).
    pub(crate) host: &'a str,
    pub(crate) service: &'a str,
    pub(crate) region: &'a str,
    /// `x-amz-date`.
    pub(crate) date: &'a str,
    /// Further headers to sign, lower-case names, besides `host` and `x-amz-date`.
    pub(crate) headers: &'a [(&'a str, &'a str)],
    pub(crate) body: &'a [u8],
    pub(crate) access_key_id: &'a str,
    pub(crate) secret_access_key: &'a str,
}

impl Signing<'_> {
    /// The value of the `Authorization` header. The path is `/` and there is no query, as every Glue call has.
    pub(crate) fn authorization(&self) -> String {
        let mut headers: Vec<(&str, &str)> = self.headers.to_vec();
        headers.push(("host", self.host));
        headers.push(("x-amz-date", self.date));
        headers.sort_by_key(|(k, _)| *k);
        let canonical_headers: String = headers
            .iter()
            .map(|(k, v)| format!("{k}:{}\n", v.trim()))
            .collect();
        let signed = headers
            .iter()
            .map(|(k, _)| *k)
            .collect::<Vec<_>>()
            .join(";");
        let canonical = format!(
            "{}\n/\n\n{canonical_headers}\n{signed}\n{}",
            self.method,
            hex(&sha256(self.body))
        );
        let day = &self.date[..8.min(self.date.len())];
        let scope = format!("{day}/{}/{}/aws4_request", self.region, self.service);
        let to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{scope}\n{}",
            self.date,
            hex(&sha256(canonical.as_bytes()))
        );
        let mut key = hmac_sha256(
            format!("AWS4{}", self.secret_access_key).as_bytes(),
            day.as_bytes(),
        );
        for part in [self.region, self.service, "aws4_request"] {
            key = hmac_sha256(&key, part.as_bytes());
        }
        let signature = hex(&hmac_sha256(&key, to_sign.as_bytes()));
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={signature}",
            self.access_key_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231() {
        // Test case 2 and the long-key case 6.
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn dates_are_utc_civil_dates() {
        assert_eq!(amz_date(0), "19700101T000000Z");
        assert_eq!(amz_date(1_440_938_160), "20150830T123600Z");
        // A leap day, and the day after.
        assert_eq!(amz_date(1_709_164_800), "20240229T000000Z");
        assert_eq!(amz_date(1_709_251_199), "20240229T235959Z");
        assert_eq!(amz_date(1_709_251_200), "20240301T000000Z");
    }

    /// The `get-vanilla` case of AWS's Signature Version 4 test suite.
    #[test]
    fn the_aws_get_vanilla_vector() {
        let s = Signing {
            method: "GET",
            host: "example.amazonaws.com",
            service: "service",
            region: "us-east-1",
            date: "20150830T123600Z",
            headers: &[],
            body: b"",
            access_key_id: "AKIDEXAMPLE",
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        };
        assert_eq!(
            s.authorization(),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }
}
