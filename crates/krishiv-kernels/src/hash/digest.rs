// SPDX-License-Identifier: Apache-2.0
//! `md5`, `sha1`, `sha2` and `crc32` over the bytes of a string or binary
//! column (a `STRING` argument is its UTF-8 bytes).

use arrow_array::builder::GenericStringBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, BinaryArray, Int64Array, StringArray};
use arrow_schema::DataType;
use md5::Md5;
use sha1::Sha1;
use sha2::{Digest, Sha224, Sha256, Sha384, Sha512};

use krishiv_common::error::{
    InvalidParameterValueHmacAlgorithm, InvalidParameterValueHmacCryptoError,
};

use crate::error::KernelError;
use crate::ops::Operand;

/// The bytes of row `i` of a string/binary column; `None` for NULL.
fn bytes_at(a: &dyn Array, i: usize) -> Option<&[u8]> {
    if a.is_null(i) {
        return None;
    }
    Some(match a.data_type() {
        DataType::Utf8 => a.as_string::<i32>().value(i).as_bytes(),
        DataType::LargeUtf8 => a.as_string::<i64>().value(i).as_bytes(),
        DataType::Binary => a.as_binary::<i32>().value(i),
        DataType::LargeBinary => a.as_binary::<i64>().value(i),
        _ => return None,
    })
}

fn check_input(a: &dyn Array) -> Result<(), KernelError> {
    match a.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => Ok(()),
        other => Err(KernelError::internal(format!(
            "a digest takes a string or binary column, not {other}"
        ))),
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn hex(bytes: &[u8], out: &mut String) {
    for b in bytes {
        out.push(char::from(HEX[usize::from(b >> 4)]));
        out.push(char::from(HEX[usize::from(b & 15)]));
    }
}

/// The output size after appending `add` bytes to `total`. The builder
/// panics on an offset past `i32::MAX`, so the kernel checks first.
fn next_offset(total: usize, add: usize) -> Result<usize, KernelError> {
    let next = total.saturating_add(add);
    i32::try_from(next)
        .map(|_| next)
        .map_err(|_| KernelError::out_of_memory(next))
}

fn hex_digests(
    input: &dyn Array,
    mut digest: impl FnMut(usize, &[u8], &mut String) -> bool,
) -> Result<StringArray, KernelError> {
    check_input(input)?;
    let mut out = GenericStringBuilder::<i32>::with_capacity(input.len(), 0);
    let mut scratch = String::new();
    let mut bytes = 0_usize;
    for i in 0..input.len() {
        scratch.clear();
        match bytes_at(input, i) {
            Some(b) if digest(i, b, &mut scratch) => {
                bytes = next_offset(bytes, scratch.len())?;
                out.append_value(&scratch);
            }
            _ => out.append_null(),
        }
    }
    Ok(out.finish())
}

/// `md5(input)`: 32 lowercase hex digits.
///
/// # Errors
/// `INTERNAL_ERROR` if `input` is not a string or binary column.
pub fn md5(input: &dyn Array) -> Result<StringArray, KernelError> {
    hex_digests(input, |_, b, out| {
        hex(&Md5::digest(b), out);
        true
    })
}

/// `sha1(input)` (also `sha`): 40 lowercase hex digits.
///
/// # Errors
/// As [`md5()`].
pub fn sha1(input: &dyn Array) -> Result<StringArray, KernelError> {
    hex_digests(input, |_, b, out| {
        hex(&Sha1::digest(b), out);
        true
    })
}

/// `sha2(input, bits)`: SHA-224/256/384/512 for `bits` 224, 256, 384, 512
/// (0 is 256); any other length, and a NULL length, gives NULL.
///
/// # Errors
/// As [`md5()`]; `INTERNAL_ERROR` if a `bits` column differs in length.
pub fn sha2(input: &dyn Array, bits: Operand<'_, Int32Type>) -> Result<StringArray, KernelError> {
    if let Operand::Array(b) = bits
        && b.len() != input.len()
    {
        return Err(KernelError::internal(format!(
            "sha2 operands differ in length: {} vs {}",
            input.len(),
            b.len()
        )));
    }
    let bits_at = |i: usize| match bits {
        Operand::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Operand::Scalar(v) => v,
    };
    hex_digests(input, |i, b, out| match bits_at(i) {
        Some(0 | 256) => {
            hex(&Sha256::digest(b), out);
            true
        }
        Some(224) => {
            hex(&Sha224::digest(b), out);
            true
        }
        Some(384) => {
            hex(&Sha384::digest(b), out);
            true
        }
        Some(512) => {
            hex(&Sha512::digest(b), out);
            true
        }
        _ => false,
    })
}

/// The CRC-32 (IEEE 802.3, reflected) table.
const fn crc_table() -> [u32; 256] {
    let mut t = [0_u32; 256];
    let mut n = 0_u32;
    while n < 256 {
        let mut c = n;
        let mut k = 0_u32;
        while k < 8 {
            c = if c & 1 == 1 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k = k.wrapping_add(1);
        }
        t[n as usize] = c;
        n = n.wrapping_add(1);
    }
    t
}

static CRC_TABLE: [u32; 256] = crc_table();

fn crc32_of(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(!0_u32, |c, &b| {
        CRC_TABLE[usize::from(u8::try_from((c ^ u32::from(b)) & 0xff).unwrap_or_default())]
            ^ (c >> 8)
    })
}

/// `crc32(input)`: the CRC-32 checksum as a `bigint`.
///
/// # Errors
/// As [`md5()`].
pub fn crc32(input: &dyn Array) -> Result<Int64Array, KernelError> {
    check_input(input)?;
    Ok((0..input.len())
        .map(|i| bytes_at(input, i).map(|b| i64::from(crc32_of(b))))
        .collect())
}

/// The bytes of `parts` hashed by `D`.
fn hashed<D: Digest>(parts: &[&[u8]]) -> Vec<u8> {
    let mut d = D::new();
    for p in parts {
        d.update(p);
    }
    d.finalize().to_vec()
}

/// HMAC (RFC 2104) of `message` under `key` with hash `D` of block size `block`.
fn hmac_of<D: Digest>(block: usize, key: &[u8], message: &[u8]) -> Vec<u8> {
    let mut k = if key.len() > block {
        hashed::<D>(&[key])
    } else {
        key.to_vec()
    };
    k.resize(block, 0);
    let ipad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k.iter().map(|b| b ^ 0x5c).collect();
    let inner = hashed::<D>(&[&ipad, message]);
    hashed::<D>(&[&opad, &inner])
}

/// `hmac(key, message[, algorithm])` (Spark 4.3): the raw MAC bytes. `algorithm` is `SHA-224`, `SHA-256`,
/// `SHA-384`, `SHA-512`, `SHA-1` (also without the hyphen) or `MD5`, in any case, and `SHA-256` when `None`.
/// NULL when any argument is NULL.
///
/// # Errors
/// `INVALID_PARAMETER_VALUE.HMAC_ALGORITHM` for another algorithm (the first failing row);
/// `INVALID_PARAMETER_VALUE.HMAC_CRYPTO_ERROR` for an empty key, which Java's `SecretKeySpec` refuses;
/// `INTERNAL_ERROR` if the columns differ in length.
pub fn hmac(
    key: &BinaryArray,
    message: &BinaryArray,
    algorithm: Option<&StringArray>,
) -> Result<BinaryArray, KernelError> {
    let rows = key.len();
    if message.len() != rows || algorithm.is_some_and(|a| a.len() != rows) {
        return Err(KernelError::internal("hmac operands differ in length"));
    }
    let mut out = arrow_array::builder::BinaryBuilder::with_capacity(rows, 0);
    for i in 0..rows {
        let name = match algorithm {
            Some(a) if a.is_null(i) => {
                out.append_null();
                continue;
            }
            Some(a) => a.value(i),
            None => "SHA-256",
        };
        if key.is_null(i) || message.is_null(i) {
            out.append_null();
            continue;
        }
        let (key, message) = (key.value(i), message.value(i));
        // Spark maps the name first (`hmacName`), then the key is refused when empty.
        let upper = name.to_uppercase();
        if !matches!(
            upper.as_str(),
            "SHA-224"
                | "SHA224"
                | "SHA-256"
                | "SHA256"
                | "SHA-384"
                | "SHA384"
                | "SHA-512"
                | "SHA512"
                | "SHA-1"
                | "SHA1"
                | "MD5"
        ) {
            return Err(KernelError::with_row(
                Some(i),
                InvalidParameterValueHmacAlgorithm {
                    function_name: "`hmac`".to_owned(),
                    parameter: "`algorithm`".to_owned(),
                    algorithm: format!("'{name}'"),
                }
                .into(),
            ));
        }
        if key.is_empty() {
            return Err(KernelError::with_row(
                Some(i),
                InvalidParameterValueHmacCryptoError {
                    function_name: "`hmac`".to_owned(),
                    parameter: "`key`".to_owned(),
                    detail_message: "Empty key".to_owned(),
                }
                .into(),
            ));
        }
        let mac = match upper.as_str() {
            "SHA-224" | "SHA224" => hmac_of::<Sha224>(64, key, message),
            "SHA-256" | "SHA256" => hmac_of::<Sha256>(64, key, message),
            "SHA-384" | "SHA384" => hmac_of::<Sha384>(128, key, message),
            "SHA-512" | "SHA512" => hmac_of::<Sha512>(128, key, message),
            "SHA-1" | "SHA1" => hmac_of::<Sha1>(64, key, message),
            _ => hmac_of::<Md5>(64, key, message),
        };
        out.append_value(mac);
    }
    Ok(out.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_offsets_are_checked() {
        assert_eq!(next_offset(10, 32).unwrap(), 42);
        let max = usize::try_from(i32::MAX).unwrap();
        assert_eq!(next_offset(max - 1, 1).unwrap(), max);
        let e = next_offset(max, 1).unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::UnableToAcquireMemory
        );
        assert!(next_offset(usize::MAX, 1).is_err());
    }
}
