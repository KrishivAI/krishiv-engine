// SPDX-License-Identifier: Apache-2.0
//! `aes_encrypt`, `aes_decrypt` and `try_aes_decrypt` (ADR 0042), checked
//! against a Spark 4.2.0 run.
//!
//! Modes: `ECB` and `CBC` with `PKCS` (or `DEFAULT`) padding, `GCM` with
//! `DEFAULT`; keys of 16, 24 or 32 bytes. The output of `CBC` and `GCM` is
//! the IV followed by the ciphertext (and, for `GCM`, the 16-byte tag); an
//! empty IV on encryption draws a random one from the operating system.
//!
//! Keys, IVs and AAD never appear in an error (Spark's own messages carry
//! only lengths and the mode text) and the ciphers zeroize their key
//! schedules on drop.

use std::sync::Arc;

use aes::{Aes128, Aes192, Aes256};
use aes_gcm::aead::consts::U12;
use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{AesGcm, Nonce};
use arrow_array::{Array, ArrayRef, BinaryArray};
use arrow_schema::DataType;
use cbc::cipher::block_padding::Pkcs7;
use cbc::cipher::{BlockModeDecrypt, BlockModeEncrypt, KeyIvInit};
use krishiv_common::error::{
    InvalidParameterValueAesCryptoError, InvalidParameterValueAesIvLength,
    InvalidParameterValueAesKeyLength, KrishivError, UnsupportedFeatureAesMode,
    UnsupportedFeatureAesModeAad, UnsupportedFeatureAesModeIv,
};

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["aes_encrypt", "aes_decrypt", "try_aes_decrypt"];

/// How Spark names the pair in its messages.
const BOTH: &str = "`aes_encrypt`/`aes_decrypt`";

type Gcm192 = AesGcm<Aes192, U12>;

fn as_block(b: &mut [u8]) -> Result<&mut aes::Block, KrishivError> {
    <&mut aes::Block>::try_from(b).map_err(|_| internal("a block of the wrong size".to_owned()))
}

fn nonce(iv: &[u8]) -> Result<Nonce<U12>, KrishivError> {
    Nonce::<U12>::try_from(iv).map_err(|_| internal("a nonce of the wrong size".to_owned()))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ecb,
    Cbc,
    Gcm,
}

/// `(mode, padding)` as written, validated: ECB and CBC take `PKCS` or
/// `DEFAULT`, GCM takes `DEFAULT`.
fn mode(mode: &str, padding: &str) -> Result<Mode, KrishivError> {
    let (m, p) = (mode.to_ascii_uppercase(), padding.to_ascii_uppercase());
    match (m.as_str(), p.as_str()) {
        ("ECB", "PKCS" | "DEFAULT") => Ok(Mode::Ecb),
        ("CBC", "PKCS" | "DEFAULT") => Ok(Mode::Cbc),
        ("GCM", "DEFAULT") => Ok(Mode::Gcm),
        _ => Err(UnsupportedFeatureAesMode {
            mode: mode.to_owned(),
            padding: padding.to_owned(),
            function_name: BOTH.to_owned(),
        }
        .into()),
    }
}

fn crypto(detail: &str) -> KrishivError {
    InvalidParameterValueAesCryptoError {
        parameter: "`expr`, `key`".to_owned(),
        function_name: BOTH.to_owned(),
        detail_message: detail.to_owned(),
    }
    .into()
}

const BAD_PADDING: &str = "Given final block not properly padded. Such issues can arise if a bad key is used during decryption.";
const BAD_LENGTH: &str = "Input length must be multiple of 16 when decrypting with padded cipher";

fn key_length(key: &[u8]) -> Result<(), KrishivError> {
    if matches!(key.len(), 16 | 24 | 32) {
        Ok(())
    } else {
        Err(InvalidParameterValueAesKeyLength {
            parameter: "`key`".to_owned(),
            function_name: BOTH.to_owned(),
            actual_length: key.len().to_string(),
        }
        .into())
    }
}

/// One AES block-cipher call over the three key sizes.
macro_rules! by_key {
    ($key:expr, $body:ident) => {
        match $key.len() {
            16 => $body!(Aes128),
            24 => $body!(Aes192),
            _ => $body!(Aes256),
        }
    };
}

fn ecb_encrypt(key: &[u8], data: &[u8]) -> Result<Vec<u8>, KrishivError> {
    use aes::cipher::{BlockCipherEncrypt, KeyInit as _};
    macro_rules! go {
        ($c:ty) => {{
            let cipher = <$c>::new_from_slice(key).map_err(|_| crypto("invalid key"))?;
            let mut out = data.to_vec();
            let pad = 16 - data.len() % 16;
            out.extend(std::iter::repeat_n(u8::try_from(pad).unwrap_or(16), pad));
            for block in out.chunks_exact_mut(16) {
                cipher.encrypt_block(as_block(block)?);
            }
            Ok(out)
        }};
    }
    by_key!(key, go)
}

fn ecb_decrypt(key: &[u8], data: &[u8]) -> Result<Vec<u8>, KrishivError> {
    use aes::cipher::{BlockCipherDecrypt, KeyInit as _};
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if !data.len().is_multiple_of(16) {
        return Err(crypto(BAD_LENGTH));
    }
    macro_rules! go {
        ($c:ty) => {{
            let cipher = <$c>::new_from_slice(key).map_err(|_| crypto("invalid key"))?;
            let mut out = data.to_vec();
            for block in out.chunks_exact_mut(16) {
                cipher.decrypt_block(as_block(block)?);
            }
            unpad(out)
        }};
    }
    by_key!(key, go)
}

/// Strips PKCS#7 padding.
fn unpad(mut out: Vec<u8>) -> Result<Vec<u8>, KrishivError> {
    let pad = usize::from(*out.last().ok_or_else(|| crypto(BAD_PADDING))?);
    if pad == 0
        || pad > 16
        || pad > out.len()
        || out[out.len() - pad..]
            .iter()
            .any(|b| usize::from(*b) != pad)
    {
        return Err(crypto(BAD_PADDING));
    }
    out.truncate(out.len() - pad);
    Ok(out)
}

fn cbc_encrypt(key: &[u8], iv: &[u8], data: &[u8]) -> Result<Vec<u8>, KrishivError> {
    macro_rules! go {
        ($c:ty) => {{
            let enc = cbc::Encryptor::<$c>::new_from_slices(key, iv)
                .map_err(|_| crypto("invalid key or IV"))?;
            Ok(enc.encrypt_padded_vec::<Pkcs7>(data))
        }};
    }
    by_key!(key, go)
}

fn cbc_decrypt(key: &[u8], iv: &[u8], data: &[u8]) -> Result<Vec<u8>, KrishivError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if !data.len().is_multiple_of(16) {
        return Err(crypto(BAD_LENGTH));
    }
    macro_rules! go {
        ($c:ty) => {{
            let dec = cbc::Decryptor::<$c>::new_from_slices(key, iv)
                .map_err(|_| crypto("invalid key or IV"))?;
            dec.decrypt_padded_vec::<Pkcs7>(data)
                .map_err(|_| crypto(BAD_PADDING))
        }};
    }
    by_key!(key, go)
}

fn gcm_encrypt(key: &[u8], iv: &[u8], aad: &[u8], data: &[u8]) -> Result<Vec<u8>, KrishivError> {
    macro_rules! go {
        ($c:ty) => {{
            let cipher = <$c>::new_from_slice(key).map_err(|_| crypto("invalid key"))?;
            cipher
                .encrypt(&nonce(iv)?, Payload { msg: data, aad })
                .map_err(|_| crypto("encryption failed"))
        }};
    }
    match key.len() {
        16 => go!(aes_gcm::Aes128Gcm),
        24 => go!(Gcm192),
        _ => go!(aes_gcm::Aes256Gcm),
    }
}

fn gcm_decrypt(key: &[u8], iv: &[u8], aad: &[u8], data: &[u8]) -> Result<Vec<u8>, KrishivError> {
    macro_rules! go {
        ($c:ty) => {{
            let cipher = <$c>::new_from_slice(key).map_err(|_| crypto("invalid key"))?;
            cipher
                .decrypt(&nonce(iv)?, Payload { msg: data, aad })
                .map_err(|_| crypto("Tag mismatch"))
        }};
    }
    match key.len() {
        16 => go!(aes_gcm::Aes128Gcm),
        24 => go!(Gcm192),
        _ => go!(aes_gcm::Aes256Gcm),
    }
}

/// One row of `aes_encrypt`.
fn encrypt_row(
    data: &[u8],
    key: &[u8],
    m: &str,
    p: &str,
    iv: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, KrishivError> {
    key_length(key)?;
    let md = mode(m, p)?;
    if md == Mode::Ecb && !iv.is_empty() {
        return Err(UnsupportedFeatureAesModeIv {
            function_name: "`aes_encrypt`".to_owned(),
            mode: m.to_owned(),
        }
        .into());
    }
    if md != Mode::Gcm && !aad.is_empty() {
        return Err(UnsupportedFeatureAesModeAad {
            function_name: "`aes_encrypt`".to_owned(),
            mode: m.to_owned(),
        }
        .into());
    }
    let want = match md {
        Mode::Ecb => 0,
        Mode::Cbc => 16,
        Mode::Gcm => 12,
    };
    if md != Mode::Ecb && !iv.is_empty() && iv.len() != want {
        return Err(InvalidParameterValueAesIvLength {
            parameter: "`iv`".to_owned(),
            function_name: BOTH.to_owned(),
            actual_length: iv.len().to_string(),
            mode: m.to_owned(),
        }
        .into());
    }
    if md == Mode::Ecb {
        return ecb_encrypt(key, data);
    }
    let mut chosen = iv.to_vec();
    if chosen.is_empty() {
        chosen = vec![0; want];
        getrandom::fill(&mut chosen).map_err(|e| internal(format!("no random bytes: {e}")))?;
    }
    let body = if md == Mode::Cbc {
        cbc_encrypt(key, &chosen, data)?
    } else {
        gcm_encrypt(key, &chosen, aad, data)?
    };
    chosen.extend_from_slice(&body);
    Ok(chosen)
}

/// One row of `aes_decrypt`.
fn decrypt_row(
    data: &[u8],
    key: &[u8],
    m: &str,
    p: &str,
    aad: &[u8],
) -> Result<Vec<u8>, KrishivError> {
    key_length(key)?;
    match mode(m, p)? {
        Mode::Ecb => ecb_decrypt(key, data),
        Mode::Cbc => {
            if data.len() < 16 {
                return Err(crypto(
                    "IV buffer too short for given offset/length combination",
                ));
            }
            let (iv, body) = data.split_at(16);
            cbc_decrypt(key, iv, body)
        }
        Mode::Gcm => {
            if data.len() < 12 {
                return Err(crypto("Invalid buffer arguments"));
            }
            if data.len() < 28 {
                return Err(crypto(
                    "Input data too short to contain an expected tag length of 16bytes",
                ));
            }
            let (iv, body) = data.split_at(12);
            gcm_decrypt(key, iv, aad, body)
        }
    }
}

/// Binds `aes_encrypt`, `aes_decrypt` or `try_aes_decrypt`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let encrypt = listed == "aes_encrypt";
    let try_ = listed == "try_aes_decrypt";
    let n = args.len();
    let max = if encrypt { 6 } else { 5 };
    if !(2..=max).contains(&n) {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let binding = Binding::new(DataType::Binary, move |x: &Inputs<'_>| {
        let data = x.binaries(0)?;
        let key = x.binaries(1)?;
        let text = |i: usize| (i < n).then(|| x.strings(i)).transpose();
        let bytes = |i: usize| (i < n).then(|| x.binaries(i)).transpose();
        let modes = text(2)?;
        let pads = text(3)?;
        let (ivs, aads) = if encrypt {
            (bytes(4)?, bytes(5)?)
        } else {
            (None, bytes(4)?)
        };
        let out: BinaryArray = (0..x.rows())
            .map(|r| -> Result<Option<Vec<u8>>, KrishivError> {
                let null = data.is_null(r)
                    || key.is_null(r)
                    || modes.as_ref().is_some_and(|a| a.is_null(r))
                    || pads.as_ref().is_some_and(|a| a.is_null(r))
                    || ivs.as_ref().is_some_and(|a| a.is_null(r))
                    || aads.as_ref().is_some_and(|a| a.is_null(r));
                if null {
                    return Ok(None);
                }
                let m = modes.as_ref().map_or("GCM", |a| a.value(r));
                let p = pads.as_ref().map_or("DEFAULT", |a| a.value(r));
                let aad = aads.as_ref().map_or(&[][..], |a| a.value(r));
                let res = if encrypt {
                    let iv = ivs.as_ref().map_or(&[][..], |a| a.value(r));
                    encrypt_row(data.value(r), key.value(r), m, p, iv, aad)
                } else {
                    decrypt_row(data.value(r), key.value(r), m, p, aad)
                };
                match res {
                    Ok(v) => Ok(Some(v)),
                    Err(_) if try_ => Ok(None),
                    Err(e) => Err(e),
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Arc::new(out) as ArrayRef)
    });
    Some((listed, Ok(binding)))
}
