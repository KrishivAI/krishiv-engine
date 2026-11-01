// SPDX-License-Identifier: Apache-2.0
//! `aes_encrypt`, `aes_decrypt` and `try_aes_decrypt` end to end: every
//! row and error class below is what a Spark 4.2.0 run returned.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

fn one(s: &Session, sql: &str) -> Result<String, KrishivError> {
    let df = block_on(s.sql(sql))?;
    let batches = block_on(df.collect())?;
    let c = batches[0].column(0);
    if c.is_null_at(0) {
        return Ok("NULL".to_owned());
    }
    Ok(match c.data_type() {
        arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(0).to_owned(),
        arrow_schema::DataType::Utf8View => c.as_string_view().value(0).to_owned(),
        arrow_schema::DataType::Int32 => c
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0)
            .to_string(),
        other => panic!("unexpected result type {other}"),
    })
}

trait NullAt {
    fn is_null_at(&self, i: usize) -> bool;
}

impl NullAt for arrow_array::ArrayRef {
    fn is_null_at(&self, i: usize) -> bool {
        arrow_array::Array::is_null(self.as_ref(), i)
    }
}

const K: &str = "'abcdefghijklmnop'";
const ZERO16: &str = "unhex('00000000000000000000000000000000')";
const ECB_SPARK: &str = "E07BF450A0B1EA77D4780A0F668D73FB";

#[test]
fn rows_follow_spark() {
    let s = Session::new().unwrap();
    let rows: Vec<(String, &str)> = vec![
        (format!("hex(aes_encrypt('Spark', {K}, 'ECB'))"), ECB_SPARK),
        (format!("hex(aes_encrypt('Spark', {K}, 'ecb', 'pkcs'))"), ECB_SPARK),
        (
            format!("hex(aes_encrypt('Spark', {K}, 'ECB', 'DEFAULT', unhex(''), ''))"),
            ECB_SPARK,
        ),
        (
            format!("hex(aes_encrypt('Spark', {K}, 'CBC', 'PKCS', {ZERO16}))"),
            "00000000000000000000000000000000E07BF450A0B1EA77D4780A0F668D73FB",
        ),
        (
            format!(
                "hex(aes_encrypt('Spark', {K}, 'GCM', 'DEFAULT', unhex('000000000000000000000000'), 'aad'))"
            ),
            "00000000000000000000000045D5E9F7BD5075212F6E004FA90D255EF0DC24F6B9",
        ),
        (
            "hex(aes_encrypt('Spark', 'abcdefghijklmnop24bytes!', 'ECB'))".to_owned(),
            "D06F931AB69C00DED910B4E96FFEAF8D",
        ),
        (
            "hex(aes_encrypt('Spark', 'abcdefghijklmnop24bytes!abcdefgh', 'ECB'))".to_owned(),
            "61A37EA8BAE809336DA72991039DC39A",
        ),
        (
            format!("hex(aes_encrypt('', {K}, 'ECB'))"),
            "8E64CE873F174DBB2423FCD814580E15",
        ),
        (
            format!("length(aes_encrypt('Spark', {K}))"),
            "33",
        ),
        (
            format!("cast(aes_decrypt(aes_encrypt('Spark', {K}), {K}) as string)"),
            "Spark",
        ),
        (
            "cast(aes_decrypt(aes_encrypt('Spark', 'abcdefghijklmnop24bytes!'), 'abcdefghijklmnop24bytes!') as string)"
                .to_owned(),
            "Spark",
        ),
        (
            format!("cast(aes_decrypt(unhex('{ECB_SPARK}'), {K}, 'ECB') as string)"),
            "Spark",
        ),
        (
            format!("cast(aes_decrypt(unhex('{ECB_SPARK}'), {K}, 'ECB', 'PKCS', 'aad') as string)"),
            "Spark",
        ),
        (
            format!("hex(aes_decrypt(unhex(''), {K}, 'ECB'))"),
            "",
        ),
        (
            format!("hex(aes_decrypt(unhex('00000000000000000000000000000000'), {K}, 'CBC'))"),
            "",
        ),
        (
            format!(
                "cast(aes_decrypt(unhex('00000000000000000000000045D5E9F7BD5075212F6E004FA90D255EF0DC24F6B9'), {K}, 'GCM', 'DEFAULT', 'aad') as string)"
            ),
            "Spark",
        ),
        (
            format!(
                "hex(aes_encrypt('Spark', {K}, 'GCM', 'NONE', unhex('000000000000000000000000')))"
            ),
            "00000000000000000000000045D5E9F7BD0392ACD0CFB20FC68C2CD3985AEA422D",
        ),
        (format!("aes_encrypt(NULL, {K})"), "NULL"),
        ("aes_encrypt('Spark', NULL)".to_owned(), "NULL"),
        (format!("aes_encrypt('Spark', {K}, NULL)"), "NULL"),
        (
            format!("aes_encrypt('Spark', {K}, 'ECB', 'PKCS', NULL, 'aad')"),
            "NULL",
        ),
        (
            format!("aes_decrypt(unhex('{ECB_SPARK}'), {K}, NULL)"),
            "NULL",
        ),
        (
            format!("try_aes_decrypt(unhex('{}A'), {K}, 'ECB')", &ECB_SPARK[..31]),
            "NULL",
        ),
        (
            format!("try_aes_decrypt(unhex('{ECB_SPARK}'), 'short')"),
            "NULL",
        ),
        (
            format!("try_aes_decrypt(unhex('{ECB_SPARK}'), {K}, 'ECB', 'XX')"),
            "NULL",
        ),
        (
            format!("cast(try_aes_decrypt(unhex('{ECB_SPARK}'), {K}, 'ECB') as string)"),
            "Spark",
        ),
        (
            format!("typeof(aes_encrypt('Spark', {K}))"),
            "binary",
        ),
    ];
    for (sql, want) in &rows {
        assert_eq!(one(&s, &format!("SELECT {sql}")).unwrap(), *want, "{sql}");
    }
}

#[test]
fn errors_follow_spark() {
    let s = Session::new().unwrap();
    let bad_pad = format!("{}A", &ECB_SPARK[..31]);
    let rows: Vec<(String, &str)> = vec![
        (
            "aes_encrypt('Spark', 'short')".to_owned(),
            "INVALID_PARAMETER_VALUE.AES_KEY_LENGTH",
        ),
        (
            "aes_encrypt('Spark', 'short', 'XYZ')".to_owned(),
            "INVALID_PARAMETER_VALUE.AES_KEY_LENGTH",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'XYZ')"),
            "UNSUPPORTED_FEATURE.AES_MODE",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'ECB', 'NONE')"),
            "UNSUPPORTED_FEATURE.AES_MODE",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'CBC', 'NONE', {ZERO16})"),
            "UNSUPPORTED_FEATURE.AES_MODE",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'GCM', 'PKCS')"),
            "UNSUPPORTED_FEATURE.AES_MODE",
        ),
        (
            format!("aes_decrypt(unhex('{ECB_SPARK}'), {K}, 'ECB', 'NONE')"),
            "UNSUPPORTED_FEATURE.AES_MODE",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'ECB', 'DEFAULT', unhex('00'))"),
            "UNSUPPORTED_FEATURE.AES_MODE_IV",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'CBC', 'DEFAULT', unhex('00'), 'x')"),
            "INVALID_PARAMETER_VALUE.AES_IV_LENGTH",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'CBC', 'PKCS', {ZERO16}, 'aad')"),
            "UNSUPPORTED_FEATURE.AES_MODE_AAD",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'CBC', 'DEFAULT', unhex('00'))"),
            "INVALID_PARAMETER_VALUE.AES_IV_LENGTH",
        ),
        (
            format!("aes_encrypt('Spark', {K}, 'GCM', 'DEFAULT', unhex('00'))"),
            "INVALID_PARAMETER_VALUE.AES_IV_LENGTH",
        ),
        (
            format!("aes_decrypt(unhex('{bad_pad}'), {K}, 'ECB')"),
            "INVALID_PARAMETER_VALUE.AES_CRYPTO_ERROR",
        ),
        (
            format!("aes_decrypt(unhex('E07BF450A0B1EA77D4780A0F668D73'), {K}, 'ECB')"),
            "INVALID_PARAMETER_VALUE.AES_CRYPTO_ERROR",
        ),
        (
            format!("aes_decrypt(unhex('00'), {K}, 'GCM')"),
            "INVALID_PARAMETER_VALUE.AES_CRYPTO_ERROR",
        ),
        (
            format!("aes_decrypt(unhex('00'), {K}, 'CBC')"),
            "INVALID_PARAMETER_VALUE.AES_CRYPTO_ERROR",
        ),
        (
            format!(
                "aes_decrypt(unhex('00000000000000000000000045D5E9F7BD5075212F6E004FA90D255EF0DC24F6B9'), {K}, 'GCM', 'DEFAULT')"
            ),
            "INVALID_PARAMETER_VALUE.AES_CRYPTO_ERROR",
        ),
        (
            format!("aes_encrypt(1, {K})"),
            "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
        ),
    ];
    for (sql, class) in &rows {
        let err = one(&s, &format!("SELECT {sql}")).expect_err(sql);
        assert_eq!(err.error_class().name(), *class, "{sql}: {err}");
    }
}

#[test]
fn the_key_never_reaches_an_error_message() {
    let s = Session::new().unwrap();
    for sql in [
        "SELECT aes_encrypt('Spark', 'secretkey-123456', 'XYZ')",
        "SELECT aes_encrypt(1, 'secretkey-123456')",
        "SELECT aes_decrypt(unhex('00'), 'secretkey-123456', 'GCM')",
        "SELECT AES_ENCRYPT(1, 'secretkey-123456')",
        "SELECT Try_Aes_Decrypt(1, 'secretkey-123456')",
        "SELECT aes_encrypt('Spark', array('secretkey-123456'))",
        "SELECT aes_decrypt(unhex('00'), map('secretkey-123456', 1))",
        "SELECT aes_encrypt(DISTINCT 1, 'secretkey-123456')",
        "SELECT aes_decrypt(DISTINCT unhex('00'), 'secretkey-123456')",
        "SELECT aes_encrypt('Spark', CAST(array('secretkey-123456') AS BINARY))",
        "SELECT try_aes_decrypt(unhex('00'), CAST(map('secretkey-123456', 1) AS BINARY))",
        "SELECT aes_encrypt('Spark', CAST(CAST(struct('secretkey-123456') AS ARRAY<INT>) AS BINARY))",
    ] {
        let err = one(&s, sql).expect_err(sql);
        assert!(
            !err.to_string().contains("secretkey-123456"),
            "{sql}: {err}"
        );
    }
}

/// A failure while evaluating the key (ANSI cast of a bad string) names no
/// value of it.
#[test]
fn an_execution_error_in_the_key_hides_the_key() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = true")).unwrap();
    for sql in [
        "SELECT aes_encrypt('Spark', CAST(CAST('secretkey-123456' AS INT) AS STRING))",
        "SELECT aes_decrypt(unhex('00'), CAST(CAST(c AS INT) AS STRING)) FROM VALUES ('secretkey-123456') t(c)",
        "SELECT try_aes_decrypt(unhex('00'), CAST(CAST(c AS INT) AS STRING)) FROM VALUES ('secretkey-123456') t(c)",
    ] {
        let err = one(&s, sql).expect_err(sql);
        assert!(
            !err.to_string().contains("secretkey-123456"),
            "{sql}: {err}"
        );
    }
}

#[test]
fn datepart_reads_the_time_precision_like_date_part() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let a = one(
        &s,
        "SELECT CAST(datepart('SECOND', CAST('09:08:01.987654' AS TIME(1))) AS STRING)",
    )
    .unwrap();
    let b = one(
        &s,
        "SELECT CAST(date_part('SECOND', CAST('09:08:01.987654' AS TIME(1))) AS STRING)",
    )
    .unwrap();
    assert_eq!(a, b);
}

fn distinct_ciphertexts(from: &str, arg: &str) -> i64 {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!(
        "SELECT count(DISTINCT hex(aes_encrypt({arg}, {K}))) FROM {from}"
    )))
    .unwrap();
    let b = block_on(df.collect()).unwrap();
    b[0].column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0)
}

/// Spark 4.2.0: constant folding computes a literal call once, so every row
/// carries the same IV (and `count(DISTINCT ...)` over it is accepted).
#[test]
fn a_literal_call_is_folded_to_one_iv() {
    assert_eq!(distinct_ciphertexts("range(10)", "'Spark'"), 1);
    assert_eq!(
        distinct_ciphertexts("(SELECT id FROM range(1000))", "'Spark'"),
        1
    );
}

/// A key that is itself a call over constants (`unhex`) is still foldable.
#[test]
fn a_constant_expression_argument_is_folded_too() {
    // Batches of several rows: the key call must not depend on their length.
    let from = "range(0, 10000, 1, 4)";
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!(
        "SELECT count(DISTINCT hex(aes_encrypt('Spark', unhex('6162636465666768696A6B6C6D6E6F70')))) FROM {from}"
    )))
    .unwrap();
    let b = block_on(df.collect()).unwrap();
    let n = b[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(n, 1);
}

/// Spark 4.2.0: a column input draws an IV for every row (4 distinct
/// ciphertexts of 4 identical values). Must never be weakened.
#[test]
fn a_column_input_draws_a_fresh_iv_for_every_row() {
    assert_eq!(
        distinct_ciphertexts(
            "VALUES ('Spark'), ('Spark'), ('Spark'), ('Spark') AS t(x)",
            "x"
        ),
        4
    );
}

#[test]
fn a_qualified_call_hides_the_key_too() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!("SELECT builtin.aes_encrypt(1, {K})")));
    let msg = match df {
        Ok(df) => format!("{:?}", block_on(df.collect()).map(|_| ())),
        Err(e) => e.to_string(),
    };
    assert!(!msg.contains("abcdefghijklmnop"), "{msg}");
}
