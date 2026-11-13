// SPDX-License-Identifier: Apache-2.0

use super::crc32;

#[test]
fn crc32_matches_zlib() {
    // zlib.crc32 values.
    assert_eq!(crc32(b""), 0);
    assert_eq!(crc32(b"a"), 0xe8b7_be43);
    assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    assert_eq!(
        crc32(b"The quick brown fox jumps over the lazy dog"),
        0x414f_a339
    );
}
