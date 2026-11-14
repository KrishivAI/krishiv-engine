// SPDX-License-Identifier: Apache-2.0
//! The client-certificate name walk (P3-T002 PR-14c; design §3.4, threat model §4.9):
//! `CertNames::from_der` reads the subject common names and the DNS and URI subject alternative
//! names out of a DER certificate. The certificate was verified by rustls before it gets here,
//! but the walk must still return names or an error for any bytes, never panic, loop or
//! allocate without bound. Seeds: the certificates of `crates/krishiv-session/tests/fixtures/tls/`.
#![no_main]

#[path = "../../crates/krishiv-session/src/auth/der.rs"]
mod der;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    if let Ok(names) = der::CertNames::from_der(data) {
        // Every name read lies inside the input.
        let total: usize = names
            .common_names
            .iter()
            .chain(&names.dns)
            .chain(&names.uris)
            .map(String::len)
            .sum();
        assert!(total <= data.len());
    }
});
