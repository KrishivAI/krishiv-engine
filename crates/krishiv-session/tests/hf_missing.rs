// SPDX-License-Identifier: Apache-2.0
//! KN-7: reading a named `hf://` file that is not there is `PATH_NOT_FOUND` naming the path. The
//! resolve endpoint answers a HEAD for a missing file with 404 and any other request with 403 and an
//! HTML page, so the read must not probe the path as a directory.

#![cfg(test)]
#![cfg(feature = "cloud")]

use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

/// A loopback server: 404 for HEAD, 403 with an HTML body for everything else; logs the request lines.
fn serve() -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let log = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&log);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let seen = Arc::clone(&seen);
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    let request = line.trim().to_owned();
                    loop {
                        let mut h = String::new();
                        if reader.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                            break;
                        }
                    }
                    let head = request.starts_with("HEAD");
                    seen.lock().unwrap().push(request);
                    let reply = if head {
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
                    } else {
                        "HTTP/1.1 403 Forbidden\r\nContent-Type: text/html\r\nContent-Length: 6\r\n\r\n<html>"
                    };
                    if stream.write_all(reply.as_bytes()).is_err() {
                        return;
                    }
                }
            });
        }
    });
    (port, log)
}

#[test]
fn a_missing_hf_file_is_path_not_found_in_any_scheme_case() {
    let (port, log) = serve();
    let s = Session::builder()
        .config("fs.hf.endpoint", format!("http://127.0.0.1:{port}"))
        .build()
        .unwrap();
    for path in [
        "hf://datasets/acme/tiny/missing.parquet",
        "HF://datasets/acme/tiny/missing.parquet",
    ] {
        let err = block_on(s.read().parquet([path]).collect()).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::PathNotFound, "{path}: {err}");
    }
    let seen = log.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(
        seen.iter().all(|r| r.starts_with("HEAD")),
        "a request beyond the HEAD: {seen:?}"
    );
}
