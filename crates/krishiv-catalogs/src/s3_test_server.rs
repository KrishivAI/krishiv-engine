// SPDX-License-Identifier: Apache-2.0
//! A loopback S3 stand-in for tests (vended credentials, design P3-T067 §4.1): path-style objects in memory behind
//! real HTTP/1.1 over TCP, a thread per connection, `Connection: close` on every answer. It takes `PUT` (with
//! `If-None-Match: *`), `GET` (whole or one `Range`), `HEAD` and `DELETE`; anything else is a 501 the test sees.
//! It checks no signature: it records the SigV4 access key id (`Credential=<id>/`) of every request, so a test can
//! say which credential each request was signed with. `deny` makes a key id get 403 `AccessDenied`, `expire` 403
//! `ExpiredToken` (what S3 answers for a revoked or expired session credential; the store retries it once).

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

/// One request as the stand-in saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Seen {
    /// `GET`, `PUT`, ...
    pub method: String,
    /// `bucket/key`, without the query.
    pub object: String,
    /// The SigV4 access key id, or `None` for an unsigned request.
    pub key_id: Option<String>,
}

#[derive(Debug, Default)]
struct State {
    objects: BTreeMap<String, Vec<u8>>,
    seen: Vec<S3Seen>,
    denied: BTreeSet<String>,
    expired: BTreeSet<String>,
}

/// A running stand-in ([`serve_s3`]); it serves until the process ends. Clones share it.
#[derive(Debug, Clone)]
pub struct S3Server {
    /// `http://127.0.0.1:<port>`, the `fs.s3a.endpoint`.
    pub endpoint: String,
    state: Arc<Mutex<State>>,
}

impl S3Server {
    /// Every request since the start or the last [`Self::clear_seen`], in arrival order.
    #[must_use]
    pub fn seen(&self) -> Vec<S3Seen> {
        self.state.lock().unwrap().seen.clone()
    }

    /// Forgets the requests seen so far.
    pub fn clear_seen(&self) {
        self.state.lock().unwrap().seen.clear();
    }

    /// Requests signed with `key_id` get 403 from now on.
    pub fn deny(&self, key_id: &str) {
        self.state.lock().unwrap().denied.insert(key_id.to_owned());
    }

    /// Requests signed with `key_id` get 403 `ExpiredToken` from now on (all but HEAD, see `handle`).
    pub fn expire(&self, key_id: &str) {
        self.state.lock().unwrap().expired.insert(key_id.to_owned());
    }

    /// The access key ids the requests since the last [`Self::clear_seen`] were signed with.
    #[must_use]
    pub fn key_ids(&self) -> BTreeSet<Option<String>> {
        self.seen().into_iter().map(|s| s.key_id).collect()
    }

    /// The `bucket/key` of every object held.
    #[must_use]
    pub fn objects(&self) -> Vec<String> {
        self.state.lock().unwrap().objects.keys().cloned().collect()
    }
}

struct Request {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn key_id(&self) -> Option<String> {
        let auth = self.header("authorization")?;
        let rest = &auth[auth.find("Credential=")? + "Credential=".len()..];
        Some(rest[..rest.find('/')?].to_owned())
    }
}

fn read_request(stream: &TcpStream) -> Option<Request> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_owned();
    let target = parts.next()?.to_owned();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).ok()?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':')?;
        headers.push((k.trim().to_ascii_lowercase(), v.trim().to_owned()));
    }
    let len = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0; len];
    reader.read_exact(&mut body).ok()?;
    Some(Request {
        method,
        target,
        headers,
        body,
    })
}

/// `bytes=a-b`, `bytes=a-` or `bytes=-n` over an object of `len` bytes, as a half-open range.
fn range(spec: &str, len: usize) -> Option<(usize, usize)> {
    let (a, b) = spec.strip_prefix("bytes=")?.split_once('-')?;
    let (start, end) = if a.is_empty() {
        let n: usize = b.parse().ok()?;
        (len.saturating_sub(n), len)
    } else {
        let start: usize = a.parse().ok()?;
        let end = if b.is_empty() {
            len
        } else {
            b.parse::<usize>().ok()?.saturating_add(1).min(len)
        };
        (start, end)
    };
    (start < end).then_some((start, end))
}

const MODIFIED: &str = "Wed, 18 Nov 2026 00:00:00 GMT";

fn answer(stream: &mut TcpStream, status: &str, headers: &[(String, String)], body: &[u8]) {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn object_headers(bytes: &[u8]) -> Vec<(String, String)> {
    vec![
        ("ETag".to_owned(), format!("\"{:x}\"", fingerprint(bytes))),
        ("Last-Modified".to_owned(), MODIFIED.to_owned()),
        ("Accept-Ranges".to_owned(), "bytes".to_owned()),
    ]
}

fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn handle(mut stream: TcpStream, state: &Mutex<State>) {
    let Some(r) = read_request(&stream) else {
        return;
    };
    let object = r
        .target
        .split('?')
        .next()
        .unwrap_or_default()
        .trim_start_matches('/')
        .to_owned();
    let key_id = r.key_id();
    let mut st = state.lock().unwrap();
    st.seen.push(S3Seen {
        method: r.method.clone(),
        object: object.clone(),
        key_id: key_id.clone(),
    });
    // A HEAD answer has no body, so it cannot name `ExpiredToken`: real S3 answers a bare 403 there, which the
    // store retries once with a fresh credential (`RefreshingStore`).
    if key_id.as_ref().is_some_and(|k| st.expired.contains(k)) {
        drop(st);
        if r.method == "HEAD" {
            answer(&mut stream, "403 Forbidden", &[], b"");
            return;
        }
        let body =
            b"<Error><Code>ExpiredToken</Code><Message>The provided token has expired.</Message></Error>";
        answer(&mut stream, "403 Forbidden", &[], body);
        return;
    }
    if key_id.as_ref().is_some_and(|k| st.denied.contains(k)) {
        drop(st);
        let body = b"<Error><Code>AccessDenied</Code><Message>denied</Message></Error>";
        answer(&mut stream, "403 Forbidden", &[], body);
        return;
    }
    match r.method.as_str() {
        "PUT" => {
            if r.header("if-none-match") == Some("*") && st.objects.contains_key(&object) {
                drop(st);
                answer(&mut stream, "412 Precondition Failed", &[], b"");
                return;
            }
            let headers = object_headers(&r.body);
            st.objects.insert(object, r.body);
            drop(st);
            answer(&mut stream, "200 OK", &headers, b"");
        }
        "GET" | "HEAD" => {
            let Some(bytes) = st.objects.get(&object).cloned() else {
                drop(st);
                let body = b"<Error><Code>NoSuchKey</Code><Message>no</Message></Error>";
                answer(&mut stream, "404 Not Found", &[], body);
                return;
            };
            drop(st);
            let mut headers = object_headers(&bytes);
            if r.method == "HEAD" {
                // Content-Length is the object's; no body follows.
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
                    bytes.len(),
                    headers
                        .iter()
                        .map(|(k, v)| format!("{k}: {v}\r\n"))
                        .collect::<String>()
                );
                let _ = stream.write_all(head.as_bytes());
                return;
            }
            match r.header("range").and_then(|s| range(s, bytes.len())) {
                Some((start, end)) => {
                    headers.push((
                        "Content-Range".to_owned(),
                        format!("bytes {start}-{}/{}", end - 1, bytes.len()),
                    ));
                    answer(
                        &mut stream,
                        "206 Partial Content",
                        &headers,
                        &bytes[start..end],
                    );
                }
                None => answer(&mut stream, "200 OK", &headers, &bytes),
            }
        }
        "DELETE" => {
            st.objects.remove(&object);
            drop(st);
            answer(&mut stream, "204 No Content", &[], b"");
        }
        _ => {
            drop(st);
            answer(&mut stream, "501 Not Implemented", &[], b"");
        }
    }
}

/// Starts a stand-in on a loopback port that serves until the process ends.
///
/// # Panics
/// When no loopback port can be bound (test code only).
#[must_use]
pub fn serve_s3() -> S3Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let state = Arc::new(Mutex::new(State::default()));
    let shared = Arc::clone(&state);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let state = Arc::clone(&shared);
            thread::spawn(move || handle(stream, &state));
        }
    });
    S3Server { endpoint, state }
}

/// `krishiv_connectors_iceberg::testing::write_table`'s table written under `warehouse` (an `s3://bucket/...` URL) on
/// `s3`, every request signed with the access key id `key_id`. Call it off any async runtime: the writer runs one of
/// its own.
///
/// # Panics
/// When the write fails (test code only).
#[must_use]
pub fn write_s3_table(
    s3: &S3Server,
    warehouse: &str,
    key_id: &str,
) -> krishiv_connectors_iceberg::testing::WrittenTable {
    let mut settings = BTreeMap::new();
    for (k, v) in [
        ("fs.s3a.endpoint", s3.endpoint.as_str()),
        ("fs.s3a.region", "us-east-1"),
        ("fs.s3a.access.key", key_id),
        ("fs.s3a.secret.key", "writer-secret"),
    ] {
        settings.insert(k.to_owned(), v.to_owned());
    }
    krishiv_connectors_iceberg::testing::write_table_with(warehouse, move |sched| {
        std::sync::Arc::new(krishiv_exec::files::ObjectPaths::new(
            krishiv_store::StoreId::new("file://"),
            "/",
            Arc::clone(sched),
            Arc::new(krishiv_store::StoreRegistry::new(Arc::new(settings))),
        ))
    })
    .unwrap()
}

#[cfg(test)]
#[test]
fn ranges() {
    assert_eq!(range("bytes=0-3", 10), Some((0, 4)));
    assert_eq!(range("bytes=4-", 10), Some((4, 10)));
    assert_eq!(range("bytes=-3", 10), Some((7, 10)));
    assert_eq!(range("bytes=8-20", 10), Some((8, 10)));
    assert_eq!(range("bytes=10-", 10), None);
}
