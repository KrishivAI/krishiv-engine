// SPDX-License-Identifier: Apache-2.0
//! The HTTP JWK set source against a loopback HTTP server (`allowInsecureIssuer`): discovery,
//! no redirects, the size cap, location and issuer checks, discovering again after a failure,
//! and the fetch deadline.

use std::io::{BufRead as _, BufReader, Write as _};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::jose::MAX_DOCUMENT_BYTES;
use super::oidc::{FETCH_DEADLINE, HttpKeySource, KeySource};
use super::oidc_config::OidcConfig;

const DISCOVERY: &str = "/.well-known/openid-configuration";
const KEYS: &str = r#"{"keys":[]}"#;

/// One canned answer: status line, extra headers, body; `slow` trickles bytes forever.
#[derive(Clone)]
struct Reply {
    status: &'static str,
    headers: String,
    body: Vec<u8>,
    slow: bool,
}

fn ok(body: impl Into<Vec<u8>>) -> Reply {
    let body = body.into();
    Reply {
        status: "200 OK",
        headers: format!("content-length: {}\r\n", body.len()),
        body,
        slow: false,
    }
}

fn status(status: &'static str, headers: &str) -> Reply {
    Reply {
        status,
        headers: format!("{headers}content-length: 0\r\n"),
        body: Vec::new(),
        slow: false,
    }
}

fn discovery(issuer: &str, jwks_uri: &str) -> Reply {
    ok(format!(
        r#"{{"issuer":"{issuer}","jwks_uri":"{jwks_uri}"}}"#
    ))
}

type Routes = Vec<(&'static str, Reply)>;

/// A loopback server answering by path; records every path asked for.
struct Server {
    base: String,
    hits: Arc<Mutex<Vec<String>>>,
}

impl Server {
    /// Starts a server whose routes are built from its own base address (`http://127.0.0.1:N`).
    fn start(routes: impl FnOnce(&str) -> Routes) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let routes = routes(&base);
        let hits = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&hits);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { return };
                let routes = routes.clone();
                let seen = Arc::clone(&seen);
                std::thread::spawn(move || serve(stream, &routes, &seen));
            }
        });
        Self { base, hits }
    }

    fn issuer(&self) -> String {
        format!("{}/", self.base)
    }

    fn hits(&self) -> Vec<String> {
        self.hits
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

fn serve(stream: TcpStream, routes: &Routes, seen: &Mutex<Vec<String>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let path = line.split(' ').nth(1).unwrap_or_default().to_owned();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
            break;
        }
    }
    seen.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(path.clone());
    let reply = routes
        .iter()
        .find(|(p, _)| *p == path)
        .map_or_else(|| status("404 Not Found", ""), |(_, r)| r.clone());
    let mut out = stream;
    let head = format!(
        "HTTP/1.1 {}\r\n{}connection: close\r\n\r\n",
        reply.status, reply.headers
    );
    if out.write_all(head.as_bytes()).is_err() {
        return;
    }
    if reply.slow {
        // One byte a second, until the client hangs up.
        while out.write_all(b" ").and_then(|()| out.flush()).is_ok() {
            std::thread::sleep(Duration::from_secs(1));
        }
        return;
    }
    out.write_all(&reply.body).ok();
}

fn config(issuer: &str, jwks_uri: Option<&str>) -> OidcConfig {
    let mut c = OidcConfig::default();
    let mut set = |k: &str, v: &str| assert!(c.apply(k, v).unwrap(), "{k}");
    set("krishiv.auth.oidc.issuer", issuer);
    set("krishiv.auth.oidc.audience", "krishiv");
    set("krishiv.auth.oidc.allowInsecureIssuer", "true");
    if let Some(uri) = jwks_uri {
        set("krishiv.auth.oidc.jwksUri", uri);
    }
    c.validate_values().unwrap();
    c
}

/// Runs one fetch on a thread of its own (a blocking client must not run inside a runtime).
fn fetch(source: HttpKeySource) -> (HttpKeySource, Result<Vec<u8>, String>) {
    std::thread::spawn(move || {
        let mut source = source;
        let r = source.fetch();
        (source, r)
    })
    .join()
    .unwrap()
}

/// A source over `server` with `jwksUri` = `path` on it.
fn fixed(server: &Server, path: &str) -> HttpKeySource {
    let uri = format!("{}{path}", server.base);
    HttpKeySource::new(&config(&server.issuer(), Some(&uri)))
}

#[test]
fn discovery_then_key_set() {
    let server = Server::start(|base| {
        vec![
            (
                DISCOVERY,
                discovery(&format!("{base}/"), &format!("{base}/keys")),
            ),
            ("/keys", ok(KEYS)),
        ]
    });
    let (source, body) = fetch(HttpKeySource::new(&config(&server.issuer(), None)));
    assert_eq!(body.unwrap(), KEYS.as_bytes());
    assert_eq!(server.hits(), [DISCOVERY, "/keys"]);
    // The location is kept: the next fetch asks for the keys only.
    let (_, body) = fetch(source);
    assert!(body.is_ok());
    assert_eq!(server.hits(), [DISCOVERY, "/keys", "/keys"]);
}

#[test]
fn the_discovery_issuer_must_match() {
    let server = Server::start(|base| {
        vec![
            (
                DISCOVERY,
                discovery("https://idp.example.test/", &format!("{base}/keys")),
            ),
            ("/keys", ok(KEYS)),
        ]
    });
    let (_, body) = fetch(HttpKeySource::new(&config(&server.issuer(), None)));
    assert!(body.unwrap_err().contains("discovery issuer"));
    assert_eq!(server.hits(), [DISCOVERY]);
}

#[test]
fn redirects_are_not_followed() {
    let server = Server::start(|_| {
        vec![
            ("/moved", status("302 Found", "location: /keys\r\n")),
            ("/keys", ok(KEYS)),
        ]
    });
    let (_, body) = fetch(fixed(&server, "/moved"));
    assert!(body.unwrap_err().contains("302"));
    assert_eq!(server.hits(), ["/moved"]);
}

#[test]
fn documents_over_the_cap_are_refused() {
    let big = vec![b' '; MAX_DOCUMENT_BYTES + 1];
    let server = Server::start(|_| {
        vec![
            ("/declared", ok(big.clone())),
            (
                "/undeclared",
                Reply {
                    headers: String::new(),
                    ..ok(big)
                },
            ),
        ]
    });
    for path in ["/declared", "/undeclared"] {
        let (_, body) = fetch(fixed(&server, path));
        assert!(body.unwrap_err().contains("larger than 1 MiB"), "{path}");
    }
}

#[test]
fn discovered_locations_are_checked() {
    for jwks in [
        "ftp://127.0.0.1/keys",
        "http://user@127.0.0.1/keys",
        "http:///keys",
    ] {
        let server = Server::start(|base| vec![(DISCOVERY, discovery(&format!("{base}/"), jwks))]);
        let (_, body) = fetch(HttpKeySource::new(&config(&server.issuer(), None)));
        assert!(
            body.unwrap_err().contains("refusing the location"),
            "{jwks}"
        );
        assert_eq!(server.hits(), [DISCOVERY], "{jwks}");
    }
    // http:// without the test flag is refused before any request.
    let server = Server::start(|_| vec![("/keys", ok(KEYS))]);
    let mut strict = config(&server.issuer(), Some(&format!("{}/keys", server.base)));
    strict.allow_insecure_issuer = false;
    let (_, body) = fetch(HttpKeySource::new(&strict));
    assert!(body.unwrap_err().contains("refusing the location"));
    assert!(server.hits().is_empty());
}

#[test]
fn a_failed_key_fetch_discovers_again() {
    let server = Server::start(|base| {
        vec![
            (
                DISCOVERY,
                discovery(&format!("{base}/"), &format!("{base}/keys")),
            ),
            ("/keys", status("500 Internal Server Error", "")),
        ]
    });
    let (source, body) = fetch(HttpKeySource::new(&config(&server.issuer(), None)));
    assert!(body.unwrap_err().contains("500"));
    let (_, body) = fetch(source);
    assert!(body.is_err());
    assert_eq!(server.hits(), [DISCOVERY, "/keys", DISCOVERY, "/keys"]);
}

#[test]
fn a_slow_body_ends_at_the_deadline() {
    let server = Server::start(|_| {
        vec![(
            "/keys",
            Reply {
                status: "200 OK",
                headers: String::new(),
                body: Vec::new(),
                slow: true,
            },
        )]
    });
    let start = Instant::now();
    let (_, body) = fetch(fixed(&server, "/keys"));
    let took = start.elapsed();
    let e = body.unwrap_err();
    assert!(e.contains("out of time"), "{e}");
    assert!(
        took >= FETCH_DEADLINE - Duration::from_secs(1)
            && took < FETCH_DEADLINE + Duration::from_secs(4),
        "{took:?}"
    );
}
