// SPDX-License-Identifier: Apache-2.0
//! A fake HTTP/JSON catalog server for tests (Iceberg REST, Unity): real HTTP/1.1 over TCP on a loopback port, one connection at a time,
//! `Connection: close` on every answer. Each request is recorded (method, target, headers, body) and answered by a
//! script, which can also stall, trickle bytes, redirect or send a body the client must refuse.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// One request as the server saw it.
#[derive(Debug, Clone)]
pub(crate) struct Seen {
    pub(crate) method: String,
    /// Path and query, as sent.
    pub(crate) target: String,
    /// Header names lower-cased.
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) body: String,
}

impl Seen {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// An answer.
pub(crate) enum Reply {
    /// A status and a JSON body.
    Json(u16, String),
    /// Raw bytes written as they are (head and body).
    Raw(Vec<u8>),
    /// Say nothing for this long, then close.
    Stall(Duration),
    /// A 200 head, then the body one byte every `Duration`.
    Trickle(String, Duration),
}

pub(crate) struct Server {
    pub(crate) base: String,
    pub(crate) seen: Arc<Mutex<Vec<Seen>>>,
}

impl Server {
    pub(crate) fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn read_request(stream: &TcpStream) -> Option<Seen> {
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
    Some(Seen {
        method,
        target,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        302 => "Found",
        401 => "Unauthorized",
        404 => "Not Found",
        _ => "Status",
    }
}

/// Starts a server answering with `script` until the test ends.
pub(crate) fn serve(mut script: impl FnMut(&Seen) -> Reply + Send + 'static) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let Some(request) = read_request(&stream) else {
                continue;
            };
            log.lock().unwrap().push(request.clone());
            match script(&request) {
                Reply::Json(status, body) => {
                    let head = format!(
                        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        reason(status),
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body.as_bytes());
                }
                Reply::Raw(bytes) => {
                    let _ = stream.write_all(&bytes);
                }
                Reply::Stall(d) => thread::sleep(d),
                Reply::Trickle(body, every) => {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    for b in body.bytes() {
                        if stream.write_all(&[b]).is_err() {
                            break;
                        }
                        let _ = stream.flush();
                        thread::sleep(every);
                    }
                }
            }
            let _ = stream.flush();
        }
    });
    Server { base, seen }
}
