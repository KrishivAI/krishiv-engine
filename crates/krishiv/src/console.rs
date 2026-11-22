// SPDX-License-Identifier: Apache-2.0
//! The web console of one session, served from the host process (ADR 0024).

use crate::Session;

/// A running console. Dropping it stops the server; so does closing the session it shows.
///
/// [`Console::url`] is the address to open, with the token; [`Console::addr`] and
/// [`Console::token`] give its parts, and [`Console::is_stopped`] tells whether it still serves.
pub use krishiv_server::session_source::EmbeddedUi as Console;

/// Why a console did not start.
pub use krishiv_server::internal::UiError as ConsoleError;

/// Starts the web console of `session` in this process and returns its handle.
///
/// The console is the one `krishiv server` serves, limited to `session`: its queries and their
/// profiles, a SQL editor that runs in the session itself (it sees the session's temporary views
/// and settings), its catalog and settings, and this process's memory and caches. Another session
/// of the same runtime is not shown.
///
/// It listens on `127.0.0.1` only, on a port the system picks, behind a token generated for this
/// start: [`Console::url`] carries it, so treat the URL as a password for the session. Nothing is
/// printed; show the URL where your program's user will see it. The server runs on a thread of
/// its own, and stops when the handle is dropped or soon after the session is closed.
///
/// ```
/// let session = krishiv::Session::new()?;
/// let console = krishiv::ui(&session).expect("the console starts");
/// assert!(console.url().starts_with("http://127.0.0.1:"));
/// assert!(!console.is_stopped());
/// # Ok::<(), krishiv::Error>(())
/// ```
///
/// # Errors
/// Returns an error when the session is closed or the server cannot start.
pub fn ui(session: &Session) -> Result<Console, ConsoleError> {
    Console::for_session(session, 0)
}

/// [`ui`] on a port of your choice (`0` picks a free one).
///
/// # Errors
/// Returns an error when the session is closed, the port is taken or the server cannot start.
pub fn ui_on_port(session: &Session, port: u16) -> Result<Console, ConsoleError> {
    Console::for_session(session, port)
}
