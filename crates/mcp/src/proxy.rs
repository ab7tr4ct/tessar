//! The stdio side of the daemon: `tessar mcp` serves one session on stdio
//! and forwards every tool call to the daemon through an MCP client.

use std::borrow::Cow;
use std::path::Path;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ErrorData, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerConfig,
};
use rmcp::service::{RequestContext, RunningService};
use rmcp::{Peer, RoleClient, RoleServer, ServerHandler, ServiceError, ServiceExt};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

use crate::daemon::Header;

/// A server whose tools are the daemon's: lists and calls are forwarded
/// verbatim, and `initialize` reports what the daemon reported.
#[derive(Clone)]
pub struct ProxyServer {
    daemon: Peer<RoleClient>,
    info: ServerConfig,
}

impl ProxyServer {
    /// The daemon's peer, for tests.
    pub fn daemon(&self) -> &Peer<RoleClient> {
        &self.daemon
    }
}

/// The daemon connection a [`ProxyServer`] forwards to. Dropping it closes
/// the connection.
pub type DaemonLink = RunningService<RoleClient, ()>;

fn forwarded(e: ServiceError) -> ErrorData {
    match e {
        ServiceError::McpError(e) => e,
        other => ErrorData::internal_error(Cow::Owned(format!("tessar daemon: {other}")), None),
    }
}

impl ServerHandler for ProxyServer {
    fn get_info(&self) -> ServerConfig {
        self.info.clone()
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        self.daemon.list_tools(request).await.map_err(forwarded)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        self.daemon.call_tool_once(request).await.map_err(forwarded)
    }
}

/// Sends the header on `stream`, runs the MCP handshake with the daemon and
/// returns the handler to serve the session with, plus the live link.
pub async fn connect<S>(
    stream: S,
    project: &str,
    cwd: &Path,
) -> Result<(ProxyServer, DaemonLink), String>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    let (reader, mut writer) = tokio::io::split(stream);
    writer
        .write_all(Header::new(project, cwd).line().as_bytes())
        .await
        .and(writer.flush().await)
        .map_err(|e| format!("cannot write the header: {e}"))?;
    let link = ()
        .serve((reader, writer))
        .await
        .map_err(|e| format!("handshake with the daemon failed: {e}"))?;
    let peer = link
        .peer()
        .peer_info()
        .ok_or("the daemon sent no server info")?;
    let mut info = ServerConfig::new(peer.capabilities.clone());
    info.server_info = peer
        .server_info
        .clone()
        .unwrap_or_else(|| Implementation::new("tessar", env!("CARGO_PKG_VERSION")));
    info.instructions.clone_from(&peer.instructions);
    let server = ProxyServer {
        daemon: link.peer().clone(),
        info,
    };
    Ok((server, link))
}

/// Serves `server` on `transport` until the session ends or the daemon
/// goes away. `Ok(true)` means the session ended normally.
pub async fn bridge<T, E, A>(
    server: ProxyServer,
    link: DaemonLink,
    transport: T,
) -> Result<bool, String>
where
    T: rmcp::transport::IntoTransport<RoleServer, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    let session = server
        .serve(transport)
        .await
        .map_err(|e| format!("cannot serve the session: {e}"))?;
    tokio::select! {
        _ = session.waiting() => Ok(true),
        _ = link.waiting() => Ok(false),
    }
}
