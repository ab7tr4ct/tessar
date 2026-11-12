//! Which `tessar` MCP servers Claude Code has registered, read from its own
//! file, `~/.claude.json` (`$CLAUDE_CONFIG_DIR/.claude.json` when that is
//! set). `claude mcp get` would tell too, but it connects to the server to
//! report its status, which starts `tessar mcp`; a setup must not do that.
//!
//! Only the user scope (`mcpServers`) and the local scope of the current
//! directory (`projects[<cwd>].mcpServers`) are looked at. The file is only
//! ever read; changes go through `claude mcp add` and `claude mcp remove`.

use std::fs;
use std::io;
use std::path::Path;

use serde_json::Value;

use super::SERVER;

/// A registered server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Registration {
    /// `user` or `local`.
    pub scope: &'static str,
    /// `None` for a server that is not a local command (http, sse).
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Lookup {
    /// The file could not be read or understood.
    Unknown(String),
    Found(Vec<Registration>),
}

impl Registration {
    /// Registered as `<exe> mcp`.
    pub fn is_setup_of(&self, exe: &Path) -> bool {
        let Some(command) = &self.command else {
            return false;
        };
        let command = Path::new(command);
        let same = command == exe || command.canonicalize().is_ok_and(|c| c == exe);
        same && self.args == ["mcp"]
    }

    pub fn describe(&self) -> String {
        match &self.command {
            Some(command) => {
                let args = self.args.join(" ");
                format!("`{command} {args}` in the {} scope", self.scope)
            }
            None => format!(
                "a server that is not a local command, in the {} scope",
                self.scope
            ),
        }
    }

    pub fn remove_args(&self) -> Vec<&str> {
        vec!["mcp", "remove", SERVER, "--scope", self.scope]
    }

    /// The `claude mcp add` that registers it again.
    pub fn add_args(&self) -> Option<Vec<String>> {
        let command = self.command.clone()?;
        let mut args: Vec<String> = ["mcp", "add", "--scope", self.scope, SERVER]
            .map(str::to_owned)
            .into();
        for (name, value) in &self.env {
            args.extend(["-e".to_owned(), format!("{name}={value}")]);
        }
        args.push("--".to_owned());
        args.push(command);
        args.extend(self.args.iter().cloned());
        Some(args)
    }
}

pub(super) fn read(path: &Path, cwd: &Path) -> Lookup {
    let shown = path.display();
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        // Claude Code has never run: nothing is registered.
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Lookup::Found(Vec::new()),
        Err(e) => return Lookup::Unknown(format!("cannot read {shown}: {e}")),
    };
    let json: Value = match serde_json::from_str(&text) {
        Ok(json) => json,
        Err(e) => return Lookup::Unknown(format!("{shown} is not valid JSON ({e})")),
    };
    let user = json.get("mcpServers").and_then(|m| m.get(SERVER));
    let local = json
        .get("projects")
        .and_then(|p| p.get(cwd.to_string_lossy().as_ref()))
        .and_then(|p| p.get("mcpServers"))
        .and_then(|m| m.get(SERVER));
    let mut found = Vec::new();
    for (scope, server) in [("user", user), ("local", local)] {
        let Some(server) = server else { continue };
        match parse(scope, server) {
            Some(r) => found.push(r),
            None => {
                return Lookup::Unknown(format!(
                    "the `{SERVER}` entry in the {scope} scope of {shown} has a shape I do not know"
                ));
            }
        }
    }
    Lookup::Found(found)
}

fn parse(scope: &'static str, server: &Value) -> Option<Registration> {
    let server = server.as_object()?;
    let stdio = server
        .get("type")
        .is_none_or(|t| t.as_str() == Some("stdio"));
    let command = match server.get("command") {
        Some(Value::String(c)) if stdio => Some(c.clone()),
        Some(Value::String(_)) | None => None,
        Some(_) => return None,
    };
    let args = match server.get("args") {
        None => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect::<Option<_>>()?,
        Some(_) => return None,
    };
    let env = match server.get("env") {
        None => Vec::new(),
        Some(Value::Object(o)) => o
            .iter()
            .map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_owned())))
            .collect::<Option<_>>()?,
        Some(_) => return None,
    };
    Some(Registration {
        scope,
        command,
        args,
        env,
    })
}
