# tessar

The `tessar` binary. `tessar mcp` serves the memory as an
[MCP](https://modelcontextprotocol.io) server on stdio (through a [daemon](#daemon)), so an agent such as
Claude Code can remember facts between conversations and recall them later.

The memory is deterministic and local: typed facts with validity periods and
version history in an append-only journal, recalled by entity, words and
meaning (voyage-4-nano running in-process, on candle). No LLM and no
network are involved.

## Install

```sh
cargo install tessar     # candle embedder, pure Rust: no cmake, no C++
```

The embedding model is not part of the package; see below.

## Build

```sh
cargo build --release -p tessar     # candle embedder, pure Rust
```

The llama.cpp embedder is not part of this binary. It lives in the
`tessar-embed-llama` crate of the repository (needs cmake, a C++ toolchain and
`scripts/setup.sh`) and is used for the comparison tests against candle.

The binary is `target/release/tessar`. It needs the embedding model at
`~/.tessar/models/voyage-4-nano-Q8_0.gguf` (or `$TESSAR_HOME/models/…`, or
the file named by `TESSAR_MODEL`). The candle embedder also needs
`tokenizer.json` from the `voyageai/voyage-4-nano` repository at
`~/.tessar/models/voyage-4-nano/tokenizer.json` (or the file named by
`TESSAR_TOKENIZER`). If the model is missing, the server says
where it expected it and exits with status 1.

## Add to Claude Code

```sh
claude mcp add tessar -- /ABSOLUTE/PATH/target/release/tessar mcp
```

Or in `.mcp.json` (project) / `~/.claude.json` (user):

```json
{
  "mcpServers": {
    "tessar": {
      "command": "/ABSOLUTE/PATH/target/release/tessar",
      "args": ["mcp"]
    }
  }
}
```

`tessar mcp` is a thin proxy: it connects to the daemon, starting it if none
runs. The first start takes a few seconds, because the daemon loads the model
and embeds the facts it has no vector for. Vectors are cached in
`<home>/index/<model>/vectors.bin`, so a later start embeds only new facts.
On macOS the first start of a build may take longer while Metal compiles the
candle kernels (seen once at about 15 s, usually well under a second); the
daemon warms them up at start and logs the time. The proxy exits with status 0
when stdin closes. As many sessions as you like can run at once.

## Daemon

One daemon per machine holds the memory and the model; every session shares
them. Previously a second Claude Code session could not connect at all, since
the journal admits one process.

- **Start.** Automatically by the first `tessar mcp` (detached, in its own
  process group, so it outlives that session), or by hand with
  `tessar daemon [--home DIR] [--idle-exit MINUTES]`, for instance from a
  launchd agent. A second `tessar daemon` on the same home prints `a tessar
  daemon is already running` and exits with status 0 (it looks twice, 50 ms
  apart, since a check by `daemon stop` or a hook holds the journal lock for
  an instant). `--takeover PID` replaces the daemon with that pid instead
  (see *Build id*).
- **Socket.** `$TESSAR_HOME/tessar.sock` (default `~/.tessar/tessar.sock`),
  mode 0600. A stale file from a daemon that died is removed on start.
- **Protocol.** A connection starts with one JSON line from the client,
  `{"tessar": 1, "project": "<name>", "cwd": "<path>", "build": "<id>"}`.
  The daemon answers with one line, `{"tessar": 1, "build": "<id>", "pid":
  <pid>}`, and then both sides speak MCP. The project is computed by the
  proxy as described under *Where data lives*, so each session gets the
  space of the directory it runs in. The project space is created on first
  use.
- **Build id.** Every binary carries `<version>+<git hash>.<build time>`;
  `tessar --version` prints it. A daemon keeps serving the code it was
  started with, so after `cargo build` the proxy compares ids on connect.
  If the daemon's differs, `tessar mcp` starts the replacement from the
  current binary as `tessar daemon --takeover <old pid>` and waits for it to
  listen. The replacement gets in line for the journal lock, then sends the
  old daemon SIGTERM itself and waits up to 20 s for it to let go. It
  signals that pid only if the socket or the pid file says it is the daemon
  of this home; otherwise it signals nobody and just waits. Being in
  line first, it has the journal the moment the old daemon releases it, so a
  daemon that another session's (older) proxy starts in that moment finds
  the journal taken and steps aside. A replacement that waits while another
  daemon already serves (two sessions upgraded together) leaves the line and
  exits with status 0. The daemon's other sessions are closed by the shutdown, and Claude
  Code shows tessar as disconnected in those sessions until you reconnect
  (`/mcp`). `tessar mcp --keep-daemon` never restarts the daemon, for
  debugging. A proxy retires only an older daemon (by build time): one that
  meets a newer daemon, started by another session's upgrade, logs `serving
  through the newer daemon` and uses it. A proxy survives daemon restarts and
  upgrades: when its connection drops it looks for a daemon at once and for
  the next 3 s, and only then starts one itself (the lost daemon is usually
  being replaced, and the replacement must get the journal first). While
  that fails it retries, first after 2 s and then with doubling pauses up to
  30 s, and answers calls made meanwhile with an error for that call only;
  it exits only when its client closes stdin. A daemon it started that
  stepped aside for another process is started again a second later if
  still nothing listens. Hooks never restart anything: on a different build they use the
  daemon as it is and write `daemon build differs` to stderr.
- **Upgrading from a daemon without the handshake.** Such a daemon sends no
  hello, so the proxy and the session hook give up after 5 s with an error
  (the prompt hook after 1 s). Stop it once by hand (`tessar daemon stop`,
  which finds it through its pid file while it holds the journal, or `pkill
  -f 'tessar daemon'`); from then on it restarts itself.
- **`tessar daemon stop`.** Asks the daemon on the socket for its pid, sends
  it SIGTERM and waits for it to go. It says so if none runs. The pid file
  `$TESSAR_HOME/tessar.pid` (written once the daemon holds the journal,
  removed when it ends) is used only when nothing answers on the socket and
  something does hold the journal lock, that is, for a daemon still loading
  its model. With the journal free, a pid file is stale (the daemon was
  killed, or the machine rebooted, and the number may belong to any process
  by now): it is deleted and nothing is signalled.
- **`tessar daemon status`.** Connects with `{"tessar": 1, "status": true}`
  and prints the daemon's build id, pid, uptime and number of open sessions.
  Exits with failure if no daemon answers.
- **Vector cache.** `$TESSAR_HOME/index/voyage-4-nano/vectors.bin` keeps the
  embeddings of facts already seen, so a start embeds only new facts. It is
  derived data: safe to delete, rebuilt on the next start, and if it cannot be
  opened the daemon logs that and runs without it.
- **Log.** `$TESSAR_HOME/daemon.log` when the proxy started it (appended): one
  line when a connection opens and one when it closes.
- **`--idle-exit N`.** Exit when no connection has been open for N minutes.
  The default is to run until stopped.
- **Stopping.** SIGTERM or SIGINT stops accepting, closes the sessions,
  removes the socket, drops the memory and exits with status 0. A connection
  that has sent no header after 5 s is closed. Every write is
  already durable, so killing it loses nothing either.
- **`tessar mcp --standalone`** serves in the process itself, as before: it
  takes the journal lock, loads its own model and cannot run beside a daemon.
  Useful for tests and single-user runs.

The daemon uses Unix sockets, so it is for macOS and Linux.

## Hooks

Hooks put memory in front of the model without the model having to ask.
`recall` and `about` return their hits as structured content next to the
text (`{"hits": [{id, space, recorded, text, kind, similarity, channels,
current}]}`), and `tessar hook` uses it. Hooks are Claude Code settings,
not part of the MCP server: add them to `~/.claude/settings.json` (or use the
`/hooks` menu). `tessar hook config` prints the snippet with the absolute
path of the running binary and writes nothing:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "/ABSOLUTE/PATH/tessar hook prompt", "timeout": 20 }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact",
        "hooks": [
          { "type": "command", "command": "/ABSOLUTE/PATH/tessar hook session", "timeout": 60 }
        ]
      }
    ]
  }
}
```

Both hooks read the hook's JSON from stdin, use the project of its `cwd`, and
talk to the daemon. Whatever goes wrong, they print
nothing, log to stderr and exit with status 0, so a memory that is down never
blocks a prompt.

- **`tessar hook prompt`** (`UserPromptSubmit`). Skips prompts of fewer than
  three words and slash commands. Otherwise it calls `recall` with the prompt
  (limit 8) and keeps the hits whose similarity is at least the threshold
  **or** that the entity channel found (an entity named in the prompt) with a
  similarity of at least the lower entity floor. At most
  `TESSAR_HOOK_ENTITY_MAX` hits pass by the entity rule alone, so a prompt
  that names the project does not pull in every fact linked to it. Facts already shown in this session are
  dropped; the ids printed are appended to `$TESSAR_HOME/hooks/<session_id>.shown`.
  The rest, best first, is cut to the budget and printed, or nothing if no hit
  passes:

  ```
  Memory (tessar) relevant to this message; call `recall` for more:
  - [2026-11-08] [user] User prefers to be talked to in Russian
  ```

  It never holds the prompt up for a daemon: if none listens it starts one
  in the background (unless one is already starting) and prints nothing, and
  the next prompt finds it running. With a daemon it gives up after 1 s, all
  told: finding the project, the handshake and the search.
- **`tessar hook session`** (`SessionStart`). Deletes the session's `.shown`
  file, since the context is fresh, and prints a briefing: a line with the
  counts and the tools, then `About you:` and `About this project:` with the
  newest current facts of each, cut to the budget (each list gets half, and
  what one leaves unused goes to the other). An empty memory gives one line
  saying so and that `remember` saves facts. It starts the daemon if none
  runs and waits up to 60 s for it.
- **`tessar hook config`** prints the snippet above.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TESSAR_HOOK_MIN_SIMILARITY` | `0.45` | Smallest similarity of a hit the `prompt` hook shows, unless the entity rule applies. |
| `TESSAR_HOOK_ENTITY_MIN_SIMILARITY` | `0.35` | Smallest similarity of a hit the entity channel found. |
| `TESSAR_HOOK_ENTITY_MAX` | `2` | Most hits that pass by the entity rule alone. |
| `TESSAR_HOOK_BUDGET` | `1500` | Characters the `prompt` hook may print. |
| `TESSAR_HOOK_SESSION_BUDGET` | `3000` | Characters the `session` hook may print. |

The thresholds 0.45 and 0.35 are first guesses; it is to be calibrated on real prompts
(experiment E13). The hooks use `$TESSAR_HOME` (default `~/.tessar`) for the
socket and the `hooks/` directory, or `--home DIR`.

## Tools

| Tool         | What it does |
|--------------|--------------|
| `remember`   | Writes facts in one transaction: `statement` (a sentence), `attribute` (subject, predicate, value; a single-valued predicate replaces the old value and keeps it in history) or `event` (needs `when`). Reports `created`, `unchanged` or `superseded` per fact. Checks new predicates: snake_case, at most four words, universal, and not a near-duplicate of an existing one; on a problem it writes no facts and says what to do (`synonym_of`, `force_new_predicate`). |
| `recall`     | Natural-language search over both spaces (or one), best first, each hit starting with its handle `fNN` and the channels that found it. `as_of` (`YYYY-MM-DD`) shows what was true on that day. |
| `vocabulary` | Lists the predicates with `single` or `many` cardinality, descriptions and synonyms. |
| `about`      | Lists the facts about one entity, oldest first, each starting with its handle `fNN`; `history: true` includes replaced, closed and corrected versions. |
| `update`     | Says a fact is outdated or wrong, by its handle `fNN` from `recall` or `about`. `reason: changed` closes it as true until `since` (default now), `wrong` marks it as never true, `redact` destroys its content (see Forgetting). With `text`, `value` or `when` it also writes the replacement, keeping the kind (and for an attribute the subject and predicate) and the fields not given. Reports `closed`, `superseded`, `corrected` or `unchanged`. Only a current version can be updated; a closed one is refused and the current version is named. |

Nothing is deleted by `changed` or `wrong`. `update` closes a version or replaces it with a new
one, and the old version stays in the journal: `about` with `history: true`
lists it, and `recall` with `as_of` finds a changed fact on the days it was
true. A fact marked `wrong` is hidden from what was true on any day, since it
never was. For a single-valued attribute with a new value, `remember` with the
same subject and predicate already supersedes the old one; `update` is for the
rest.

Dates are days in the local time zone of the daemon process: the date shown for
a recorded moment, "today" for `past` / `now` / `future`, and the days in `when`
and `as_of`. The journal itself stores UTC milliseconds.

Every fact also mentions the person (`user` space) or the project entity
(project space), unless it is already its subject or mentioned, so `about`
finds statements that name no entity. Explicit `mentions` are kept.

The vocabulary at startup is part of the `remember` tool description, so the
agent sees it without a call. The tool descriptions are the agent's manual:
they explain spaces, fact kinds and predicate rules.

## Forgetting

Three levels of taking a fact out of use:

- **`changed`**: it was true and no longer is. The old version stays, true
  until `since`.
- **`wrong`**: it was never true. It leaves "what was true" but memory still
  shows it knew it.
- **`redact`**: the content is destroyed. Use it for secrets and for personal
  data the person wants gone, not for mistakes.

`update` with `reason: "redact"` takes only `fact`; `text`, `value`, `when` and
`since` are refused. It works on any version, current or closed. The text
leaves the keyword, entity and meaning indexes at once, the mentions and the
source link are cleared, and `recall` and `about` never show the fact again.
With `history: true`, `about` lists the version as `[redacted, version N]` when
other versions of its chain are still about the entity. It cannot be undone.

The text is still in `journal.jsonl`, because the journal is append-only.
`tessar purge [--home DIR]` finishes the job on disk: it stops the daemon if
one runs, rewrites the journal without the redacted content, and deletes the
vector cache (`index/<model>/vectors.bin`, rebuilt on the next start). The
next `tessar mcp` or hook starts the daemon again. It prints one line:

```
purged 2 fact(s) and 0 episode(s); journal 1843 bytes before, 1520 after; vector cache files deleted: 1
```

With nothing to purge it says `nothing to purge: the journal holds no redacted
content (N bytes); vector cache files deleted: M` and still succeeds. When it
stopped a daemon the line starts with `stopped the daemon (it restarts with
the next session); `.

From a shell, `tessar redact f12 f13 [--purge] [--home DIR]` does both steps
for a person: it starts the daemon if needed, redacts each handle through the
same `update` code path and prints the tool's line for each. Every handle is
tried; the exit status is 1 if any failed. With `--purge` it then runs
`tessar purge` and prints its line.

What stays: fact ids, the chain structure and times, entity keys and aliases
(an entity created for a secret keeps its name; redaction does not touch
entities). The episode a fact was taken from keeps its text: redacting the
fact does not blank the source, and there is no tool for that yet. Known
limit from the core: a chain whose every version is purged loses its
(subject, predicate) pair, so the next write of that pair starts a new chain.

## Where data lives

- `$TESSAR_HOME` (or `--home DIR`), default `~/.tessar`: `journal.jsonl` is
  the whole memory; vectors are recomputed on start. `tessar.sock` and
  `daemon.log` are the daemon's.
- Two spaces. `user` holds facts about the person and the entity `user`
  (kind `person`), plus the predicate vocabulary shared by every project. The
  **project space** is named after the repository: the last path component
  of `git remote get-url origin` without `.git`, lowercased (so a second clone
  shares the space; `https://…/owner/repo.git` and `git@host:owner/repo.git`
  both give `repo`). Without an origin it is the name of the main checkout
  (the parent of `git rev-parse --git-common-dir`, so worktrees agree), and
  outside git the basename of the session's working directory.
  Claude Code starts MCP servers in the project directory. `recall` searches
  all spaces by default, including other repositories; each hit shows its
  space.
- Facts are attributed to `claude-code`. Both spaces get the entity kinds
  `person, place, organization, item, project, event, topic, other`.

## Logging

Stdout carries only the protocol. Messages for humans go to stderr.

## Tests

```sh
cargo test -p tessar                                  # no model, no GPU
cargo test -p tessar --test stdio -- --ignored        # real model, GPU
cargo test -p tessar --test daemon_e2e -- --ignored   # real model, GPU
```

The default tests include the daemon and the proxy over a real socket in a
temporary directory, with a fake embedder. The ignored `stdio` test spawns
`tessar mcp --standalone` on a temporary home, runs `initialize`,
`tools/list`, a `remember` and a `recall`, and checks that stdout holds
JSON-RPC lines only. The ignored `daemon_e2e` tests start `tessar daemon`
and two concurrent `tessar mcp` proxies, then the daemon from the first
proxy alone.
