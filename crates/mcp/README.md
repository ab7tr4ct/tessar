# tessar-mcp

The `tessar` binary. `tessar mcp` serves the memory as an
[MCP](https://modelcontextprotocol.io) server on stdio (through a [daemon](#daemon)), so an agent such as
Claude Code can remember facts between conversations and recall them later.

The memory is deterministic and local: typed facts with validity periods and
version history in an append-only journal, recalled by entity, words and
meaning (voyage-4-nano running in-process through llama.cpp). No LLM and no
network are involved.

## Build

```sh
scripts/setup.sh                        # llama.cpp submodule, sparse checkout
cargo build --release -p tessar-mcp     # needs cmake and a C++ toolchain
```

The binary is `target/release/tessar`. It needs the embedding model at
`~/.tessar/models/voyage-4-nano-Q8_0.gguf` (or `$TESSAR_HOME/models/…`, or
the file named by `TESSAR_MODEL`). If the model is missing, the server says
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
and embeds every fact again (vectors are not stored yet), so startup grows
with the journal; a vector cache is planned. The proxy exits with status 0
when stdin closes. As many sessions as you like can run at once.

## Daemon

One daemon per machine holds the memory and the model; every session shares
them. Previously a second Claude Code session could not connect at all, since
the journal admits one process.

- **Start.** Automatically by the first `tessar mcp` (detached, in its own
  process group, so it outlives that session), or by hand with
  `tessar daemon [--home DIR] [--idle-exit MINUTES]`, for instance from a
  launchd agent. A second `tessar daemon` on the same home prints `a tessar
  daemon is already running` and exits with status 0.
- **Socket.** `$TESSAR_HOME/tessar.sock` (default `~/.tessar/tessar.sock`),
  mode 0600. A stale file from a daemon that died is removed on start.
- **Protocol.** A connection starts with one JSON line,
  `{"tessar": 1, "project": "<name>", "cwd": "<path>"}`, then speaks MCP. The
  project is computed by the proxy as described under *Where data lives*, so
  each session gets the space of the directory it runs in. The project space
  is created on first use.
- **Log.** `$TESSAR_HOME/daemon.log` when the proxy started it (appended): one
  line when a connection opens and one when it closes.
- **`--idle-exit N`.** Exit when no connection has been open for N minutes.
  The default is to run until stopped.
- **Stopping.** SIGTERM or SIGINT stops accepting, closes the sessions, drops
  the memory, removes the socket and exits with status 0. Every write is
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
talk to the daemon (starting it if none runs). Whatever goes wrong, they print
nothing, log to stderr and exit with status 0, so a memory that is down never
blocks a prompt.

- **`tessar hook prompt`** (`UserPromptSubmit`). Skips prompts of fewer than
  three words and slash commands. Otherwise it calls `recall` with the prompt
  (limit 8) and keeps the hits whose similarity is at least the threshold
  **or** that the entity channel found, since an entity named in the prompt is
  a strong signal whatever the cosine. Facts already shown in this session are
  dropped; the ids printed are appended to `$TESSAR_HOME/hooks/<session_id>.shown`.
  The rest, best first, is cut to the budget and printed, or nothing if no hit
  passes:

  ```
  Memory (tessar) relevant to this message; call `recall` for more:
  - [2026-11-08] [user] User prefers to be talked to in Russian
  ```

  It gives up after 15 s (a daemon that has to load the model first is not
  waited for; the next prompt finds it running).
- **`tessar hook session`** (`SessionStart`). Deletes the session's `.shown`
  file, since the context is fresh, and prints a briefing: a line with the
  counts and the tools, then `About you:` and `About this project:` with the
  newest current facts of each, cut to the budget (each list gets half, and
  what one leaves unused goes to the other). An empty memory gives one line
  saying so and that `remember` saves facts. It waits up to 60 s for a daemon.
- **`tessar hook config`** prints the snippet above.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TESSAR_HOOK_MIN_SIMILARITY` | `0.45` | Smallest similarity of a hit the `prompt` hook shows, unless the entity channel found it. |
| `TESSAR_HOOK_BUDGET` | `1500` | Characters the `prompt` hook may print. |
| `TESSAR_HOOK_SESSION_BUDGET` | `3000` | Characters the `session` hook may print. |

The threshold 0.45 is a first guess; it is to be calibrated on real prompts
(experiment E13). The hooks use `$TESSAR_HOME` (default `~/.tessar`) for the
socket and the `hooks/` directory, or `--home DIR`.

## Tools

| Tool         | What it does |
|--------------|--------------|
| `remember`   | Writes facts in one transaction: `statement` (a sentence), `attribute` (subject, predicate, value; a single-valued predicate replaces the old value and keeps it in history) or `event` (needs `when`). Reports `created`, `unchanged` or `superseded` per fact. Checks new predicates: snake_case, at most four words, universal, and not a near-duplicate of an existing one; on a problem it writes no facts and says what to do (`synonym_of`, `force_new_predicate`). |
| `recall`     | Natural-language search over both spaces (or one), best first, each hit starting with its handle `fNN` and the channels that found it. `as_of` (`YYYY-MM-DD`) shows what was true on that day. |
| `vocabulary` | Lists the predicates with `single` or `many` cardinality, descriptions and synonyms. |
| `about`      | Lists the facts about one entity, oldest first, each starting with its handle `fNN`; `history: true` includes replaced, closed and corrected versions. |
| `update`     | Says a fact is outdated or wrong, by its handle `fNN` from `recall` or `about`. `reason: changed` closes it as true until `since` (default now), `wrong` marks it as never true. With `text`, `value` or `when` it also writes the replacement, keeping the kind (and for an attribute the subject and predicate) and the fields not given. Reports `closed`, `superseded`, `corrected` or `unchanged`. Only a current version can be updated; a closed one is refused and the current version is named. |

Nothing is ever deleted. `update` closes a version or replaces it with a new
one, and the old version stays in the journal: `about` with `history: true`
lists it, and `recall` with `as_of` finds a changed fact on the days it was
true. A fact marked `wrong` is hidden from what was true on any day, since it
never was. For a single-valued attribute with a new value, `remember` with the
same subject and predicate already supersedes the old one; `update` is for the
rest.

Every fact also mentions the person (`user` space) or the project entity
(project space), unless it is already its subject or mentioned, so `about`
finds statements that name no entity. Explicit `mentions` are kept.

The vocabulary at startup is part of the `remember` tool description, so the
agent sees it without a call. The tool descriptions are the agent's manual:
they explain spaces, fact kinds and predicate rules.

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

Stdout carries only the protocol. Messages for humans, including llama.cpp's
errors, go to stderr. Set `TESSAR_LLAMA_LOG=1` to see all llama.cpp output.

## Tests

```sh
cargo test -p tessar-mcp                                  # no model, no GPU
cargo test -p tessar-mcp --test stdio -- --ignored        # real model, GPU
cargo test -p tessar-mcp --test daemon_e2e -- --ignored   # real model, GPU
```

The default tests include the daemon and the proxy over a real socket in a
temporary directory, with a fake embedder. The ignored `stdio` test spawns
`tessar mcp --standalone` on a temporary home, runs `initialize`,
`tools/list`, a `remember` and a `recall`, and checks that stdout holds
JSON-RPC lines only. The ignored `daemon_e2e` tests start `tessar daemon`
and two concurrent `tessar mcp` proxies, then the daemon from the first
proxy alone.
