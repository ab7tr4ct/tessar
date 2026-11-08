# tessar-mcp

The `tessar` binary. `tessar mcp` serves the memory as an
[MCP](https://modelcontextprotocol.io) server on stdio, so an agent such as
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

The server loads the model before it starts serving, so the first start takes
a few seconds. Opening the memory also embeds every fact again (vectors are not
stored yet), so startup grows with the journal; a vector cache is planned. It exits with status 0 when stdin closes. Only one server can
use a journal at a time: a second one prints `another tessar is using …` and
exits with status 1.

## Tools

| Tool         | What it does |
|--------------|--------------|
| `remember`   | Writes facts in one transaction: `statement` (a sentence), `attribute` (subject, predicate, value; a single-valued predicate replaces the old value and keeps it in history) or `event` (needs `when`). Reports `created`, `unchanged` or `superseded` per fact. Checks new predicates: snake_case, at most four words, universal, and not a near-duplicate of an existing one; on a problem it writes no facts and says what to do (`synonym_of`, `force_new_predicate`). |
| `recall`     | Natural-language search over both spaces (or one), best first, each hit with the channels that found it. `as_of` (`YYYY-MM-DD`) shows what was true on that day. |
| `vocabulary` | Lists the predicates with `single` or `many` cardinality, descriptions and synonyms. |
| `about`      | Lists the facts about one entity, oldest first; `history: true` includes replaced versions. |

The vocabulary at startup is part of the `remember` tool description, so the
agent sees it without a call. The tool descriptions are the agent's manual:
they explain spaces, fact kinds and predicate rules.

## Where data lives

- `$TESSAR_HOME` (or `--home DIR`), default `~/.tessar`: `journal.jsonl` is
  the whole memory; vectors are recomputed on start.
- Two spaces. `user` holds facts about the person and the entity `user`
  (kind `person`), plus the predicate vocabulary shared by every project. The
  **project space** is named after the repository: the last path component
  of `git remote get-url origin` without `.git`, lowercased (so a second clone
  shares the space; `https://…/owner/repo.git` and `git@host:owner/repo.git`
  both give `repo`). Without an origin it is the name of the main checkout
  (the parent of `git rev-parse --git-common-dir`, so worktrees agree), and
  outside git the basename of the server's working directory.
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
```

The ignored test spawns the built binary on a temporary home, runs
`initialize`, `tools/list`, a `remember` and a `recall`, and checks that
stdout holds JSON-RPC lines only.
