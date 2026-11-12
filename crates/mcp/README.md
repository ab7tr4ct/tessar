# tessar

A memory for AI agents, served to Claude Code over
[MCP](https://modelcontextprotocol.io). The agent writes typed facts
(statements, attributes, events) with validity periods, and every
version is kept in an append-only journal; it recalls them in later
conversations by entity, by words and by meaning, and each hit says why
it was found. The memory is local and deterministic: no LLM and no
network are involved, and the embeddings come from `voyage-4-nano`
running in the same process, in pure Rust on candle. Tested on macOS
with Apple Silicon; Linux is untested. The API is not stable.

## Install

```sh
cargo install --locked tessar
tessar setup
```

Rust 1.89 or newer, `curl` and the Claude Code CLI (`claude`) are
needed. `tessar setup` does three things, in this order, and ends each
with one line (done, in place, would, skipped or FAILED):

1. Downloads the embedding model (about 375 MB) and its tokenizer
   (about 7 MB) into `~/.tessar/models` with `curl`, from Hugging Face
   files pinned to a commit and checked against SHA-256 sums.
2. Registers this binary as the MCP server `tessar` in the user scope of
   Claude Code. It asks first.
3. Merges four hooks into `~/.claude/settings.json`, leaving the rest of
   the file as it is, after copying the old file to
   `settings.json.tessar-backup-<UTC time>` beside it. It asks first.

Options: `--yes` answers yes to the questions (without a terminal the
answer is no, and the command to run by hand is printed), `--dry-run`
changes nothing, `--no-models`, `--no-mcp` and `--no-hooks` skip a step,
and `--allow-unverified` accepts a model or tokenizer file whose SHA-256
differs from the pinned one. `--home DIR` is accepted only together with
`--no-mcp --no-hooks`, because the registered server and hooks use the
default home. The exit status is 0 when every requested step is done or
was in place, 1 when one failed and 2 when some were left to you. Run it
again at any time: every step is idempotent. Afterwards restart Claude
Code, or run `/mcp` in it.

### By hand

```sh
mkdir -p ~/.tessar/models/voyage-4-nano
curl -L --fail -o ~/.tessar/models/voyage-4-nano-Q8_0.gguf \
  https://huggingface.co/MongoDB/voyage-4-nano-GGUF/resolve/03547551689253732502486d62de4cd0970d6564/voyage-4-nano-Q8_0.gguf
curl -L --fail -o ~/.tessar/models/voyage-4-nano/tokenizer.json \
  https://huggingface.co/MongoDB/voyage-4-nano/resolve/5fb34e870f49f19e79c94d5e118f5ee730133279/tokenizer.json

claude mcp add --scope user tessar -- "$(command -v tessar)" mcp
tessar hook config      # prints the hooks; merge them into ~/.claude/settings.json
```

The SHA-256 sums that `setup` checks are
`c6ed6cc898d50ae1a4492809affadc6c0d023f8ac05c07a37abe20c563152560` for
the model and
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` for
the tokenizer.

### Undo

```sh
tessar daemon stop                    # if a daemon is running
claude mcp remove tessar --scope user
rm -r ~/.tessar                       # the memory itself, journal included
cargo uninstall tessar
```

Then take the four tessar hooks (`PreToolUse`, `SessionStart`,
`SubagentStart`, `UserPromptSubmit`) out of `~/.claude/settings.json`, or
put back the `settings.json.tessar-backup-<UTC time>` file that `setup`
left, which also drops any later change to the settings. If
`CLAUDE_CONFIG_DIR` is set, Claude Code's files are in that directory
instead of `~/.claude`.

## Tools

The tool descriptions are the agent's manual.

- `remember`: writes facts in one transaction. A fact is a `statement`, an
  `attribute` (subject, predicate, value; a single-valued predicate
  replaces the old value and keeps it in history) or an `event` (needs
  `when`). New predicates are checked: short, universal, not a duplicate
  of an existing one.
- `lesson`: records a lesson from a mistake, to be shown later when a
  prompt resembles it.
- `recall`: searches in natural language, best first, with the channels
  that found each hit; `as_of` (`YYYY-MM-DD`) shows what was true on that
  day.
- `vocabulary`: lists the predicates with their cardinality, descriptions
  and synonyms.
- `about`: lists the facts about one entity; `history: true` includes
  replaced, closed and corrected versions.
- `update`: marks a fact, by its handle `fNN`, as `changed` (it was true
  and no longer is), `wrong` (it never was) or `redact` (its content is
  destroyed), optionally writing the replacement.

Nothing is deleted by `changed` or `wrong`: the old version stays in the
journal. `redact` destroys the content in memory at once; because the
journal is append-only, the text stays in `journal.jsonl` until
`tessar purge` (or `tessar redact fNN... --purge`) rewrites the file.

## Hooks

Hooks are Claude Code settings, not part of the MCP server; `setup`
installs them and `tessar hook config` prints them. Whatever goes wrong,
a hook prints nothing and exits with status 0, so a memory that is down
never blocks a prompt.

- `prompt` (`UserPromptSubmit`): prints the facts and lessons that fit
  the user's message.
- `session` (`SessionStart`): prints a short summary of what memory holds.
- `agent` (`PreToolUse` on the `Agent` tool) and `subagent`
  (`SubagentStart`): give a subagent the memory that fits its brief. The
  first leaves the selection in a short-lived file in `~/.tessar/hooks/`,
  the second hands it to the agent.

The hooks keep a local decision log, `~/.tessar/decisions.jsonl`: what
they showed or held back and which tools were called afterwards. It holds
the text of messages, briefs and `recall` queries (each cut to 2000
characters) and the error text of failed tool calls, never the texts of
facts. It is never sent anywhere, and `tessar purge` removes it.

## Commands

As `tessar --help` shows them:

```
tessar mcp [--home DIR] [--standalone] [--keep-daemon]
tessar daemon [--home DIR] [--idle-exit MINUTES] [--takeover PID]
tessar daemon stop|status [--home DIR]
tessar redact fNN... [--purge] [--home DIR]
tessar lesson "TEXT" [--space project|user] [--home DIR]
tessar purge [--home DIR]
tessar hook prompt|session|agent|subagent [--home DIR]
tessar hook config
tessar setup [--yes] [--dry-run] [--no-models] [--no-mcp] [--no-hooks]
             [--allow-unverified] [--home DIR]
tessar --version
tessar --help
```

`tessar mcp` serves the memory on stdio as a proxy to a daemon, which it
starts when none runs and replaces when it is another build than the
binary. One daemon per machine holds the memory and the model, so any
number of sessions can share them; `--standalone` serves in the process
itself instead. The daemon listens on `tessar.sock` in the home. The
first start takes a few seconds, while the model loads and the facts
without a vector are embedded.

## Environment variables

| Variable | Meaning |
|---|---|
| `TESSAR_HOME` | The home directory, default `~/.tessar`; `--home DIR` overrides it. |
| `TESSAR_MODEL` | The model file, instead of `<home>/models/voyage-4-nano-Q8_0.gguf`. |
| `TESSAR_TOKENIZER` | The tokenizer file, instead of `<home>/models/voyage-4-nano/tokenizer.json`. |
| `TESSAR_EMBEDDER` | `candle`, the only embedder in this binary; `llama` is an error. |
| `TESSAR_HOOK_MIN_SIMILARITY` | Smallest similarity of a hit the `prompt` hook shows, default `0.45`. |
| `TESSAR_HOOK_ENTITY_MIN_SIMILARITY` | Smallest similarity of a hit found through an entity named in the prompt, default `0.35`. |
| `TESSAR_HOOK_ENTITY_MAX` | Most hits that pass by the entity rule alone, default `2`. |
| `TESSAR_HOOK_BUDGET` | Characters the `prompt` hook may print, default `1500`. |
| `TESSAR_HOOK_SESSION_BUDGET` | Characters the `session` hook may print, default `3000`. |

## Where data lives

In the home, `~/.tessar` by default (created `0700`, files `0600`):

- `journal.jsonl` is the whole memory.
- `index/<fingerprint>/vectors.bin` is the vector cache. The fingerprint
  covers the model file, the tokenizer, the embedding settings and a
  revision of the embedding code, so a change in any of them starts a
  fresh cache. It is derived data: safe to delete, rebuilt on the next
  start.
- `models/` holds the model and the tokenizer.
- `tessar.sock`, `tessar.pid` and `daemon.log` belong to the daemon;
  `hooks/` and `decisions.jsonl` to the hooks.

There are two kinds of space. `user` holds facts about the person and the
predicate vocabulary shared by every project. The project space is named
after the repository: the last path component of `git remote get-url
origin` without `.git`, lowercased; without an origin, the name of the
main checkout; outside git, the name of the working directory. `recall`
searches all spaces by default and shows each hit's space.

## More

The data model, the results, the library crates and the source are in the
repository: <https://github.com/ab7tr4ct/tessar>.
