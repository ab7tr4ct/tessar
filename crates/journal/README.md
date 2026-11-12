# tessar-journal

The file journal of [tessar](https://github.com/ab7tr4ct/tessar): an
append-only log of commits in JSON Lines, one CRC-32 per line, fsync per
commit. Implements `tessar_core::log::Log`.

A torn last line is cut off on open; damage anywhere else fails the open with
the line number. One process holds the journal at a time. The format is
described in the crate documentation.

Most people want the binary: `cargo install tessar`.
