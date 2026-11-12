# tessar-embed-candle

Pure-Rust embedder for [tessar](https://github.com/ab7tr4ct/tessar) on
[candle](https://github.com/huggingface/candle): runs `voyage-4-nano`
(GGUF, Q8_0) in-process, with no cmake and no C++. Implements
`tessar_core::embed::Embedder`. Metal on macOS with the `metal` feature, CPU
otherwise.

The model file and `tokenizer.json` are not part of the crate; see the
[`tessar`](https://crates.io/crates/tessar) README for where to get them.

Most people want the binary: `cargo install tessar`.
