#!/bin/sh
# Fetches the llama.cpp submodule (needed only to build tessar-mcp with
# `--features llama`; the default candle build does not use it) and checks out only what the build needs:
# ggml, the library sources, headers, cmake files and vendored headers.
# Tools, examples, tests, docs, models and the web UI stay out (about
# 40 MB instead of 185).
set -eu
cd "$(dirname "$0")/.."
git submodule update --init --depth 1 third_party/llama.cpp
git -C third_party/llama.cpp sparse-checkout set --cone ggml src include cmake vendor
echo "llama.cpp ready: $(git -C third_party/llama.cpp describe --tags --always)"
