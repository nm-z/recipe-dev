#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p /home/nate/codex/rnj-chat
cargo build --release --lib
rustc --edition=2024 -O rnj-1.rs --extern recipe=target/release/librecipe.rlib -L dependency=target/release/deps -o /home/nate/codex/rnj-chat/rnj-chat
exec node rnj-chat/server.mjs
