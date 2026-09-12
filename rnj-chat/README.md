# RNJ-1 local chat

Run `./rnj-chat/start.sh` from this checkout, then open http://127.0.0.1:8766.
Requires the RNJ-1 GGUF at the path in `rnj-1.rs`, Node.js, Rust, and the AMD GPU/toolchain used by Recipe.

The page streams generated text and keeps the conversation in browser local storage. Each request prefills the complete conversation again. The model context is 32,768 tokens, including the selected reply budget. Requests exceeding that limit return a context-full message; New chat clears the browser conversation. Stop cancels the active generation.

The current background instance runs as the user service `rnj-chat`. Stop it with `systemctl --user stop rnj-chat` before running the launcher manually. Runtime binaries and logs live in `/home/nate/codex/rnj-chat`.

At full context capacity on the 12 GiB GPU, the launcher enables `RECIPE_HOST_SPILL=1`: AMD allocations that exhaust VRAM use GPU-accessible system memory. The short-prompt verification used 4,812 MiB of system memory and generated at about 7.3 tok/s. Long conversations require longer prefill; the 32k-capacity check does not benchmark a filled 32k prompt.
