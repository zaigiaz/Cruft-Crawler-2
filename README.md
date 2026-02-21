# Cruft-Crawler
Cruft Crawler is an LLM-first background agent that runs entirely offline from a 64 GB USB drive. It profiles the filesystem slowly over time with imperceptible CPU load.
It uses a local quantized LLM to help recommend safe deletions, and delivers a concise AI-generated report.

# AI models used / tested
- https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

## Crates needed
- filetime: https://docs.rs/filetime/latest/filetime/ (cross-platform time-dates)
- Walkdir: https://docs.rs/walkdir/latest/walkdir/ (for recursive traversal of file-system)
- sled: https://docs.rs/sled/latest/sled/ (for long term storage)
- steady-state: https://docs.rs/steady_state/latest/steady_state/ (project architecture)
- llama-cpp: https://docs.rs/llama_cpp/latest/llama_cpp/ (interact with llama-cpp bindings for LLM)
- SHA-2: https://docs.rs/sha2/latest/sha2/ (hash the contents of files for preformant storage)
- hex: https://docs.rs/hex/latest/hex/ (decode hash from bytes into string)
- serde: serialization of struct into u8 bytes
- encoding-rs: for LLM
- anyhow: for LLM
- llama-cpp-2: for LLM

