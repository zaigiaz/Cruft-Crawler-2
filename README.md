# Cruft-Crawler
Cruft Crawler is an LLM-first background agent that runs entirely offline from a 64 GB USB drive. It profiles the filesystem slowly over time with imperceptible CPU load.
It uses a local quantized LLM to help recommend safe deletions, and delivers a concise AI-generated report.

# TODO 

## main functionality
- [ ] TODO: parse config file to get settings data
- [ ] TODO: cpu scheduling with LLM

## crawler actor
- [ ] TODO: Batching and Time-Based sending of Data
- [ ] TODO: write function to compare hashes and then add that change to 'is_dupe' flag

## db_manager actor
- [ ] TODO: Make Function for Write-Ahead Log
- [ ] TODO: use sled_view to see database
- [ ] TODO: create DB schema for Sled

## LLM actor
- [ ] TODO: Linear Regression Model or No? 
- [ ] TODO: make Max's llama code actor compliant
- [ ] TODO: port over Max's llama actor
- [ ] TODO: figure out pre-prompts and style
- [ ] TODO: find limitations of Context Window

## UI actor
- [ ] Figure out inteface layout, and learn Tauri
- [ ] Tauri or QT application
- [ ] how to integrate LLM into Tauri window

## Stretch Goals && Misc 
- [ ] <5% CPU usage
- [ ] implement niceness and priority scheduling
- [ ] set config file usage
- [ ] screensaver API (X11 or Win32 api)

## Crates needed
- filetime: https://docs.rs/filetime/latest/filetime/ (cross-platform time-dates)
- Walkdir: https://docs.rs/walkdir/latest/walkdir/ (for recursive traversal of file-system)
- sled: https://docs.rs/sled/latest/sled/ (for long term storage)
- steady-state: https://docs.rs/steady_state/latest/steady_state/ (project architecture)
- llama-cpp: https://docs.rs/llama_cpp/latest/llama_cpp/ (interact with llama-cpp bindings for LLM)
- SHA-2: https://docs.rs/sha2/latest/sha2/ (hash the contents of files for preformant storage)
- hex: https://docs.rs/hex/latest/hex/ (decode hash from bytes into string)
- serde: serialization of struct into u8 bytes
