# pi-coding-agent — Coding Agent CLI

[English](README.en.md) | [中文](README.md)

The application layer that assembles `pi_agent` + `pi_ai` into a usable coding agent: CLI (print / RPC / TUI), configuration, trust, session persistence, extensions, package management, and the remote model catalog overlay.

## Quick Start

```bash
# one-shot print mode
uv run python -m pi_coding_agent -p "read README.md and summarize it"

# TUI interactive mode (default when stdin is a TTY and no message is given)
uv run python -m pi_coding_agent --mode tui

# RPC mode (stdin/stdout JSONL)
uv run python -m pi_coding_agent --mode rpc
```

## CLI

### Sessions

```bash
--no-session                       # don't persist
--session <file>                   # open a session file
-c / --continue                    # continue the most recent session
--fork <path|id>                   # fork a session into a new one
--resume                           # interactive session picker
--session-id <id>                  # exact project session id
--session-dir <dir>                # session storage directory
--name <name>                      # name a newly created session
```

### Modes

```bash
--mode print|text|json|rpc|tui     # text/json are TS-compatible aliases
--json                             # JSON Lines output in print mode
```

### Package management

```bash
pi install <source> [-l]          # npm:/git:@ref/local dir; -l writes project config
pi remove <source> [-l]
pi update [<source>]              # reinstall configured sources
pi update --models                # force-refresh model catalogs
pi list                           # user/project grouped package listing
pi config [-l]                    # read-only package config view
```

Sources install into `~/.pi/agent/packages` (user scope) or `.pi/packages` (project scope, trust-gated). Persisted in `settings.packages` (global/project split).

### Auth

```bash
pi login <provider>
pi logout <provider>
pi auth list
pi auth print-api-key --model <model>
pi auth print-bearer-token --model <model>
```

## Modes

### Print (default)

One-shot query with plain-text or `--json` event output; session header as the first JSONL record; SIGTERM/SIGHUP handled with conventional exit codes.

### RPC (`--mode rpc`)

Headless stdin/stdout JSONL with 32 commands (prompt / steer / follow_up / abort / bash / new_session / get_tree / ...). `prompt` emits success only after preflight succeeds (TS `preflightResult` semantics); failures before preflight emit error responses.

### TUI (`--mode tui`)

Built-in engine interface: slash commands, model/tree/settings/scoped-models selectors (centered scroll window + `(n/total)` indicator), mermaid terminal diagrams (`off|final|streaming`), startup changelog notice, Ctrl+Z suspend (POSIX), usage cost breakdown in `/session`.

## Remote model catalog overlay

Built-in static providers get a pi.dev catalog overlay (`remote_catalog_provider.py`): ETag conditional requests, 4-hour freshness window, connection failures also record `checkedAt` (offline startup stays fast after the first attempt).

## Tests

```bash
uv run pytest src/pi_coding_agent/tests/ -v
```

## License

MIT
