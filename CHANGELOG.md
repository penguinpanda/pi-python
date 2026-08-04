# Changelog

## [Unreleased]

### Added

- `/trust` slash command and CLI startup project-trust resolution (`trust.json` persistence)
- `/changelog` slash command backed by `CHANGELOG.md` parsing
- TUI trust selector and settings selector entry points
- Structured system-prompt builder with project context files (AGENTS.md/CLAUDE.md)
- Turn-level timings and prompt-cache waste statistics in session stats
- Typed SettingsManager with file/in-memory storage, migration, and project-trust gating
- Unified resource loader aggregating skills/prompts/extensions/themes/context files
- Tool-scope constraints against whole-disk searches (read/bash/find/grep guidance)
- Image pipeline (EXIF orientation, resize, multi-format to PNG) wired into read/clipboard

## [0.1.0]

### Added

- Unified LLM API layer (`pi-ai`) with providers, auth, models, and streaming
- Agent core loop (`pi-agent`) with tools, retry, compaction, and branching
- Coding agent (`pi-coding-agent`) with print/RPC/TUI modes and slash commands
- Textual-based TUI (`pi-tui`) with selectors, keybindings, and themes
- Docker development environment and session JSONL persistence
