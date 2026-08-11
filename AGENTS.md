# Development Rules (pi-python)

## Conversational Style

- Keep answers short and concise.
- No emojis in commits, issues, PR comments, or code.
- No fluff or cheerful filler text.
- Technical prose only, be direct.
- When the user asks a question, answer it first before making edits or running implementation commands.
- When responding to user feedback or an analysis, explicitly say whether you agree or disagree before saying what you changed.

## Code Quality

- Read files in full before wide-ranging changes, before editing files you have not fully inspected, and when asked to investigate or audit. Do not rely on search snippets for broad changes.
- Avoid `Any`; when it is truly necessary, add a comment explaining why.
- Python 3.10+: prefer `dataclass`, `pathlib`, and type annotations. Add `from __future__ import annotations` to every source file.
- Prefer top-level imports. Local imports are allowed only for circular dependencies or lazy loading, with a comment.
- Always ask before removing functionality or code that appears intentional.
- Do not preserve backward compatibility unless the user asks for it.
- Never hardcode key bindings or shortcuts; add them to `DEFAULT_*` constants (e.g. `DEFAULT_APP_KEYBINDINGS`) so they stay configurable.
- Never modify generated artifacts by hand (e.g. model metadata generated files); update the generator script and regenerate.

## Commands

- After code changes (not docs): run `python scripts/check.py` (ruff lint + ruff format + mypy + pytest). Read the full output and fix all errors, warnings, and infos.
- Fast iteration: `pytest -q src/pi_coding_agent/tests/test_x.py -k ...`. If you create or modify a test file, run it and iterate on test or implementation until it passes.
- Tests always use the faux provider. No real provider APIs, keys, or paid tokens.
- For ad-hoc scripts, write them to a temp file, run, edit if needed, and remove when done. Do not embed multi-line scripts in shell commands.
- Never commit unless the user asks.

## Dependency and Install Security

- Treat dependency and `uv.lock` changes as reviewed code. Add dependencies with `uv add` / `uv add --dev`.
- Do not silently bypass pre-commit; never commit with `--no-verify`.

## Git

Multiple pi sessions may be running in this cwd at the same time, each modifying different files. Git operations that touch unstaged, staged, or untracked files outside your own changes will stomp on other sessions' work. Follow these rules:

Committing:

- Only commit files YOU changed in THIS session.
- Stage explicit paths (`git add <path1> <path2>`); never `git add -A` / `git add .`.
- Before committing, run `git status` and verify you are only staging your files.
- Message format: `{feat,fix,docs,test,chore}[(ai,agent,coding-agent,tui,protocol,storage,server,evals)]: <summary>`. Message is informative and concise.

Never run (destroys other agents' work or bypasses checks):

- `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, `git add -A`, `git add .`, `git commit --no-verify`.

If rebase conflicts occur:

- Resolve conflicts only in files you modified.
- If a conflict is in a file you did not modify, abort and ask the user.
- Never force push.

## Repository Structure

- src layout: `pi_ai` / `pi_agent` / `pi_coding_agent` / `pi_tui` / `pi_protocol` / `pi_storage` / `pi_server` / `pi_evals`.
- `pi_tui` is a standalone reusable TUI framework with its own engine (`src/pi_tui/engine/`) and overlay runtime (`src/pi_tui/overlay/`); there is no Textual dependency. Core (model/layout/focus/manager) stays unit-testable without a terminal.
- Docs: `docs/tui.md` (user docs), `docs/nd_upload/tui-ts-feature-gap.md` (TS gap and roadmap); `src/pi_coding_agent/examples/extensions/` 对齐 TS `packages/coding-agent/examples/extensions/`（含 TS 原文与 Python 移植），状态见 `STATUS.md`。
- Tests: `src/pi_tui/tests/` (framework), `src/pi_coding_agent/tests/` (application), plus per-package `tests/` under `src/pi_*`.

## Issues and PRs

This repo has no `CONTRIBUTING.md`. The CI gate is `.github/workflows/ci.yml`: uv sync --frozen → ruff lint/format → mypy → pytest (with a PostgreSQL service).

When reviewing PRs:

- Do not run `gh pr checkout`, `git switch`, or otherwise move the worktree to the PR branch unless the user explicitly asks.
- Use `gh pr view`, `gh pr diff`, `gh api`, and local `git show`/`git diff` against fetched refs to inspect PR metadata, commits, and patches without changing branches.
- If you need PR file contents, fetch/read them into temporary files or use `git show <ref>:<path>` without switching branches.

When posting issue/PR comments:

- Write the comment to a temp file and post with `gh issue/pr comment --body-file` (never multi-line markdown via `--body`).
- Keep comments concise, technical, in the user's tone.
- End every AI-posted comment with the AI-generated disclaimer line specified by the originating prompt.

When closing issues via commit:

- Include `fixes #<number>` or `closes #<number>` in the message so merging auto-closes the issue. For multiple issues, repeat the keyword per issue (`closes #1, closes #2`); a shared keyword (`closes #1, #2`) only closes the first.

## Testing pi Interactive Mode

TUI framework tests run headless via `FakeTerminal` (`src/pi_tui/tests/`). For an interactive smoke test in a real terminal (when tmux is available, e.g. WSL), run from the repo root:

```bash
tmux new-session -d -s pi-test -x 80 -y 24
tmux send-keys -t pi-test "uv run pi --mode tui" Enter
sleep 3 && tmux capture-pane -t pi-test -p     # capture after startup
tmux send-keys -t pi-test "your prompt here" Enter
tmux send-keys -t pi-test Escape               # special keys (also C-o for ctrl+o, etc.)
tmux kill-session -t pi-test
```

On Windows, run `python -m pi_coding_agent --mode tui` directly in a terminal.

## Changelog

Location: `CHANGELOG.md` at the repo root (single changelog; not one per package).

Sections under `## [Unreleased]`: `### Breaking Changes` (API changes requiring migration), `### Added`, `### Changed`, `### Fixed`, `### Removed`.

Rules:

- All new entries go under `## [Unreleased]`. Read the full section first and append to existing subsections; never duplicate them.
- Released version sections (e.g. `## [0.1.0]`) are immutable; never modify them.
- Entries are concise, technical summaries; no attribution links are required in this repo's format.

## Releasing

No automated release pipeline is configured (no npm-style release scripts; the version lives in `pyproject.toml`, currently `0.1.0`). Do not bump versions or publish packages unless the user asks. If asked, update `CHANGELOG.md` under `[Unreleased]`, run `python scripts/check.py`, then bump the version manually.

## User Override

If the user's instructions conflict with any rule in this document, ask for explicit confirmation before overriding. Only then execute their instructions.
