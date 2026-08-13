# pi-tui — Terminal UI Framework (Built-in Engine)

[English](README.en.md) | [中文](README.md)

A Python port of [pi-mono/packages/tui](https://github.com/earendil-works/pi-mono): a reusable terminal UI framework with **no Textual dependency** — built-in engine, themes, keybindings, base components, selectors, and clipboard image handling. The application layer (AgentSession binding, slash commands, session switching) lives in `pi_coding_agent.modes.interactive`.

## Structure

```text
pi_tui
├── engine/                        # built-in terminal engine
│   ├── cells.py                   # Cell / Line model + ANSI/OSC8 output
│   ├── text.py                    # Rich renderable → fixed-width Line
│   ├── keys.py                    # UTF-8 / CSI / SS3 / kitty / SGR mouse / paste / OSC
│   ├── terminal.py                # raw / alt-screen / size / line diff / OSC 133/2026/52
│   ├── widgets.py                 # Widget / Container / Input / Editor / ScrollView / SelectList
│   ├── overlay_widget.py          # OverlayWidget (line-text / component dual mode)
│   └── app.py                     # App base (event loop, focus, overlay composition, keybindings)
├── components.py                  # PiHeader / PiChatContainer / PiEditor / PiStatusBar / PiFooter
├── keybindings.py                 # KeybindingsManager + DEFAULT_APP_KEYBINDINGS
├── selectors.py                   # ModelSelector / SessionPicker / TreeSelector / SettingsSelector
├── theme.py                       # ThemeLoader + DARK_THEME / LIGHT_THEME / custom JSON
└── mermaid.py                     # mermaid → Unicode terminal diagrams (flowchart/sequence)
```

## Highlights

- **Rendering**: alt-screen (fullscreen) and main-screen (regular) modes; per-line diff; OSC 2026 synchronized output; hardware cursor
- **Input**: UTF-8 / CSI / SS3 / kitty keyboard protocol / SGR mouse (wheel + scrollbar) / bracketed paste / OSC 52 clipboard / OSC 11 background
- **Model selector** (TS-aligned): `model-id [provider]` layout, ✓ current marker, centered scroll window, `(n/total)` indicator
- **Tree selector**: fold/unfold (ctrl+left/right, `⊞`/`⊟` markers), shift+l label editing
- **Scoped-models selector**: ctrl+a all / ctrl+x clear / ctrl+p provider toggle / alt+up/down reorder / ctrl+s persist
- **Mermaid diagrams**: flowchart TD/LR + sequenceDiagram, `off|final|streaming` modes, width checks, warning display
- **Suspend**: Ctrl+Z (POSIX) — exits alt-screen, ignores SIGINT while suspended, restores on SIGCONT
- **Dialogs**: visible countdown (`auto-cancel in Xs`) on timeout

## Tests

```bash
uv run pytest src/pi_tui/tests/ -v
```

## License

MIT
