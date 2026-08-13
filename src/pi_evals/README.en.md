# pi-evals — Evaluation Framework

[English](README.en.md) | [中文](README.md)

A full port of TS `packages/evals`: `createPiCodingAgentHarness` (isolated workspaces / transform / output / session snapshots), the vitest-evals equivalent (judge / harness table / artifacts / summary), and the `pi-evals` CLI runner.

## Run

```bash
# CLI model selection or PI_PROVIDER/PI_MODEL env vars; faux by default
uv run pi-evals
```

## Tests

```bash
uv run pytest src/pi_evals/ -v
```

## License

MIT
