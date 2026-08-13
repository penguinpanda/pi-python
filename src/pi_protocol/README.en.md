# pi-protocol — Wire Protocol v2

[English](README.en.md) | [中文](README.md)

Pydantic schemas + JSONL framing for the pi protocol v2: Command / Result / Snapshot / Progress / Error.

```python
from pi_protocol import Command, Result, Snapshot, Progress, Error
```

## Tests

```bash
uv run pytest src/pi_protocol/tests/ -v
```

## License

MIT
