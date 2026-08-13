# pi-storage — PostgreSQL Session Storage

[English](README.en.md) | [中文](README.md)

PostgreSQL-backed session storage (asyncpg + migrations + tsvector/pg_trgm search) with a PostgreSQL v4 session backend: lanes / records / lane_moves / facts / branch cache / session_stats / writer lease (TTL 30s + heartbeat + fence).

## Setup

```bash
docker compose -f docker/compose.yaml up -d pg
export PI_PG_DSN="postgresql://pi:pi@127.0.0.1:5432/pi"
```

## Tests

```bash
uv run pytest src/pi_storage/tests/ -v
```

## License

MIT
