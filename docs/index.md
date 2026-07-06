# FORGE Documentation

- [CLI reference](cli.md)
- [Settings](settings.md)
- [Logging](logging.md)
- [Schema reference & versioning policy](schemas.md)

## Package Layout

```
src/forge/          Installable Python package
  cli.py            Typer CLI entry point
  settings.py       pydantic-settings configuration
  logging.py        structlog JSON logging
  schemas/          Versioned Parquet table definitions
tests/              Unit tests mirroring package structure
tests/fixtures/     Bundled CI fixtures (synthetic in Phase 0)
configs/            Pipeline configs (Phase 1+)
docker/             Dockerfile and compose stack
scripts/            Maintenance scripts (e.g. make_fixture.py)
```

## Phase Roadmap

See the [README](../README.md) checklist for implementation status.
