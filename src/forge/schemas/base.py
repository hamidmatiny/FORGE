"""Base schema types and Parquet round-trip helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

TRecord = TypeVar("TRecord", bound=BaseModel)


class SchemaVersion(BaseModel):
    """Semantic version tag for a data-lake table schema."""

    model_config = ConfigDict(frozen=True)

    major: int
    minor: int

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}"


class BaseTable(ABC, Generic[TRecord]):
    """Pydantic record model paired with an explicit PyArrow schema."""

    schema_version: ClassVar[SchemaVersion]
    table_name: ClassVar[str]
    record_model: ClassVar[type[TRecord]]

    @classmethod
    @abstractmethod
    def arrow_schema(cls) -> pa.Schema:
        """Return the canonical PyArrow schema for this table."""

    @classmethod
    def to_arrow(cls, records: list[TRecord]) -> pa.Table:
        """Convert Pydantic records to a PyArrow table."""
        if not records:
            return pa.Table.from_pylist([], schema=cls.arrow_schema())
        rows = [record.model_dump(mode="python") for record in records]
        table = pa.Table.from_pylist(rows)
        return table.cast(cls.arrow_schema())

    @classmethod
    def from_arrow(cls, table: pa.Table) -> list[TRecord]:
        """Convert a PyArrow table to validated Pydantic records."""
        validated = table.cast(cls.arrow_schema())
        rows: list[dict[str, object]] = validated.to_pylist()
        return [cls.record_model.model_validate(row) for row in rows]

    @classmethod
    def write_parquet(cls, records: list[TRecord], path: str) -> None:
        """Write records to a Parquet file with embedded schema metadata."""
        table = cls.to_arrow(records)
        metadata = table.schema.metadata or {}
        enriched = {
            **metadata,
            b"forge.table": cls.table_name.encode(),
            b"forge.schema_version": str(cls.schema_version).encode(),
        }
        table = table.replace_schema_metadata(enriched)
        pq.write_table(table, path)  # type: ignore[no-untyped-call]

    @classmethod
    def read_parquet(cls, path: str) -> list[TRecord]:
        """Read and validate records from a Parquet file."""
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        return cls.from_arrow(table)
