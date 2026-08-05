"""Ingestion adapters — raw sensor logs to the FORGE Parquet data lake."""

from forge.ingest.nuscenes import IngestResult, ingest_nuscenes

__all__ = ["IngestResult", "ingest_nuscenes"]
