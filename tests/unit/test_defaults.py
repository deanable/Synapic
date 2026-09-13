"""
Tests for Default Configuration Values
======================================

These tests make sure the application starts with sensible defaults for engine
selection, thresholds, and related baseline settings.
"""

from collections import deque


from src.core.session import DatasourceConfig, EngineConfig, Session


def test_default_datasource_and_engine_configs():
    ds = DatasourceConfig()
    eng = EngineConfig()

    # Defaults as defined in the dataclasses
    assert ds.type == "local"
    assert isinstance(ds.local_path, str)
    assert eng.provider == "local"


def test_session_initializes_defaults():
    s = Session()
    assert isinstance(s.datasource, DatasourceConfig)
    assert isinstance(s.engine, EngineConfig)
    # Daminion client starts as None
    assert s.daminion_client is None
    # Basic processing state defaults
    assert s.total_items == 0
    assert s.processed_items == 0
    assert s.failed_items == 0
    # results is a bounded deque (keeps the last 500 entries for export)
    assert isinstance(s.results, deque)
    assert len(s.results) == 0
