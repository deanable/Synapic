"""
Tests for Session.validate_engine
=================================

These tests verify the engine-configuration validation that gates the start of
a processing job: the provider must be the local-only 'local' provider, a model
must be selected, and local models must be downloaded.
"""

from unittest.mock import patch


from src.core.session import Session


def _session_with(**engine_overrides):
    s = Session()
    for key, value in engine_overrides.items():
        setattr(s.engine, key, value)
    return s


def test_missing_model_id_fails():
    s = _session_with(provider="local", model_id="")
    assert s.validate_engine() is False


def test_unknown_provider_fails():
    s = _session_with(provider="made_up", model_id="some/model")
    assert s.validate_engine() is False


def test_non_local_provider_fails():
    """Synapic tags images with local models only; cloud providers are gone
    and any non-local provider is invalid."""
    for provider in ("huggingface", "openrouter", "groq_package", "cerebras", "ollama"):
        s = _session_with(provider=provider, model_id="some/model")
        assert s.validate_engine() is False


def test_local_requires_downloaded_model():
    s = _session_with(provider="local", model_id="some/model")

    with patch("src.core.huggingface_utils.is_model_downloaded", return_value=False):
        assert s.validate_engine() is False

    with patch("src.core.huggingface_utils.is_model_downloaded", return_value=True):
        assert s.validate_engine() is True