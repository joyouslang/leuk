"""Tests for leuk.cli.models — dynamic model catalog and selection dialog."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from leuk.cli.models import run_model_selector
from leuk.providers.catalog import (
    PROVIDER_NAMES,
    _CRED_KEYS,
    fetch_all_available,
    fetch_models,
    has_credentials,
    invalidate_cache,
)


# ------------------------------------------------------------------
# Catalog metadata
# ------------------------------------------------------------------


class TestProviderNames:
    def test_all_expected_providers_present(self):
        for key in ("zen", "anthropic", "openai", "google", "openrouter", "local"):
            assert key in PROVIDER_NAMES, f"Missing display name for {key}"

    def test_all_providers_have_cred_keys(self):
        for key in PROVIDER_NAMES:
            assert key in _CRED_KEYS, f"Missing credential keys for {key}"


# ------------------------------------------------------------------
# has_credentials
# ------------------------------------------------------------------


class TestHasCredentials:
    def test_with_api_key(self):
        assert has_credentials("openai", {"openai_api_key": "sk-test"})

    def test_with_auth_token(self):
        assert has_credentials("anthropic", {"anthropic_auth_token": "tok"})

    def test_missing_creds(self):
        assert not has_credentials("openai", {})

    def test_local_always_available(self):
        assert has_credentials("local", {})

    def test_zen_with_key(self):
        assert has_credentials("zen", {"zen_api_key": "zk-test"})

    def test_zen_without_key(self):
        assert not has_credentials("zen", {})


# ------------------------------------------------------------------
# fetch_models (with mocked HTTP)
# ------------------------------------------------------------------


class TestFetchModels:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        invalidate_cache()
        yield
        invalidate_cache()

    @pytest.mark.asyncio
    async def test_returns_cached_on_second_call(self):
        """Second call should return the cached result without hitting HTTP."""
        from leuk.config import LLMConfig

        config = LLMConfig(openai_api_key="sk-test")
        fake_models = [("gpt-4o", "GPT-4o")]

        with patch(
            "leuk.providers.catalog._fetch_from_provider",
            new_callable=AsyncMock,
            return_value=fake_models,
        ) as mock_fetch:
            first = await fetch_models("openai", config)
            second = await fetch_models("openai", config)

        assert first == fake_models
        assert second == fake_models
        mock_fetch.assert_called_once()  # only fetched once

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        from leuk.config import LLMConfig

        config = LLMConfig()

        with patch(
            "leuk.providers.catalog._fetch_from_provider",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network error"),
        ):
            result = await fetch_models("openai", config)

        assert result == []

    @pytest.mark.asyncio
    async def test_invalidate_clears_cache(self):
        from leuk.config import LLMConfig

        config = LLMConfig(openai_api_key="sk-test")
        fake = [("m1", "Model 1")]

        with patch(
            "leuk.providers.catalog._fetch_from_provider",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mock_fetch:
            await fetch_models("openai", config)
            invalidate_cache("openai")
            await fetch_models("openai", config)

        assert mock_fetch.call_count == 2


# ------------------------------------------------------------------
# fetch_all_available
# ------------------------------------------------------------------


class TestFetchAllAvailable:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        invalidate_cache()
        yield
        invalidate_cache()

    @pytest.mark.asyncio
    async def test_only_fetches_for_credentialed_providers(self):
        from leuk.config import LLMConfig

        config = LLMConfig(openai_api_key="sk-test")
        creds = {"openai_api_key": "sk-test"}

        with patch(
            "leuk.providers.catalog._fetch_from_provider",
            new_callable=AsyncMock,
            return_value=[("gpt-4o", "GPT-4o")],
        ):
            result = await fetch_all_available(config, creds)

        # Should include openai and local (always available)
        assert "openai" in result
        # Should NOT include providers without credentials
        assert "anthropic" not in result
        assert "google" not in result


# ------------------------------------------------------------------
# run_model_selector (dialog mocked)
# ------------------------------------------------------------------


class TestRunModelSelector:
    def _run(self, dialog_return, provider_models=None, **kwargs):
        """Run the selector with a mocked radiolist_dialog."""
        if provider_models is None:
            provider_models = {
                "anthropic": [
                    ("claude-sonnet-4-20250514", "Claude Sonnet 4"),
                    ("claude-opus-4-20250514", "Claude Opus 4"),
                ],
                "openai": [
                    ("gpt-4.1", "GPT-4.1"),
                    ("gpt-4.1-mini", "GPT-4.1 Mini"),
                ],
            }
        defaults = {
            "current_provider": "anthropic",
            "current_model": "claude-sonnet-4-20250514",
            "provider_models": provider_models,
        }
        defaults.update(kwargs)
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = dialog_return
            return run_model_selector(**defaults)

    def test_returns_provider_and_model(self):
        result = self._run("claude-opus-4-20250514")
        assert result == ("anthropic", "claude-opus-4-20250514")

    def test_returns_none_on_cancel(self):
        result = self._run(None)
        assert result is None

    def test_ignores_header_selection(self):
        result = self._run("__header__anthropic")
        assert result is None

    def test_cross_provider_selection(self):
        """Selecting a model from a different provider returns that provider."""
        result = self._run("gpt-4.1")
        assert result == ("openai", "gpt-4.1")

    def test_includes_current_model_not_in_catalog(self):
        """If current model isn't in any catalog, it should still appear."""
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = "my-custom-model"
            result = run_model_selector(
                current_provider="anthropic",
                current_model="my-custom-model",
                provider_models={
                    "anthropic": [("claude-sonnet-4-20250514", "Claude Sonnet 4")],
                },
            )
        assert result == ("anthropic", "my-custom-model")
        call_kwargs = mock_dialog.call_args[1]
        model_ids = [v[0] for v in call_kwargs["values"]]
        assert "my-custom-model" in model_ids

    def test_current_provider_shown_first(self):
        """Current provider's models should be listed first."""
        provider_models = {
            "anthropic": [("claude-sonnet-4-20250514", "Claude Sonnet 4")],
            "openai": [("gpt-4.1", "GPT-4.1")],
        }
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = None
            run_model_selector(
                current_provider="openai",
                current_model="gpt-4.1",
                provider_models=provider_models,
            )
        call_kwargs = mock_dialog.call_args[1]
        values = call_kwargs["values"]
        first_header = values[0][0]
        assert first_header == "__header__openai"

    def test_returns_none_when_no_providers(self):
        result = self._run("anything", provider_models={})
        assert result is None
