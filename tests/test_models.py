"""Tests for leuk.cli.models — model catalog and selection dialog."""

from __future__ import annotations

from unittest.mock import patch

from leuk.cli.models import (
    PROVIDER_MODELS,
    get_available_models,
    run_model_selector,
)


# ------------------------------------------------------------------
# Model catalog
# ------------------------------------------------------------------


class TestProviderModels:
    def test_all_providers_have_models(self):
        for key in ("anthropic", "openai", "google", "openrouter", "local"):
            assert key in PROVIDER_MODELS, f"Missing models for {key}"
            assert len(PROVIDER_MODELS[key]) > 0

    def test_model_tuples_are_well_formed(self):
        for provider, models in PROVIDER_MODELS.items():
            for model_id, display_name in models:
                assert isinstance(model_id, str) and model_id, \
                    f"Empty model_id in {provider}"
                assert isinstance(display_name, str) and display_name, \
                    f"Empty display_name in {provider}"

    def test_no_duplicate_model_ids_within_provider(self):
        for provider, models in PROVIDER_MODELS.items():
            ids = [mid for mid, _ in models]
            assert len(ids) == len(set(ids)), \
                f"Duplicate model IDs in {provider}"


# ------------------------------------------------------------------
# get_available_models
# ------------------------------------------------------------------


class TestGetAvailableModels:
    def test_returns_models_with_api_key(self):
        creds = {"openai_api_key": "sk-test"}
        result = get_available_models("openai", creds)
        assert len(result) > 0
        assert all(isinstance(m, tuple) and len(m) == 2 for m in result)

    def test_returns_models_with_auth_token(self):
        creds = {"anthropic_auth_token": "tok"}
        result = get_available_models("anthropic", creds)
        assert len(result) > 0

    def test_returns_empty_without_creds(self):
        result = get_available_models("openai", {})
        assert result == []

    def test_local_always_available(self):
        result = get_available_models("local", {})
        assert len(result) > 0

    def test_unknown_provider_returns_empty(self):
        result = get_available_models("nonexistent", {"nonexistent_api_key": "k"})
        assert result == []


# ------------------------------------------------------------------
# run_model_selector (dialog mocked)
# ------------------------------------------------------------------


class TestRunModelSelector:
    def _run(self, dialog_return, **kwargs):
        """Run the selector with a mocked radiolist_dialog."""
        defaults = {
            "current_provider": "anthropic",
            "current_model": "claude-sonnet-4-20250514",
            "creds": {"anthropic_auth_token": "tok"},
        }
        defaults.update(kwargs)
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = dialog_return
            return run_model_selector(**defaults)

    def test_returns_selected_model(self):
        result = self._run("claude-opus-4-20250514")
        assert result == "claude-opus-4-20250514"

    def test_returns_none_on_cancel(self):
        result = self._run(None)
        assert result is None

    def test_ignores_header_selection(self):
        result = self._run("__header__anthropic")
        assert result is None

    def test_still_shows_local_without_creds(self):
        """Even with no credentials, 'local' models are always available."""
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = "llama3.1:8b"
            result = run_model_selector(
                current_provider="nonexistent",
                current_model="",
                creds={},
            )
        assert result == "llama3.1:8b"
        call_kwargs = mock_dialog.call_args[1]
        model_ids = [v[0] for v in call_kwargs["values"]]
        assert "__header__local" in model_ids

    def test_includes_current_model_not_in_catalog(self):
        """If current model isn't in any catalog, it should still appear."""
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = "my-custom-model"
            result = run_model_selector(
                current_provider="anthropic",
                current_model="my-custom-model",
                creds={"anthropic_auth_token": "tok"},
            )
        assert result == "my-custom-model"
        # Verify the dialog was called with values containing the custom model
        call_kwargs = mock_dialog.call_args[1]
        model_ids = [v[0] for v in call_kwargs["values"]]
        assert "my-custom-model" in model_ids

    def test_shows_multiple_providers(self):
        """When multiple providers have creds, all their models appear."""
        creds = {
            "anthropic_auth_token": "tok",
            "openai_api_key": "sk-test",
        }
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = "gpt-4.1"
            result = run_model_selector(
                current_provider="anthropic",
                current_model="claude-sonnet-4-20250514",
                creds=creds,
            )
        assert result == "gpt-4.1"
        call_kwargs = mock_dialog.call_args[1]
        model_ids = [v[0] for v in call_kwargs["values"]]
        # Should have models from both providers
        assert any("claude" in m for m in model_ids)
        assert any("gpt" in m for m in model_ids)

    def test_current_provider_shown_first(self):
        """Current provider's models should be listed first."""
        creds = {
            "anthropic_auth_token": "tok",
            "openai_api_key": "sk-test",
        }
        with patch("leuk.cli.models.radiolist_dialog") as mock_dialog:
            mock_dialog.return_value.run.return_value = None
            run_model_selector(
                current_provider="openai",
                current_model="gpt-4.1",
                creds=creds,
            )
        call_kwargs = mock_dialog.call_args[1]
        values = call_kwargs["values"]
        # First non-header entry should be from openai
        first_header = values[0][0]
        assert first_header == "__header__openai"
