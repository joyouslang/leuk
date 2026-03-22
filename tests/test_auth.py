"""Tests for the Anthropic OAuth PKCE flow in leuk.cli.auth."""

from __future__ import annotations

import base64
import hashlib
from unittest.mock import MagicMock, patch

import httpx
import pytest

from leuk.cli.auth import (
    PROVIDERS,
    _ANTHROPIC_AUTHORIZE_URL,
    _ANTHROPIC_CLIENT_ID,
    _ANTHROPIC_REDIRECT_URI,
    _ANTHROPIC_SCOPES,
    _ANTHROPIC_TOKEN_URL,
    _build_authorize_url,
    _credential_summary,
    _exchange_code_for_token,
    _generate_pkce,
    _generate_state,
    _has_credentials,
    _mask_key,
    _refresh_access_token,
    refresh_anthropic_token,
)
from leuk.config import LLMConfig
from leuk.providers import create_provider
from leuk.providers.base import NoCredentialsError


# ------------------------------------------------------------------
# Pure helper tests
# ------------------------------------------------------------------


class TestMaskKey:
    def test_short_key(self):
        assert _mask_key("abcd") == "****"
        assert _mask_key("12345678") == "****"

    def test_long_key(self):
        assert _mask_key("sk-ant-abcdefghij") == "...ghij"

    def test_empty_key(self):
        assert _mask_key("") == "****"


class TestGeneratePkce:
    def test_returns_two_strings(self):
        verifier, challenge = _generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_verifier_length(self):
        verifier, _ = _generate_pkce()
        # secrets.token_urlsafe(64) => ~86 chars, within RFC 7636 43-128
        assert 43 <= len(verifier) <= 128

    def test_challenge_is_s256_of_verifier(self):
        verifier, challenge = _generate_pkce()
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_unique(self):
        """Each call should produce a different verifier."""
        v1, _ = _generate_pkce()
        v2, _ = _generate_pkce()
        assert v1 != v2


class TestGenerateState:
    def test_returns_string(self):
        s = _generate_state()
        assert isinstance(s, str)
        assert len(s) > 20

    def test_unique(self):
        s1 = _generate_state()
        s2 = _generate_state()
        assert s1 != s2


# ------------------------------------------------------------------
# URL builder
# ------------------------------------------------------------------


class TestBuildAuthorizeUrl:
    def test_contains_required_params(self):
        url = _build_authorize_url(code_challenge="CHALLENGE", state="STATE123")
        assert url.startswith(_ANTHROPIC_AUTHORIZE_URL)
        assert f"client_id={_ANTHROPIC_CLIENT_ID}" in url
        assert "response_type=code" in url
        assert "code_challenge=CHALLENGE" in url
        assert "code_challenge_method=S256" in url
        assert "state=STATE123" in url
        assert "code=true" in url

    def test_redirect_uri_is_correct(self):
        url = _build_authorize_url(code_challenge="C", state="S")
        # URL-encoded version of the redirect URI
        assert "platform.claude.com" in url

    def test_scope_included(self):
        url = _build_authorize_url(code_challenge="C", state="S")
        # Scopes are URL-encoded (spaces become +)
        assert "user%3Ainference" in url or "user:inference" in url


# ------------------------------------------------------------------
# Token exchange (mocked HTTP)
# ------------------------------------------------------------------


class TestExchangeCodeForToken:
    def test_sends_json_body(self):
        """Token exchange must POST JSON (not form-encoded) to the token URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "at_123",
            "refresh_token": "rt_456",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("leuk.cli.auth.httpx.Client") as MockClient:
            client_instance = MagicMock()
            client_instance.post.return_value = mock_response
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = client_instance

            result = _exchange_code_for_token("AUTH_CODE", "VERIFIER", "STATE")

        assert result == {"access_token": "at_123", "refresh_token": "rt_456"}

        # Verify it was called with json= (not data=)
        call_kwargs = client_instance.post.call_args
        assert call_kwargs[1]["json"]["grant_type"] == "authorization_code"
        assert call_kwargs[1]["json"]["code"] == "AUTH_CODE"
        assert call_kwargs[1]["json"]["code_verifier"] == "VERIFIER"
        assert call_kwargs[1]["json"]["state"] == "STATE"
        assert call_kwargs[1]["json"]["client_id"] == _ANTHROPIC_CLIENT_ID
        assert call_kwargs[1]["json"]["redirect_uri"] == _ANTHROPIC_REDIRECT_URI
        assert "data" not in call_kwargs[1]

    def test_posts_to_correct_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "tok"}
        mock_response.raise_for_status = MagicMock()

        with patch("leuk.cli.auth.httpx.Client") as MockClient:
            client_instance = MagicMock()
            client_instance.post.return_value = mock_response
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = client_instance

            _exchange_code_for_token("CODE", "VERIFIER", "STATE")

        url_arg = client_instance.post.call_args[0][0]
        assert url_arg == _ANTHROPIC_TOKEN_URL

    def test_raises_on_http_error(self):
        with patch("leuk.cli.auth.httpx.Client") as MockClient:
            client_instance = MagicMock()
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401", request=MagicMock(), response=MagicMock()
            )
            client_instance.post.return_value = resp
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.HTTPStatusError):
                _exchange_code_for_token("CODE", "V", "S")


# ------------------------------------------------------------------
# Token refresh (mocked HTTP)
# ------------------------------------------------------------------


class TestRefreshAccessToken:
    def test_sends_json_refresh_request(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_at",
            "refresh_token": "new_rt",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("leuk.cli.auth.httpx.Client") as MockClient:
            client_instance = MagicMock()
            client_instance.post.return_value = mock_response
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = client_instance

            result = _refresh_access_token("old_rt_token")

        assert result["access_token"] == "new_at"

        call_kwargs = client_instance.post.call_args
        assert call_kwargs[1]["json"]["grant_type"] == "refresh_token"
        assert call_kwargs[1]["json"]["refresh_token"] == "old_rt_token"
        assert call_kwargs[1]["json"]["client_id"] == _ANTHROPIC_CLIENT_ID
        assert call_kwargs[1]["json"]["scope"] == _ANTHROPIC_SCOPES


# ------------------------------------------------------------------
# Public refresh_anthropic_token (mocked credentials + HTTP)
# ------------------------------------------------------------------


class TestRefreshAnthropicToken:
    def test_returns_none_without_refresh_token(self):
        with patch("leuk.cli.auth.load_credentials", return_value={"anthropic_auth_token": "at_old"}):
            result = refresh_anthropic_token()
        assert result is None

    def test_refreshes_and_saves_new_token(self):
        creds = {
            "anthropic_auth_token": "at_old",
            "anthropic_refresh_token": "rt_old",
        }

        with (
            patch("leuk.cli.auth.load_credentials", return_value=dict(creds)),
            patch(
                "leuk.cli.auth._refresh_access_token",
                return_value={"access_token": "at_new", "refresh_token": "rt_new"},
            ),
            patch("leuk.cli.auth.save_credentials") as mock_save,
        ):
            result = refresh_anthropic_token()

        assert result == "at_new"
        saved = mock_save.call_args[0][0]
        assert saved["anthropic_auth_token"] == "at_new"
        assert saved["anthropic_refresh_token"] == "rt_new"

    def test_returns_none_on_token_refresh_failure(self):
        with (
            patch(
                "leuk.cli.auth.load_credentials",
                return_value={"anthropic_refresh_token": "rt_old"},
            ),
            patch(
                "leuk.cli.auth._refresh_access_token",
                side_effect=httpx.HTTPError("fail"),
            ),
        ):
            result = refresh_anthropic_token()
        assert result is None

    def test_returns_none_if_response_missing_access_token(self):
        with (
            patch(
                "leuk.cli.auth.load_credentials",
                return_value={"anthropic_refresh_token": "rt_old"},
            ),
            patch(
                "leuk.cli.auth._refresh_access_token",
                return_value={"error": "bad_grant"},
            ),
        ):
            result = refresh_anthropic_token()
        assert result is None

    def test_keeps_old_refresh_if_not_rotated(self):
        """If the server doesn't return a new refresh_token, keep the old one."""
        with (
            patch(
                "leuk.cli.auth.load_credentials",
                return_value={
                    "anthropic_auth_token": "at_old",
                    "anthropic_refresh_token": "rt_old",
                },
            ),
            patch(
                "leuk.cli.auth._refresh_access_token",
                return_value={"access_token": "at_new"},
            ),
            patch("leuk.cli.auth.save_credentials") as mock_save,
        ):
            result = refresh_anthropic_token()

        assert result == "at_new"
        saved = mock_save.call_args[0][0]
        assert saved["anthropic_auth_token"] == "at_new"
        # Original refresh token is preserved since server didn't rotate it
        assert saved["anthropic_refresh_token"] == "rt_old"


# ------------------------------------------------------------------
# Constants correctness
# ------------------------------------------------------------------


class TestOAuthConstants:
    """Verify that the hardcoded OAuth constants match the expected values."""

    def test_client_id(self):
        assert _ANTHROPIC_CLIENT_ID == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

    def test_authorize_url(self):
        assert _ANTHROPIC_AUTHORIZE_URL == "https://claude.ai/oauth/authorize"

    def test_token_url(self):
        assert _ANTHROPIC_TOKEN_URL == "https://platform.claude.com/v1/oauth/token"

    def test_redirect_uri(self):
        assert _ANTHROPIC_REDIRECT_URI == "https://platform.claude.com/oauth/code/callback"

    def test_scopes(self):
        assert "user:profile" in _ANTHROPIC_SCOPES
        assert "user:inference" in _ANTHROPIC_SCOPES
        # org:create_api_key is not needed — bearer tokens work directly
        assert "org:create_api_key" not in _ANTHROPIC_SCOPES


# ------------------------------------------------------------------
# Auth menu helpers
# ------------------------------------------------------------------


class TestHasCredentials:
    def test_has_api_key(self):
        assert _has_credentials({"anthropic_api_key": "sk-ant-123"}, "anthropic")

    def test_has_auth_token(self):
        assert _has_credentials({"anthropic_auth_token": "tok"}, "anthropic")

    def test_missing(self):
        assert not _has_credentials({}, "anthropic")

    def test_empty_string(self):
        assert not _has_credentials({"openai_api_key": ""}, "openai")


class TestCredentialSummary:
    def test_api_key(self):
        result = _credential_summary({"openai_api_key": "sk-test12345678"}, "openai")
        assert "API key" in result
        assert "5678" in result  # last 4 chars

    def test_auth_token(self):
        result = _credential_summary({"anthropic_auth_token": "token12345678"}, "anthropic")
        assert "auth token" in result

    def test_not_configured(self):
        result = _credential_summary({}, "google")
        assert "not configured" in result


class TestProvidersList:
    def test_all_providers_present(self):
        keys = [k for k, _ in PROVIDERS]
        assert "anthropic" in keys
        assert "openai" in keys
        assert "google" in keys
        assert "openrouter" in keys
        assert "local" in keys


# ------------------------------------------------------------------
# NoCredentialsError and credential validation
# ------------------------------------------------------------------


class TestNoCredentialsError:
    def test_error_message(self):
        err = NoCredentialsError("anthropic")
        assert err.provider == "anthropic"
        assert "anthropic" in str(err)
        assert "/auth" in str(err)

    def test_is_runtime_error(self):
        assert issubclass(NoCredentialsError, RuntimeError)


class TestCreateProviderValidation:
    def test_anthropic_no_creds_raises(self):
        cfg = LLMConfig(provider="anthropic")
        with pytest.raises(NoCredentialsError) as exc_info:
            create_provider(cfg)
        assert exc_info.value.provider == "anthropic"

    def test_anthropic_with_api_key_ok(self):
        cfg = LLMConfig(provider="anthropic", anthropic_api_key="sk-ant-test")
        # Should not raise
        provider = create_provider(cfg)
        assert provider is not None

    def test_anthropic_with_auth_token_ok(self):
        """auth_token alone is valid — used directly with oauth-2025-04-20 beta."""
        cfg = LLMConfig(provider="anthropic", anthropic_auth_token="tok")
        provider = create_provider(cfg)
        assert provider is not None

    def test_openai_no_creds_raises(self):
        cfg = LLMConfig(provider="openai")
        with pytest.raises(NoCredentialsError) as exc_info:
            create_provider(cfg)
        assert exc_info.value.provider == "openai"

    def test_google_no_creds_raises(self):
        cfg = LLMConfig(provider="google")
        with pytest.raises(NoCredentialsError) as exc_info:
            create_provider(cfg)
        assert exc_info.value.provider == "google"

    def test_openrouter_no_creds_raises(self):
        cfg = LLMConfig(provider="openrouter")
        with pytest.raises(NoCredentialsError) as exc_info:
            create_provider(cfg)
        assert exc_info.value.provider == "openrouter"

    def test_local_no_creds_ok(self):
        """Local provider (ollama) should work without explicit credentials."""
        cfg = LLMConfig(provider="local")
        provider = create_provider(cfg)
        assert provider is not None

    def test_unknown_provider_raises_value_error(self):
        cfg = LLMConfig(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider(cfg)
